#!/usr/bin/env bash
# 训练看护（阶段4收尾自动化）：
#   1. 监控在跑的 jab / guard_idle 训练进程；
#   2. 某个训练退出后：从其 run 目录取最新 ONNX → 提取部署到
#      vendor/policy/boxing_<clip>.* → 逐向量对齐 + 单机 sim2sim 验收，
#      任一环节失败自动回滚旧策略；
#   3. 空出的 GPU 槽位自动开训队列里的下一段（cross → hook，各 20k 迭代）；
#   4. 全部完成后跑最终双机回归并退出。
#
# 启动：  nohup tools/train_watchdog.sh > /dev/null 2>&1 &
# 进度：  tail -f /tmp/train_watchdog.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_DIR="$ROOT/.workbuddy/gpu/g1_boxing_task"
NPZ="$ROOT/.workbuddy/gpu/data/npz"
WATCH_LOG=/tmp/train_watchdog.log

declare -A RUNS     # clip -> "pid|rundir|trainlog"
QUEUE=("cross:4096" "hook:4096")

log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$WATCH_LOG"; }
alive() { kill -0 "$1" 2>/dev/null; }
# newest run dir whose dumped env config references this clip's motion npz
# (survives restarts/resumes: each run creates a fresh timestamped dir)
find_rundir() { # clip-motion-name (e.g. boxing_jab)
  for d in $(ls -td "$TRAIN_DIR"/logs/rsl_rl/g1_boxing/*/ 2>/dev/null); do
    if grep -aq "$1.npz" "$d/params/env.yaml" 2>/dev/null; then echo "$d"; return 0; fi
  done
  return 1
}
backup() { mkdir -p "/tmp/policy_bak_boxing_$1"; cp "$ROOT"/vendor/policy/boxing_"$1".* "/tmp/policy_bak_boxing_$1/" 2>/dev/null || true; }
restore() { cp "/tmp/policy_bak_boxing_$1"/boxing_"$1".* "$ROOT/vendor/policy/" 2>/dev/null || true; }

deploy_clip() { # clip rundir
  local clip=$1 rundir=$2 onnx
  onnx=$(ls -t "$rundir"/*.onnx 2>/dev/null | head -1)
  if [ -z "$onnx" ]; then log "deploy $clip: no ONNX in $rundir"; return 1; fi
  log "deploy $clip from $onnx"
  backup "$clip"
  if ! (cd "$ROOT" && uv run --no-project --with onnx --with onnxruntime \
        python tools/extract_tracking_onnx.py "$onnx" "vendor/policy/boxing_$clip") >> "$WATCH_LOG" 2>&1; then
    log "deploy $clip: extract FAILED → rollback"; restore "$clip"; return 1
  fi
  if ! (cd "$ROOT" && node tools/test_tracking.mjs "vendor/policy/boxing_$clip") >> "$WATCH_LOG" 2>&1; then
    log "deploy $clip: alignment FAIL → rollback"; restore "$clip"; return 1
  fi
  log "deploy $clip: alignment PASS"
  return 0
}

start_training() { # clip envs
  local clip=$1 envs=$2 pid rundir
  log "queue → starting $clip ($envs envs, 20k iters)"
  ( cd "$TRAIN_DIR" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline nohup \
      uv run train Mjlab-Boxing-Unitree-G1 \
      --env.commands.motion.motion-file "$NPZ/boxing_$clip.npz" \
      --env.scene.num-envs "$envs" --agent.max-iterations 20000 \
      > "/tmp/train_$clip.log" 2>&1 & echo $! > "/tmp/train_$clip.pid" )
  sleep 150                       # 等 mjlab 初始化并创建 run 目录
  pid=$(cat "/tmp/train_$clip.pid" 2>/dev/null || echo 0)
  rundir=$(find_rundir "boxing_$clip")
  RUNS[$clip]="$pid|$rundir|/tmp/train_$clip.log"
  log "started $clip pid=$pid rundir=$rundir"
}

# --- 接管当前在跑的两个训练（pid 按 motion 名匹配，run 目录按 env.yaml 发现）
RUNS[jab]="$(pgrep -f 'motion-file.*boxing_jab.npz' | head -1)|$(find_rundir boxing_jab)|/tmp/train_jab.log"
RUNS[guard]="$(pgrep -f 'motion-file.*boxing_guard_idle.npz' | head -1)|$(find_rundir boxing_guard_idle)|/tmp/train_guard.log"

log "=== watchdog start: 监控 ${!RUNS[@]}，队列 ${QUEUE[*]} ==="
while :; do
  for clip in "${!RUNS[@]}"; do
    IFS='|' read -r pid rundir logf <<< "${RUNS[$clip]}"
    alive "$pid" && continue
    log "$clip 训练进程退出 (pid $pid, 日志 $logf)"
    tail -3 "$logf" >> "$WATCH_LOG" 2>/dev/null
    sleep 90                       # 等最终 ONNX/checkpoint 落盘
    if deploy_clip "$clip" "$rundir"; then
      if (cd "$ROOT" && timeout 500 node tools/test_tracking_sim.mjs "boxing_$clip") >> "$WATCH_LOG" 2>&1; then
        log "$clip ✓ 部署完成，sim2sim PASS"
      else
        log "$clip sim2sim FAIL → 回滚到上一版策略"; restore "$clip"
      fi
    else
      log "$clip 部署失败（保留上一版策略），请查看 $WATCH_LOG"
    fi
    unset "RUNS[$clip]"
  done
  while [ "${#RUNS[@]}" -lt 2 ] && [ "${#QUEUE[@]}" -gt 0 ]; do
    IFS=: read -r c e <<< "${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
    start_training "$c" "$e"
  done
  if [ "${#RUNS[@]}" -eq 0 ] && [ "${#QUEUE[@]}" -eq 0 ]; then
    log "=== 全部片段完成 — 最终双机回归 ==="
    (cd "$ROOT" && node tools/test_tracking_boxing.mjs 22 boxing) >> "$WATCH_LOG" 2>&1 \
      && log "最终双机回归 PASS ✓ 浏览器 ?scene=tracking 即为完整形态" \
      || log "最终双机回归 FAIL — 手动检查 vendor/policy/boxing_*"
    break
  fi
  sleep 300
done
