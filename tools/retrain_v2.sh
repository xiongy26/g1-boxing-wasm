#!/usr/bin/env bash
# 阶段4 v2 重训链：等 finish_training.sh 收尾退出后，用"直立重切片段 + 旧策略
# 热启动"重训 jab/cross/hook（跟踪技能跨动作迁移，+8000 迭代/片段足够收敛；
# 若 8k 后误差仍 >0.15 可再续）。每片段完成即部署+验收（失败回滚），最后双机回归 ×2。
#   nohup tools/retrain_v2.sh > /dev/null 2>&1 &
#   tail -f /tmp/retrain_v2.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_DIR="$ROOT/.workbuddy/gpu/g1_boxing_task"
LOG=/tmp/retrain_v2.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
find_rundir() {
  for d in $(ls -td "$TRAIN_DIR"/logs/rsl_rl/g1_boxing/*/ 2>/dev/null); do
    if grep -aq "boxing_$1.npz" "$d/params/env.yaml" 2>/dev/null; then echo "$d"; return 0; fi
  done
  return 1
}
restore() { cp "/tmp/policy_bak_$1"/boxing_"$1".* "$ROOT/vendor/policy/" 2>/dev/null || true; }

# ---- 1. 等 finisher 退出（hook 旧窗口训练 + 部署 + 回归）----
while pgrep -f "tools/finish_training.sh" > /dev/null; do sleep 120; done
log "finisher 已退出，开始 v2 重训（直立片段，热启动 +8000 迭代/片段）"

# ---- 2. 顺序重训（热启动源 = 各片段当前最新 checkpoint）----
declare -A SRC
SRC[jab]="2026-09-24_20-41-29|model_19999.pt"
SRC[cross]="2026-09-26_11-18-18|model_19999.pt"
# hook 的热启动源 = finisher 刚跑完的 run 目录（旧窗口 20k），按 env.yaml 现查
hook_dir=$(find_rundir "boxing_hook_old" || true)
[ -z "${hook_dir:-}" ] && hook_dir=$(ls -td "$TRAIN_DIR"/logs/rsl_rl/g1_boxing/*/ | head -3 | while read d; do grep -aq "boxing_hook" "$d/params/env.yaml" 2>/dev/null && { echo "$d"; break; }; done)
SRC[hook]="$(basename "$hook_dir")|$(ls "$hook_dir" 2>/dev/null | grep -oE "model_[0-9]+\.pt" | sort -t_ -k2 -n | tail -1)"

for clip in jab cross hook; do
  IFS='|' read -r load_run ckpt <<< "${SRC[$clip]}"
  log "=== $clip：热启动 $load_run/$ckpt，新直立片段 +8000 迭代 ==="
  ( cd "$TRAIN_DIR" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
      uv run train Mjlab-Boxing-Unitree-G1 \
      --env.commands.motion.motion-file "$ROOT/.workbuddy/gpu/data/npz/boxing_$clip.npz" \
      --env.scene.num-envs 4096 --agent.max-iterations 8000 \
      --agent.resume True --agent.load_run "$load_run" --agent.load_checkpoint "$ckpt" ) \
      >> "/tmp/train_${clip}_v2.log" 2>&1
  log "$clip 训练段退出 exit=$?"
  # 部署 + 验收
  rundir=$(find_rundir "boxing_$clip")
  onnx=$(ls -t "$rundir"/*.onnx 2>/dev/null | head -1)
  if [ -n "$onnx" ]; then
    log "deploy $clip from $onnx"
    mkdir -p "/tmp/policy_bak_$clip"; cp "$ROOT"/vendor/policy/boxing_"$clip".* "/tmp/policy_bak_$clip/" 2>/dev/null || true
    if ( cd "$ROOT" && uv run --no-project --with onnx --with onnxruntime \
         python tools/extract_tracking_onnx.py "$onnx" "vendor/policy/boxing_$clip" ) >> "$LOG" 2>&1 \
       && ( cd "$ROOT" && node tools/test_tracking.mjs "vendor/policy/boxing_$clip" ) >> "$LOG" 2>&1 \
       && ( cd "$ROOT" && node tools/test_tracking_sim.mjs "boxing_$clip" ) >> "$LOG" 2>&1; then
      log "$clip ✓ v2 部署完成，验收 PASS"
    else
      log "$clip v2 验收 FAIL → 回滚（保留上一版）"; restore "$clip"
    fi
  else
    log "$clip 无 ONNX（$rundir），跳过部署"
  fi
done

# ---- 3. 最终回归 ×2 ----
log "=== v2 最终双机回归 ×2 ==="
(cd "$ROOT" && node tools/test_tracking_boxing.mjs 22 boxing) >> "$LOG" 2>&1 && log "回归#1 PASS" || log "回归#1 FAIL"
(cd "$ROOT" && node tools/test_tracking_boxing.mjs 22 boxing) >> "$LOG" 2>&1 && log "回归#2 PASS" || log "回归#2 FAIL"
log "=== v2 重训链结束 ==="
