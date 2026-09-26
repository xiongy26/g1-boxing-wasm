#!/usr/bin/env bash
# 阶段4收尾（v2，替代已退出的 train_watchdog.sh）：
# 顺序恢复 cross(13500→20k) 和 hook(11000→20k) 的训练，各自完成后立即部署+验收，
# 全部完成后跑两遍最终双机回归。顺序执行（不并行），避免内存过载。
#   nohup tools/finish_training.sh > /dev/null 2>&1 &
#   tail -f /tmp/finish_training.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_DIR="$ROOT/.workbuddy/gpu/g1_boxing_task"
LOG=/tmp/finish_training.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

find_rundir() { # clip-motion-name
  for d in $(ls -td "$TRAIN_DIR"/logs/rsl_rl/g1_boxing/*/ 2>/dev/null); do
    if grep -aq "boxing_$1.npz" "$d/params/env.yaml" 2>/dev/null; then echo "$d"; return 0; fi
  done
  return 1
}

train_resume() { # clip envs load_run ckpt budget   — 前台运行，直到该段训练结束
  local clip=$1 envs=$2 load_run=$3 ckpt=$4 budget=$5
  log "resume $clip from $ckpt (+$budget iters)"
  ( cd "$TRAIN_DIR" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
      uv run train Mjlab-Boxing-Unitree-G1 \
      --env.commands.motion.motion-file "$ROOT/.workbuddy/gpu/data/npz/boxing_$clip.npz" \
      --env.scene.num-envs "$envs" --agent.max-iterations "$budget" \
      --agent.resume True --agent.load_run "$load_run" --agent.load_checkpoint "$ckpt" ) \
      >> "/tmp/train_$clip.log" 2>&1
  log "$clip 训练段退出，exit=$?"
}

deploy_and_test() { # clip
  local clip=$1 rundir onnx
  rundir=$(find_rundir "boxing_$clip")
  onnx=$(ls -t "$rundir"/*.onnx 2>/dev/null | head -1)
  [ -n "$onnx" ] || { log "deploy $clip: no ONNX in $rundir"; return 1; }
  log "deploy $clip from $onnx"
  mkdir -p "/tmp/policy_bak_boxing_$clip"
  cp "$ROOT"/vendor/policy/boxing_"$clip".* "/tmp/policy_bak_boxing_$clip/" 2>/dev/null || true
  ( cd "$ROOT" && uv run --no-project --with onnx --with onnxruntime \
      python tools/extract_tracking_onnx.py "$onnx" "vendor/policy/boxing_$clip" ) >> "$LOG" 2>&1 || { restore "$clip"; return 1; }
  ( cd "$ROOT" && node tools/test_tracking.mjs "vendor/policy/boxing_$clip" ) >> "$LOG" 2>&1 || { log "$clip 对齐 FAIL → 回滚"; restore "$clip"; return 1; }
  ( cd "$ROOT" && node tools/test_tracking_sim.mjs "boxing_$clip" ) >> "$LOG" 2>&1 || { log "$clip sim2sim FAIL → 回滚"; restore "$clip"; return 1; }
  log "$clip ✓ 部署完成，验收 PASS"
}
restore() { cp "/tmp/policy_bak_boxing_$1"/boxing_"$1".* "$ROOT/vendor/policy/" 2>/dev/null || true; }

log "=== 收尾开始：cross(13500→20k) → hook(11000→20k) → 最终回归 ==="
train_resume cross 4096 2026-09-25_02-00-06 model_13500.pt 6500
deploy_and_test cross
train_resume hook 4096 2026-09-25_06-04-17 model_11000.pt 9000
deploy_and_test hook
log "=== 最终双机回归 ×2 ==="
(cd "$ROOT" && node tools/test_tracking_boxing.mjs 22 boxing) >> "$LOG" 2>&1 && log "回归#1 PASS" || log "回归#1 FAIL"
(cd "$ROOT" && node tools/test_tracking_boxing.mjs 22 boxing) >> "$LOG" 2>&1 && log "回归#2 PASS" || log "回归#2 FAIL"
log "=== 收尾结束 ==="
