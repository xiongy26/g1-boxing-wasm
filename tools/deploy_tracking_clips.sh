#!/usr/bin/env bash
# Deploy trained boxing clip policies from mjlab training runs into the browser
# app (final step of milestone D/E, docs/plan-mjlab-gpu-training.md §5).
#
# Each training run dir (logs/rsl_rl/g1_boxing/<timestamp>/) holds the latest
# checkpoint's ONNX export. Re-extract it with the exported metadata and drop
# it into vendor/policy/boxing_<clip>.{bin,meta.json}, then re-run the
# regressions:
#   node tools/test_tracking.mjs vendor/policy/boxing_guard
#   node tools/test_tracking_sim.mjs boxing_guard
#   node tools/test_tracking_boxing.mjs 22 boxing
#
# Usage: tools/deploy_tracking_clips.sh <guard_run_dir> [jab_run_dir] [cross_run_dir] [hook_run_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

GUARD_DIR=${1:?usage: deploy_tracking_clips.sh <guard_run> [jab_run] [cross_run] [hook_run]}
JAB_DIR=${2:-$GUARD_DIR}
CROSS_DIR=${3:-$JAB_DIR}
HOOK_DIR=${4:-$CROSS_DIR}

dir_onnx() { ls -t "$1"/*.onnx 2>/dev/null | head -1; }

deploy() {
  local clip=$1 dir=$2
  local onnx; onnx=$(dir_onnx "$dir")
  [ -n "$onnx" ] || { echo "no ONNX in $dir for $clip" >&2; return 1; }
  echo "boxing_$clip <- $onnx"
  uv run --no-project --with onnx --with onnxruntime \
    python tools/extract_tracking_onnx.py "$onnx" "vendor/policy/boxing_$clip"
}

deploy guard "$GUARD_DIR"
deploy jab "$JAB_DIR"
deploy cross "$CROSS_DIR"
deploy hook "$HOOK_DIR"

echo
for clip in guard jab cross hook; do
  node "tools/test_tracking.mjs" "vendor/policy/boxing_$clip" | tail -1
done
echo "deployed. next: node tools/test_tracking_sim.mjs boxing_guard && node tools/test_tracking_boxing.mjs 22 boxing"
