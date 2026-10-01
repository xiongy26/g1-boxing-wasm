#!/usr/bin/env bash
# Deploy trained boxing clip policies from mjlab training runs into the browser
# app (final step of milestone D/E; updated for plan-realistic-boxing §4.1/§7.6).
#
# Each training run dir (logs/rsl_rl/g1_boxing/<timestamp>/) holds the latest
# checkpoint's ONNX export. Re-extract it with the exported metadata and drop
# it into vendor/policy/boxing_<clip>.{bin,meta.json,test_vectors.json}.
#
# v3 行为：
#   - 把 clips 清单（.workbuddy/gpu/data/clips_v3/manifest.json）的片段元数据
#     （阶段秒标/有效出拳区间/来源）注入策略 meta 的 motion.clip 字段；
#   - 部署后把训练 run 目录写回清单的 policy_run 字段（来源可追溯）；
#   - 检测不同槽位部署了相同权重的哈希碰撞（jab==hook 事故，见方案 §2）并
#     以非零码失败。
#
# Usage: tools/deploy_tracking_clips.sh <guard_run_dir> [jab_run_dir] [cross_run_dir] [hook_run_dir] [combo_run_dir]
#
# combo 槽位（plan-revamp-20260928 §5.2）：第 5 参数给出 35s 组合策略训练 run
# 目录时部署 boxing_combo.*。元数据用整文件注入：存在
# .workbuddy/gpu/data/combo_clip_meta.json 时经 --clip-meta-direct 写进
# motion.clip（非 clips_v3 清单条目），否则省略（浏览器回退内建标定）。
set -euo pipefail
cd "$(dirname "$0")/.."

GUARD_DIR=${1:?usage: deploy_tracking_clips.sh <guard_run> [jab_run] [cross_run] [hook_run] [combo_run]}
JAB_DIR=${2:-$GUARD_DIR}
CROSS_DIR=${3:-$JAB_DIR}
HOOK_DIR=${4:-$CROSS_DIR}
COMBO_DIR=${5:-}

CLIP_MANIFEST=.workbuddy/gpu/data/clips_v3/manifest.json
[ -f "$CLIP_MANIFEST" ] || CLIP_MANIFEST=.workbuddy/gpu/data/clips/manifest.json
COMBO_META=.workbuddy/gpu/data/combo_clip_meta.json

dir_onnx() { ls -t "$1"/*.onnx 2>/dev/null | head -1; }

deploy() {
  local clip=$1 dir=$2
  local onnx; onnx=$(dir_onnx "$dir")
  [ -n "$onnx" ] || { echo "no ONNX in $dir for $clip" >&2; return 1; }
  echo "boxing_$clip <- $onnx"
  uv run --no-project --with onnx --with onnxruntime \
    python tools/extract_tracking_onnx.py "$onnx" "vendor/policy/boxing_$clip" \
    --clip-meta "$CLIP_MANIFEST" --clip-name "boxing_$clip"
  # 来源写回清单（§4.1 对 jab/cross/hook 分别记录数据来源与策略来源）
  python3 - "$CLIP_MANIFEST" "$clip" "$dir" <<'PY'
import json, sys
path, clip, run = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f: m = json.load(f)
key = f"boxing_{clip}" if f"boxing_{clip}" in m else clip
if key in m:
    m[key]["policy_run"] = run
    with open(path, "w") as f: json.dump(m, f, indent=1, ensure_ascii=False)
PY
}

deploy_combo() {
  # combo 槽位：单动作单策略，不走 clips_v3 清单（其 manifest 无此条目）
  local dir=$1
  local onnx; onnx=$(dir_onnx "$dir")
  [ -n "$onnx" ] || { echo "no ONNX in $dir for combo" >&2; return 1; }
  echo "boxing_combo <- $onnx"
  local meta_args=()
  [ -f "$COMBO_META" ] && meta_args=(--clip-meta-direct "$COMBO_META")
  uv run --no-project --with onnx --with onnxruntime \
    python tools/extract_tracking_onnx.py "$onnx" "vendor/policy/boxing_combo" \
    "${meta_args[@]+"${meta_args[@]}"}"
}

deploy guard "$GUARD_DIR"
deploy jab "$JAB_DIR"
deploy cross "$CROSS_DIR"
deploy hook "$HOOK_DIR"
[ -z "$COMBO_DIR" ] || deploy_combo "$COMBO_DIR"

echo
echo "-- weight hash collision check (不同槽位必须不同权重) --"
fail=0
declare -A seen
clips=(guard jab cross hook)
[ -z "$COMBO_DIR" ] || clips+=(combo)
for clip in "${clips[@]}"; do
  h=$(sha256sum "vendor/policy/boxing_$clip.bin" | cut -d' ' -f1)
  if [ -n "${seen[$h]:-}" ]; then
    echo "COLLISION: boxing_$clip.bin == boxing_${seen[$h]}.bin ($h)"
    fail=1
  else
    seen[$h]=$clip
  fi
done
if [ "$fail" = 1 ]; then
  echo "RESULT: FAIL (duplicate weights deployed to different slots)" >&2
  exit 1
fi
echo "all ${#clips[@]} slots distinct"

for clip in "${clips[@]}"; do
  node "tools/test_tracking.mjs" "vendor/policy/boxing_$clip" | tail -1
done
echo "deployed. next: node tools/test_tracking_sim.mjs boxing_guard && node tools/test_tracking_boxing.mjs 22 boxing"
