#!/usr/bin/env python3
"""Cut boxing skill clips out of the LAFAN1-retargeted G1 fight CSVs
(milestone C of docs/plan-mjlab-gpu-training.md; v3 per
docs/plan-realistic-boxing.md §4.1).

Input : HuggingFace lvhaidong/LAFAN1_Retargeting_Dataset g1/*.csv
        (30 fps, no header, [root_pos(3), root_quat_xyzw(4), 29 joints])
Output: per-clip CSVs in the same format + manifest.json (schema v2).

v3 changes (§4.1 重做拳击架势与基础动作数据):
  - 书端守卫换成 FK 验证过的前后脚拳击架势 BOXING_GUARD（拳套护头 0.14-0.16m、
    收肘、左脚在前 0.26m/横距 0.29m，几何见 tools/check_clip.mjs 输出），
    `--stance v2` 可复现旧部署权重的训练数据。
  - 缩短人工保持段：pad 1.0s -> 0.4s、blend 0.5s -> 0.4s（v2 出拳片段 ~3s 的
    附加时长压到 ~1.6s）。
  - 每个动作自带完整阶段：准备（pad+blend）/ 出拳（核心窗）/ 收拳恢复
    （blend+pad），manifest 以秒记录，并按 50Hz 策略频率换算帧索引。
  - 来源标注：source 源文件、窗口、版本、姿态、拳法（左右手）、支撑脚；
    策略来源（训练 run）由部署时填写，消除未标注的权重复用。

之后用 tools/check_clip.mjs 做 FK 验证（护头距离/骨盆高度/滑脚/关节范围），
训练完成 deploy 时 tools/extract_tracking_onnx.py 把 manifest 条目注入策略
meta 的 motion.clip 字段，运行时调度器按元数据调度（src/boxing_ai.mjs）。

Usage:
  uv run --no-project --with numpy python tools/cut_boxing_clips.py \
      --data-dir .workbuddy/gpu/data --out-dir .workbuddy/gpu/data/clips
"""
import argparse
import json
import os

import numpy as np

FPS = 30.0
POLICY_HZ = 50  # mjlab tracking env rate; manifest frame indices use this

# ---- v2 stance: spinkick example's validated standing pose (== ONNX default) --
GUARD_JOINTS_V2 = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,   # left leg
    -0.312, 0, 0, 0.669, -0.363, 0,   # right leg
    0, 0, 0,                          # waist yaw/roll/pitch
    0.2, 0.2, 0, 0.6, 0, 0, 0,        # left arm  (sp, sr, sy, elbow, wrist r/p/y)
    0.2, -0.2, 0, 0.6, 0, 0, 0,       # right arm
])

# ---- v3 stance: 前后脚拳击架势（FK 验证：tools/check_clip.mjs）----------------
# 29 关节顺序：L腿6 R腿6 腰3 L臂7 R臂7。
#   前后脚：L 脚在前（hip_pitch -0.25），R 脚在后（+0.18），膝微屈，双脚贴地
#   护头  ：肩前屈 -0.55/-0.65 + 肘 -1.05/-1.15，双拳距头 0.14-0.16m、下颌高度
#   收肘  ：肩展 ±0.15 + 肩旋 ∓0.25；腰微转 0.15 呈刃状站架
GUARD_JOINTS_V3 = np.array([
    -0.25, -0.0, -0.12, 0.45, -0.20, 0,    # left leg  （前脚）
    0.18, -0.0, 0.12, 0.40, -0.35, 0,      # right leg （后脚）
    0.15, 0, 0,                            # waist yaw/roll/pitch
    -0.55, 0.15, -0.25, -1.00, 0, 0, 0,    # left arm  （前手，肘留 0.05 裕量）
    -0.65, -0.15, 0.25, -1.05, 0, 0, 0,    # right arm （后手，肘到限位 -1.05）
])
GUARD_Z_V2 = 0.76
GUARD_Z_V3 = 0.74   # 微屈膝的运动架势高度（FK 骨盆 0.678 直立 → 屈膝下沉）

STANCES = {
    "v2": {"joints": GUARD_JOINTS_V2, "z": GUARD_Z_V2},
    "v3": {"joints": GUARD_JOINTS_V3, "z": GUARD_Z_V3},
}

# v3 缩短过渡（§4.1："重新设计片段过渡，缩短人工保持段"）
TRANSITION_DURATION = 0.4    # s, blend between guard and clip (v2: 0.5)
PAD_DURATION = 0.4           # s, hold guard before/after the blend (v2: 1.0)

# windows picked from the extension-event scan over g1_fight1_subject2
# (name, source stem, t0, t1, hand) — times in seconds into the source clip.
# v2 直立窗口（2026-09-26）：v1 的深蹲窗口（骨盆压到 0.487m）是 20k 策略的
# 失稳源——双机对练里频繁在深蹲段自摔。v2 只选骨盆全程 >0.6m 的直立出拳
# 窗口（见 docs/stage4-tracking-notes.md 待办）。
WINDOWS = [
    ("boxing_jab",   "g1_fight1_subject2", 234.90, 235.90, "left"),
    ("boxing_cross", "g1_fight1_subject2", 21.75, 22.75, "right"),
    ("boxing_hook",  "g1_fight1_subject2", 219.50, 220.30, "left"),
]


def quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def yaw_of(quat_xyzw):
    R = quat_xyzw_to_mat(quat_xyzw)
    return np.arctan2(R[1, 0], R[0, 0])


def guard_frame(yaw, stance):
    q = mat_to_quat_xyzw(np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ]))
    return np.concatenate([[0, 0, stance["z"]], q, stance["joints"]])


def ease_in(t):
    return t ** 3


def ease_out(t):
    return 1 - (1 - t) ** 3


def add_transitions(frames, stance, transition_s=TRANSITION_DURATION, pad_s=PAD_DURATION):
    """Bookend `frames` with guard holds + eased blends; returns (frames, phases).

    phases 是动作阶段的秒级标注（§4.1 动作元数据）：
      prepare [0, p+t)   守卫保持 + 入场混合
      strike  [p+t, p+t+core)  截取的出拳核心（有效出拳区间的粗标定，FK 精标由
                         tools/check_clip.mjs 按拳速回写 manifest）
      recover [p+t+core, end)  出场混合 + 守卫保持
    """
    tf, pf = int(transition_s * FPS), int(pad_s * FPS)
    first, last = frames[0], frames[-1]
    start = guard_frame(yaw_of(first[3:7]), stance)
    start[0:2] = first[0:2]          # guard at the clip's XY start
    end = guard_frame(yaw_of(last[3:7]), stance)
    end[0:2] = last[0:2]
    # keep synthesized quats in the same hemisphere as their targets so the
    # linear blend takes the short path
    if np.dot(start[3:7], first[3:7]) < 0:
        start[3:7] *= -1
    if np.dot(end[3:7], last[3:7]) < 0:
        end[3:7] *= -1

    out = [start.copy() for _ in range(pf)]
    for i in range(tf):
        t = ease_in(i / max(tf - 1, 1))
        out.append(start * (1 - t) + first * t)   # quat lerp ok: yaw-only diff, same sign
    out.extend(frames)
    for i in range(tf):
        t = ease_out(i / max(tf - 1, 1))
        out.append(last * (1 - t) + end * t)
    out.extend([end.copy() for _ in range(pf)])
    # renormalize quaternions
    for f in out:
        q = f[3:7]
        f[3:7] = q / np.linalg.norm(q)
    n = len(out)
    phases = {
        "prepare_s": [0.0, round((pf + tf) / FPS, 3)],
        "strike_s": [round((pf + tf) / FPS, 3), round((n - pf - tf) / FPS, 3)],
        "recover_s": [round((n - pf - tf) / FPS, 3), round(n / FPS, 3)],
    }
    return np.array(out), phases


def synthesize_guard_idle(stance, seconds=6.0):
    """In-place guard bounce around the boxing stance: root bob + gentle waist
    sway + arm micro-guard. 不做腿部关节振荡——根节点弹跳叠加腿关节摆动在
    FK 下会让脚在地面扫 1m+（check_clip 滑移检查的合成伪影）；关节活动留在
    腰与手臂，腿只跟随根节点整体下沉（物理上由踝/膝柔顺吸收）。"""
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    freq = 1.4
    phase = 2 * np.pi * freq * t
    bounce = 0.012 * np.sin(phase)

    base = stance["joints"]
    frames = np.zeros((n, 36))
    frames[:, 2] = stance["z"] + bounce
    frames[:, 6] = 1.0   # quat xyzw identity
    j = frames[:, 7:]
    j[:] = base
    j[:, 12] = base[12] + 0.05 * np.sin(2 * np.pi * 0.35 * t)   # waist yaw sway
    j[:, 14] = base[14] + 0.03 * np.sin(phase + 0.5)            # waist pitch
    j[:, 15] += 0.02 * np.sin(phase + 1.0)           # arm micro-guard
    j[:, 22] += 0.02 * np.sin(phase + 1.2)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".workbuddy/gpu/data")
    ap.add_argument("--out-dir", default=".workbuddy/gpu/data/clips")
    ap.add_argument("--stance", choices=sorted(STANCES), default="v3",
                    help="书端守卫姿态版本（v3=拳击架势；v2 复现旧部署训练数据）")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="subset of window names to cut (default: all)")
    args = ap.parse_args()
    stance = STANCES[args.stance]
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {}
    cache = {}

    for name, stem, t0, t1, hand in WINDOWS:
        if args.sources and name not in args.sources:
            continue
        if stem not in cache:
            cache[stem] = np.loadtxt(os.path.join(args.data_dir, stem + ".csv"), delimiter=",")
        d = cache[stem]
        i0, i1 = int(t0 * FPS), int(t1 * FPS)
        clip = d[i0:i1].copy()
        net = np.hypot(*(clip[-1, :2] - clip[0, :2]))
        out, phases = add_transitions(clip, stance)
        path = os.path.join(args.out_dir, name + ".csv")
        np.savetxt(path, out, delimiter=",", fmt="%.6f")
        manifest[name] = {
            "version": "v3" if args.stance == "v3" else "v2-stance",
            "source": stem, "window_s": [t0, t1],
            "hand": hand,                       # 出拳手（§4.1 动作元数据）
            "support_foot": "right" if hand == "left" else "left",  # 反手侧支撑
            "stance": args.stance,
            "duration_s": round(out.shape[0] / FPS, 2),
            "net_displacement_m": round(float(net), 3),
            "phases_s": phases,
            # 50Hz 策略频率帧索引（§4.1：时间标注统一为秒，导出换算帧索引）
            "phases_frame50": {k: [int(v[0] * POLICY_HZ), int(v[1] * POLICY_HZ)]
                               for k, v in phases.items()},
            # forward_drive_m / strike_window_s 由 tools/check_clip.mjs FK 回写
            # policy_run 由 deploy_tracking_clips.sh 部署时填写
            "policy_run": None,
            "frames": int(out.shape[0]),
        }
        print(f"{name}: {out.shape[0]} frames ({out.shape[0] / FPS:.2f}s), "
              f"net root travel {net:.2f}m, strike {phases['strike_s']} -> {path}")

    if not args.sources or "boxing_guard_idle" in args.sources:
        idle = add_transitions(synthesize_guard_idle(stance), stance)
        # 合成片段：strike 阶段无意义（无出拳），标注为空窗口
        idle_frames, idle_phases = idle
        idle_phases["strike_s"] = [0.0, 0.0]
        path = os.path.join(args.out_dir, "boxing_guard_idle.csv")
        np.savetxt(path, idle_frames, delimiter=",", fmt="%.6f")
        manifest["boxing_guard_idle"] = {
            "version": "v3" if args.stance == "v3" else "v2-stance",
            "source": "synthesized", "stance": args.stance,
            "frames": int(idle_frames.shape[0]),
            "duration_s": round(idle_frames.shape[0] / FPS, 2),
            "phases_s": idle_phases,
            "phases_frame50": {k: [int(v[0] * POLICY_HZ), int(v[1] * POLICY_HZ)]
                               for k, v in idle_phases.items()},
            "policy_run": None,
        }
        print(f"boxing_guard_idle: {idle_frames.shape[0]} frames "
              f"({idle_frames.shape[0] / FPS:.2f}s) -> {path}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"manifest (stance {args.stance}) -> {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
