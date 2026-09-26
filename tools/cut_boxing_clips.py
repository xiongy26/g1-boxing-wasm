#!/usr/bin/env python3
"""Cut boxing skill clips out of the LAFAN1-retargeted G1 fight CSVs
(milestone C of docs/plan-mjlab-gpu-training.md).

Input : HuggingFace lvhaidong/LAFAN1_Retargeting_Dataset g1/*.csv
        (30 fps, no header, [root_pos(3), root_quat_xyzw(4), 29 joints])
Output: per-clip CSVs in the same format, each bookended with the spinkick
        example's validated standing pose ("guard"): 0.5 s ease-in blend into
        the clip, 1.0 s hold, ... clip ... , 1.0 s hold, 0.5 s ease-out blend.
        These transition segments are what makes later clip-to-clip switching
        physically smooth (the tracking policy starts/ends every clip parked
        in the same stable stance).

Also synthesizes `boxing_guard_idle`: an in-place guard bounce cycle, used as
the scheduler's default/idle clip.

Usage:
  uv run --no-project --with numpy python tools/cut_boxing_clips.py \
      --data-dir .workbuddy/gpu/data --out-dir .workbuddy/gpu/data/clips
"""
import argparse
import json
import os

import numpy as np

FPS = 30.0

# validated standing pose from g1_spinkick_example/pkl_to_csv.py (== the ONNX
# default_joint_pos), G1 29-joint order
GUARD_JOINTS = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,   # left leg
    -0.312, 0, 0, 0.669, -0.363, 0,   # right leg
    0, 0, 0,                          # waist yaw/roll/pitch
    0.2, 0.2, 0, 0.6, 0, 0, 0,        # left arm  (sp, sr, sy, elbow, wrist r/p/y)
    0.2, -0.2, 0, 0.6, 0, 0, 0,       # right arm
])
GUARD_Z = 0.76

TRANSITION_DURATION = 0.5   # s, blend between guard and clip
PAD_DURATION = 1.0          # s, hold guard before the blend

# windows picked from the extension-event scan over g1_fight1_subject2
# (name, source stem, t0, t1) — times in seconds into the source clip.
# v2 直立窗口（2026-09-26）：v1 的深蹲窗口（骨盆压到 0.487m）是 20k 策略的
# 失稳源——双机对练里频繁在深蹲段自摔。v2 只选骨盆全程 >0.6m 的直立出拳
# 窗口（见 docs/stage4-tracking-notes.md 待办）。
WINDOWS = [
    ("boxing_jab",   "g1_fight1_subject2", 234.90, 235.90),  # L 直拳，几乎原地（0.19m），z≥0.72
    ("boxing_cross", "g1_fight1_subject2", 21.75, 22.75),    # R 后手直拳，0.35m，z≥0.63
    ("boxing_hook",  "g1_fight1_subject2", 219.50, 220.30),  # L 弧线拳，0.80m，z≥0.69
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


def guard_frame(yaw):
    q = mat_to_quat_xyzw(np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ]))
    return np.concatenate([[0, 0, GUARD_Z], q, GUARD_JOINTS])


def ease_in(t):
    return t ** 3


def ease_out(t):
    return 1 - (1 - t) ** 3


def add_transitions(frames, transition_s=TRANSITION_DURATION, pad_s=PAD_DURATION):
    """Bookend `frames` with guard holds + eased blends (spinkick recipe)."""
    tf, pf = int(transition_s * FPS), int(pad_s * FPS)
    first, last = frames[0], frames[-1]
    start = guard_frame(yaw_of(first[3:7]))
    start[0:2] = first[0:2]          # guard at the clip's XY start
    end = guard_frame(yaw_of(last[3:7]))
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
    return np.array(out)


def synthesize_guard_idle(seconds=6.0):
    """In-place guard bounce: leg springs + root bob, gentle waist sway."""
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    freq = 1.4
    phase = 2 * np.pi * freq * t
    bounce = 0.012 * np.sin(phase)

    frames = np.zeros((n, 36))
    frames[:, 2] = GUARD_Z + bounce
    frames[:, 6] = 1.0   # quat xyzw identity
    j = frames[:, 7:]
    j[:] = GUARD_JOINTS
    j[:, 0] += 0.045 * np.sin(phase)    # L hip pitch
    j[:, 6] += 0.045 * np.sin(phase)    # R hip pitch
    j[:, 3] += 0.075 * np.sin(phase)    # L knee
    j[:, 9] += 0.075 * np.sin(phase)    # R knee
    j[:, 4] -= 0.030 * np.sin(phase)    # L ankle pitch
    j[:, 10] -= 0.030 * np.sin(phase)   # R ankle pitch
    j[:, 12] = 0.05 * np.sin(2 * np.pi * 0.35 * t)   # waist yaw sway
    j[:, 14] = 0.03 * np.sin(phase + 0.5)            # waist pitch
    j[:, 15] += 0.02 * np.sin(phase + 1.0)           # arm micro-guard
    j[:, 22] += 0.02 * np.sin(phase + 1.2)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".workbuddy/gpu/data")
    ap.add_argument("--out-dir", default=".workbuddy/gpu/data/clips")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="subset of window names to cut (default: all)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {}
    cache = {}

    for name, stem, t0, t1 in WINDOWS:
        if args.sources and name not in args.sources:
            continue
        if stem not in cache:
            cache[stem] = np.loadtxt(os.path.join(args.data_dir, stem + ".csv"), delimiter=",")
        d = cache[stem]
        i0, i1 = int(t0 * FPS), int(t1 * FPS)
        clip = d[i0:i1].copy()
        net = np.hypot(*(clip[-1, :2] - clip[0, :2]))
        out = add_transitions(clip)
        path = os.path.join(args.out_dir, name + ".csv")
        np.savetxt(path, out, delimiter=",", fmt="%.6f")
        manifest[name] = {
            "source": stem, "window_s": [t0, t1],
            "duration_s": round(out.shape[0] / FPS, 2),
            "net_displacement_m": round(float(net), 3),
            "frames": int(out.shape[0]),
        }
        print(f"{name}: {out.shape[0]} frames ({out.shape[0] / FPS:.2f}s), "
              f"net root travel {net:.2f}m -> {path}")

    if not args.sources or "boxing_guard_idle" in args.sources:
        idle = add_transitions(synthesize_guard_idle())
        path = os.path.join(args.out_dir, "boxing_guard_idle.csv")
        np.savetxt(path, idle, delimiter=",", fmt="%.6f")
        manifest["boxing_guard_idle"] = {
            "source": "synthesized", "frames": int(idle.shape[0]),
            "duration_s": round(idle.shape[0] / FPS, 2),
        }
        print(f"boxing_guard_idle: {idle.shape[0]} frames ({idle.shape[0] / FPS:.2f}s) -> {path}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
