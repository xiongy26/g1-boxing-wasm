#!/usr/bin/env python3
"""Extract a BeyondMimic/mjlab motion-tracking ONNX (see docs/plan-mjlab-gpu-training.md)
into the browser runtime format: vendor/policy/<name>.bin + <name>_meta.json +
test vectors for the JS forward-pass alignment check (tools/test_tracking.mjs).

The ONNX graph (mjlab whole_body_tracking export, validated against the
g1_spinkick_example spinkick_safe.onnx) is:

    time_step -> int -> clip(min 0, max motion_len-1) -> motion tables (Gather)
    obs -> (obs - mean) / std -> Gemm/Elu x3 -> Gemm -> actions

Inputs : obs [1,154], time_step [1,1]
Outputs: actions [1,29], joint_pos/joint_vel [1,29], body_pos_w [1,14,3],
         body_quat_w [1,14,4], body_lin_vel_w/body_ang_vel_w [1,14,3]

Metadata (custom_metadata_map) carries joint_names, default_joint_pos,
joint_stiffness, joint_damping, action_scale, anchor_body_name, body_names,
observation_names. Test vectors are generated with onnxruntime when available,
otherwise with an exact numpy replica of the graph.

Usage:
  uv run --no-project --with onnx --with onnxruntime python \
      tools/extract_tracking_onnx.py <onnx_file> <out_prefix> \
      [--clip-meta manifest.json] [--clip-name boxing_jab]

--clip-meta/--clip-name: 把动作清单条目（阶段秒标、有效出拳区间、来源，见
tools/cut_boxing_clips.py v3）写进 meta 的 motion.clip 字段，浏览器调度器
（src/boxing_ai.mjs punchRangeFor/punchActive）按元数据调度；缺失时回退
内建标定值。

--clip-meta-direct <file>：整份 JSON 即单条 clip 元数据（不经清单按键索引）。
combo 槽位（plan-revamp-20260928 §5.2）用——35s 单动作不属于 clips_v3 清单，
来源/窗口等元数据直接落一个 JSON 文件注入。
"""
import json
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def graph_constant(m, name):
    for n in m.graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    raise KeyError(name)


def initializer(m, name):
    for init in m.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(name)


def find_initializer(m, pred):
    for init in m.graph.initializer:
        if pred(init.name):
            return numpy_helper.to_array(init)
    raise KeyError("no initializer matches")


def extract_policy_tensors(m):
    """Locate normalizer / MLP / motion tensors in both ONNX export variants
    (deployment export: `normalizer._mean`, `actor.N.*`, motion in Constant
    nodes; training export: `policy.obs_normalizer._mean`, `policy.mlp.N.*`,
    motion as `joint_pos.1`-style initializers)."""
    mean = find_initializer(m, lambda n: "_mean" in n)
    std = find_initializer(m, lambda n: n.startswith("onnx::Div"))
    weights, biases = [], []
    import re
    layers = {}
    for init in m.graph.initializer:
        mt = re.search(r"(?:policy\.mlp|actor)\.(\d+)\.(weight|bias)$", init.name)
        if mt:
            layers.setdefault(int(mt.group(1)), {})[mt.group(2)] = numpy_helper.to_array(init)
    for k in sorted(layers):
        weights.append(layers[k]["weight"].astype(np.float32))
        biases.append(layers[k]["bias"].astype(np.float32))
    if not weights:
        raise KeyError("no actor MLP weights found")

    motion = {}
    for init in m.graph.initializer:
        mt = re.match(r"(joint_pos|joint_vel|body_pos_w|body_quat_w|body_lin_vel_w|body_ang_vel_w)\b", init.name)
        if mt:
            motion[mt.group(1)] = numpy_helper.to_array(init).astype(np.float32)
    if not motion:  # deployment variant: Constant nodes
        motion = {
            "joint_pos": graph_constant(m, "/Constant_1_output_0").astype(np.float32),
            "joint_vel": graph_constant(m, "/Constant_2_output_0").astype(np.float32),
            "body_pos_w": graph_constant(m, "/Constant_3_output_0").astype(np.float32),
            "body_quat_w": graph_constant(m, "/Constant_4_output_0").astype(np.float32),
            "body_lin_vel_w": graph_constant(m, "/Constant_5_output_0").astype(np.float32),
            "body_ang_vel_w": graph_constant(m, "/Constant_6_output_0").astype(np.float32),
        }
    # clip max: the third input of the Clip node (max bound), produced by a Constant
    clip_max = None
    for n in m.graph.node:
        if n.op_type == "Clip" and len(n.input) == 3:
            clip_max = float(graph_constant(m, n.input[2]))
    assert clip_max is not None, "Clip max not found"
    return mean, std, weights, biases, motion, clip_max


def numpy_forward(obs, arrays):
    x = (obs - arrays["mean"]) / arrays["std"]
    for w, b in zip(arrays["weights"], arrays["biases"]):
        x = x @ w.T + b
        if w is not arrays["weights"][-1]:
            x = np.where(x > 0, x, np.expm1(x))  # Elu
    return x


def main():
    args = sys.argv[1:]
    onnx_file, out_prefix = args[0], args[1]
    clip_meta = None
    if "--clip-meta-direct" in args:
        # combo 槽位直通注入：整份 JSON 就是 motion.clip 条目
        direct_path = args[args.index("--clip-meta-direct") + 1]
        with open(direct_path) as f:
            clip_meta = json.load(f)
    elif "--clip-meta" in args:
        manifest_path = args[args.index("--clip-meta") + 1]
        clip_name = args[args.index("--clip-name") + 1] if "--clip-name" in args else \
            __import__("os").path.basename(out_prefix)
        with open(manifest_path) as f:
            entry = json.load(f).get(clip_name)
        if entry:
            clip_meta = {
                "manifest_version": entry.get("version"),
                "source": entry.get("source"),
                "window_s": entry.get("window_s"),
                "hand": entry.get("hand"),
                "support_foot": entry.get("support_foot"),
                "stance": entry.get("stance"),
                "phases_s": entry.get("phases_s"),
                "strike_window_s": (entry.get("measured") or {}).get("strike_window_s")
                    or entry.get("phases_s", {}).get("strike_s"),
                "punch_range_m": (entry.get("measured") or {}).get("punch_range_m")
                    or entry.get("punch_range_m"),
                "forward_drive_m": (entry.get("measured") or {}).get("forward_drive_m"),
                "policy_run": entry.get("policy_run"),
            }
        else:
            print(f"warning: {clip_name} not in {manifest_path}; no clip metadata embedded")

    m = onnx.load(onnx_file)
    meta = {p.key: p.value for p in m.metadata_props}

    mean, std, weights, biases, motion, clip_max = extract_policy_tensors(m)
    motion_len = motion["joint_pos"].shape[0]
    assert clip_max == motion_len - 1, (clip_max, motion_len)

    obs_dim = weights[0].shape[1]
    num_joints = weights[-1].shape[0]

    # ---- pack a single .bin (float32 stream), record byte offsets ----
    blobs = {
        "mean": mean, "std": std,
        **{f"W:{i}": w for i, w in enumerate(weights)},
        **{f"b:{i}": b for i, b in enumerate(biases)},
        **{f"motion:{k}": v.reshape(-1) for k, v in motion.items()},
    }
    offsets, parts, off = {}, [], 0
    for name, arr in blobs.items():
        offsets[name] = off * 4  # byte offset
        parts.append(arr.reshape(-1))
        off += arr.size
    with open(out_prefix + ".bin", "wb") as f:
        np.concatenate(parts).astype("<f4").tofile(f)

    parse_floats = lambda s: [float(v) for v in s.split(",")]
    parse_strings = lambda s: s.split(",")
    obs_names = parse_strings(meta["observation_names"])
    # obs layout per observation_names: command(2*nq) anchor_ori(6) [anchor_pos(3)]
    # [lin_vel(3)] ang_vel(3) joint_pos(nq) joint_vel(nq) actions(nq)
    # + opponent_state(14) (P2 双机对打, obs_dim 168, 契约 plan-fight §5.1)
    layout, cursor = {}, 0
    for name in obs_names:
        size = {"command": 2 * num_joints, "motion_anchor_ori_b": 6,
                "motion_anchor_pos_b": 3, "base_lin_vel": 3, "base_ang_vel": 3,
                "joint_pos": num_joints, "joint_vel": num_joints,
                "actions": num_joints, "opponent_state": 14}[name]
        layout[name] = [cursor, size]
        cursor += size
    assert cursor == obs_dim, (cursor, obs_dim)

    meta_out = {
        "format": "tracking_v1",
        "source": onnx_file,
        "obs_dim": obs_dim,
        "num_joints": num_joints,
        "num_bodies": motion["body_pos_w"].shape[1],
        "joint_names": parse_strings(meta["joint_names"]),
        "default_joint_pos": parse_floats(meta["default_joint_pos"]),
        "kp": parse_floats(meta["joint_stiffness"]),
        "kd": parse_floats(meta["joint_damping"]),
        "action_scale": parse_floats(meta["action_scale"]),
        "anchor_body_name": meta["anchor_body_name"],
        "body_names": parse_strings(meta["body_names"]),
        "obs_names": obs_names,
        "obs_layout": layout,
        "motion": {"fps": 50, "length": motion_len, "max_step": int(clip_max),
                   **({"clip": clip_meta} if clip_meta else {})},
        "offsets": offsets,        "shapes": {
            "mean": [obs_dim], "std": [obs_dim],
            **{f"W:{i}": list(w.shape) for i, w in enumerate(weights)},
            **{f"b:{i}": [b.size] for i, b in enumerate(biases)},
            **{f"motion:{k}": list(v.shape) for k, v in motion.items()},
        },
    }
    with open(out_prefix + "_meta.json", "w") as f:
        json.dump(meta_out, f, indent=1)

    # ---- test vectors: onnxruntime when available, else exact numpy replica ----
    rng = np.random.default_rng(42)
    n_vec = 16
    obs_vec = rng.standard_normal((n_vec, obs_dim)).astype(np.float32) * 0.5
    ts_vec = rng.integers(0, motion_len + 30, n_vec)  # includes clip range
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        acts, refs = [], []
        for i in range(n_vec):
            outs = sess.run(None, {
                "obs": obs_vec[i:i + 1],
                "time_step": np.array([[ts_vec[i]]], dtype=np.float32),
            })
            acts.append(outs[0][0])
            refs.append([outs[1][0], outs[2][0], outs[3][0], outs[4][0]])
        mode = "onnxruntime"
    except ImportError:
        acts, refs = [], []
        arrays = {"mean": mean, "std": std, "weights": weights, "biases": biases}
        for i in range(n_vec):
            acts.append(numpy_forward(obs_vec[i], arrays))
            t = min(int(ts_vec[i]), motion_len - 1)
            refs.append([motion["joint_pos"][t], motion["joint_vel"][t],
                         motion["body_pos_w"][t], motion["body_quat_w"][t]])
        mode = "numpy"

    vectors = {
        "mode": mode, "onnx": onnx_file,
        "obs": obs_vec.tolist(), "time_step": ts_vec.tolist(),
        "actions": np.asarray(acts, dtype=np.float32).tolist(),
        "ref_joint_pos": np.asarray([r[0] for r in refs], dtype=np.float32).tolist(),
        "ref_joint_vel": np.asarray([r[1] for r in refs], dtype=np.float32).tolist(),
        "ref_body_pos_w": np.asarray([r[2] for r in refs], dtype=np.float32).tolist(),
        "ref_body_quat_w": np.asarray([r[3] for r in refs], dtype=np.float32).tolist(),
    }
    with open(out_prefix + "_test_vectors.json", "w") as f:
        json.dump(vectors, f)
    print(f"wrote {out_prefix}.bin ({off * 4} bytes), "
          f"{out_prefix}_meta.json, {out_prefix}_test_vectors.json ({n_vec} vectors, {mode})")


if __name__ == "__main__":
    main()
