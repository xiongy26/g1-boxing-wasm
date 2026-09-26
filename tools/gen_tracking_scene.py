#!/usr/bin/env python3
# Generate models/scene_boxing_tracking.xml: two mjlab G1 29-DoF robots (the
# BeyondMimic/mjlab tracking model, stage 4 of docs/plan-mjlab-gpu-training.md)
# + boxing gloves. The tracking policies were trained on the mjlab g1.xml
# (collision capsules included), so the robot tree is reused verbatim (A_/B_
# prefixes); only the glove spheres are added (massless — keeps dynamics
# in-distribution for the tracking policies).
#
# Physics mirrors the training env (mjlab tracking cfg + compiled model):
#   timestep 0.005 (200 Hz sim, 50 Hz policy, decimation 4), Euler, Newton,
#   iterations 100, joint damping 0 / frictionloss 0 / armature 0,
#   floor friction 1 / 0.005 / 0.0001.
# Torque actuators: the browser computes PD torques in JS from the ONNX
# metadata gains (kp/kd per joint, see g1_tracking_actuators.json), exactly
# like the AMO scene but with the tracking robot's softer gains.
import xml.etree.ElementTree as ET

SRC = "models/tracking_g1/g1.xml"
OUT = "models/scene_boxing_tracking.xml"

REF_ATTRS = ("joint", "site", "material", "objname", "childclass",
             "subtreename", "mesh", "body1", "body2", "sensor")


def prefix_elem(elem, p):
    for k, v in list(elem.attrib.items()):
        if k == "name" or k in REF_ATTRS:
            elem.set(k, p + v)
        elif k == "class":
            elem.set(k, p + v)
    for child in elem:
        prefix_elem(child, p)


tree = ET.parse(SRC)
root = tree.getroot()
for tag in ("keyframe", "sensor"):
    for el in root.findall(tag):
        root.remove(el)

robot_src = ET.ElementTree(root)

JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# rotor reflected inertia per joint, dumped from the mjlab-compiled training
# model (BuiltinPositionActuatorCfg armature). CRITICAL for stability: without
# it the low-inertia ankle/wrist dofs are explicit-Euler unstable at 200 Hz.
ARMATURE = {
    "hip_pitch": 0.010178, "hip_yaw": 0.010178, "waist_yaw": 0.010178,
    "hip_roll": 0.025102, "knee": 0.025102,
    "ankle_pitch": 0.007219, "ankle_roll": 0.007219,
    "waist_roll": 0.007219, "waist_pitch": 0.007219,
    "shoulder_pitch": 0.003610, "shoulder_roll": 0.003610, "shoulder_yaw": 0.003610,
    "elbow": 0.003610, "wrist_roll": 0.003610,
    "wrist_pitch": 0.004250, "wrist_yaw": 0.004250,
}
import re as _re
JOINT_ARMATURE = {j: ARMATURE[_re.sub(r"^(left|right)_", "", _re.sub(r"_joint$", "", j))] for j in JOINTS}
# guard standing pose == the model's default_joint_pos (validated safe pose,
# == SAFE_POSE_JOINTS in the spinkick example / ONNX default_joint_pos)
GUARD = [
    -0.312, 0, 0, 0.669, -0.363, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0,
    0.2, -0.2, 0, 0.6, 0, 0, 0,
]


def make_robot(p, pos, quat):
    r = ET.fromstring(ET.tostring(robot_src.getroot()))
    prefix_elem(r, p)
    # rotor reflected inertia per joint (see ARMATURE note above)
    for jname, arm in JOINT_ARMATURE.items():
        jel = r.find(f".//joint[@name='{p}{jname}']")
        assert jel is not None, p + jname
        jel.set("armature", str(arm))
    # contact config = mjlab FULL_COLLISION (g1_constants.py): every collision
    # geom condim=1 except the feet (condim=3, priority=1, friction 0.6 — the
    # priority makes the foot's OWN friction win over the floor's)
    for g in r.findall(".//geom[@class='collision']"):
        name = g.get("name", "")
        if "_foot" in name:
            g.set("condim", "3"); g.set("priority", "1")
            g.set("friction", "0.6 0.005 0.0001")
        else:
            g.set("condim", "1")
    # the controller's hit detection expects the trained collision shapes under
    # the boxing-demo names: rename the mjlab torso/head collision geoms
    for old, new in ((f"{p}torso_collision", f"{p}torso_col"),
                     (f"{p}head_collision", f"{p}head_col")):
        el = r.find(f".//*[@name='{old}']")
        assert el is not None, old
        el.set("name", new)
    # hit-detection sites (positions mirror the collision geoms)
    torso = r.find(f".//body[@name='{p}torso_link']")
    ET.SubElement(torso, "site", {"name": f"{p}head", "pos": "0 0 0.43",
                                  "size": "0.015", "rgba": "1 0 0 0.4"})
    # gloves: massless collision spheres at the palm anchors (same size/anchor
    # as the AMO demo gloves so hit reach is unchanged). The mjlab G1 has no
    # hand body — the palm site lives on the wrist_yaw_link.
    for side in ("left", "right"):
        hand = r.find(f".//body[@name='{p}{side}_wrist_yaw_link']")
        assert hand is not None, p + side
        col = "0.88 0.16 0.12 1" if p == "A_" else "0.18 0.36 0.95 1"
        ET.SubElement(hand, "geom", {
            "name": f"{p}{side}_fist", "type": "sphere", "size": "0.047",
            "pos": "0.082 0.003 0", "rgba": col, "density": "0", "group": "2",
            "contype": "1", "conaffinity": "1", "friction": "0.8 0.005 0.0001",
            "priority": "1"})
        ET.SubElement(hand, "site", {
            "name": f"{p}{side}_fist_site", "pos": "0.082 0.003 0",
            "size": "0.015", "rgba": "1 1 0 0.4"})
    pelvis = r.find(f".//body[@name='{p}pelvis']")
    pelvis.set("pos", pos)
    if quat:
        pelvis.set("quat", quat)
    return r


mujoco = ET.Element("mujoco", {"model": "g1_boxing_tracking"})
ET.SubElement(mujoco, "compiler", {"angle": "radian", "meshdir": "unitree_g1/assets",
                                   "autolimits": "true"})
ET.SubElement(mujoco, "option", {"timestep": "0.005", "integrator": "Euler",
                                 "solver": "Newton", "iterations": "100",
                                 "gravity": "0 0 -9.81"})
dflt = ET.SubElement(mujoco, "default")
# the mjlab g1.xml trains its policies with free hinges: no damping/frictionloss
# (torque clamping happens in JS from the ONNX metadata effort limits)
ET.SubElement(dflt, "joint", {"type": "hinge"})
ET.SubElement(dflt, "geom", {"condim": "3"})

asset = ET.SubElement(mujoco, "asset")
ET.SubElement(asset, "material", {"name": "floor_mat", "rgba": "0.32 0.33 0.38 1"})

world = ET.SubElement(mujoco, "worldbody")
ET.SubElement(world, "light", {"pos": "0 0 3.2", "dir": "0 0 -1", "directional": "true"})
ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "6 6 0.1",
                              "material": "floor_mat", "condim": "3",
                              "friction": "1 0.005 0.0001"})

robots = [make_robot("A_", "-0.50 0 0.761", None),
          make_robot("B_", "0.50 0 0.761", "0 0 0 1")]

# collect mesh assets + robot materials (dedupe by prefixed name)
seen = set()
for r in robots:
    for m in r.findall(".//asset/mesh"):
        if m.get("name") not in seen:
            seen.add(m.get("name"))
            asset.append(m)
    for mat in r.findall(".//asset/material"):
        if mat.get("name") not in seen:
            seen.add(mat.get("name"))
            asset.append(mat)

for r in robots:
    for child in list(r):
        if child.tag in ("compiler", "option", "size", "asset", "keyframe",
                         "actuator", "sensor"):
            continue
        if child.tag == "worldbody":
            for sub in list(child):
                if sub.tag == "body":
                    world.append(sub)
            continue
        mujoco.append(child)

act_all = ET.SubElement(mujoco, "actuator")
for p in ("A_", "B_"):
    for j in JOINTS:
        ET.SubElement(act_all, "motor", {"name": f"{p}{j}", "joint": f"{p}{j}"})

key = ET.SubElement(ET.SubElement(mujoco, "keyframe"), "key", {"name": "ready"})
qa = ["-0.50", "0", "0.761", "1", "0", "0", "0"] + [f"{v}" for v in GUARD]
qb = ["0.50", "0", "0.761", "0", "0", "0", "1"] + [f"{v}" for v in GUARD]
key.set("qpos", " ".join(qa + qb))
key.set("ctrl", " ".join(["0"] * 58))

xml_out = ET.tostring(mujoco, encoding="unicode")
open(OUT, "w", encoding="utf-8").write(xml_out)
print("written", OUT, len(xml_out), "bytes")

# sanity: load with mujoco
import mujoco as mj
model = mj.MjModel.from_xml_path(OUT)
data = mj.MjData(model)
mj.mj_resetDataKeyframe(model, data, 0)
mj.mj_forward(model, data)
print("nu:", model.nu, "nq:", model.nq, "nbody:", model.nbody,
      "timestep:", model.opt.timestep)
zA = float(data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, "A_pelvis"), 2])
print("pelvis z:", zA)
for s in ("A_left_fist_site", "A_head", "B_right_fist_site"):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, s)
    assert sid >= 0, s
print("sites ok")
