# Generate models/scene_boxing_amo.xml: two AMO-official G1 robots + boxing gloves.
# The official AMO g1.xml (23 torque actuators, wrists fused) is the exact model the
# policy was trained/deployed on, so the scene reuses it verbatim (prefixed A_/B_).
import xml.etree.ElementTree as ET

TMP = r"C:/Users/33448/.zcode/workspace/default/g1-boxing-wasm/.workbuddy/tmp"
OUT = r"C:/Users/33448/.zcode/workspace/default/g1-boxing-wasm/models/scene_boxing_amo.xml"

NAME_ATTRS = ("name",)
REF_ATTRS = ("joint", "site", "material", "objname", "childclass", "subtreename", "mesh")

def prefix_elem(elem, p):
    for k, v in list(elem.attrib.items()):
        if k == "name" or k in REF_ATTRS:
            elem.set(k, p + v)
        elif k == "class":
            elem.set(k, p + v)
        elif k == "file" and elem.tag == "mesh":
            pass  # shared mesh files, name attribute carries the prefix
    for child in elem:
        prefix_elem(child, p)

def rename_mesh_asset_refs(elem):
    """mesh name= gets prefix; geoms referencing material= handled by REF_ATTRS."""
    pass

tree = ET.parse(TMP + "/g1_official.xml")
root = tree.getroot()

# remove keyframes from the robot tree (scene defines its own)
for kf in root.findall("keyframe"):
    root.remove(kf)

robot_src = ET.ElementTree(root)

def make_robot(p, pos, quat):
    r = ET.fromstring(ET.tostring(robot_src.getroot()))
    prefix_elem(r, p)
    # gloves + hit-detection sites
    for side in ("left", "right"):
        hand = r.find(f".//body[@name='{p}{side}_rubber_hand']")
        assert hand is not None, p + side
        col = "0.88 0.16 0.12 1" if p == "A_" else "0.18 0.36 0.95 1"
        g = ET.SubElement(hand, "geom", {
            "name": f"{p}{side}_fist", "type": "sphere", "size": "0.047",
            "pos": "0.082 0.003 0", "rgba": col, "mass": "0.12", "group": "2",
            "contype": "1", "conaffinity": "1", "friction": "0.8 0.005 0.0001", "priority": "1"})
        if p == "B":
            g.set("rgba", col)
        ET.SubElement(hand, "site", {
            "name": f"{p}{side}_fist_site", "pos": "0.082 0.003 0",
            "size": "0.015", "rgba": "1 1 0 0.4"})
    torso = r.find(f".//body[@name='{p}torso_link']")
    assert torso is not None
    # official xml comments out the head body; head is a visual mesh anchored at
    # pos="0.0039635 0 -0.044" but its actual center sits ~+0.45 above that anchor
    # (commented head_link inertial pos z=0.449869, mass 1.036). The collision sphere
    # MUST go at the real head center (z≈+0.406): at the anchor z=-0.044 it grinds
    # against the waist links while walking and the AMO policy falls (verified by
    # mass/position sensitivity test). Mass matches the official head_link (1.036).
    # group=3: collision-only geoms stay invisible (official head mesh renders).
    ET.SubElement(torso, "geom", {
        "name": f"{p}head_col", "type": "sphere", "size": "0.085", "group": "3",
        "pos": "0.0039635 0 0.406", "rgba": "0.9 0.75 0.7 1", "mass": "1.036",
        "contype": "1", "conaffinity": "1", "friction": "0.8 0.005 0.0001", "priority": "1"})
    ET.SubElement(torso, "site", {"name": f"{p}head", "pos": "0.0039635 0 0.406",
                                  "size": "0.015", "rgba": "1 0 0 0.4"})
    # torso hit capsule (chest/belly). Official mesh geoms are contype=0 (visual
    # only), so without this the opponents' fists pass through the body. density=0
    # keeps it massless: every body has an explicit <inertial>, so dynamics are
    # unchanged and the policy stays in-distribution.
    ET.SubElement(torso, "geom", {
        "name": f"{p}torso_col", "type": "capsule", "size": "0.10 0.11", "group": "3",
        "pos": "0 0 0.20", "rgba": "0.9 0.75 0.7 0.3", "density": "0",
        "contype": "1", "conaffinity": "1", "friction": "0.8 0.005 0.0001", "priority": "1"})
    pelvis = r.find(f".//body[@name='{p}pelvis']")
    pelvis.set("pos", pos)
    if quat:
        pelvis.set("quat", quat)
    return r

mujoco = ET.Element("mujoco", {"model": "g1_boxing_amo"})
ET.SubElement(mujoco, "compiler", {"angle": "radian", "meshdir": "unitree_g1/assets"})
ET.SubElement(mujoco, "option", {"timestep": "0.002", "integrator": "Euler", "iterations": "50", "solver": "PGS", "gravity": "0 0 -9.81"})
dflt = ET.SubElement(mujoco, "default")
# joint defaults MUST match the official g1.xml the policy was trained on
# (<joint actuatorfrclimited="true" type="hinge" frictionloss="0.2" damping="2"/>).
# damping=0/frictionloss=0 broke walking (stand OK, walk falls).
ET.SubElement(dflt, "joint", {"actuatorfrclimited": "true", "type": "hinge",
                              "frictionloss": "0.2", "damping": "2"})
ET.SubElement(dflt, "geom", {"condim": "3"})

asset = ET.SubElement(mujoco, "asset")
ET.SubElement(asset, "material", {"name": "metal", "rgba": "0.7 0.7 0.7 1"})
ET.SubElement(asset, "material", {"name": "black", "rgba": "0.2 0.2 0.2 1"})
ET.SubElement(asset, "material", {"name": "floor_mat", "rgba": "0.32 0.33 0.38 1"})

world = ET.SubElement(mujoco, "worldbody")
ET.SubElement(world, "light", {"pos": "0 0 3.2", "dir": "0 0 -1", "directional": "true"})
ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "6 6 0.1",
                              "material": "floor_mat", "condim": "3", "friction": "1 0.005 0.0001"})

robots = [make_robot("A_", "-0.42 0 0.78", None), make_robot("B_", "0.42 0 0.78", "0 0 0 1")]

# collect assets (meshes) from both robots - dedupe by prefixed name
seen = set()
for r in robots:
    for m in r.findall(".//asset/texture") + r.findall(".//asset/mesh"):
        if m.get("name") and ("MatPlane" in m.get("name") or "texplane" in m.get("name")):
            continue  # robot's own ground plane - our scene provides the floor
        if m.get("name") not in seen:
            seen.add(m.get("name"))
            asset.append(m)
    for mat in r.findall(".//asset/material"):
        if mat.get("name") and ("MatPlane" in mat.get("name") or "texplane" in mat.get("name")):
            continue
        if mat.get("name") not in seen:
            seen.add(mat.get("name"))
            asset.append(mat)
    r.find("asset") is not None and r.remove(r.find("asset"))

for r in robots:
    for child in list(r):
        if child.tag in ("compiler", "option", "size", "asset", "keyframe", "actuator", "sensor"):
            continue
        if child.tag == "worldbody":
            for sub in list(child):
                if sub.get("name") and sub.tag == "body":
                    world.append(sub)
            continue
        mujoco.append(child)
    # move worldbody leftovers (nothing expected)

act_all = ET.SubElement(mujoco, "actuator")
for r in robots:
    for a in r.findall("actuator/motor"):
        act_all.append(a)
    r.remove(r.find("actuator"))

key = ET.SubElement(ET.SubElement(mujoco, "keyframe"), "key", {"name": "ready"})
# qpos: A(7+23) + B(7+23); use AMO default pose for joints
LEG = [-0.1, 0, 0, 0.3, -0.2, 0]
WAIST = [0, 0, 0]
ARM = [0.5, 0, 0.2, 0.3]
joints = LEG + LEG + WAIST + ARM + ARM
qa = ["-0.42", "0", "0.78", "1", "0", "0", "0"] + [f"{v}" for v in joints]
qb = ["0.42", "0", "0.78", "0", "0", "0", "1"] + [f"{v}" for v in joints]
key.set("qpos", " ".join(qa + qb))
key.set("ctrl", " ".join(["0"] * 46))

xml_out = ET.tostring(mujoco, encoding="unicode")
# pretty-ish: newlines between top-level sections
open(OUT, "w", encoding="utf-8").write(xml_out)
print("written", OUT, len(xml_out), "bytes")

# sanity: load with mujoco
import mujoco as mj
import numpy as np
model = mj.MjModel.from_xml_path(OUT)
data = mj.MjData(model)
mj.mj_resetDataKeyframe(model, data, 0)
mj.mj_forward(model, data)
print("nu:", model.nu, "nq:", model.nq, "pelvis z:", float(data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, "A_pelvis"), 2]))
print("max z of fist sites:", max(float(data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, s)][2]) for s in ("A_left_fist_site", "A_right_fist_site")))
