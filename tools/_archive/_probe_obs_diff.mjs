// One-off probe: compare actor obs between yawDelta=0 and yawDelta=computed
// starts for the combo policy in the dual scene. Prints the first differing
// observation indices and the delta value. Diagnostics only.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(path.join(root, 'models/unitree_g1/assets'))) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(root, 'models/unitree_g1/assets', f))));
  }
}
const xml = fs.readFileSync(path.join(root, 'models/scene_boxing_tracking.xml'), 'utf8');
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

const { TrackingNetwork, TrackingFighter, yawOfQuat } = await import('../src/tracking_policy.mjs');
const bin = fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo.bin'));
const meta = JSON.parse(fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo_meta.json'), 'utf8'));
const net = new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);

const mk = (yawOffset) => {
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);
  const f = new TrackingFighter(mujoco, model, data, 'A', net, { yawOffset });
  const delta = yawOffset; // setTracking computes cur - ref0; probe feeds it directly
  f.rebaseToReference(delta);
  mujoco.mj_forward(model, data);
  f.reset(0);
  // one policy tick, capture obs
  const obs = f.buildObsForProbe?.() ?? null;
  return { f, delta };
};

// instrument: replicate policyTick's obs construction via a wrapper
function buildObs(fighter, yawOffset) {
  // copy of TrackingFighter.policyTick obs construction (read-only)
  const d = fighter.data, net = fighter.net, n = fighter.n;
  const ref = net.refAt(fighter.timeStep);
  const obs = new Float32Array(net.meta.obs_dim);
  obs.set(ref.joint_pos, 0);
  obs.set(ref.joint_vel, n);
  let o = 2 * n;
  const tq = d.xquat;
  const rqw = tq[4 * fighter.torsoBody], rqx = tq[4 * fighter.torsoBody + 1],
        rqy = tq[4 * fighter.torsoBody + 2], rqz = tq[4 * fighter.torsoBody + 3];
  const rj = ref.body_quat_w;
  const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
  let qw = rj[4 * ai], qx = rj[4 * ai + 1], qy = rj[4 * ai + 2], qz = rj[4 * ai + 3];
  if (fighter._qz) {
    const [zw, zx, zy, zz] = fighter._qz;
    const rw = zw * qw - zz * qz, rx = zw * qx + zz * qy, ry = zw * qy - zz * qx, rz = zw * qz + zz * qw;
    qw = rw; qx = rx; qy = ry; qz = rz;
  }
  const iw = rqw, ix = -rqx, iy = -rqy, iz = -rqz;
  const cw = iw * qw - ix * qx - iy * qy - iz * qz;
  const cx = iw * qx + ix * qw + iy * qz - iz * qy;
  const cy = iw * qy - ix * qz + iy * qw + iz * qx;
  const cz = iw * qz + ix * qy - iy * qx + iz * qw;
  const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const xx = cx * cx, yy = cy * cy, zz2 = cz * cz;
  M[0] = 1 - 2 * (yy + zz2); M[1] = 2 * (cx * cy - cz * cw); M[2] = 2 * (cx * cz + cy * cw);
  M[3] = 2 * (cx * cy + cz * cw); M[4] = 1 - 2 * (xx + zz2); M[5] = 2 * (cy * cz - cx * cw);
  M[6] = 2 * (cx * cz - cy * cw); M[7] = 2 * (cy * cz + cx * cw); M[8] = 1 - 2 * (xx + yy);
  obs[o++] = M[0]; obs[o++] = M[1];
  obs[o++] = M[3]; obs[o++] = M[4];
  obs[o++] = M[6]; obs[o++] = M[7];
  obs[o++] = d.qvel[fighter.freeDof + 3];
  obs[o++] = d.qvel[fighter.freeDof + 4];
  obs[o++] = d.qvel[fighter.freeDof + 5];
  for (let i = 0; i < n; i++) obs[o++] = d.qpos[fighter.qposadr[i]] - net.defaultQ[i];
  for (let i = 0; i < n; i++) obs[o++] = d.qvel[fighter.dofadr[i]];
  obs.set(fighter.lastAction, o);
  return obs;
}

// A-side: what setTracking would compute
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);
const tmp = new TrackingFighter(mujoco, model, data, 'A', net, {});
const tq = data.xquat, o4 = 4 * tmp.torsoBody;
const cur = yawOfQuat(tq[o4], tq[o4 + 1], tq[o4 + 2], tq[o4 + 3]);
const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
const r0 = net.refAt(0).body_quat_w, r4 = 4 * ai;
const refYaw = yawOfQuat(r0[r4], r0[r4 + 1], r0[r4 + 2], r0[r4 + 3]);
let delta = cur - refYaw;
delta = ((delta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
console.log(`ref0 torso yaw=${refYaw.toFixed(3)} rad, scene A yaw=${cur.toFixed(3)}, delta=${delta.toFixed(3)} rad`);

const runs = {};
for (const y of [0, -2.33]) {
  const { f } = mk(y);
  f.timeStep = 5; // step a few frames in
  f.policyTick();
  runs[y] = { obs: buildObs(f, y), action: Float32Array.from(f.lastAction) };
}
const a0 = runs[0].obs, a1 = runs[-2.33].obs;
const names = ['command[0..57]', 'anchor_ori[58..63]', 'ang_vel[64..66]', 'joint_pos[67..95]', 'joint_vel[96..124]', 'actions[125..153]'];
const bounds = [[0, 58], [58, 64], [64, 67], [67, 96], [96, 125], [125, 154]];
for (const [nm, [s, e]] of names.map((n, i) => [n, bounds[i]])) {
  let mx = 0, mi = -1;
  for (let i = s; i < e; i++) {
    const d = Math.abs(a0[i] - a1[i]);
    if (d > mx) { mx = d; mi = i; }
  }
  console.log(`${nm}: max|diff|=${mx.toExponential(3)} at idx ${mi} (values ${a0[mi].toFixed(4)} vs ${a1[mi].toFixed(4)})`);
}
