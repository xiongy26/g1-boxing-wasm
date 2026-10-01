// One-off probe: survival sweep over start bins for the combo policy in the
// browser runtime (single robot, RSI at arbitrary reference frame). Diagnoses
// which motion segments the policy can play. Not part of the regression suite.
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

const { TrackingNetwork, TrackingFighter, yawOfQuat } = await import('../src/tracking_policy.mjs');
const bin = fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo.bin'));
const meta = JSON.parse(fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo_meta.json'), 'utf8'));
const net = new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);

const BINS = [0.0, 0.25, 0.4, 0.55, 0.65, 0.75, 0.85, 0.95];
const SIM_S = 20;

for (const frac of BINS) {
  const startStep = Math.round(frac * net.maxStep);
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);
  const fighter = new TrackingFighter(mujoco, model, data, 'A', net);

  // RSI from an arbitrary frame: state <- refAt(startStep), yaw-anchored
  const tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_torso_link');
  const tq = data.xquat, o = 4 * tb;
  const cur = yawOfQuat(tq[o], tq[o + 1], tq[o + 2], tq[o + 3]);
  const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
  const r0 = net.refAt(startStep).body_quat_w, ro = 4 * ai;
  const ref = yawOfQuat(r0[ro], r0[ro + 1], r0[ro + 2], r0[ro + 3]);
  const delta = ((cur - ref + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

  const d = data, refS = net.refAt(startStep);
  for (let i = 0; i < net.n; i++) {
    d.qpos[fighter.qposadr[i]] = refS.joint_pos[i];
    d.qvel[fighter.dofadr[i]] = refS.joint_vel[i];
  }
  const zw = Math.cos(delta / 2), zz = Math.sin(delta / 2);
  const qw = refS.body_quat_w[0], qx = refS.body_quat_w[1], qy = refS.body_quat_w[2], qz = refS.body_quat_w[3];
  d.qpos[fighter.freeDof + 3] = zw * qw - zz * qz;
  d.qpos[fighter.freeDof + 4] = zw * qx + zz * qy;
  d.qpos[fighter.freeDof + 5] = zw * qy - zz * qx;
  d.qpos[fighter.freeDof + 6] = zw * qz + zz * qw;
  d.qpos[fighter.freeDof + 2] = refS.body_pos_w[2];
  for (let k = 0; k < 6; k++) d.qvel[fighter.freeDof + k] = 0;
  fighter.reset(startStep);
  mujoco.mj_forward(model, data);

  const aPelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_pelvis');
  const dt = model.opt.timestep;
  const steps = Math.round(SIM_S / dt);
  let minZ = 1e9, fellAt = -1;
  for (let i = 0; i < steps; i++) {
    fighter.physicsStep(4);
    mujoco.mj_step(model, data);
    const z = data.xpos[3 * aPelvis + 2];
    minZ = Math.min(minZ, z);
    if (z < 0.35 && fellAt < 0) { fellAt = i * dt; break; }
  }
  const status = fellAt < 0 ? `SURVIVED ${SIM_S}s` : `fell at t+${fellAt.toFixed(2)}s`;
  console.log(`bin ${frac.toFixed(2)} (frame ${startStep}, t=${(startStep / 50).toFixed(1)}s): ` +
    `${status} minZ=${minZ.toFixed(3)}`);
}
