// One-off probe: track the combo policy's first-2s behaviour in the browser
// runtime (single robot, RSI start). Diagnoses boot alignment vs policy
// divergence. Not part of the regression suite.
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

const fighter = new TrackingFighter(mujoco, model, data, 'A', net);
// RSI: pose+vel exactly at reference frame 0 (rebaseToReference keeps the
// robot's world x/y, takes ref z/quat/joints/vels — the combo branch boot path)
const delta = (() => {
  const tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_torso_link');
  const tq = data.xquat, o = 4 * tb;
  const cur = yawOfQuat(tq[o], tq[o + 1], tq[o + 2], tq[o + 3]);
  const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
  const r = net.refAt(0).body_quat_w, ro = 4 * ai;
  const ref = yawOfQuat(r[ro], r[ro + 1], r[ro + 2], r[ro + 3]);
  let d = cur - ref;
  return ((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
})();
fighter.rebaseToReference(delta);
mujoco.mj_forward(model, data);

const aPelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_pelvis');
const aTorso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_torso_link');
const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);

const dt = model.opt.timestep;
const report = new Set([0, 25, 50, 100, 150, 200, 250, 300, 400].map(s => s));
for (let i = 0; i <= 400; i++) {
  if (i % 4 === 0) fighter.policyTick();
  fighter.writeTorque();
  mujoco.mj_step(model, data);
  if (report.has(i)) {
    const ref = net.refAt(Math.round(i / 4));
    const tq = data.xquat, o = 4 * aTorso;
    let qw = tq[o], qx = tq[o + 1], qy = tq[o + 2], qz = tq[o + 3];
    const r = ref.body_quat_w, ro = 4 * ai;
    let rw = r[ro], rx = r[ro + 1], ry = r[ro + 2], rz = r[ro + 3];
    // relative quat magnitude (w close to 1 = aligned)
    const cw = qw * rw + qx * rx + qy * ry + qz * rz;
    let jerr = 0, jmax = 0;
    for (let k = 0; k < 29; k++) {
      const e = Math.abs(data.qpos[fighter.qposadr[k]] - ref.joint_pos[k]);
      jerr += e; if (e > jmax) jmax = e;
    }
    let a2 = 0;
    for (let k = 0; k < 29; k++) a2 += fighter.lastAction[k] * fighter.lastAction[k];
    console.log(`t=${(i * dt).toFixed(2)}s z=${data.xpos[3 * aPelvis + 2].toFixed(3)} ` +
      `oriAlign=${cw.toFixed(4)} jerrMean=${(jerr / 29).toFixed(3)} jerrMax=${jmax.toFixed(3)} ` +
      `|action|rms=${Math.sqrt(a2 / 29).toFixed(3)} pdTarget0=${fighter.pdTarget[0].toFixed(2)} q0=${data.qpos[fighter.qposadr[0]].toFixed(2)}`);
  }
}
