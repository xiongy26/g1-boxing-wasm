// Headless sim2sim test for the tracking stack (milestone F infra check):
// the pretrained spinkick ONNX (via TrackingNetwork/TrackingFighter, the same
// runtime the boxing clip policies will use) drives ONE 29-DoF robot in the
// WASM dual-robot tracking scene. PASS = the robot tracks the motion through
// the clip without falling (pelvis z > 0.4), no NaN, no engine auto-reset.
//
//   node tools/test_tracking_sim.mjs [clips]
//
// clips: space-separated prefixes under vendor/policy (default: spinkick).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const prefixes = process.argv.slice(2).length ? process.argv.slice(2) : ['spinkick'];

const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
const assetsDir = path.join(root, 'models/unitree_g1/assets');
for (const f of fs.readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(assetsDir, f))));
  }
}
const xml = fs.readFileSync(path.join(root, 'models/scene_boxing_tracking.xml'), 'utf8');
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
console.log(`scene: nu=${model.nu} nq=${model.nq} timestep=${model.opt.timestep}`);

const { TrackingNetwork, TrackingFighter } = await import('../src/tracking_policy.mjs');

let anyFail = false;
for (const prefix of prefixes) {
  const bin = fs.readFileSync(path.join(root, `vendor/policy/${prefix}.bin`));
  const meta = JSON.parse(fs.readFileSync(path.join(root, `vendor/policy/${prefix}_meta.json`), 'utf8'));
  const net = new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);

  const fighter = new TrackingFighter(mujoco, model, data, 'A', net);
  // RSI: initialize robot A exactly at the clip's frame-0 pose (pelvis pose +
  // joint targets), like the training env does. The keyframe spawn faces yaw 0
  // but clips may start at any yaw — the policy is blind to anchor position
  // error (no_state_estimation export), so this must match.
  const ref0 = net.refAt(0);
  const aFj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, 'A_floating_base_joint');
  const aq = model.jnt_qposadr[aFj];
  for (let k = 0; k < 3; k++) data.qpos[aq + k] = ref0.body_pos_w[k];
  for (let k = 0; k < 4; k++) data.qpos[aq + 3 + k] = ref0.body_quat_w[k];
  for (let i = 0; i < net.n; i++) data.qpos[fighter.qposadr[i]] = ref0.joint_pos[i];
  // move B out of the way — this test is single-robot tracking
  const bPelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, 'B_pelvis');
  const bq = model.jnt_qposadr[bPelvis];
  data.qpos[bq] = 5.0;
  mujoco.mj_forward(model, data);
  fighter.reset(0);

  const dt = model.opt.timestep;
  const clipFrames = net.motionLen;
  const steps = Math.round((clipFrames / net.meta.motion.fps + 0.5) / dt);  // clip + 0.5s margin
  const aPelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_pelvis');
  const aTorso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_torso_link');

  let minZ = 1e9, nan = false, maxAbsCtrl = 0, divergences = 0, lastT = -1;
  let sumAnchorErr = 0, anchorSamples = 0, sumAnchorErrLate = 0, lateSamples = 0;
  const clipSeconds = clipFrames / net.meta.motion.fps;

  for (let i = 0; i < steps; i++) {
    fighter.physicsStep(4);
    mujoco.mj_step(model, data);
    const t = data.time;
    if (t < lastT) divergences++;
    lastT = t;
    const z = data.xpos[3 * aPelvis + 2];
    if (!Number.isFinite(z)) { nan = true; break; }
    minZ = Math.min(minZ, z);
    for (let k = 0; k < 29; k++) {
      const v = Math.abs(data.ctrl[fighter.actId[k]]);
      if (!Number.isFinite(v)) { nan = true; break; }
      if (v > maxAbsCtrl) maxAbsCtrl = v;
    }
    // anchor tracking error (torso world position vs reference at current step)
    const ref = net.refAt(Math.min(fighter.timeStep, net.maxStep));
    const nb = net.nBody;
    const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
    const dx = data.xpos[3 * aTorso] - ref.body_pos_w[3 * ai];
    const dy = data.xpos[3 * aTorso + 1] - ref.body_pos_w[3 * ai + 1];
    const dz = data.xpos[3 * aTorso + 2] - ref.body_pos_w[3 * ai + 2];
    const err = Math.hypot(dx, dy, dz);
    sumAnchorErr += err; anchorSamples++;
    if (t > clipSeconds * 0.5) { sumAnchorErrLate += err; lateSamples++; }
    if (fighter.clipDone && t > clipSeconds + 0.4) break;
  }
  const meanErr = sumAnchorErr / Math.max(anchorSamples, 1);
  const meanErrLate = sumAnchorErrLate / Math.max(lateSamples, 1);
  console.log(`${prefix}: clip ${clipFrames}f/${clipSeconds.toFixed(2)}s t=${data.time.toFixed(2)} ` +
    `minZ=${minZ.toFixed(3)} meanTorsoErr=${meanErr.toFixed(3)} lateErr=${meanErrLate.toFixed(3)} ` +
    `maxCtrl=${maxAbsCtrl.toFixed(1)} divergences=${divergences}`);

  let fail = null;
  if (nan) fail = 'NaN in sim state';
  else if (divergences > 0) fail = 'engine auto-reset';
  else if (minZ < 0.4) fail = 'robot fell (pelvis z < 0.4)';
  else if (!fighter.healthy) fail = 'policy produced non-finite actions';
  else if (meanErrLate > 0.5) fail = `torso tracking error too large (${meanErrLate.toFixed(2)}m)`;
  if (fail) { console.log(`${prefix}: FAIL (${fail})`); anyFail = true; }
  else console.log(`${prefix}: PASS`);
}
console.log('RESULT: ' + (anyFail ? 'FAIL' : 'PASS'));
process.exit(anyFail ? 1 : 0);
