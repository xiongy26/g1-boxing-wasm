// Face-off probe (Phase 2, 2026-09-29): re-test non-zero yaw rebase delta with
// the CURRENT deployed combo policy. Round 1 measured "non-zero delta falls in
// 1.5s" BEFORE the rebaseToReference freeDof→freeQpos fix; this probe re-runs
// the experiment post-fix. Production code untouched — B-side delta is applied
// here by rebuilding the fighter (mirrors setTracking's combo branch).
//
//   node tools/_probe_faceoff.mjs obsdiff           # obs invariance delta=0 vs delta=π
//   node tools/_probe_faceoff.mjs singleB [30]      # B alone, yaw π, ±1.2m slot
//   node tools/_probe_faceoff.mjs dual [30]         # A delta=0 + B delta=π face-off
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MODE = process.argv[2] || 'dual';
const DURATION = Number(process.argv[3] || 30);
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(path.join(root, 'models/unitree_g1/assets'))) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(root, 'models/unitree_g1/assets', f))));
  }
}
// COMBO_MODE spawn isolation, same replacement as main.js; singleB parks A far
let xml = fs.readFileSync(path.join(root, 'models/scene_boxing_tracking.xml'), 'utf8');
const xml0 = xml;
xml = xml.split('-0.50 0 0.761').join(MODE === 'singleB' ? '-5.00 0 0.761' : '-1.20 0 0.761')
         .split(' 0.50 0 0.761').join(' 1.20 0 0.761');
if (xml === xml0) { console.error('spawn isolation did not apply'); process.exit(2); }
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);
console.log(`faceoff probe mode=${MODE} dur=${DURATION}s spawn=±1.2${MODE === 'singleB' ? ' (A parked -5m)' : ''}`);

const { TrackingNetwork, TrackingFighter, yawOfQuat } = await import('../src/tracking_policy.mjs');
const bin = fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo.bin'));
const meta = JSON.parse(fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo_meta.json'), 'utf8'));
const net = new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
const r0 = net.refAt(0).body_quat_w, r4 = 4 * ai;
const ref0Yaw = yawOfQuat(r0[r4], r0[r4 + 1], r0[r4 + 2], r0[r4 + 3]);
console.log(`clip ref0 torso yaw=${ref0Yaw.toFixed(3)} rad, clip ${net.motionLen}f`);
// B spawn yaw = π (scene keyframe quat 0 0 0 1); face-off delta = spawn - ref0,
// same formula as rebaseYawDelta — B ends up facing ref0+delta ≈ π (toward A)
const DELTA_B = ((Math.PI - ref0Yaw + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
console.log(`B-side face-off delta=${DELTA_B.toFixed(3)} rad (spawn π − ref0) → B world yaw=${(ref0Yaw + DELTA_B).toFixed(3)}`);

// ---- obs invariance check: same spawn state, delta 0 vs π → obs must match --
function buildObs(fighter) {
  // read-only replica of TrackingFighter.policyTick obs construction
  const d = fighter.data, n = fighter.n;
  const ref = fighter.net.refAt(fighter.timeStep);
  const obs = new Float32Array(fighter.net.meta.obs_dim);
  obs.set(ref.joint_pos, 0);
  obs.set(ref.joint_vel, n);
  let o = 2 * n;
  const tq = d.xquat;
  const rqw = tq[4 * fighter.torsoBody], rqx = tq[4 * fighter.torsoBody + 1],
        rqy = tq[4 * fighter.torsoBody + 2], rqz = tq[4 * fighter.torsoBody + 3];
  const rj = ref.body_quat_w;
  let qw = rj[4 * ai], qx = rj[4 * ai + 1], qy = rj[4 * ai + 2], qz = rj[4 * ai + 3];
  if (fighter._qz) {
    const [zw, , , zz] = fighter._qz;
    const fx = fighter._faceoffFix;
    const rw = zw * qw - zz * qz;
    const rx = fx ? zw * qx - zz * qy : zw * qx + zz * qy;
    const ry = fx ? zw * qy + zz * qx : zw * qy - zz * qx;
    const rz = zw * qz + zz * qw;
    qw = rw; qx = rx; qy = ry; qz = rz;
  }
  const iw = rqw, ix = -rqx, iy = -rqy, iz = -rqz;
  // q_rel = q_robot^-1 ⊗ q_ref (wxyz Hamilton), same as policyTick
  const W = iw * qw - ix * qx - iy * qy - iz * qz;
  const X = iw * qx + ix * qw + iy * qz - iz * qy;
  const Y = iw * qy - ix * qz + iy * qw + iz * qx;
  const Z = iw * qz + ix * qy - iy * qx + iz * qw;
  const M = [
    1 - 2 * (Y * Y + Z * Z), 2 * (X * Y - Z * W), 2 * (X * Z + Y * W),
    2 * (X * Y + Z * W), 1 - 2 * (X * X + Z * Z), 2 * (Y * Z - X * W),
    2 * (X * Z - Y * W), 2 * (Y * Z + X * W), 1 - 2 * (X * X + Y * Y),
  ];
  obs[o++] = M[0]; obs[o++] = M[1];
  obs[o++] = M[3]; obs[o++] = M[4];
  obs[o++] = M[6]; obs[o++] = M[7];
  obs[o++] = d.qvel[fighter.freeDof + 3];
  obs[o++] = d.qvel[fighter.freeDof + 4];
  obs[o++] = d.qvel[fighter.freeDof + 5];
  for (let i = 0; i < n; i++) obs[o++] = d.qpos[fighter.qposadr[i]] - fighter.net.defaultQ[i];
  for (let i = 0; i < n; i++) obs[o++] = d.qvel[fighter.dofadr[i]];
  obs.set(fighter.lastAction, o);
  return obs;
}

if (MODE === 'obsdiff') {
  const mk = (delta) => {
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    const f = new TrackingFighter(mujoco, model, data, 'B', net, { yawOffset: delta, faceoff: delta !== 0 });
    f.reset(0);
    f.rebaseToReference(delta);
    mujoco.mj_forward(model, data);
    f.timeStep = 5;
    f.policyTick();
    return { obs: buildObs(f), act: Float32Array.from(f.lastAction), q: Float32Array.from(data.qpos) };
  };
  const z0 = mk(0), p1 = mk(DELTA_B);
  const names = ['command[0..57]', 'anchor_ori[58..63]', 'ang_vel[64..66]', 'joint_pos[67..95]', 'joint_vel[96..124]', 'actions[125..153]'];
  const bounds = [[0, 58], [58, 64], [64, 67], [67, 96], [96, 125], [125, 154]];
  let worst = 0;
  for (const [nm, [s, e]] of names.map((n2, i) => [n2, bounds[i]])) {
    let mx = 0, mi = -1;
    for (let i = s; i < e; i++) {
      const d2 = Math.abs(z0.obs[i] - p1.obs[i]);
      if (d2 > mx) { mx = d2; mi = i; }
    }
    worst = Math.max(worst, mx);
    if (mi < 0) { console.log(`${nm}: max|diff|=0 (identical)`); continue; }
    console.log(`${nm}: max|diff|=${mx.toExponential(3)} at idx ${mi} (${z0.obs[mi].toFixed(4)} vs ${p1.obs[mi].toFixed(4)})`);
  }
  let amx = 0;
  for (let i = 0; i < z0.act.length; i++) amx = Math.max(amx, Math.abs(z0.act[i] - p1.act[i]));
  console.log(`action max|diff|=${amx.toExponential(3)} (obs worst=${worst.toExponential(3)})`);
  console.log('OBS-INVARIANT: ' + (amx < 1e-3 ? 'YES (face-off obs identical to delta=0)' : 'NO — leakage above'));
  process.exit(0);
}

// ---- survival runs -----------------------------------------------------------
const DT = model.opt.timestep;
const steps = Math.round(DURATION / DT);
const mkFighter = (side, delta) => {
  const f = new TrackingFighter(mujoco, model, data, side, net, { yawOffset: delta, faceoff: delta !== 0 });
  f.reset(0);
  f.rebaseToReference(delta);
  return f;
};
const A = MODE === 'dual' ? mkFighter('A', 0) : null;
const B = mkFighter('B', DELTA_B);
mujoco.mj_forward(model, data);
const pelA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'A_pelvis');
const pelB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'B_pelvis');
const torB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'B_torso_link');
const clipS = net.motionLen / net.meta.motion.fps;
let minZA = 1e9, minZB = 1e9, nan = false, div = 0, lastT = -1, wrapsA = 0, wrapsB = 0, lastPrint = -1;
for (let i = 0; i < steps; i++) {
  if (A) { if (A.stepCount % 4 === 0) A.policyTick(); A.stepCount++; A.writeTorque(); }
  if (B.stepCount % 4 === 0) B.policyTick();
  B.stepCount++; B.writeTorque();
  mujoco.mj_step(model, data);
  const t = data.time;
  if (t < lastT) div++;
  lastT = t;
  if (A) minZA = Math.min(minZA, data.xpos[3 * pelA + 2]);
  minZB = Math.min(minZB, data.xpos[3 * pelB + 2]);
  if (!Number.isFinite(minZB)) { nan = true; break; }
  // combo wrap: in-place rebase with the side's own delta (production wraps delta=0)
  if (!B.clipDone && B.timeStep >= net.motionLen - 1) { B.clipDone = true; }
  if (B.timeStep >= net.motionLen) { B.timeStep = 0; B.clipDone = false; B.rebaseToReference(DELTA_B); wrapsB++; }
  if (A) {
    if (A.timeStep >= net.motionLen) { A.timeStep = 0; A.clipDone = false; A.rebaseToReference(0); wrapsA++; }
  }
  if (t - lastPrint >= 2 || i === 0) {
    lastPrint = t;
    const zb = data.xpos[3 * pelB + 2], xb = data.xpos[3 * pelB], yb = data.xpos[3 * pelB + 1];
    const tq = data.xquat, o4 = 4 * torB;
    const yawB = yawOfQuat(tq[o4], tq[o4 + 1], tq[o4 + 2], tq[o4 + 3]);
    const gap = A ? Math.hypot(data.xpos[3 * pelB] - data.xpos[3 * pelA], data.xpos[3 * pelB + 1] - data.xpos[3 * pelA + 1]) : NaN;
    console.log(`t=${t.toFixed(1)} zB=${zb.toFixed(2)} yawB=${yawB.toFixed(2)} posB=(${xb.toFixed(2)},${yb.toFixed(2)})` +
      (A ? ` gap=${gap.toFixed(2)} zA=${data.xpos[3 * pelA + 2].toFixed(2)}` : '') +
      ` step=${B.timeStep} ncon=${data.ncon} wraps=${wrapsA}/${wrapsB}`);
  }
}
console.log(`minZ A=${A ? minZA.toFixed(3) : 'n/a'} B=${minZB.toFixed(3)} wraps=${wrapsA}/${wrapsB} div=${div} clipLoop=${clipS.toFixed(2)}s`);
const fallB = minZB < 0.4;
const fallA = A && minZA < 0.4;
if (nan || div > 0) { console.log('RESULT: FAIL (NaN/engine reset)'); process.exit(1); }
if (fallB || fallA) { console.log(`RESULT: FAIL (fell: A=${fallA} B=${fallB})`); process.exit(1); }
console.log('RESULT: PASS (no fall across window)');
