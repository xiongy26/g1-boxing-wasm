// Face-off integration probe (Phase 2, 2026-09-29): full BoxingController
// scheduler path with comboMode+faceoff wired — validates setTracking AND the
// swapTo wrap (comboDelta + faceoff-flagged TrackingFighter) in the real
// control loop, unlike _probe_faceoff.mjs dual (direct fighters).
//
//   node tools/_probe_faceoff_ctl.mjs [seconds=42]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DURATION = Number(process.argv[2] || 42);
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(path.join(root, 'models/unitree_g1/assets'))) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(root, 'models/unitree_g1/assets', f))));
  }
}
// same spawn isolation replacement as main.js COMBO_MODE
const xml0 = fs.readFileSync(path.join(root, 'models/scene_boxing_tracking.xml'), 'utf8');
const xml = xml0.split('-0.50 0 0.761').join('-1.20 0 0.761').split(' 0.50 0 0.761').join(' 1.20 0 0.761');
if (xml === xml0) { console.error('spawn isolation did not apply'); process.exit(2); }
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);
console.log(`faceoff ctl probe dur=${DURATION}s combo=ON faceoff=ON spawn=±1.2`);

const { BoxingController, mulberry32 } = await import('../src/boxing_ai.mjs');
const { TrackingNetwork } = await import('../src/tracking_policy.mjs');
const bin = fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo.bin'));
const meta = JSON.parse(fs.readFileSync(path.join(root, 'vendor/policy/boxing_combo_meta.json'), 'utf8'));
const net = new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);

const OBJ = mujoco.mjtObj;
const id = (t, n) => mujoco.mj_name2id(model, t.value ?? t, n);
const ctl = new BoxingController(model, data, {
  mujoco,
  ids: {
    pelvisA: id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB: id(OBJ.mjOBJ_BODY, 'B_pelvis'),
    headA: id(OBJ.mjOBJ_SITE, 'A_head'), headB: id(OBJ.mjOBJ_SITE, 'B_head'),
    fistLA: id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA: id(OBJ.mjOBJ_SITE, 'A_right_fist_site'),
    fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site'),
  },
  assist: false,
  rng: mulberry32(1),
  comboMode: true,
  faceoff: true,
});
const nets = { combo: net };
ctl.setTracking('A', nets);
ctl.setTracking('B', nets);
console.log(`tracking on: yawOffset A=${ctl.fighters.A.tracking.yawOffset.toFixed(3)} B=${ctl.fighters.B.tracking.yawOffset.toFixed(3)} (faceoff B)`);
const pelA = id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelB = id(OBJ.mjOBJ_BODY, 'B_pelvis');
const torB = id(OBJ.mjOBJ_BODY, 'B_torso_link');
const DT = model.opt.timestep;
const steps = Math.round(DURATION / DT);
let minZA = 1e9, minZB = 1e9, nan = false, lastT = -1, lastPrint = -1;
for (let i = 0; i < steps; i++) {
  ctl.update(DT);
  mujoco.mj_step(model, data);
  const t = data.time;
  if (t < lastT) { nan = true; break; }  // engine auto-reset
  lastT = t;
  minZA = Math.min(minZA, data.xpos[3 * pelA + 2]);
  minZB = Math.min(minZB, data.xpos[3 * pelB + 2]);
  if (!Number.isFinite(minZB) || !ctl.fighters.B.tracking.healthy) { nan = true; break; }
  if (t - lastPrint >= 4 || i === 0) {
    lastPrint = t;
    const tq = data.xquat, o4 = 4 * torB;
    const yawB = Math.atan2(2 * (tq[o4] * tq[o4 + 3] + tq[o4 + 1] * tq[o4 + 2]), 1 - 2 * (tq[o4 + 2] ** 2 + tq[o4 + 3] ** 2));
    const gap = Math.hypot(data.xpos[3 * pelB] - data.xpos[3 * pelA], data.xpos[3 * pelB + 1] - data.xpos[3 * pelA + 1]);
    console.log(`t=${t.toFixed(1)} zA=${data.xpos[3 * pelA + 2].toFixed(2)} zB=${data.xpos[3 * pelB + 2].toFixed(2)} ` +
      `yawB=${yawB.toFixed(2)} gap=${gap.toFixed(2)} swaps=${ctl.fighters.A.clipSwaps}/${ctl.fighters.B.clipSwaps} clip=${ctl.fighters.A.clip}@${ctl.fighters.A.tracking.timeStep} ncon=${data.ncon}`);
  }
}
let fail = null;
if (nan) fail = 'NaN / engine auto-reset / unhealthy policy';
else if (minZA < 0.40 || minZB < 0.40) fail = `fell (minZ A=${minZA.toFixed(3)} B=${minZB.toFixed(3)})`;
else if (ctl.fighters.A.clipSwaps < 1 || ctl.fighters.B.clipSwaps < 1) fail = `no wrap in ${DURATION}s (swaps ${ctl.fighters.A.clipSwaps}/${ctl.fighters.B.clipSwaps})`;
else if (ctl.fighters.A.clip !== 'combo' || ctl.fighters.B.clip !== 'combo') fail = 'left combo clip';
console.log(`minZ A=${minZA.toFixed(3)} B=${minZB.toFixed(3)} swaps=${ctl.fighters.A.clipSwaps}/${ctl.fighters.B.clipSwaps}`);
if (fail) { console.log('RESULT: FAIL (' + fail + ')'); process.exit(1); }
console.log('RESULT: PASS (faceoff scheduler path)');
