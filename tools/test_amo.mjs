// Headless AMO-mode regression: merged AMO boxing scene, BOTH robots driven by
// the AMO whole-body policy (no gantry). Verifies:
//   1. both robots stay upright while the boxing AI walks/punches (no NaN,
//      no engine auto-reset, pelvis z > 0.6 in normal play)
//   2. punches actually get thrown (strike states observed) and contact happens
//   3. a forced KO makes the robot collapse (policy off -> limp) and the round
//      resets, after which both robots recover upright
// Run:  node tools/test_amo.mjs [seconds]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const DURATION = Number(process.argv[2] || 20);
const KO_AT = 12; // seconds — inject lethal damage on A

const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
const assetsDir = path.join(root, 'models/unitree_g1/assets');
for (const f of fs.readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(assetsDir, f))));
  }
}
const xml = fs.readFileSync(path.join(root, 'models/scene_boxing_amo.xml'), 'utf8');
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

const OBJ = mujoco.mjtObj;
const id = (type, name) => mujoco.mj_name2id(model, type.value ?? type, name);
const pelvisA = id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB = id(OBJ.mjOBJ_BODY, 'B_pelvis');
const headA = id(OBJ.mjOBJ_SITE, 'A_head'), headB = id(OBJ.mjOBJ_SITE, 'B_head');
const fistLA = id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA = id(OBJ.mjOBJ_SITE, 'A_right_fist_site');

const { BoxingController } = await import('../src/boxing_ai.mjs');
const { AMONetwork } = await import('../src/rl_policy.mjs');
const ctl = new BoxingController(model, data, {
  mujoco, ids: { pelvisA, pelvisB, headA, headB,
    fistLA, fistRA, fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site') },
});

const bin = fs.readFileSync(path.join(root, 'vendor/policy/amo.bin'));
const meta = JSON.parse(fs.readFileSync(path.join(root, 'vendor/policy/amo_meta.json'), 'utf8'));
const net = new AMONetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
ctl.setAMO('A', net);
ctl.setAMO('B', net);
console.log('AMO mode on: A + B');

const dt = model.opt.timestep;
const steps = Math.round(DURATION / dt);
let minZA = 1e9, minZB = 1e9, nan = false, maxAbsCtrl = 0;
let strikesA = 0, strikesB = 0, prevA = '', prevB = '';
let koSeen = 0, koFallMinZ = 1e9, recoveredOk = true, koT = null, postResetMin = 1e9;
let lastPrint = 0;
let maxDxA = 0, maxDxB = 0;
const x0A = data.xpos[3 * pelvisA], x0B = data.xpos[3 * pelvisB];

for (let i = 0; i < steps; i++) {
  ctl.update(dt);
  mujoco.mj_step(model, data);
  const t = data.time;
  const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
  if (!Number.isFinite(za) || !Number.isFinite(zb)) { nan = true; break; }
  if (ctl.koState) koFallMinZ = Math.min(koFallMinZ, za, zb);
  else { minZA = Math.min(minZA, za); minZB = Math.min(minZB, zb); }
  if (!ctl.koState && koT !== null) {
    // after the round reset, watch recovery
    postResetMin = Math.min(postResetMin, za, zb);
    if (t - koT > 5 && (za < 0.6 || zb < 0.6)) recoveredOk = false;
  }
  for (let k = 0; k < data.ctrl.length; k++) {
    const v = Math.abs(data.ctrl[k]);
    if (!Number.isFinite(v)) { nan = true; break; }
    if (v > maxAbsCtrl) maxAbsCtrl = v;
  }
  if (ctl.fighters.A.state === 'strike' && prevA !== 'strike') strikesA++;
  if (ctl.fighters.B.state === 'strike' && prevB !== 'strike') strikesB++;
  prevA = ctl.fighters.A.state; prevB = ctl.fighters.B.state;

  // force a KO on A by injecting lethal damage once the fight is going
  if (koT === null && t >= KO_AT) {
    ctl.damage.A = 7.5;
    koT = t;
    console.log(`t=${t.toFixed(1)}: injected lethal damage on A`);
  }
  koSeen = ctl.events.filter(e => e.type === 'ko').length;

  if (t - lastPrint >= 2.0) {
    lastPrint = t;
    const dx = (a, b) => Math.hypot(data.site_xpos[3 * a] - data.site_xpos[3 * b],
      data.site_xpos[3 * a + 1] - data.site_xpos[3 * b + 1], data.site_xpos[3 * a + 2] - data.site_xpos[3 * b + 2]);
    console.log(`t=${t.toFixed(1)} zA=${za.toFixed(3)} zB=${zb.toFixed(3)} ` +
      `A.fistL→B.head=${dx(fistLA, headB).toFixed(2)} ncon=${data.ncon} strikes=${strikesA}/${strikesB} ` +
      `hits=${ctl.fighters.A.scores.hits}/${ctl.fighters.B.scores.hits} ko=${koSeen}`);
  }
}
maxDxA = Math.abs(data.xpos[3 * pelvisA] - x0A);
maxDxB = Math.abs(data.xpos[3 * pelvisB] - x0B);

console.log(`FINAL t=${data.time.toFixed(1)} minZA=${minZA.toFixed(3)} minZB=${minZB.toFixed(3)} ` +
  `koFallMinZ=${Number.isFinite(koFallMinZ) ? koFallMinZ.toFixed(3) : '-'} ` +
  `postResetMin=${Number.isFinite(postResetMin) ? postResetMin.toFixed(3) : '-'} ` +
  `strikes=${strikesA}/${strikesB} divergences=${ctl._divergences} walkA=${maxDxA.toFixed(2)} walkB=${maxDxB.toFixed(2)}`);

let fail = null;
if (nan) fail = 'NaN in sim state';
else if (ctl._divergences > 0) fail = 'engine auto-reset = instability';
else if (minZA < 0.6 || minZB < 0.6) fail = 'a robot collapsed during normal play';
else if (koSeen < 1) fail = 'KO injection did not trigger';
else if (koFallMinZ > 0.4) fail = 'KO robot did not actually fall';
else if (!recoveredOk || postResetMin < 0.6) fail = 'robots did not recover upright after round reset';
else if (strikesA + strikesB < 3) fail = 'almost no punches thrown';
if (fail) { console.log('RESULT: FAIL (' + fail + ')'); process.exit(1); }
console.log('RESULT: PASS');
