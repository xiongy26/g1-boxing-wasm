// Headless RL-mode regression: loads the merged boxing scene, switches both
// robots to pretrained KungfuBot (PBHC) policies (A = horse-stance punch,
// B = horse-stance stance) and verifies the sim stays physical.
// Run:  node tools/test_rl.mjs [seconds]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const DURATION = Number(process.argv[2] || 20);

const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
const assetsDir = path.join(root, 'models/unitree_g1/assets');
for (const f of fs.readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('assets/' + f, new Uint8Array(fs.readFileSync(path.join(assetsDir, f))));
  }
}
const xml = fs.readFileSync(path.join(root, 'models/scene_boxing.xml'), 'utf8');
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
const ctl = new BoxingController(model, data, {
  mujoco, ids: { pelvisA, pelvisB, headA, headB,
    fistLA, fistRA, fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site') },
});
ctl.assist = true; // RL robots ignore the gantry anyway; keeps scripted fallback sane

// A: punch policy, B: stance policy
const punchWeights = fs.readFileSync(path.join(root, 'vendor/policy/horse_stance_punch.bin')).buffer;
const poseWeights = fs.readFileSync(path.join(root, 'vendor/policy/horse_stance_pose.bin')).buffer;
ctl.setRL('A', { name: 'horse_stance_punch', weights: punchWeights, motionDur: 200 / 30 });
ctl.setRL('B', { name: 'horse_stance_pose', weights: poseWeights, motionDur: 210 / 30 });
console.log('RL mode on: A=horse_stance_punch, B=horse_stance_pose');

const dt = model.opt.timestep;
const steps = Math.round(DURATION / dt);
let minZA = 1e9, minZB = 1e9, maxAbsCtrl = 0, nan = false, koEvents = 0;
let lastPrint = 0;
for (let i = 0; i < steps; i++) {
  ctl.update(dt);
  mujoco.mj_step(model, data);
  const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
  if (Number.isFinite(za)) minZA = Math.min(minZA, za);
  if (Number.isFinite(zb)) minZB = Math.min(minZB, zb);
  if (!Number.isFinite(za) || !Number.isFinite(zb)) { nan = true; break; }
  for (let k = 0; k < data.ctrl.length; k++) {
    const v = Math.abs(data.ctrl[k]);
    if (!Number.isFinite(v)) { nan = true; break; }
    if (v > maxAbsCtrl) maxAbsCtrl = v;
  }
  koEvents = ctl.events.filter(e => e.type === 'ko').length;
  if (data.time - lastPrint >= 1.0) {
    lastPrint = data.time;
    const dx = (a, b) => Math.hypot(data.site_xpos[3 * a] - data.site_xpos[3 * b],
      data.site_xpos[3 * a + 1] - data.site_xpos[3 * b + 1], data.site_xpos[3 * a + 2] - data.site_xpos[3 * b + 2]);
    console.log(`t=${data.time.toFixed(1)} zA=${za.toFixed(3)} zB=${zb.toFixed(3)} ` +
      `A.fistL→B.head=${dx(fistLA, headB).toFixed(2)} ncon=${data.ncon} ko=${koEvents} ` +
      `A.hits=${ctl.fighters.A.scores.hits} B.hits=${ctl.fighters.B.scores.hits}`);
  }
}
console.log(`FINAL t=${data.time.toFixed(1)} minZA=${minZA.toFixed(3)} minZB=${minZB.toFixed(3)} ` +
  `maxAbsCtrl=${maxAbsCtrl.toFixed(2)} koRounds=${koEvents} divergences=${ctl._divergences}`);
if (nan) { console.log('RESULT: FAIL (NaN in sim state)'); process.exit(1); }
if (ctl._divergences > 0) { console.log('RESULT: FAIL (engine auto-reset = instability)'); process.exit(1); }
if (minZA < 0.45 || minZB < 0.45) { console.log('RESULT: FAIL (a robot collapsed)'); process.exit(1); }
if (ctl.fighters.A.scores.hits + ctl.fighters.B.scores.hits < 1 && DURATION >= 20) {
  console.log('RESULT: WARN (no hits — policies ran but never made contact)');
  process.exit(0);
}
console.log('RESULT: PASS');
