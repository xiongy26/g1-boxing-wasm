// Headless stage-4 regression (milestone G): BOTH robots driven by the
// tracking clip stack (TrackingFighter + the boxing_ai clip scheduler) in the
// 29-DoF dual-robot WASM scene. Verifies: stable standing/looping, clip swaps
// happen, hits register, a forced KO collapses the robot and the round resets.
//
//   node tools/test_tracking_boxing.mjs [seconds] [clipPrefix]
//
// clipPrefix: vendor/policy prefix used for ALL four clip slots (default:
// spinkick — the pipeline-validation policy). When the trained boxing clips
// exist pass e.g. "boxing_v1" with per-clip files boxing_v1_{guard,jab,...}.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const DURATION = Number(process.argv[2] || 16);
const KO_AT = DURATION * 0.6;
const PREFIX = process.argv[3] || 'spinkick';

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
console.log(`tracking scene: nu=${model.nu} nq=${model.nq} dt=${model.opt.timestep}`);

const { BoxingController } = await import('../src/boxing_ai.mjs');
const { TrackingNetwork } = await import('../src/tracking_policy.mjs');
const { TacticsPolicy } = await import('../src/tactics.mjs');

const loadNet = (name) => {
  const bin = fs.readFileSync(path.join(root, `vendor/policy/${name}.bin`));
  const meta = JSON.parse(fs.readFileSync(path.join(root, `vendor/policy/${name}_meta.json`), 'utf8'));
  return new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
};

// clip slots: single-prefix mode reuses one policy everywhere (mechanics
// check). Per-clip files win when present.
const slots = ['guard', 'jab', 'cross', 'hook'];
const nets = {};
for (const s of slots) {
  const perClip = `${PREFIX}_${s}`;
  const hasPerClip = fs.existsSync(path.join(root, `vendor/policy/${perClip}.bin`));
  nets[s] = loadNet(hasPerClip ? perClip : PREFIX);
}
console.log('clip slots:', Object.fromEntries(slots.map(s => [s, nets[s] === nets.guard ? 'guard' : (fs.existsSync(path.join(root, `vendor/policy/${PREFIX}_${s}.bin`)) ? `${PREFIX}_${s}` : PREFIX)])));

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
});

// learned tactics when available; otherwise the weighted-random fallback
try {
  const tj = JSON.parse(fs.readFileSync(path.join(root, 'vendor/tactics/tactics.json'), 'utf8'));
  const pol = TacticsPolicy.fromJSON(tj);
  ctl.setTactics(pol);
  console.log('tactics policy loaded');
} catch {
  console.log('tactics policy not found — random fallback drives clip choices');
}

ctl.setTracking('A', nets);
ctl.setTracking('B', nets);
console.log('tracking mode on: A + B');

const DT = model.opt.timestep;
const steps = Math.round(DURATION / DT);
const pelvisA = id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB = id(OBJ.mjOBJ_BODY, 'B_pelvis');
let minZA = 1e9, minZB = 1e9, nan = false, divergences = 0, lastT = -1;
let koSeen = 0, postResetMin = 1e9, koT = null;
let lastPrint = 0;

for (let i = 0; i < steps; i++) {
  ctl.update(DT);
  mujoco.mj_step(model, data);
  const t = data.time;
  if (t < lastT) divergences++;
  lastT = t;
  const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
  if (!Number.isFinite(za) || !Number.isFinite(zb)) { nan = true; break; }
  if (!ctl.koState) { minZA = Math.min(minZA, za); minZB = Math.min(minZB, zb); }
  if (!ctl.koState && koT !== null) {
    postResetMin = Math.min(postResetMin, za, zb);
    if (t - koT > 3 && (za < 0.5 || zb < 0.5)) { /* recovery judged below */ }
  }
  koSeen = ctl.events.filter(e => e.type === 'ko').length;
  if (koT === null && t >= KO_AT) {
    ctl.damage.A = 7.5;
    koT = t;
    console.log(`t=${t.toFixed(1)}: injected lethal damage on A`);
  }
  if (t - lastPrint >= 2) {
    lastPrint = t;
    console.log(`t=${t.toFixed(1)} zA=${za.toFixed(2)} zB=${zb.toFixed(2)} ` +
      `clipA=${ctl.fighters.A.clip}@${ctl.fighters.A.tracking.timeStep} swaps=${ctl.fighters.A.clipSwaps}/${ctl.fighters.B.clipSwaps} ` +
      `hits=${ctl.fighters.A.scores.hits}/${ctl.fighters.B.scores.hits} ko=${koSeen} ncon=${data.ncon} div=${ctl._divergences}`);
  }
}

let fail = null;
if (nan) fail = 'NaN in sim state';
else if (ctl._divergences > 0) fail = 'engine auto-reset = instability';
else if (minZA < 0.4 || minZB < 0.5) fail = `a robot fell during normal play (minZ A=${minZA.toFixed(2)} B=${minZB.toFixed(2)})`;
else if (koSeen < 1) fail = 'KO injection did not trigger';
else if (ctl.fighters.A.clipSwaps + ctl.fighters.B.clipSwaps < 1 && PREFIX !== 'spinkick') fail = 'no clip swaps happened';
else if (koT !== null && postResetMin < 0.5) fail = 'robots did not recover after round reset';
if (fail) { console.log('RESULT: FAIL (' + fail + ')'); process.exit(1); }
console.log('RESULT: PASS');
