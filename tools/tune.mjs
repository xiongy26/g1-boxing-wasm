// Headless tuning harness: loads the merged boxing scene with the WASM bindings,
// runs the shared boxing controller, prints diagnostics. No browser needed.
// Run:  node tools/tune.mjs [seconds]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const DURATION = Number(process.argv[2] || 6);

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
console.log(`loaded: nq=${model.nq} nu=${model.nu} nbody=${model.nbody} ngeom=${model.ngeom} nkey=${model.nkey}`);

mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

// name lookups
const OBJ = mujoco.mjtObj;
const id = (type, name) => mujoco.mj_name2id(model, type.value ?? type, name);
const pelvisA = id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB = id(OBJ.mjOBJ_BODY, 'B_pelvis');
const fistLA = id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA = id(OBJ.mjOBJ_SITE, 'A_right_fist_site');
const fistLB = id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB = id(OBJ.mjOBJ_SITE, 'B_right_fist_site');
const headA = id(OBJ.mjOBJ_SITE, 'A_head'), headB = id(OBJ.mjOBJ_SITE, 'B_head');
const geomFistA = new Set(['A_left_fist', 'A_right_fist'].map(n => id(OBJ.mjOBJ_GEOM, n)));
const geomFistB = new Set(['B_left_fist', 'B_right_fist'].map(n => id(OBJ.mjOBJ_GEOM, n)));

const { BoxingController } = await import('../src/boxing_ai.mjs');
const ctl = new BoxingController(model, data, {
  mujoco, ids: { pelvisA, pelvisB, headA, headB, fistLA, fistRA, fistLB, fistRB },
});

const dt = model.opt.timestep;
const steps = Math.round(DURATION / dt);
let hits = 0, blocks = 0, lastPrint = 0;
for (let i = 0; i < steps; i++) {
  ctl.update(dt);
  mujoco.mj_step(model, data);
  hits += ctl.takeHitCount(); blocks += ctl.takeBlockCount();
  if (data.time - lastPrint >= 0.5) {
    lastPrint = data.time;
    const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
    const dLH = dist3(data.site_xpos, fistLA, headB);
    const dRH = dist3(data.site_xpos, fistRA, headB);
    const dBH = dist3(data.site_xpos, fistLB, headA);
    console.log(`t=${data.time.toFixed(2)} zA=${za.toFixed(3)} zB=${zb.toFixed(3)} ` +
      `A→Bhead L/R=${dLH.toFixed(2)}/${dRH.toFixed(2)} B→Ahead=${dBH.toFixed(2)} hits=${hits} blocks=${blocks}`);
  }
}
const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
console.log(`FINAL t=${data.time.toFixed(2)} zA=${za.toFixed(3)} zB=${zb.toFixed(3)} hits=${hits} blocks=${blocks} koRounds=${ctl.events.filter(e => e.type === 'ko').length}`);
if (hits < 3) { console.log('RESULT: FAIL (too few hits)'); process.exit(1); }
console.log('RESULT: PASS (falls during KO knockdowns are part of the fight)');

function dist3(arr, a, b) {
  const dx = arr[3 * a] - arr[3 * b], dy = arr[3 * a + 1] - arr[3 * b + 1], dz = arr[3 * a + 2] - arr[3 * b + 2];
  return Math.hypot(dx, dy, dz);
}
