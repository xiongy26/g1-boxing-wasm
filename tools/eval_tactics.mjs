// Evaluate the trained tactics policy vs the old weighted-random picker.
//   node tools/eval_tactics.mjs [episodes] [epLenSeconds]
// A = tactics policy (temp 0.7 sampling), B = legacy hardcoded behaviour.
// Reports hits/episode, damage dealt and reward-like score per side.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const N = Number(process.argv[2] || 20);
const EP_LEN = Number(process.argv[3] || 24);

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(path.join(root, 'models/unitree_g1/assets'))) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(root, 'models/unitree_g1/assets', f))));
  }
}
const model = mujoco.MjModel.from_xml_string(
  fs.readFileSync(path.join(root, 'models/scene_boxing_amo.xml'), 'utf8'), vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);
const OBJ = mujoco.mjtObj;
const id = (t, n) => mujoco.mj_name2id(model, t.value ?? t, n);

const { BoxingController } = await import('../src/boxing_ai.mjs');
const { TacticsPolicy } = await import('../src/tactics.mjs');
const ctl = new BoxingController(model, data, {
  mujoco,
  ids: {
    pelvisA: id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB: id(OBJ.mjOBJ_BODY, 'B_pelvis'),
    headA: id(OBJ.mjOBJ_SITE, 'A_head'), headB: id(OBJ.mjOBJ_SITE, 'B_head'),
    fistLA: id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA: id(OBJ.mjOBJ_SITE, 'A_right_fist_site'),
    fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site'),
  },
});
// AMO mode for both robots (scripted fallback can't stand in this scene)
const bin = fs.readFileSync(path.join(root, 'vendor/policy/amo.bin'));
const meta = JSON.parse(fs.readFileSync(path.join(root, 'vendor/policy/amo_meta.json'), 'utf8'));
const { AMONetwork } = await import('../src/rl_policy.mjs');
const amoNet = new AMONetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
ctl.setAMO('A', amoNet);
ctl.setAMO('B', amoNet);

const json = JSON.parse(fs.readFileSync(path.join(root, 'vendor/tactics/tactics.json'), 'utf8'));
const pol = TacticsPolicy.fromJSON(json);
pol.tacticsTemp = 0.7;
ctl.tactics = { A: pol }; // B side: no policy -> legacy weighted-random fallback
console.log(`eval: tactics-A vs legacy-B, ${N} episodes x ${EP_LEN}s (side swapped halfway)`);
const dt = model.opt.timestep;

const acc = {
  pol: { hits: 0, points: 0, ko: 0 },
  leg: { hits: 0, points: 0, ko: 0 },
};
const swap = Math.floor(N / 2); // second half: policy plays B to cancel side bias
for (let ep = 0; ep < N; ep++) {
  const policySide = ep < swap ? 'A' : 'B';
  ctl.tactics = { [policySide]: pol };
  ctl.aggression = 1;
  const t0 = ctl.time;
  const evStart = ctl.events.length;
  for (let i = 0, steps = Math.round(EP_LEN / dt); i < steps; i++) {
    ctl.update(dt);
    mujoco.mj_step(model, data);
  }
  const evs = ctl.events.splice(evStart).filter(e => e.t >= t0 - 1e-9);
  for (const e of evs) {
    if (e.type === 'hit') {
      const isPol = e.attacker === policySide;
      acc[isPol ? 'pol' : 'leg'].hits++;
      acc[isPol ? 'pol' : 'leg'].points += e.points;
    }
    if (e.type === 'ko' && e.winner) acc[e.winner === policySide ? 'pol' : 'leg'].ko++;
  }
}
console.log(`tactics policy : hits/ep=${(acc.pol.hits / N).toFixed(2)}  pts/ep=${(acc.pol.points / N).toFixed(2)}  KO=${acc.pol.ko}`);
console.log(`legacy random  : hits/ep=${(acc.leg.hits / N).toFixed(2)}  pts/ep=${(acc.leg.points / N).toFixed(2)}  KO=${acc.leg.ko}`);
const ratio = acc.leg.hits > 0 ? (acc.pol.hits / acc.leg.hits).toFixed(2) : '(legacy: no hits)';
console.log(`offense ratio (pol/leg hits): ${ratio}`);
