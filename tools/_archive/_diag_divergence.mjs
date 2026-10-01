// TEMP diagnostic v4: count REAL MuJoCo auto-resets (qpos0 signature) vs
// resetRound (keyframe signature) across controller variants. Delete after use.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const VARIANT = process.argv[2] || 'baseline';
const DURATION = Number(process.argv[3] || 90);

const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
const assetsDir = path.join(root, 'models/unitree_g1/assets');
for (const f of fs.readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl'))
    vfs.addBuffer('assets/' + f, new Uint8Array(fs.readFileSync(path.join(assetsDir, f))));
}
const xml = fs.readFileSync(path.join(root, 'models/scene_boxing.xml'), 'utf8');
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

const OBJ = mujoco.mjtObj;
const id = (t, n) => mujoco.mj_name2id(model, t.value ?? t, n);
const { BoxingController } = await import('../src/boxing_ai.mjs');
const ctl = new BoxingController(model, data, {
  mujoco,
  ids: {
    pelvisA: id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB: id(OBJ.mjOBJ_BODY, 'B_pelvis'),
    headA: id(OBJ.mjOBJ_SITE, 'A_head'), headB: id(OBJ.mjOBJ_SITE, 'B_head'),
    fistLA: id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA: id(OBJ.mjOBJ_SITE, 'A_right_fist_site'),
    fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site'),
  },
});
const dt = model.opt.timestep;
const steps = Math.round(DURATION / dt);
let prevTime = 0, mjResets = 0, roundResets = 0, koCount = 0, prevKo = null, worstQvel = 0;
const chaosDur = { inKo: 0, outKo: 0 };

for (let i = 0; i < steps; i++) {
  ctl.update(dt);
  if (VARIANT === 'zeroXfrc' && ctl.koState) data.xfrc_applied.fill(0);
  if (VARIANT === 'ragdoll' && ctl.koState) {
    // fully passive: no assist wrench, no servo targets (hold current pose)
    data.xfrc_applied.fill(0);
    for (let a = 0; a < model.nu; a++) data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[2 * a]]] ?? data.ctrl[a];
  }
  mujoco.mj_step(model, data);

  let mv = 0;
  for (let j = 0; j < model.nv; j++) { const v = Math.abs(data.qvel[j]); if (v > mv) mv = v; }
  worstQvel = Math.max(worstQvel, mv);
  if (mv > 10) chaosDur[ctl.koState ? 'inKo' : 'outKo']++;

  if (data.time < prevTime - 1e-9) {
    // arm-joint qpos signature: keyframe has -0.72, qpos0 has 0
    const qArm = data.qpos[7 + 15]; // A: 7 floatbase + 15 (legs+waist) = left_shoulder_pitch
    if (Math.abs(qArm + 0.72) < 0.05) roundResets++;
    else mjResets++;
  }
  prevTime = data.time;
  if (prevKo === null && ctl.koState !== null) koCount++;
  prevKo = ctl.koState;
}
console.log(`VARIANT=${VARIANT.padEnd(9)} ${DURATION}s: KOs=${koCount} resetRounds=${roundResets} ` +
  `MuJoCoAutoResets=${mjResets} worstQvel=${worstQvel.toFixed(0)} ` +
  `chaosSteps(qvel>10) inKO=${chaosDur.inKo} outKO=${chaosDur.outKo}`);
