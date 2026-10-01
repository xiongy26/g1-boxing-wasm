// 诊断：追踪模式双机互刺摔倒的机理——机间接触时刻/部位/力度 + 骨盆轨迹
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

const { BoxingController, mulberry32 } = await import('../src/boxing_ai.mjs');
const { TrackingNetwork } = await import('../src/tracking_policy.mjs');
const { TacticsPolicy } = await import('../src/tactics.mjs');

const OBJ = mujoco.mjtObj;
const id = (t, n) => mujoco.mj_name2id(model, t.value, n);
const nets = {};
for (const c of ['guard', 'jab', 'cross', 'hook']) {
  const bin = fs.readFileSync(path.join(root, `vendor/policy/boxing_${c}.bin`));
  nets[c] = new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength),
    JSON.parse(fs.readFileSync(path.join(root, `vendor/policy/boxing_${c}_meta.json`), 'utf8')));
}
const ctl = new BoxingController(model, data, {
  mujoco,
  ids: {
    pelvisA: id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB: id(OBJ.mjOBJ_BODY, 'B_pelvis'),
    fistLA: id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA: id(OBJ.mjOBJ_SITE, 'A_right_fist_site'),
    fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site'),
  },
  assist: false,
  rng: mulberry32(1),
});
try { ctl.setTactics(TacticsPolicy.fromJSON(JSON.parse(fs.readFileSync(path.join(root, 'vendor/tactics/tactics.json'), 'utf8')))); } catch {}

ctl.setTracking('A', nets);
ctl.setTracking('B', nets);

const gname = (g) => {
  const b = model.geom_bodyid[g];
  return mujoco.mj_id2name(model, OBJ.mjOBJ_BODY.value, b);
};
const pelvisA = id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB = id(OBJ.mjOBJ_BODY, 'B_pelvis');
const DT = model.opt.timestep;
let lastPairContact = false;
let lastPrint = 0;

for (let i = 0; i < Math.round(8 / DT); i++) {
  ctl.update(DT);
  mujoco.mj_step(model, data);
  const t = data.time;

  // robot-to-robot contacts only
  let pairs = [];
  if (data.ncon > 0) {
    const con = data.contact;
    for (let k = 0; k < data.ncon; k++) {
      const c = con.get(k);
      const ga = gname(c.geom1), gb = gname(c.geom2);
      c.delete();
      if ((ga.startsWith('A_') && gb.startsWith('B_')) || (ga.startsWith('B_') && ga && gb.startsWith('A_'))) {
        pairs.push(`${ga.replace(/_(link)?$/, '')}|${gb.replace(/_(link)?$/, '')}`);
      }
    }
  }
  const pairNow = pairs.length > 0;
  if (pairNow !== lastPairContact) {
    const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
    console.log(`t=${t.toFixed(2)} 机间接触 ${pairNow ? '开始' : '结束'} n=${pairs.length} ` +
      `[${pairs.slice(0, 4).join(' ')}] zA=${za.toFixed(2)} zB=${zb.toFixed(2)} ` +
      `clipA=${ctl.fighters.A.clip}@${ctl.fighters.A.tracking.timeStep} clipB=${ctl.fighters.B.clip}@${ctl.fighters.B.tracking.timeStep}`);
    lastPairContact = pairNow;
  }

  if (t - lastPrint >= 0.5) {
    lastPrint = t;
    const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
    const vxA = data.qvel[0], vxB = data.qvel[7];
    console.log(`t=${t.toFixed(1)} zA=${za.toFixed(3)} zB=${zb.toFixed(3)} ` +
      `pelvisVxA=${vxA.toFixed(2)} pelvisVxB=${vxB.toFixed(2)} ` +
      `clipA=${ctl.fighters.A.clip}@${ctl.fighters.A.tracking.timeStep} clipB=${ctl.fighters.B.clip}@${ctl.fighters.B.tracking.timeStep} ` +
      `dist=${Math.hypot(data.xpos[3*pelvisA]-data.xpos[3*pelvisB], data.xpos[3*pelvisA+1]-data.xpos[3*pelvisB+1]).toFixed(2)} ncon=${data.ncon}`);
  }
  if (ctl.koState) { console.log(`t=${t.toFixed(2)} KO 判定: ${JSON.stringify(ctl.koState)}`); break; }
}
