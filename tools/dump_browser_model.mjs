// 浏览器侧运行时 mjModel dump（fight 物理补丁后状态），与训练侧
// sparring_model_dump.json 同字段对照。只读，不跑仿真。
//   node tools/dump_browser_model.mjs [out.json]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const outPath = process.argv[2] ?? '/home/xy/zcode/g1-boxing-wasm/.workbuddy/gpu/logs/browser_model_dump.json';

const loadMujoco = (await import(path.join(root, 'vendor/mujoco/mujoco.js'))).default;
const mujoco = await loadMujoco({
  locateFile: f => 'file://' + path.join(root, 'vendor/mujoco', f),
});

// 与 main.js FIGHT_MODE 分支完全相同的运行时替换管线（含全部物理补丁）
let xml = fs.readFileSync(path.join(root, 'models/scene_boxing_tracking.xml'), 'utf8');
xml = xml
  .split('-0.50 0 0.761').join('-0.60 0 0.761')
  .split(' 0.50 0 0.761').join(' 0.60 0 0.761')
  .split('solver="Newton" iterations="100"')
    .join('solver="Newton" iterations="10" ls_iterations="20"')
  .split('<geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1" />')
    .join('<geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1" condim="1" />')
  .split('<geom type="capsule" size="0.01" />')
    .join('<geom type="capsule" size="0.01" condim="3" friction="0.6 0.005 0.0001" priority="1" />')
  .split('contype="1" conaffinity="1" friction="0.8 0.005 0.0001" priority="1"')
    .join('contype="1" conaffinity="1" condim="1"');

const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(path.join(root, 'models/unitree_g1/assets'))) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(root, 'models/unitree_g1/assets', f))));
  }
}
const model = mujoco.MjModel.from_xml_string(xml, vfs);

const name = (t, i) => mujoco.mj_id2name(model, t, i) || `<${i}>`;
const dump = {
  source: 'browser fight-patched scene_boxing_tracking.xml (runtime replace pipeline)',
  option: {
    timestep: model.opt.timestep, integrator: model.opt.integrator, solver: model.opt.solver,
    iterations: model.opt.iterations, tolerance: model.opt.tolerance,
    ls_iterations: model.opt.ls_iterations, ls_tolerance: model.opt.ls_tolerance,
    impratio: model.opt.impratio, cone: model.opt.cone,
    gravity: Array.from(model.opt.gravity),
  },
  dofs: [], joints: [], geoms: [], actuators: [], bodies: [],
};
for (let i = 0; i < model.nv; i++) {
  dump.dofs.push({ dofadr: i, damping: model.dof_damping[i],
    frictionloss: model.dof_frictionloss[i], armature: model.dof_armature[i] });
}
for (let i = 0; i < model.njnt; i++) {
  dump.joints.push({ name: name(mujoco.mjtObj.mjOBJ_JOINT.value, i), type: model.jnt_type[i],
    qposadr: model.jnt_qposadr[i], dofadr: model.jnt_dofadr[i],
    stiffness: model.jnt_stiffness[i] });
}
for (let i = 0; i < model.ngeom; i++) {
  dump.geoms.push({ name: name(mujoco.mjtObj.mjOBJ_GEOM.value, i),
    contype: model.geom_contype[i], conaffinity: model.geom_conaffinity[i],
    condim: model.geom_condim[i], priority: model.geom_priority[i],
    friction: Array.from(model.geom_friction.slice ? model.geom_friction.slice(3 * i, 3 * i + 3) : [model.geom_friction[3 * i], model.geom_friction[3 * i + 1], model.geom_friction[3 * i + 2]]),
    solref: Array.from([model.geom_solref[2 * i], model.geom_solref[2 * i + 1]]),
    solimp: Array.from([model.geom_solimp[5 * i], model.geom_solimp[5 * i + 1], model.geom_solimp[5 * i + 2], model.geom_solimp[5 * i + 3], model.geom_solimp[5 * i + 4]]),
    margin: model.geom_margin[i], gap: model.geom_gap[i] });
}
for (let i = 0; i < model.nu; i++) {
  dump.actuators.push({ name: name(mujoco.mjtObj.mjOBJ_ACTUATOR.value, i),
    gainprm: [model.actuator_gainprm[10 * i], model.actuator_gainprm[10 * i + 1], model.actuator_gainprm[10 * i + 2]],
    biasprm: [model.actuator_biasprm[10 * i], model.actuator_biasprm[10 * i + 1], model.actuator_biasprm[10 * i + 2]],
    forcerange: [model.actuator_forcerange[2 * i], model.actuator_forcerange[2 * i + 1]],
    ctrlrange: [model.actuator_ctrlrange[2 * i], model.actuator_ctrlrange[2 * i + 1]] });
}
for (let i = 0; i < model.nbody; i++) dump.bodies.push({ name: name(mujoco.mjtObj.mjOBJ_BODY.value, i) });

fs.writeFileSync(outPath, JSON.stringify(dump, null, 1));
const condimDist = {};
for (const g of dump.geoms) condimDist[g.condim] = (condimDist[g.condim] || 0) + 1;
console.log(`dumped: ${outPath}`);
console.log(`nv=${model.nv} njnt=${model.njnt} ngeom=${model.ngeom} nu=${model.nu}`);
console.log('condim dist:', JSON.stringify(condimDist));
console.log('option:', JSON.stringify(dump.option));
