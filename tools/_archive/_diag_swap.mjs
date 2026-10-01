// 实验：jab 摔倒与切入方式的关系（无接触、单机）
//   A: RSI 起步（= test_tracking_sim，对照组）
//   B: 守卫播 127 帧后切换（= 新调度器 early-exit：参考过零即切，机器人带弹跳速度）
//   C: 守卫播完 449 帧后切换（= 旧调度器：静态书端切换）
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

const { TrackingNetwork, TrackingFighter } = await import('../src/tracking_policy.mjs');

const loadNet = (name) => {
  const bin = fs.readFileSync(path.join(root, `vendor/policy/boxing_${name}.bin`));
  return new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength),
    JSON.parse(fs.readFileSync(path.join(root, `vendor/policy/boxing_${name}_meta.json`), 'utf8')));
};
const guardNet = loadNet('guard');
const clipName = process.argv[2] || 'jab';
const clipNet = loadNet(clipName);
const jabNet = clipNet;   // 复用脚本主体

const OBJ = mujoco.mjtObj;
const data = new mujoco.MjData(model);
const aFj = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, 'A_floating_base_joint');
const aq = model.jnt_qposadr[aFj];
const bFj = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, 'B_floating_base_joint');
const bq = model.jnt_qposadr[bFj];
const aPelvis = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY.value, 'A_pelvis');
const DT = model.opt.timestep;

function rsi() {
  const ref0 = jabNet.refAt(0);
  for (let k = 0; k < 3; k++) data.qpos[aq + k] = ref0.body_pos_w[k];
  for (let k = 0; k < 4; k++) data.qpos[aq + 3 + k] = ref0.body_quat_w[k];
  return ref0;
}

function run(label, mode) {
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);
  data.qpos[bq] = 5.0;   // B 挪开

  let fighter = new TrackingFighter(mujoco, model, data, 'A', guardNet);
  let swapAt = -1;

  if (mode === 'RSI') {
    const ref0 = rsi();
    for (let i = 0; i < jabNet.n; i++) data.qpos[fighter.qposadr[i]] = ref0.joint_pos[i];
    mujoco.mj_forward(model, data);
    fighter = new TrackingFighter(mujoco, model, data, 'A', jabNet);
    fighter.reset(0);
    swapAt = 0;
  } else {
    // 先播守卫 N 帧（机器人真实跟踪弹跳），然后按 swapTo 的方式切换
    const N = mode === 'EARLY' ? 127 : 449;
    for (let i = 0; i < N * 4; i++) { fighter.physicsStep(4); mujoco.mj_step(model, data); }
    const prev = fighter;
    fighter = new TrackingFighter(mujoco, model, data, 'A', jabNet);
    fighter.reset(0);
    fighter.lastAction.set(prev.lastAction);
    fighter.pdTarget.set(prev.pdTarget);
    swapAt = N;
  }

  let minZ = 1e9, minZPostSwap = 1e9, nan = false, fallFrame = -1;
  const totalSteps = Math.round((9 / DT));
  for (let i = 0; i < totalSteps; i++) {
    fighter.physicsStep(4);
    mujoco.mj_step(model, data);
    const z = data.xpos[3 * aPelvis + 2];
    if (!Number.isFinite(z)) { nan = true; break; }
    minZ = Math.min(minZ, z);
    if (swapAt >= 0) minZPostSwap = Math.min(minZPostSwap, z);
    if (z < 0.4 && fallFrame < 0) fallFrame = i * DT;
  }
  const zEnd = data.xpos[3 * aPelvis + 2];
  const fallInfo = fallFrame >= 0
    ? `摔倒@t=${fallFrame.toFixed(2)}s（切后${(fallFrame - swapAt / 50).toFixed(2)}s，参考帧${Math.round((fallFrame - swapAt / 50) * 50)}）`
    : '存活';
  console.log(`${label.padEnd(30)} minZ=${minZ.toFixed(3)} 切换后minZ=${minZPostSwap.toFixed(3)} 终z=${zEnd.toFixed(3)} ${fallInfo} ${nan ? 'NaN!' : ''}`);
}

run('A: RSI 直接进 ' + clipName, 'RSI');
run('B: guard@127 early-exit 切 ' + clipName, 'EARLY');
run('C: guard 书端 clipDone 切 ' + clipName, 'BOOKEND');
