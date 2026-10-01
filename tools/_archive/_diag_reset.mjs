// 验证：切换片段时做"训练对齐状态重置"（RSI-lite）能否消除拳法片段切换摔倒。
// 机理（2026-09-28 诊断）：swapTo 只接续 lastAction/pdTarget，参考时钟归零但
// 机器人实际关节状态/速度不归零——jab/hook 策略对这种状态-参考失配零鲁棒性。
// 修复设计：swapTo 时把 29 关节 qpos/qvel 设为目标片段第 0 帧参考值、基座
// 6 维速度清零、lastAction/pdTarget 归零（= 训练回合起点状态），世界位姿保留
// （基座位置/朝向不动，观测全在体坐标系，yaw 不变性保证严格成立）。
//
//   node tools/_diag_reset.mjs [clip]   clip: jab|cross|hook|guard（默认 jab）
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

// ---- 数据体检：守卫片段的书端/中途帧与目标片段第 0 帧的失配有多大 ----
{
  const g0 = guardNet.refAt(0), gEnd = guardNet.refAt(guardNet.motionLen - 1), gMid = guardNet.refAt(127);
  const t0 = clipNet.refAt(0);
  const maxDiff = (a, b) => {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
  };
  const maxAbs = (a) => {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
    return m;
  };
  console.log(`[失配体检] guard帧0 vs ${clipName}帧0: max|Δ关节|=${maxDiff(g0.joint_pos, t0.joint_pos).toFixed(4)} rad`);
  console.log(`[失配体检] guard末帧 vs ${clipName}帧0: max|Δ关节|=${maxDiff(gEnd.joint_pos, t0.joint_pos).toFixed(4)} rad, guard末帧 max|关节速度|=${maxAbs(gEnd.joint_vel).toFixed(3)} rad/s`);
  console.log(`[失配体检] guard帧127 vs ${clipName}帧0: max|Δ关节|=${maxDiff(gMid.joint_pos, t0.joint_pos).toFixed(4)} rad, guard帧127 max|关节速度|=${maxAbs(gMid.joint_vel).toFixed(3)} rad/s`);
  // 基座差异：骨盆高度 + 双脚相对骨盆位置（判断"只重置关节"时脚会跳多大）
  const pelvisZ = (r) => r.body_pos_w[2];
  const footRel = (r, idx) => {
    const px = r.body_pos_w[0], py = r.body_pos_w[1];
    return `(${(r.body_pos_w[3 * idx] - px).toFixed(3)},${(r.body_pos_w[3 * idx + 1] - py).toFixed(3)},${(r.body_pos_w[3 * idx + 2]).toFixed(3)})`;
  };
  const L_ANKLE = 3, R_ANKLE = 6;   // body_names: left/right_ankle_roll_link
  console.log(`[失配体检] 骨盆高度: guard书端=${pelvisZ(gEnd).toFixed(3)} ${clipName}帧0=${pelvisZ(t0).toFixed(3)}`);
  console.log(`[失配体检] 左脚@guard书端=${footRel(gEnd, L_ANKLE)} vs @${clipName}帧0=${footRel(t0, L_ANKLE)}（相对骨盆 x,y）`);
  console.log(`[失配体检] 右脚@guard书端=${footRel(gEnd, R_ANKLE)} vs @${clipName}帧0=${footRel(t0, R_ANKLE)}（相对骨盆 x,y）`);
}

// ---- 修复设计原型的状态重置 ----
function resetToReference(fighter, step = 0) {
  const d = fighter.data, ref = fighter.net.refAt(step);
  for (let i = 0; i < fighter.n; i++) {
    d.qpos[fighter.qposadr[i]] = ref.joint_pos[i];
    d.qvel[fighter.dofadr[i]] = ref.joint_vel[i];
  }
  for (let k = 0; k < 6; k++) d.qvel[fighter.freeDof + k] = 0;
  fighter.lastAction.fill(0);
  fighter.pdTarget.set(fighter.net.defaultQ);
}

// 原位 RSI：保留机器人当前世界水平位置与朝向（战斗站位不动），基座高度/
// 姿态与全部关节状态采用参考第 0 帧（绕世界 z 旋转到当前朝向）；基座速度清零。
function resetToReferenceInPlace(fighter, step = 0) {
  const d = fighter.data, ref = fighter.net.refAt(step);
  for (let i = 0; i < fighter.n; i++) {
    d.qpos[fighter.qposadr[i]] = ref.joint_pos[i];
    d.qvel[fighter.dofadr[i]] = ref.joint_vel[i];
  }
  // 基座四元数：q_new = Rz(yaw_cur) ⊗ q_ref（保持当前世界朝向，采用参考 roll/pitch）
  const fw = d.qpos[fighter.freeDof + 3], fx = d.qpos[fighter.freeDof + 4],
        fy = d.qpos[fighter.freeDof + 5], fz = d.qpos[fighter.freeDof + 6];
  const yaw = Math.atan2(2 * (fw * fz + fx * fy), 1 - 2 * (fy * fy + fz * fz));
  const [zw, zx, zy, zz] = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
  const qw = ref.body_quat_w[3], qx = ref.body_quat_w[4], qy = ref.body_quat_w[5], qz = ref.body_quat_w[6];
  d.qpos[fighter.freeDof + 3] = zw * qw - zz * qz;
  d.qpos[fighter.freeDof + 4] = zw * qx + zz * qy;
  d.qpos[fighter.freeDof + 5] = zw * qy - zz * qx;
  d.qpos[fighter.freeDof + 6] = zw * qz + zz * qw;
  d.qpos[fighter.freeDof + 2] = ref.body_pos_w[2];   // 高度用参考值，x/y 保留
  for (let k = 0; k < 6; k++) d.qvel[fighter.freeDof + k] = 0;
  fighter.lastAction.fill(0);
  fighter.pdTarget.set(fighter.net.defaultQ);
}

const OBJ = mujoco.mjtObj;
const data = new mujoco.MjData(model);
const bFj = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, 'B_floating_base_joint');
const bq = model.jnt_qposadr[bFj];
const aPelvis = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY.value, 'A_pelvis');
const DT = model.opt.timestep;

function run(label, mode) {
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);
  data.qpos[bq] = 5.0;   // B 挪开

  let fighter = new TrackingFighter(mujoco, model, data, 'A', guardNet);
  let swapAt = -1;
  const N = mode === 'EARLY' ? 127 : guardNet.motionLen - 1;
  for (let i = 0; i < N * 4; i++) { fighter.physicsStep(4); mujoco.mj_step(model, data); }
  const prev = fighter;
  fighter = new TrackingFighter(mujoco, model, data, 'A', clipNet);
  fighter.reset(0);
  if (mode.endsWith('RESET')) {
    resetToReference(fighter, 0);
    mujoco.mj_forward(model, data);
  } else if (mode.endsWith('FULL')) {
    resetToReferenceInPlace(fighter, 0);
    mujoco.mj_forward(model, data);
  } else {
    fighter.lastAction.set(prev.lastAction);
    fighter.pdTarget.set(prev.pdTarget);
  }
  swapAt = N;

  let minZPostSwap = 1e9, nan = false, fallFrame = -1;
  const clipSeconds = (fighter.net.motionLen - 1) / fighter.net.meta.motion.fps;
  const totalSteps = Math.round(clipSeconds / DT);   // 只评片段播放窗口（真实调度器在 clipDone 切回守卫，不冻结末帧）
  for (let i = 0; i < totalSteps; i++) {
    fighter.physicsStep(4);
    mujoco.mj_step(model, data);
    const z = data.xpos[3 * aPelvis + 2];
    if (!Number.isFinite(z)) { nan = true; break; }
    minZPostSwap = Math.min(minZPostSwap, z);
    if (z < 0.4 && fallFrame < 0) fallFrame = i * DT;
  }
  const zEnd = data.xpos[3 * aPelvis + 2];
  const fallInfo = fallFrame >= 0
    ? `片段内摔倒@切后${fallFrame.toFixed(2)}s`
    : '存活';
  console.log(`${label.padEnd(34)} 片段内minZ=${minZPostSwap.toFixed(3)} 片段末z=${zEnd.toFixed(3)} ${fallInfo} ${nan ? 'NaN!' : ''}`);
}

console.log(`\n[实验] 切入片段 = ${clipName}`);
run('B: guard中途切+carryOver(现状)', 'EARLY');
run('B+FULL: guard中途切+原位RSI', 'EARLY_FULL');
run('C+FULL: guard书端切+原位RSI', 'BOOKEND_FULL');
