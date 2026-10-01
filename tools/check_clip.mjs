// FK validation for retargeted boxing clip CSVs (plan-realistic-boxing §4.1:
// "重定向后检查脚底接触、拳头轨迹、躯干旋转、关节范围、速度及执行器力矩限制").
//
// Replays a clip CSV ([root_pos(3), root_quat_xyzw(4), 29 joints] per frame,
// 30 fps) through the tracking scene's A robot with plain mj_forward per frame
// (kinematic replay — no policy, no dynamics) and measures:
//   - guard quality: min fist→head distance per hand, fist height vs head
//     (拳套护头), fist-to-torso horizontal offset (收肘)
//   - pelvis z range (深蹲失稳检查 — v1 教训: 0.487m 深蹲是策略失稳源)
//   - foot contact slide: XY drift while the foot is near the ground (滑脚)
//   - joint ranges vs model.jnt_range, peak joint speed (deg/s)
//   - forward drive: root XY displacement during the strike window (前冲标定)
//   - stance bookends: first/last hold frames match the guard stance
//
// --enrich writes measured values back into the clips manifest:
//   strike_window_s (fist-speed detection), forward_drive_m, guard metrics.
//
// Usage:
//   node tools/check_clip.mjs <clip.csv> [--fps 30] [--hand left|right]
//   node tools/check_clip.mjs <clips-dir> --enrich   # whole dir + manifest
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const FPS = 30;
const FOOT_GROUND_Z = 0.012;   // foot site "in contact" threshold (kinematic)

const args = process.argv.slice(2);
const enrich = args.includes('--enrich');
const inputs = args.filter(a => !a.startsWith('--'));

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
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

const OBJ = mujoco.mjtObj;
const qadr = n => model.jnt_qposadr[mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, 'A_' + n)];
const site = n => {
  const i = 3 * mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, 'A_' + n);
  return [data.site_xpos[i], data.site_xpos[i + 1], data.site_xpos[i + 2]];
};
const JOINTS = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
  'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
  'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
  'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
  'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];
const jadr = JOINTS.map(qadr);
// move B out of the way
const bqid = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, 'B_floating_base_joint');
data.qpos[model.jnt_qposadr[bqid]] = 5.0;

function setFrame(frame) {
  data.qpos[qadr('floating_base_joint') + 0] = frame[0];
  data.qpos[qadr('floating_base_joint') + 1] = frame[1];
  data.qpos[qadr('floating_base_joint') + 2] = frame[2];
  // CSV quat is xyzw, MuJoCo qpos is wxyz
  data.qpos[qadr('floating_base_joint') + 3] = frame[6];
  data.qpos[qadr('floating_base_joint') + 4] = frame[3];
  data.qpos[qadr('floating_base_joint') + 5] = frame[4];
  data.qpos[qadr('floating_base_joint') + 6] = frame[5];
  for (let i = 0; i < 29; i++) data.qpos[jadr[i]] = frame[7 + i];
  mujoco.mj_forward(model, data);
}

const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function checkClip(csvPath, hand = null) {
  const text = fs.readFileSync(csvPath, 'utf8').trim();
  const lines = text.split('\n');
  const frames = lines.map(l => l.split(',').map(Number));
  const n = frames.length;
  const dur = n / FPS;

  let minHeadL = 1e9, minHeadR = 1e9, minZL = 1e9, minZR = 1e9;
  let pelZmin = 1e9, pelZmax = -1e9;  const foot = { left: { trail: [], slide: 0, contactT: 0 }, right: { trail: [], slide: 0, contactT: 0 } };
  const jmin = new Float64Array(29).fill(1e9), jmax = new Float64Array(29).fill(-1e9);
  let peakJointSpeed = 0;
  let rootXY = null, rootTravel = 0;
  const fistSpeedL = [], fistSpeedR = [];
  let prev = null;

  for (let i = 0; i < n; i++) {
    setFrame(frames[i]);
    const head = site('head'), fL = site('left_fist_site'), fR = site('right_fist_site');
    // 骨盆高度取根节点原点（CSV root z；imu site 比骨盆原点低 ~0.08m）
    const pz = frames[i][2];
    minHeadL = Math.min(minHeadL, d3(fL, head));
    minHeadR = Math.min(minHeadR, d3(fR, head));
    minZL = Math.min(minZL, fL[2] - head[2]);
    minZR = Math.min(minZR, fR[2] - head[2]);
    pelZmin = Math.min(pelZmin, pz);
    pelZmax = Math.max(pelZmax, pz);
    for (let j = 0; j < 29; j++) {
      const q = frames[i][7 + j];
      jmin[j] = Math.min(jmin[j], q); jmax[j] = Math.max(jmax[j], q);
    }
    const r = [frames[i][0], frames[i][1]];
    if (rootXY) rootTravel += Math.hypot(r[0] - rootXY[0], r[1] - rootXY[1]);
    rootXY = r;

    for (const [name, sname] of [['left', 'left_foot'], ['right', 'right_foot']]) {
      const p = site(sname);
      const f = foot[name];
      if (p[2] < FOOT_GROUND_Z) {
        f.contactT += 1 / FPS;
        if (f.trail.length) {
          const q = f.trail[f.trail.length - 1];
          f.slide += Math.hypot(p[0] - q[0], p[1] - q[1]);
        }
        f.trail.push(p);
      } else {
        f.trail = [];   // contact broken: start a new contact segment
      }
    }
    if (prev) {
      const vL = d3(fL, prev.fL) * FPS, vR = d3(fR, prev.fR) * FPS;
      fistSpeedL.push(vL); fistSpeedR.push(vR);
      for (let j = 0; j < 29; j++) peakJointSpeed = Math.max(peakJointSpeed, Math.abs(frames[i][7 + j] - frames[i - 1][7 + j]) * FPS);
    } else { fistSpeedL.push(0); fistSpeedR.push(0); }
    prev = { fL, fR };
  }

  // strike window: 第一段连续超阈区（允许 ≤3 帧间隙）——出拳发力从起手到
  // 收拳前；收拳段的臂部回缩速度不再并入窗口。
  const speeds = hand === 'right' ? fistSpeedR : hand === 'left' ? fistSpeedL
    : (fistSpeedL.map((v, i) => Math.max(v, fistSpeedR[i])));
  const peak = Math.max(...speeds);
  const TH = 0.35 * peak, GAP = 3;
  let w0 = -1, w1 = -1, gap = 0;
  for (let i = 0; i < n; i++) {
    if (w0 < 0) {
      if (speeds[i] > TH) w0 = i;
    } else if (speeds[i] > TH) {
      w1 = i; gap = 0;
    } else if (++gap > GAP) break;
  }
  if (w0 >= 0 && w1 < 0) w1 = w0;
  const strike_s = w0 >= 0 ? [+(w0 / FPS).toFixed(2), +((w1 + 1) / FPS).toFixed(2)] : [0, 0];

  // joint range violations
  const violations = [];
  for (let j = 0; j < 29; j++) {
    const jid = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, 'A_' + JOINTS[j]);
    const lo = model.jnt_range[2 * jid], hi = model.jnt_range[2 * jid + 1];
    if (hi > lo) {  // limited joint（5e-3 rad 容差：源重定向数据在肘限位上有
      // ~2e-4 的轻微越界，物理无关；真实越界仍报）
      if (jmin[j] < lo - 5e-3 || jmax[j] > hi + 5e-3) {
        violations.push(`${JOINTS[j]} [${jmin[j].toFixed(2)},${jmax[j].toFixed(2)}] vs range [${lo.toFixed(2)},${hi.toFixed(2)}]`);
      }
    }
  }

  return {
    file: path.basename(csvPath), frames: n, duration_s: +dur.toFixed(2),
    guard: {
      min_fist_head_m: { left: +minHeadL.toFixed(3), right: +minHeadR.toFixed(3) },
      fist_below_head_m: { left: +(-minZL).toFixed(3), right: +(-minZR).toFixed(3) },
    },
    pelvis_z_m: [+pelZmin.toFixed(3), +pelZmax.toFixed(3)],
    foot_slide_m: { left: +foot.left.slide.toFixed(3), right: +foot.right.slide.toFixed(3) },
    foot_contact_s: { left: +foot.left.contactT.toFixed(2), right: +foot.right.contactT.toFixed(2) },
    root_travel_m: +rootTravel.toFixed(3),
    strike_window_s: strike_s, peak_fist_speed_mps: +peak.toFixed(2),
    peak_joint_speed_dps: +(peakJointSpeed * 180 / Math.PI).toFixed(0),
    joint_range_violations: violations,
  };
}

let failures = 0;
function judge(r, { requireGuard = true } = {}) {
  const problems = [];
  if (requireGuard) {
    const worst = Math.min(r.guard.min_fist_head_m.left, r.guard.min_fist_head_m.right);
    if (worst > 0.30) problems.push(`拳套不护头 (min fist-head ${worst.toFixed(2)}m > 0.30)`);
  }
  if (r.pelvis_z_m[0] < 0.60 && !r.file.includes('guard')) problems.push(`骨盆过低 (${r.pelvis_z_m[0]}m, v1 深蹲失稳区)`);
  for (const [k, v] of Object.entries(r.foot_slide_m)) {
    if (v > 0.12) problems.push(`${k} 脚滑移 ${v.toFixed(2)}m > 0.12`);
  }
  if (r.joint_range_violations.length) problems.push(`关节越界: ${r.joint_range_violations.join('; ')}`);
  if (problems.length) { failures++; console.log(`  FAIL: ${problems.join(' | ')}`); }
  else console.log('  PASS');
}

if (enrich) {
  const dir = inputs[0] || path.join(root, '.workbuddy/gpu/data/clips');
  const manifestPath = path.join(dir, 'manifest.json');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  for (const [name, entry] of Object.entries(manifest)) {
    const csv = path.join(dir, name + '.csv');
    if (!fs.existsSync(csv)) { console.log(`${name}: csv missing, skip`); continue; }
    const r = checkClip(csv, entry.hand ?? null);
    const isPunch = name !== 'boxing_guard_idle';
    entry.measured = {
      forward_drive_m: r.root_travel_m,
      strike_window_s: isPunch ? r.strike_window_s : [0, 0],
      peak_fist_speed_mps: isPunch ? r.peak_fist_speed_mps : 0,
      min_fist_head_m: r.guard.min_fist_head_m,
      foot_slide_m: r.foot_slide_m,
      pelvis_z_m: r.pelvis_z_m,
    };
    // 出拳片段的 punch_range_m：以实测前冲为依据的起始间距窗口。
    // 标定原则（与 boxing_ai PUNCH_RANGE 一致）：拳-头可达 ≈ 起始间距 + 双方前倾，
    // 前冲越大，可发起的间距越远；窗口宽度 ±0.13m。
    if (isPunch) {
      const drive = r.root_travel_m;
      const lo = Math.max(0.6, 0.82 + 0.45 * (drive - 0.2));
      entry.punch_range_m = [+lo.toFixed(2), +(lo + 0.26).toFixed(2)];
    }
    console.log(`${name}: strike=${r.strike_window_s} drive=${r.root_travel_m}m ` +
      `fist-head L/R=${r.guard.min_fist_head_m.left}/${r.guard.min_fist_head_m.right}m ` +
      `pelvisZ=[${r.pelvis_z_m}] slide L/R=${r.foot_slide_m.left}/${r.foot_slide_m.right}m`);
    judge(r, { requireGuard: isPunch || (entry.stance === 'v3') });
  }
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1));
  console.log(`manifest enriched -> ${manifestPath}`);
} else {
  const handFlag = args.includes('--hand') ? args[args.indexOf('--hand') + 1] : null;
  for (const p of inputs) {
    const r = checkClip(path.resolve(p), handFlag);
    console.log(JSON.stringify(r, null, 1));
    judge(r);
  }
}
process.exit(failures ? 1 : 0);
