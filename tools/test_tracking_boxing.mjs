// Headless stage-4 regression (milestone G): BOTH robots driven by the
// tracking clip stack (TrackingFighter + the boxing_ai clip scheduler) in the
// 29-DoF dual-robot WASM scene. Verifies: stable standing/looping, clip swaps
// happen, hits register, a forced KO collapses the robot and the round resets.
//
//   node tools/test_tracking_boxing.mjs [seconds] [clipPrefix] [--assist] [--seed N]
//   node tools/test_tracking_boxing.mjs [seconds] --fight [--no-heading-servo] [--seed N]
//
// clipPrefix: vendor/policy prefix used for ALL four clip slots (default:
// spinkick — the pipeline-validation policy). When the trained boxing clips
// exist pass e.g. "boxing" with per-clip files boxing_{guard,jab,...}.
// --assist: run WITH the balance assist on (browser default); the default is
// assist OFF. 方案 §7.4：辅助开关分别报告结果。
// --seed N: deterministic RNG for decisions/timers (default: random; fight
// 模式默认固定 1——方案 B 要求对打回归可复现)。
// --fight: FIGHT_MODE 对打回归（2026-10-01 航向伺服验收工具，方案 B）——双机
// 挂同一 boxing_fight 单策略（168 维 obs），场景走与浏览器 src/main.js 共用
// 的 src/fight_scene_patch.mjs 补丁（出生 ±0.6 + 物理对齐，两路径逐字节同
// 源）。KO 注入禁用（60s 真实对打，KO 由摔倒/伤害系统自然触发并重置）；
// 输出 facing 误差/交战率/命中/KO/双机站立量化指标。伺服开启时执行验收门
// 槛（median facing ≤35°、>90° ≤15%、命中 ≥3、双机站立 ≥60%，2026-10-02
// 增补形态门槛：贴身占比 ≤15%、抱架保持率 ≥70%）；关闭时为
// A/B 基线诊断跑（预期暴露缺陷：median facing >60°），不设质量门槛。
// --no-heading-servo: 伺服关闭基线（等价 FIGHT_HEADING_SERVO=0，构造项优
// 先于 env）。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
// COMBO_MODE 回归（plan-revamp-20260928 §5.2 D9）：BOXING_COMBO=1 时两拳手
// 常驻循环 combo 片段，默认跑 ≥42s（覆盖至少一个完整循环 + 回绕），断言为
// 无 NaN / divergences=0 / minZ≥0.40 / 回绕发生；KO 注入断言仅默认模式执行。
// --fight / FIGHT=1 时 fight 优先（与浏览器 URL 契约 §5.1 同规则）。
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2).filter(a => !a.startsWith('--'));
const raw = process.argv.slice(2);
const flags = raw.filter(a => a.startsWith('--'));
const FIGHT = flags.includes('--fight') || process.env.FIGHT === '1';
// --combo 旗标（2026-10-02）：与 BOXING_COMBO=1 等价的命令行入口（world.run
// 类沙箱无法注入环境变量）；默认行为不变，环境变量仍然生效。
const COMBO = !FIGHT && (process.env.BOXING_COMBO === '1' || flags.includes('--combo'));
const SERVO_OFF = flags.includes('--no-heading-servo');
const DURATION = Number(argv[0]) > 0 ? Number(argv[0]) : (FIGHT ? 60 : COMBO ? 42 : 16);
const KO_AT = DURATION * 0.6;
// PREFIX 规则：argv[0] 为数字时长时前缀取 argv[1]，否则 argv[0] 即前缀
const PREFIX = Number(argv[0]) > 0 ? (argv[1] || 'spinkick') : (argv[0] || 'spinkick');
const ASSIST = flags.includes('--assist');
const seedIdx = raw.indexOf('--seed');
const SEED = seedIdx >= 0 ? Number(raw[seedIdx + 1] ?? 1) : (FIGHT ? 1 : (Math.random() * 2 ** 31) | 0);

const mujoco = await loadMujoco();
// fight 场景补丁函数在下方 XML 补丁处使用——import 必须先于使用点（顶层
// await 动态导入顺序即执行顺序，TDZ 不可前引）。
const { applyFightScenePatch } = await import('../src/fight_scene_patch.mjs');
const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(path.join(root, 'models/unitree_g1/assets'))) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(root, 'models/unitree_g1/assets', f))));
  }
}
let xml = fs.readFileSync(path.join(root, "models/scene_boxing_tracking.xml"), "utf8");
// FIGHT 场景补丁（出生 ±0.6 + 求解器/condim 物理对齐）：与浏览器 src/main.js
// 共用 src/fight_scene_patch.mjs 同一函数（方案 B：两路径逐字节同源，杜绝
// harness 特判）。锚串失配（上游 XML 改版 → 补丁未生效）直接退出——fight
// 指标在错误场景下无意义，不静默带病运行。
if (FIGHT) {
  const patched = applyFightScenePatch(xml);
  if (patched === xml || !patched.includes('ls_iterations="20"') || !patched.includes('-0.60 0 0.761')) {
    console.error('fight scene patch did not apply (anchor strings mismatch)');
    process.exit(1);
  }
  xml = patched;
}
// COMBO 模式出生隔离（与 main.js 的加载时替换同一规则）：±0.5 → ±1.2
if (COMBO) {
  xml = xml.split('-0.50 0 0.761').join('-1.20 0 0.761').split(' 0.50 0 0.761').join(' 1.20 0 0.761');
}
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);
console.log(`tracking scene: nu=${model.nu} nq=${model.nq} dt=${model.opt.timestep}`);

const { BoxingController, mulberry32, FIGHT_PUNCH_SPEED } =
  await import('../src/boxing_ai.mjs');
const { TrackingNetwork, yawOfQuat } = await import('../src/tracking_policy.mjs');
const { TacticsPolicy } = await import('../src/tactics.mjs');

const loadNet = (name) => {
  const bin = fs.readFileSync(path.join(root, `vendor/policy/${name}.bin`));
  const meta = JSON.parse(fs.readFileSync(path.join(root, `vendor/policy/${name}_meta.json`), 'utf8'));
  return new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
};

// clip slots: single-prefix mode reuses one policy everywhere (mechanics
// check). Per-clip files win when present. COMBO adds the combo slot (falls
// back to PREFIX when boxing_combo is not deployed — mechanics-only run).
// FIGHT：单策略 fight 槽（168 维 obs，opponent_state@[154,14]）——双机挂同
// 一 boxing_fight 权重，与浏览器 main.js loadFightNet 同名同路径。
const nets = {};
const slotSource = {};
if (FIGHT) {
  nets.fight = loadNet('boxing_fight');
  slotSource.fight = 'boxing_fight';
} else {
  const slots = ['guard', 'jab', 'cross', 'hook'];
  if (COMBO) slots.push('combo');
  for (const s of slots) {
    // 候选链：常规槽 = per-clip → 裸 PREFIX；combo 槽优先固定部署名
    // boxing_combo（main.js 即按此名加载），再退 PREFIX 家族做机制代理。
    const candidates = s === 'combo'
      ? ['boxing_combo', `${PREFIX}_combo`, `${PREFIX}`, `${PREFIX}_guard`]
      : [`${PREFIX}_${s}`, `${PREFIX}`];
    const pick = candidates.find(c => fs.existsSync(path.join(root, `vendor/policy/${c}.bin`)))
      || `${PREFIX}_${s}`;
    slotSource[s] = pick;
    nets[s] = loadNet(pick);
  }
}
console.log('clip slots:', JSON.stringify(slotSource));

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
  assist: ASSIST,
  rng: mulberry32(SEED),
  comboMode: COMBO,
  fightMode: FIGHT,
  // A/B 基线开关（方案 A）：构造项显式传入，优先于 env（FIGHT_HEADING_SERVO）。
  headingServo: !SERVO_OFF,
});
console.log(`assist=${ASSIST ? 'ON' : 'OFF'} seed=${SEED} combo=${COMBO ? 'ON' : 'off'}` +
  (FIGHT ? ` fight=ON servo=${SERVO_OFF ? 'OFF (baseline)' : `ON tau=${ctl.headingTau.toFixed(2)}s`}` : ''));

// learned tactics when available; otherwise the weighted-random fallback
if (FIGHT) {
  // fight 决策层短路（decideTactics 的 fight 分支只维持常驻片段），战术权重
  // 不被消费——跳过加载，保持输出诚实。
  console.log('tactics skipped in fight mode (policy-driven, no scripted orchestration)');
} else {
try {
  const tj = JSON.parse(fs.readFileSync(path.join(root, 'vendor/tactics/tactics.json'), 'utf8'));
  const pol = TacticsPolicy.fromJSON(tj);
  ctl.setTactics(pol);
  console.log('tactics policy loaded');
} catch {
  console.log('tactics policy not found — random fallback drives clip choices');
}
}

ctl.setTracking('A', nets);
ctl.setTracking('B', nets);
console.log('tracking mode on: A + B');

const DT = model.opt.timestep;
const steps = Math.round(DURATION / DT);
const pelvisA = id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB = id(OBJ.mjOBJ_BODY, 'B_pelvis');
const torsoA = id(OBJ.mjOBJ_BODY, 'A_torso_link'), torsoB = id(OBJ.mjOBJ_BODY, 'B_torso_link');
let minZA = 1e9, minZB = 1e9, nan = false, divergences = 0, lastT = -1;
let koSeen = 0, postResetMin = 1e9, koT = null;
let lastPrint = 0;
// 拳法门禁期（全部拳法禁用）不会排队任何拳法意图、不会发生片段切换：
// crossQueued/crossPlayed 恒为 false；保留观察项供门禁逐个恢复后的回归读取。
let crossQueued = false, crossPlayed = false;

// FIGHT 量化指标（方案 B）：facing 误差 = |torso yaw − 指向对手 bearing|（角
// 度制，双方 torso 世界位置）；交战 = 距离∈[0.4,1.2]m 且双方 cos(facing 误
// 差)>0.5；双机站立 = 双骨盆 z≥0.40（与 KO 判读阈同口径）。每物理步采样。
// 形态指标（2026-10-02）：贴身占比 = 躯干水平间距 <0.35m 时间占比；抱架保持
// 率 = 非出拳状态（与运行时 applyFightGuard 同口径：任一臂拳速 EMA >
// FIGHT_PUNCH_SPEED 即算出拳，整步不计入分母）下双拳均"高于肘部 +0.03m 且
// 接近头高（|拳套 z − 头 site z|≤0.25m）"的时间占比；步法 = 骨盆水平路径
// 长度/时长（观察项）与躯干间距极差。
const wrapPi = a => Math.atan2(Math.sin(a), Math.cos(a));
const fight = FIGHT ? {
  n: 0, errA: [], errB: [], over90A: 0, over90B: 0,
  distWin: 0, engaged: 0, bothStand: 0,
  close: 0, distSum: 0, distMin: Infinity, distMax: 0,
  guardSamples: { A: 0, B: 0 }, guardOK: { A: 0, B: 0 },
  path: { A: 0, B: 0 }, prev: { A: null, B: null },
} : null;

// 抱架探针：返回 true/false（非出拳状态下的判定）或 null（出拳状态/倒地，
// 不计入分母）。与运行时 applyFightGuard 共享 boxing_ai 的同一组阈值常量，
// 浏览器与 headless 行为同源、无 harness 特判。
const elbowIds = FIGHT ? {
  A: [id(OBJ.mjOBJ_BODY, 'A_left_elbow_link'), id(OBJ.mjOBJ_BODY, 'A_right_elbow_link')],
  B: [id(OBJ.mjOBJ_BODY, 'B_left_elbow_link'), id(OBJ.mjOBJ_BODY, 'B_right_elbow_link')],
} : null;
const probeGuard = FIGHT ? (sd) => {
  const f = ctl.fighters[sd];
  const m = data.xpos, P = data.site_xpos;
  const pb = sd === 'A' ? pelvisA : pelvisB;
  if (m[3 * pb + 2] < 0.35) return null;   // 倒地/深度失衡不采样
  if (f.fistSpeed.left > FIGHT_PUNCH_SPEED || f.fistSpeed.right > FIGHT_PUNCH_SPEED) {
    return null;                           // 出拳状态：整步不算"非出拳状态"
  }
  const hz = P[3 * (sd === 'A' ? ctl.ids.headA : ctl.ids.headB) + 2];
  const fists = sd === 'A' ? [ctl.ids.fistLA, ctl.ids.fistRA] : [ctl.ids.fistLB, ctl.ids.fistRB];
  let guardOK = true;
  for (let k = 0; k < 2; k++) {
    const fz = P[3 * fists[k] + 2], ez = m[3 * elbowIds[sd][k] + 2];
    if (!(fz > ez + 0.03 && Math.abs(fz - hz) <= 0.25)) guardOK = false;
  }
  return guardOK;
} : null;

for (let i = 0; i < steps; i++) {
  ctl.update(DT);
  mujoco.mj_step(model, data);
  const t = data.time;
  if (t < lastT) divergences++;
  lastT = t;
  const za = data.xpos[3 * pelvisA + 2], zb = data.xpos[3 * pelvisB + 2];
  if (!Number.isFinite(za) || !Number.isFinite(zb)) { nan = true; break; }
  if (!ctl.koState) { minZA = Math.min(minZA, za); minZB = Math.min(minZB, zb); }
  if (fight) {
    const m = data.xpos, xq = data.xquat;
    const ax = m[3 * torsoA], ay = m[3 * torsoA + 1], bx = m[3 * torsoB], by = m[3 * torsoB + 1];
    const qa = 4 * torsoA, qb = 4 * torsoB;
    const yawA = yawOfQuat(xq[qa], xq[qa + 1], xq[qa + 2], xq[qa + 3]);
    const yawB = yawOfQuat(xq[qb], xq[qb + 1], xq[qb + 2], xq[qb + 3]);
    const errA = Math.abs(wrapPi(yawA - Math.atan2(by - ay, bx - ax))) * 180 / Math.PI;
    const errB = Math.abs(wrapPi(yawB - Math.atan2(ay - by, ax - bx))) * 180 / Math.PI;
    fight.errA.push(errA); fight.errB.push(errB);
    if (errA > 90) fight.over90A++;
    if (errB > 90) fight.over90B++;
    const dist = Math.hypot(bx - ax, by - ay);
    if (dist >= 0.4 && dist <= 1.2) {
      fight.distWin++;
      if (Math.cos(errA * Math.PI / 180) > 0.5 && Math.cos(errB * Math.PI / 180) > 0.5) fight.engaged++;
    }
    // 形态指标采样（2026-10-02）
    if (dist < 0.35) fight.close++;
    fight.distSum += dist;
    if (dist < fight.distMin) fight.distMin = dist;
    if (dist > fight.distMax) fight.distMax = dist;
    for (const sd of ['A', 'B']) {
      const pb = sd === 'A' ? pelvisA : pelvisB;
      const px2 = m[3 * pb], py2 = m[3 * pb + 1];
      const prev = fight.prev[sd];
      if (prev) fight.path[sd] += Math.hypot(px2 - prev[0], py2 - prev[1]);
      fight.prev[sd] = [px2, py2];
      const g = probeGuard(sd);
      if (g !== null) { fight.guardSamples[sd]++; if (g) fight.guardOK[sd]++; }
    }
    if (za >= 0.40 && zb >= 0.40) fight.bothStand++;
    fight.n++;
  }
  if (!ctl.koState && koT !== null) {
    postResetMin = Math.min(postResetMin, za, zb);
    if (t - koT > 3 && (za < 0.5 || zb < 0.5)) { /* recovery judged below */ }
  }
  koSeen = ctl.events.filter(e => e.type === 'ko').length;
  if (ctl.fighters.A.punchIntent?.clip === 'cross' || ctl.fighters.B.punchIntent?.clip === 'cross') crossQueued = true;
  if (ctl.fighters.A.clip === 'cross' || ctl.fighters.B.clip === 'cross') crossPlayed = true;
  // fight 模式不注入 KO（方案 B：60s 真实对打——KO 由摔倒/伤害系统自然触发
  // 并经 resetRound 重开，注入会把命中/站立统计污染成 KO 结算序列）。
  if (koT === null && t >= KO_AT && !COMBO && !FIGHT) {
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
const grazes = ctl.events.filter(e => e.type === 'graze').length;
const blocks = ctl.events.filter(e => e.type === 'block').length;
const au = ctl.assistUsage;
const fmtU = u => `F|${(u.fx + u.fy + u.fz).toFixed(0)}N·s T|${(u.tx + u.ty + u.tz).toFixed(0)}N·m·s`;
console.log(`grazes=${grazes} blocks=${blocks} ` +
  `assistUsage: A[${fmtU(au.A)}] B[${fmtU(au.B)}]`);
// 拳法门禁期（全部拳法禁用）swaps=0、零出拳即预期行为，"必须发生切换"断言
// 已改为纯观察（摘要 crossQueued/crossPlayed，门禁期恒为 false）；门禁逐个
// 恢复后再按需加回硬断言。
if (nan) fail = 'NaN in sim state';
else if (ctl._divergences > 0) fail = 'engine auto-reset = instability';
else if (FIGHT) {
  // fight：KO 系统启用——自然摔倒=KO 事件（结算后 resetRound 重开），摔倒
  // 瞬间的 minZ 过渡值只记录不判失败（minZ/crossQueued 在 fight 下均为观察
  // 项）。质量门槛仅伺服开启时执行（验收标准 2 的硬口径，不放宽；交战率
  // ≥50% 尽力达标、不设硬门）。--no-heading-servo 为基线诊断跑（验收标准
  // 1：预期暴露缺陷 median facing >60°），只输出指标不设门槛。
  const n = fight.n;
  const mean = arr => arr.reduce((s, v) => s + v, 0) / (arr.length || 1);
  const quantile = (arr, q) => {
    if (!arr.length) return NaN;
    const s = [...arr].sort((a, b) => a - b);
    const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  };
  const pct = (num, den) => den ? (100 * num / den).toFixed(1) + '%' : 'n/a';
  const comb = [...fight.errA, ...fight.errB];
  const medC = quantile(comb, 0.5);
  const overPct = 100 * (fight.over90A + fight.over90B) / (2 * n || 1);
  const hitsTotal = ctl.fighters.A.scores.hits + ctl.fighters.B.scores.hits;
  const standFrac = fight.bothStand / (n || 1);
  const rounds = ctl.events.filter(e => e.type === 'round').length;
  console.log(`fight metrics (servo=${SERVO_OFF ? 'OFF baseline' : 'ON'}, samples=${n}):`);
  console.log(`  facing err deg A: mean=${mean(fight.errA).toFixed(1)} median=${quantile(fight.errA, 0.5).toFixed(1)} p90=${quantile(fight.errA, 0.9).toFixed(1)} >90=${pct(fight.over90A, n)}`);
  console.log(`  facing err deg B: mean=${mean(fight.errB).toFixed(1)} median=${quantile(fight.errB, 0.5).toFixed(1)} p90=${quantile(fight.errB, 0.9).toFixed(1)} >90=${pct(fight.over90B, n)}`);
  console.log(`  facing combined: median=${medC.toFixed(1)} p90=${quantile(comb, 0.9).toFixed(1)} >90=${overPct.toFixed(1)}%`);
  console.log(`  engagement(dist 0.4-1.2 && both cos>0.5)=${pct(fight.engaged, n)} (distInWin only=${pct(fight.distWin, n)})`);
  console.log(`  both-standing(z>=0.40)=${pct(fight.bothStand, n)} hits=${ctl.fighters.A.scores.hits}/${ctl.fighters.B.scores.hits} (total ${hitsTotal}) ko=${koSeen} rounds=${rounds}`);
  // 形态指标（2026-10-02）：贴身占比 / 抱架保持率 / 步法观察项 / 整形力量
  const closePct = 100 * fight.close / (n || 1);
  const guardPct = sd => fight.guardSamples[sd]
    ? 100 * fight.guardOK[sd] / fight.guardSamples[sd] : NaN;
  const guardA = guardPct('A'), guardB = guardPct('B');
  const guardVals = [guardA, guardB].filter(Number.isFinite);
  const guardMean = guardVals.length ? guardVals.reduce((s, v) => s + v, 0) / guardVals.length : NaN;
  const fmtPct = v => Number.isFinite(v) ? v.toFixed(1) + '%' : 'n/a';
  const fu = ctl.fightForceUsage;
  console.log(`  clinch(torsoDist<0.35)=${closePct.toFixed(1)}%  guard-hold(nonPunch) A=${fmtPct(guardA)} B=${fmtPct(guardB)} (samples A=${fight.guardSamples.A} B=${fight.guardSamples.B})`);
  console.log(`  footwork(observe): pelvis speed A=${(fight.path.A / (DURATION || 1)).toFixed(3)} B=${(fight.path.B / (DURATION || 1)).toFixed(3)} m/s  torsoDist min/mean/max=${fight.distMin.toFixed(2)}/${(fight.distSum / (n || 1)).toFixed(2)}/${fight.distMax.toFixed(2)} m`);
  // 整形力量（观察项）：验收场景（出生 1.2m）躯干水平间距 0.62-1.23m，不进
  // 分离(<0.42m)/接近(>1.25m)力触发区间——两项为 0 N·s 属预期（P1-1 覆盖边界，
  // 见 README"覆盖边界说明"；生效性由浏览器长局观察覆盖）
  console.log(`  fight-shaping forces: A[sep ${(fu.A.sx + fu.A.sy).toFixed(0)}N·s app ${(fu.A.ax + fu.A.ay).toFixed(0)}N·s] B[sep ${(fu.B.sx + fu.B.sy).toFixed(0)}N·s app ${(fu.B.ax + fu.B.ay).toFixed(0)}N·s]`);
  if (!SERVO_OFF) {
    const G = { medianFacingDeg: 35, over90Pct: 15, minHits: 3, minBothStanding: 0.60,
                maxClinchPct: 15, minGuardHoldPct: 70 };
    if (!(medC <= G.medianFacingDeg)) fail = `gate median facing ${medC.toFixed(1)}deg > ${G.medianFacingDeg}deg`;
    else if (!(overPct <= G.over90Pct)) fail = `gate facing>90deg ${overPct.toFixed(1)}% > ${G.over90Pct}%`;
    else if (!(hitsTotal >= G.minHits)) fail = `gate hits ${hitsTotal} < ${G.minHits}`;
    else if (!(standFrac >= G.minBothStanding)) fail = `gate both-standing ${(standFrac * 100).toFixed(1)}% < ${G.minBothStanding * 100}%`;
    else if (!(closePct <= G.maxClinchPct)) fail = `gate clinch(torsoDist<0.35) ${closePct.toFixed(1)}% > ${G.maxClinchPct}%`;
    else if (!(guardMean >= G.minGuardHoldPct)) fail = `gate guard-hold ${fmtPct(guardMean)} < ${G.minGuardHoldPct}%`;
  } else {
    console.log('baseline run (servo off): quality gates skipped — expect median facing >60deg (defect exposed)');
  }
}
else if (minZA < 0.40 || minZB < 0.40) fail = `a robot fell during normal play (minZ A=${minZA.toFixed(2)} B=${minZB.toFixed(2)})`;
else if (COMBO) {
  // combo 断言：42s 内至少完成一次完整循环回绕（clipSwaps≥1），且片段恒为
  // combo（KO/回合系统被禁用，无 ko 事件、无 round 事件）
  const wraps = [ctl.fighters.A.clipSwaps, ctl.fighters.B.clipSwaps];
  const koEvents = ctl.events.filter(e => e.type === 'ko' || e.type === 'round').length;
  if (wraps[0] < 1 || wraps[1] < 1) fail = `combo wrap did not happen in ${DURATION}s (swaps A=${wraps[0]} B=${wraps[1]})`;
  else if (ctl.fighters.A.clip !== 'combo' || ctl.fighters.B.clip !== 'combo') fail = `fighters left the combo clip (A=${ctl.fighters.A.clip} B=${ctl.fighters.B.clip})`;
  else if (koEvents > 0) fail = `KO/round system fired in combo mode (${koEvents} events)`;
}
else if (koSeen < 1) fail = 'KO injection did not trigger';
else if (koT !== null && postResetMin < 0.5) fail = 'robots did not recover after round reset';
const koTimes = ctl.events.filter(e => e.type === 'ko').map(e => `${e.down}@${e.t.toFixed(1)}`).join(',');
const minZSummary = `minZ A=${minZA.toFixed(3)} B=${minZB.toFixed(3)} swaps=${ctl.fighters.A.clipSwaps}/${ctl.fighters.B.clipSwaps}` +
  ` hits=${ctl.fighters.A.scores.hits}/${ctl.fighters.B.scores.hits} ko=${koSeen}[${koTimes}]` +
  ` crossQueued=${crossQueued} crossPlayed=${crossPlayed}`;
if (fail) { console.log('RESULT: FAIL (' + fail + ') [' + minZSummary + ']'); process.exit(1); }
console.log('RESULT: PASS [' + minZSummary + ']');
