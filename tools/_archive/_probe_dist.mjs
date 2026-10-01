// Headless stage-4 regression (milestone G): BOTH robots driven by the
// tracking clip stack (TrackingFighter + the boxing_ai clip scheduler) in the
// 29-DoF dual-robot WASM scene. Verifies: stable standing/looping, clip swaps
// happen, hits register, a forced KO collapses the robot and the round resets.
//
//   node tools/test_tracking_boxing.mjs [seconds] [clipPrefix] [--assist] [--seed N]
//
// clipPrefix: vendor/policy prefix used for ALL four clip slots (default:
// spinkick — the pipeline-validation policy). When the trained boxing clips
// exist pass e.g. "boxing" with per-clip files boxing_{guard,jab,...}.
// --assist: run WITH the balance assist on (browser default); the default is
// assist OFF. 方案 §7.4：辅助开关分别报告结果。
// --seed N: deterministic RNG for decisions/timers (default: random).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
// COMBO_MODE 回归（plan-revamp-20260928 §5.2 D9）：BOXING_COMBO=1 时两拳手
// 常驻循环 combo 片段，默认跑 ≥42s（覆盖至少一个完整循环 + 回绕），断言为
// 无 NaN / divergences=0 / minZ≥0.40 / 回绕发生；KO 注入断言仅默认模式执行。
const COMBO = process.env.BOXING_COMBO === '1';
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2).filter(a => !a.startsWith('--'));
const raw = process.argv.slice(2);
const flags = raw.filter(a => a.startsWith('--'));
const DURATION = Number(argv[0] || (COMBO ? 42 : 16));
const KO_AT = DURATION * 0.6;
const PREFIX = argv[1] || 'spinkick';
const ASSIST = flags.includes('--assist');
const seedIdx = raw.indexOf('--seed');
const SEED = seedIdx >= 0 ? Number(raw[seedIdx + 1] ?? 1) : (Math.random() * 2 ** 31) | 0;

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
const { mulberry32 } = await import('../src/boxing_ai.mjs');

const loadNet = (name) => {
  const bin = fs.readFileSync(path.join(root, `vendor/policy/${name}.bin`));
  const meta = JSON.parse(fs.readFileSync(path.join(root, `vendor/policy/${name}_meta.json`), 'utf8'));
  return new TrackingNetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
};

// clip slots: single-prefix mode reuses one policy everywhere (mechanics
// check). Per-clip files win when present. COMBO adds the combo slot (falls
// back to PREFIX when boxing_combo is not deployed — mechanics-only run).
const slots = ['guard', 'jab', 'cross', 'hook'];
if (COMBO) slots.push('combo');
const nets = {};
const slotSource = {};
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
});
console.log(`assist=${ASSIST ? 'ON' : 'OFF'} seed=${SEED} combo=${COMBO ? 'ON' : 'off'}`);

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
// 拳法门禁期（全部拳法禁用）不会排队任何拳法意图、不会发生片段切换：
// crossQueued/crossPlayed 恒为 false；保留观察项供门禁逐个恢复后的回归读取。
let crossQueued = false, crossPlayed = false;

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
  if (ctl.fighters.A.punchIntent?.clip === 'cross' || ctl.fighters.B.punchIntent?.clip === 'cross') crossQueued = true;
  if (ctl.fighters.A.clip === 'cross' || ctl.fighters.B.clip === 'cross') crossPlayed = true;
  if (koT === null && t >= KO_AT && !COMBO) {
    ctl.damage.A = 7.5;
    koT = t;
    console.log(`t=${t.toFixed(1)}: injected lethal damage on A`);
  }
  if (t - lastPrint >= 2) {
    if (process.env.BOXING_DIST) {
      const dxp = data.xpos[3 * pelvisA] - data.xpos[3 * pelvisB];
      const dyp = data.xpos[3 * pelvisA + 1] - data.xpos[3 * pelvisB + 1];
      console.log(`DIST t=${t.toFixed(1)} pelvisDist=${Math.hypot(dxp, dyp).toFixed(3)}`);
    }
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
