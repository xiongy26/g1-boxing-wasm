// fight ONNX 部署物一致性自检（plan-fight-20260929 §5.2 W6，可选件）。
// 校验 fight meta 的 obs 布局与浏览器侧构造契约（§5.1 数据契约）逐项一致，
// 并对测试向量做可行的数值检查；用法：
//
//   node tools/check_fight_meta.mjs [vendor/policy/boxing_fight_dev]
//
// 参数是 <basename> 前缀（_meta.json / _test_vectors.json / .bin 同前缀）。
// 缺省依次探测 boxing_fight_dev / boxing_fight；都不存在时报错退出（训练侧
// 产物尚未交付属预期，此时以非零码标明"未验证"而非布局错误）。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = process.argv[2];
const candidates = arg ? [arg] : ['vendor/policy/boxing_fight_dev', 'vendor/policy/boxing_fight'];
const prefix = candidates.find(c => fs.existsSync(path.join(root, c + '_meta.json')));
if (!prefix) {
  console.log(`fight meta not found (tried: ${candidates.join(', ')})`);
  console.log('RESULT: SKIP (fight weights not deployed yet — 未验证)');
  process.exit(2);
}

const meta = JSON.parse(fs.readFileSync(path.join(root, prefix + '_meta.json'), 'utf8'));
// 契约 §5.1：段序与 [偏移, 宽度] 固定；opponent_state 154..168 为 fight 新段
const CONTRACT = {
  command: [0, 58],
  motion_anchor_ori_b: [58, 6],
  base_ang_vel: [64, 3],
  joint_pos: [67, 29],
  joint_vel: [96, 29],
  actions: [125, 29],
  opponent_state: [154, 14],
};

const fails = [];
const check = (ok, msg) => { console.log(`  ${ok ? 'ok ' : 'FAIL'} ${msg}`); if (!ok) fails.push(msg); };

console.log(`fight meta check: ${prefix}_meta.json`);
check(meta.obs_dim === 168, `obs_dim === 168 (got ${meta.obs_dim})`);
for (const [name, [off, w]] of Object.entries(CONTRACT)) {
  const got = meta.obs_layout?.[name];
  check(Array.isArray(got) && got[0] === off && got[1] === w,
    `obs_layout.${name} === [${off}, ${w}] (got ${JSON.stringify(got) ?? 'missing'})`);
}
const names = meta.obs_names ?? meta.observation_names;
check(Array.isArray(names) && names.includes('opponent_state'),
  `obs_names includes opponent_state (got ${JSON.stringify(names)})`);
// 一致性锚点（两侧审计点）：训练侧 anchor = MotionCommand cfg 的
// anchor_body_name = "torso_link"（config/g1/env_cfgs.py:42）——浏览器侧
// opponent_state 的自系构造用 side_torso_link（tracking_policy.mjs），meta
// 的 anchor_body_name 必须同值，否则 anchor_ori 段与对手段的参考系不一致。
check(meta.anchor_body_name === 'torso_link',
  `anchor_body_name === "torso_link" (got ${JSON.stringify(meta.anchor_body_name)})`);
check(JSON.stringify(meta.shapes?.mean) === '[168]' && JSON.stringify(meta.shapes?.std) === '[168]',
  `mean/std shapes === [168] (got ${JSON.stringify(meta.shapes?.mean)} / ${JSON.stringify(meta.shapes?.std)})`);
check(JSON.stringify(meta.shapes?.['W:0']) === '[512,168]',
  `W:0 input dim === 168 (got ${JSON.stringify(meta.shapes?.['W:0'])})`);

// 测试向量（若已导出）：obs 全维长度 + opponent_state 段有限性检查。
// 说明：段内数值无法在此复现——训练侧向量由双实体训练场景（对手位姿/速度
// 随训练回合状态而定）生成，向量文件不携带对手场景状态，浏览器侧重构造需要
// 完整双机 WASM 场景复放，故只做布局/有限性 spot-check，数值一致性由
// test_tracking.mjs（net.infer 全维前向对 ONNX ~1e-6）+ e2e（?fight=1 接线）
// 分工覆盖。
let vpath = path.join(root, prefix + '_test_vectors.json');
if (fs.existsSync(vpath)) {
  const vectors = JSON.parse(fs.readFileSync(vpath, 'utf8'));
  const n = vectors.obs?.length ?? 0;
  let lenOk = n > 0, finiteOk = true, nanSeg = 0;
  for (const obs of vectors.obs) {
    if (obs.length !== 168) lenOk = false;
    for (let i = 154; i < 168; i++) {
      if (!Number.isFinite(obs[i])) { finiteOk = false; nanSeg++; }
    }
  }
  check(lenOk, `${n} vectors: obs length === 168`);
  check(finiteOk, `opponent_state segment [154:168] all finite (${nanSeg} non-finite)`);
} else {
  console.log(`  --  ${prefix}_test_vectors.json not found (vector check skipped — 未验证)`);
}

console.log(`RESULT: ${fails.length ? 'FAIL' : 'PASS'} (${fails.length} layout violation(s))`);
process.exit(fails.length ? 1 : 0);
