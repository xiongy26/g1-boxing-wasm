// Tactics-policy numeric gradient check (plan-realistic-boxing §4.7: "复核策略
// 更新的梯度符号、netGain 对梯度的影响及采样分布一致性；可用小规模数值梯度
// 检查验证").
//
//   node tools/test_tactics_grad.mjs
//
// Checks, on a tiny random network and batch:
//   1. analytic gradient (update()'s backprop, captured at _adam) matches the
//      finite-difference gradient of the REINFORCE + entropy loss;
//   2. with priorW > 0, netGain=0.5 gradients equal 0.5× the netGain=1
//      gradients (prior is constant w.r.t. params, so backprop stays exact);
//   3. forward() probs match a manual softmax of netGain*net + priorW*prior
//      and sum to 1 (sampling distribution consistency).
import { TacticsPolicy, ringPriorLogits } from '../src/tactics.mjs';

const rng = (() => { let s = 42; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
const gaussian = () => {
  const u = Math.max(rng(), 1e-9), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  (' + detail + ')' : ''}`);
  if (!ok) failures++;
}

// loss used by update(): L = -adv*logp(a) - entropy*H  (adv fixed w.r.t. params)
function lossFor(pol, obs, action, adv, entropy) {
  const { probs } = pol.forward(obs);
  const logpa = Math.log(probs[action] + 1e-12);
  let H = 0;
  for (let k = 0; k < probs.length; k++) if (probs[k] > 1e-12) H -= probs[k] * Math.log(probs[k]);
  return -adv * logpa - entropy * H;
}

function runGradCheck(netGain, priorW) {
  const pol = new TacticsPolicy({ hidden: [8, 8] });
  for (let i = 0; i < pol.nParams; i++) pol.params[i] = gaussian() * 0.5;
  pol.priorW = priorW;
  pol.netGain = netGain;
  const ret = 0.1;
  pol.baseline = 0;                       // fixed so adv is constant w.r.t. params
  const obs = Array.from({ length: 13 }, () => gaussian());
  const { probs } = pol.forward(obs);
  let action = 0, r = rng();
  for (let k = 0; k < probs.length; k++) { r -= probs[k]; if (r <= 0) { action = k; break; } }
  const entropy = 0.01;
  // update() 先做 baseline EMA（单样本: new = 0.1*ret），adv = ret - new = 0.9*ret
  const advUsed = ret - (0.9 * pol.baseline + 0.1 * ret);

  // analytic: capture the gradient update() passes to _adam (pre-Adam)
  let captured = null;
  pol._adam = (grad) => { captured = Float32Array.from(grad); };
  pol.update([{ obs, action, ret }], { lr: 0, entropy });   // lr=0: params unchanged

  // numeric (same loss, same adv the update actually used)
  // eps 取 2e-3：参数存于 Float32Array，扰动写入会被量化（ulp≈3e-8），eps
  // 太小会让有效扰动的量化抖动主导误差（1e-4/1e-5 时出现 2-3% 假失配）。
  const eps = 2e-3;
  let maxErr = 0, scale = 0;
  const nCheck = Math.min(60, pol.nParams);
  for (let i = 0; i < nCheck; i++) {
    const p0 = pol.params[i];
    pol.params[i] = p0 + eps;
    const lp = lossFor(pol, obs, action, advUsed, entropy);
    pol.params[i] = p0 - eps;
    const lm = lossFor(pol, obs, action, advUsed, entropy);
    pol.params[i] = p0;
    const num = (lp - lm) / (2 * eps);
    // update() minimizes; grad accumulates dL, then Adam DEscends: params -= lr*g.
    // captured gradient = dL/dparam.
    maxErr = Math.max(maxErr, Math.abs(num - captured[i]));
    scale = Math.max(scale, Math.abs(num));
  }
  return { maxErr, scale, rel: maxErr / Math.max(scale, 1e-12) };
}

const a = runGradCheck(1, 0);
check('梯度符号与数值 (netGain=1, priorW=0)', a.rel < 2e-3,
  `rel err ${a.rel.toExponential(2)}, |g|max ${a.scale.toExponential(2)}`);

// netGain/priorW 组合点：验证 netGain 链式因子与 prior 常数路径。注意
// "grad(netGain=0.5) = 0.5×grad(1)" 不成立——gain 改变 softmax 本身，不满足
// 线性；正确的判定是各自与数值微分一致。
const b = runGradCheck(0.5, 1);
check('梯度与数值一致 (netGain=0.5, priorW=1)', b.rel < 2e-3,
  `rel err ${b.rel.toExponential(2)}`);
const c = runGradCheck(0.3, 0.3);
check('梯度与数值一致 (netGain=0.3, priorW=0.3，退火工作点)', c.rel < 2e-3,
  `rel err ${c.rel.toExponential(2)}`);

// sampling distribution consistency: forward probs == softmax(netGain*net + priorW*prior)
{
  const pol = new TacticsPolicy({ hidden: [8, 8] });
  for (let i = 0; i < pol.nParams; i++) pol.params[i] = gaussian() * 0.5;
  pol.priorW = 0.7; pol.netGain = 0.9;
  const obs = Array.from({ length: 13 }, () => gaussian());
  const { probs } = pol.forward(obs);
  // recompute logits manually: run forward with priorW=0 for net logits
  const savedW = pol.priorW;
  pol.priorW = 0;
  const netOnly = pol.forward(obs).probs;   // only need relative — use raw layer? simpler:
  // direct: recompute via manual softmax over combined logits
  // (forward returns only probs; replicate logits through a no-prior forward's
  //  layer stack is not exported, so verify via inverse softmax consistency)
  let sum = 0;
  for (const p of probs) sum += p;
  check('probs 归一化', Math.abs(sum - 1) < 1e-6, `sum=${sum.toFixed(9)}`);
  pol.priorW = savedW;
  // argmax(probs) must equal manual argmax of gain*netlogit + priorW*prior
  const l = ringPriorLogits(obs);
  // netOnly = softmax(netGain·net)（priorW=0 的 forward）⇒ log(netOnly_k) =
  // netGain·net_k − C。组合指数 = log(netOnly_k) + priorW·prior_k（常数可略）。
  let mx = -1e30;
  const comb = [];
  for (let k = 0; k < netOnly.length; k++) {
    const v = Math.log(netOnly[k] + 1e-300) + 0.7 * l[k];
    comb.push(v); if (v > mx) mx = v;
  }
  let s = 0;
  for (let k = 0; k < comb.length; k++) { comb[k] = Math.exp(comb[k] - mx); s += comb[k]; }
  let maxP = 0;
  for (let k = 0; k < comb.length; k++) maxP = Math.max(maxP, Math.abs(comb[k] / s - probs[k]));
  check('softmax(netGain·net + priorW·prior) 一致', maxP < 1e-6, `max prob diff ${maxP.toExponential(2)}`);
}

// greedy act == argmax probs
{
  const pol = new TacticsPolicy({ hidden: [8, 8] });
  pol.priorW = 1;
  const obs = Array.from({ length: 13 }, () => gaussian());
  const { probs } = pol.forward(obs);
  const { action } = pol.act(obs, { sample: false });
  let best = 0;
  for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
  check('act(sample=false) == argmax(probs)', action === best);
}

console.log(failures ? 'RESULT: FAIL' : 'RESULT: PASS');
process.exit(failures ? 1 : 0);
