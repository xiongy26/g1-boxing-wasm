// Tactics policy: a tiny pure-JS MLP that replaces the weighted-random combo
// picker in boxing_ai.mjs. Shared by the browser app and the headless trainer.
//
//   obs (13)  -> tanh(48) -> tanh(48) -> logits(10) -> softmax
//
// Actions (see TACTIC_ACTIONS): footwork + attack family selection. The exact
// combo within a family (left/right) is still randomized for variety.
// Training: REINFORCE with a moving-average baseline and an entropy bonus,
// self-play in the headless MuJoCo sim (tools/train_tactics.mjs).

export const OBS_DIM = 13;

export const TACTIC_ACTIONS = [
  'approach',    // 0  close the distance
  'retreat',     // 1  pull back out of range
  'strafe_left', // 2  lateral step (robot frame left; meaningful in AMO mode)
  'strafe_right',// 3  lateral step
  'jab',         // 4  jab_left / jab_right
  'double',      // 5  jab_left + jab_right
  'cross',       // 6  cross_left / cross_right
  'hook',        // 7  hook_left / hook_right
  'jab_hook',    // 8  jab_left + hook_right
  'wait',        // 9  hold guard briefly
];

// combo indices in boxing_ai.mjs COMBOS for each attack action
export const ACTION_COMBOS = {
  jab:      [0, 1],
  double:   [2],
  cross:    [3, 4],
  hook:     [5, 6],
  jab_hook: [7],
};

// ---- ring-craft prior ---------------------------------------------------------
// Distance-band + reaction priors as differentiable logits, blended with the
// network logits before the softmax (forward): final = net + priorW * prior.
// Why: three pure-REINFORCE runs (400+ episodes each) collapsed to the mutual
// standoff equilibrium — with ~10 decisions/episode there is far too little
// signal to discover "close distance, then attack" from sparse hit rewards.
// The prior guarantees a sane floor (a fighter that understands range); RL only
// fine-tunes around it. priorW is annealed 1.0 -> 0.3 during training and the
// deploy weight is stored in the JSON, so browser behaviour matches evaluation.
// obs layout (boxing_ai.buildTacticsObs): [1]=dist/1.5 [7]=opp attacking [8]=opp staggered
export function ringPriorLogits(obs) {
  const dist = obs[1] * 1.5;
  const l = new Float32Array(10);
  if (dist > 0.85) {          // out of range: close in decisively
    l[0] = 2.2; l[2] = 0.2; l[3] = 0.2;
    l[4] = -1.2; l[5] = -1.4; l[6] = -1.2; l[7] = -1.4; l[8] = -1.6; l[1] = -1.0; l[9] = -0.6;
  } else if (dist > 0.55) {   // edge of range: probe with the jab while closing
    l[0] = 1.2; l[2] = 0.4; l[3] = 0.4; l[4] = 0.6; l[9] = -0.4; l[1] = -0.6;
  } else if (dist > 0.38) {   // pocket: throw — this is what hits are made of
    l[4] = 2.0; l[5] = 1.5; l[6] = 1.6; l[7] = 1.4; l[8] = 1.5;
    l[2] = 0.15; l[3] = 0.15; l[0] = -0.6; l[9] = -0.8; l[1] = -0.4;
  } else {                    // clinch range: create space or punish
    l[1] = 1.2; l[4] = 0.6; l[6] = 0.5; l[2] = 0.3; l[3] = 0.3; l[0] = -1.4; l[9] = -0.4;
  }
  if (obs[7] > 0.5) { l[1] += 0.6; l[2] += 0.3; l[3] += 0.3; }              // respect incoming punch
  if (obs[8] > 0.5) { l[4] += 1.0; l[5] += 0.8; l[6] += 0.8; l[0] += 0.4; } // swarm the staggered
  return l;
}

const tanh = Math.tanh;

export class TacticsPolicy {
  constructor({ hidden = [48, 48], params = null } = {}) {
    this.sizes = [OBS_DIM, ...hidden, TACTIC_ACTIONS.length];
    this.shapes = [];
    let n = 0;
    for (let l = 0; l < this.sizes.length - 1; l++) {
      this.shapes.push([this.sizes[l + 1], this.sizes[l] + 1]); // W (+bias col)
      n += this.sizes[l + 1] * (this.sizes[l] + 1);
    }
    this.nParams = n;
    this.params = params ? Float32Array.from(params) : this._init();
    this._initAdam();
    this.baseline = 0;
    this.priorW = 0;  // ring-craft prior weight; >0 blends ringPriorLogits into the softmax
    this.netGain = 1; // net logit gain; <1 bounds how far the net can bend the prior
  }

  _init() {
    const p = new Float32Array(this.nParams);
    let off = 0;
    for (const [out, inPlus] of this.shapes) {
      const scale = Math.sqrt(2 / (inPlus - 1));
      for (let i = 0; i < out * inPlus; i++) p[off + i] = (Math.random() * 2 - 1) * scale;
      off += out * inPlus;
    }
    return p;
  }

  _initAdam() {
    this.m = new Float32Array(this.nParams);
    this.v = new Float32Array(this.nParams);
    this.adamT = 0;
  }

  // forward: returns {logits, probs, acts} where acts = per-layer activations
  forward(obs) {
    const p = this.params;
    const layers = [Float32Array.from(obs)];
    let off = 0;
    for (let l = 0; l < this.shapes.length; l++) {
      const [out, inPlus] = this.shapes[l];
      const x = layers[l];
      const y = new Float32Array(out);
      const last = l === this.shapes.length - 1;
      for (let o = 0; o < out; o++) {
        let s = p[off + o * inPlus + inPlus - 1]; // bias
        const wr = off + o * inPlus;
        for (let i = 0; i < x.length; i++) s += p[wr + i] * x[i];
        y[o] = last ? s : tanh(s);
      }
      off += out * inPlus;
      layers.push(y);
    }
    const logits = layers[layers.length - 1];
    // blend: final = netGain * net + priorW * prior, then softmax. The prior is
    // a constant w.r.t. params, so backprop through the net logits stays exact.
    // netGain < 1 structurally bounds the net's influence — with the engage
    // decision owned by the prior, the net can only pick *within* each band.
    if (this.netGain !== 1) for (let k = 0; k < logits.length; k++) logits[k] *= this.netGain;
    if (this.priorW > 0) {
      const pl = ringPriorLogits(layers[0]);
      for (let k = 0; k < logits.length; k++) logits[k] += this.priorW * pl[k];
    }
    let mx = -1e30;
    for (const v of logits) if (v > mx) mx = v;
    let sum = 0;
    const probs = new Float32Array(logits.length);
    for (let k = 0; k < logits.length; k++) { probs[k] = Math.exp(logits[k] - mx); sum += probs[k]; }
    for (let k = 0; k < probs.length; k++) probs[k] /= sum;
    return { probs, layers };
  }

  act(obs, { sample = true, temp = 1, rng = Math.random } = {}) {
    const { probs } = this.forward(obs);
    if (!sample) {
      let best = 0;
      for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
      return { action: best, probs };
    }
    // temperature-scaled sampling
    let sum = 0;
    const adj = new Float32Array(probs.length);
    for (let k = 0; k < probs.length; k++) { adj[k] = Math.exp(Math.log(probs[k] + 1e-12) / temp); sum += adj[k]; }
    let r = rng() * sum, action = probs.length - 1;
    for (let k = 0; k < adj.length; k++) { r -= adj[k]; if (r <= 0) { action = k; break; } }
    return { action, probs };
  }

  // One REINFORCE step on a batch of trajectories.
  // traj: [{obs, action, ret}]  — ret is the discounted return from that decision
  // opts: {lr, entropy (beta), gradClip}
  update(traj, { lr = 2e-3, entropy = 0.01, gradClip = 5 } = {}) {
    if (!traj.length) return { loss: 0, entropy: 0 };
    let bSum = 0;
    for (const s of traj) bSum += s.ret;
    const baseline = bSum / traj.length;
    this.baseline = 0.9 * this.baseline + 0.1 * baseline;

    const grad = new Float32Array(this.nParams);
    let lossSum = 0, entSum = 0;
    for (const s of traj) {
      const adv = s.ret - this.baseline;
      const { probs, layers } = this.forward(s.obs);
      const a = s.action;
      const logpa = Math.log(probs[a] + 1e-12);
      let H = 0;
      for (let k = 0; k < probs.length; k++) if (probs[k] > 1e-12) H -= probs[k] * Math.log(probs[k]);
      lossSum += -adv * logpa - entropy * H;
      entSum += H;
      // dLogits = dL/dlogits for DESCENT on L = -adv·logp(a) - entropy·H:
      //   dL/dlogit_k = adv·(p_k - 1[k=a]) + entropy·p_k·(log p_k + H)
      // （2026-09-26 修复：原先写的是 J = -L 的上升方向，交给 params -= lr·g
      //   的 Adam 等于反着训——tools/test_tactics_grad.mjs 数值梯度检查抓出。）
      const dLogits = new Float32Array(probs.length);
      for (let k = 0; k < probs.length; k++) {
        let g = adv * (probs[k] - (k === a ? 1 : 0));
        if (probs[k] > 1e-12) g += entropy * probs[k] * (Math.log(probs[k]) + H);
        dLogits[k] = g;
      }
      // 链式法则：forward 的最终 logits = netGain·net + priorW·prior，反向
      // 传播要对 net logits 求 dL，必须乘 netGain（否则 netGain<1 时梯度被
      // 放大 1/netGain——tools/test_tactics_grad.mjs 抓出的第二个 bug）。
      const gain = this.netGain;
      if (gain !== 1) for (let k = 0; k < dLogits.length; k++) dLogits[k] *= gain;
      this._backprop(layers, dLogits, grad);
    }
    const n = traj.length;
    let sq = 0;
    for (let i = 0; i < grad.length; i++) { grad[i] /= n; sq += grad[i] * grad[i]; }
    if (sq > gradClip * gradClip) {
      const s = gradClip / Math.sqrt(sq);
      for (let i = 0; i < grad.length; i++) grad[i] *= s;
    }
    this._adam(grad, lr);
    return { loss: lossSum / n, entropy: entSum / n, baseline: this.baseline };
  }

  _backprop(layers, dLogits, grad) {
    let delta = dLogits;
    let off = this.nParams;
    for (let l = this.shapes.length - 1; l >= 0; l--) {
      const [out, inPlus] = this.shapes[l];
      off -= out * inPlus;
      const x = layers[l];
      const ndelta = l > 0 ? new Float32Array(x.length) : null;
      for (let o = 0; o < out; o++) {
        const wr = off + o * inPlus;
        const d = delta[o];
        if (d !== 0) {
          grad[wr + inPlus - 1] += d;
          for (let i = 0; i < x.length; i++) {
            grad[wr + i] += d * x[i];
            if (ndelta) ndelta[i] += d * this.params[wr + i];
          }
        } else if (ndelta) {
          for (let i = 0; i < x.length; i++) ndelta[i] += 0;
        }
      }
      if (ndelta) {
        for (let i = 0; i < ndelta.length; i++) {
          const a = layers[l][i];
          ndelta[i] *= 1 - a * a; // tanh'
        }
        delta = ndelta;
      }
    }
  }

  _adam(grad, lr) {
    this.adamT++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.adamT), c2 = 1 - Math.pow(b2, this.adamT);
    for (let i = 0; i < this.params.length; i++) {
      this.m[i] = b1 * this.m[i] + (1 - b1) * grad[i];
      this.v[i] = b2 * this.v[i] + (1 - b2) * grad[i] * grad[i];
      this.params[i] -= lr * (this.m[i] / c1) / (Math.sqrt(this.v[i] / c2) + eps);
    }
  }

  // ---- serialization ---------------------------------------------------------
  toJSON(extra = {}) {
    return {
      version: 1, obsDim: OBS_DIM, actions: TACTIC_ACTIONS,
      sizes: this.sizes, params: Array.from(this.params, v => Math.round(v * 1e6) / 1e6),
      priorW: this.priorW, netGain: this.netGain,
      ...extra,
    };
  }

  static fromJSON(json) {
    if (!json || json.obsDim !== OBS_DIM) throw new Error('tactics json: obsDim mismatch');
    const pol = new TacticsPolicy({ hidden: json.sizes.slice(1, -1) });
    if (json.params.length !== pol.nParams) throw new Error('tactics json: param count mismatch');
    pol.params = Float32Array.from(json.params);
    pol.priorW = json.priorW ?? 0;
    pol.netGain = json.netGain ?? 1;
    return pol;
  }
}
