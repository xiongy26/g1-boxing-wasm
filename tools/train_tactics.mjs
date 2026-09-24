// Self-play REINFORCE trainer for the boxing tactics policy
// (src/tactics.mjs). Spawns N headless MuJoCo workers (scripted mode), each
// running one episode per message; the learner updates the shared policy from
// the learner-side trajectory and keeps a small pool of past snapshots as
// opponents.
//
//   node tools/train_tactics.mjs [--episodes 3000] [--workers 6] [--ep-len 24]
//                                 [--out vendor/tactics/tactics.json] [--minutes 30]
//
// The output JSON is loaded by the browser (src/main.js) and replaces the
// weighted-random combo picker. Without it, boxing_ai.mjs falls back to the
// old behaviour, so this is fully optional.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const arg = (k, d) => {
  const i = args.indexOf('--' + k);
  return i >= 0 ? Number(args[i + 1]) : d;
};
const EPISODES = arg('episodes', 3000);
const NWORKERS = Math.max(1, Math.min(arg('workers', Math.max(2, os.cpus().length - 1)), 12));
const EP_LEN = arg('ep-len', 24);
const MINUTES = arg('minutes', 0); // 0 = no time limit
// curriculum: first PHASE1 episodes fight the hardcoded weighted-random AI
// (teaches engagement — pure self-play from scratch collapses to mutual
// standoff), then the opponent pool of past snapshots takes over.
const PHASE1 = arg('phase1', 250);
const OUT = path.join(root, args.includes('--out') ? args[args.indexOf('--out') + 1] : 'vendor/tactics/tactics.json');

const { TacticsPolicy } = await import('../src/tactics.mjs');

const policy = new TacticsPolicy();
const pool = [];            // opponent snapshots (Float32Array params)
const POOL_MAX = 8;
const GAMMA = 0.97;

// ---- reward shaping (learner perspective) -----------------------------------
// Landed >> taken on purpose: in symmetric self-play a "trading" exchange must
// be locally positive or both policies converge to mutual standoff avoidance
// (observed: hit reward 1.5x/2x => hits collapse to zero within 50 episodes).
function eventReward(e, side) {
  switch (e.type) {
    case 'hit':
      if (e.attacker === side) return 3 * e.points;     // head +6 / body +3
      if (e.victim === side) return -0.5;
      return 0;
    case 'block': return 0.3;
    case 'ko':
      if (e.winner === side) return 8;
      if (e.down === side) return -8;
      return 0;
    default: return 0;
  }
}

// assign event rewards to the decision that was in control, then discount.
// Shaping: small per-decision proximity bonus keeps "mutual retreat" (the
// degenerate equilibrium of sparse hit rewards in symmetric self-play) from
// being optimal — someone must close the distance for hits to ever happen.
function buildTrajectory(traj, events, side) {
  if (!traj.length) return [];
  const rew = new Float64Array(traj.length);
  let di = 0;
  for (const e of events) {
    const r = eventReward(e, side);
    if (r === 0) continue;
    while (di + 1 < traj.length && traj[di + 1].t <= e.t) di++;
    rew[di] += r;
  }
  for (let i = 0; i < traj.length; i++) {
    const dist = traj[i].obs[1] * 1.5;         // undo the obs normalization
    const prox = Math.max(0, 1 - dist / 0.9);  // 1 at range 0, 0 beyond 0.9m
    // Distance pressure: hovering far away must be clearly WORSE than engaging,
    // or both policies converge to mutual standoff (observed twice: avgRet
    // pinned at exactly -0.20 = 7 decisions x -0.03 with zero hits). With the
    // ring prior driving engagement the pressure is secondary, so keep it mild.
    if (dist > 0.75) rew[i] -= 0.25;           // standoff is expensive
    else rew[i] += 0.12 * prox - 0.02;         // pocket zone stays mildly positive
    // commitment bonus: attacking while in range fights the standoff collapse
    const a = traj[i].action;
    if (prox > 0 && a >= 4 && a <= 8) rew[i] += 0.10;
  }
  const G = new Float64Array(traj.length);
  let acc = 0;
  for (let i = traj.length - 1; i >= 0; i--) {
    acc = rew[i] + GAMMA * acc;
    G[i] = acc;
  }
  return traj.map((s, i) => ({ obs: s.obs, action: s.action, ret: G[i] }));
}

// ---- worker pool --------------------------------------------------------------
const workers = [];
const idle = [];
const inflight = new Map();
let epId = 0, ready = 0, errors = 0;
let shuttingDown = false;

function spawnWorker() {
  return new Promise(resolve => {
    const w = new Worker(new URL('./train_worker.mjs', import.meta.url), { workerData: { root } });
    w.on('message', msg => {
      if (msg.type === 'ready') { ready++; resolve(w); return; }
      if (msg.type === 'error') {
        console.error('worker error:', msg.error);
        errors++;
      }
      const rec = inflight.get(msg.id);
      if (rec) { inflight.delete(msg.id); idle.push(w); rec(msg); }
    });
    w.on('error', err => { console.error('worker crashed:', err); process.exit(1); });
    workers.push(w);
  });
}

console.log(`training tactics: ${EPISODES} episodes, ${NWORKERS} workers, ${EP_LEN}s sim per episode`);
console.time('boot');
for (let i = 0; i < NWORKERS; i++) idle.push(await spawnWorker());
console.timeEnd('boot');

// ---- training loop --------------------------------------------------------------
let eps = 0, sumRet = 0, sumHits = 0, sumBlocks = 0, sumEnt = 0, wins = 0, kos = 0;
let bestAvg = -Infinity;
const tStart = Date.now();
const ENT_BETA = 0.02;          // fixed mild entropy bonus (annealing caused premature collapse)
// Ring prior owns the engage decision (priorW constant 1.0); the net output is
// scaled by NET_GAIN so it can only bend probabilities WITHIN each distance
// band. Six runs showed the net otherwise learns to cancel any prior weight
// and collapses to mutual standoff — the residual is bounded by construction.
const DEPLOY_PRIORW = 1.0;
const NET_GAIN = 0.1;
const priorWAt = () => DEPLOY_PRIORW;

function dispatch() {
  while (idle.length && eps + inflight.size < EPISODES && !shuttingDown) {
    const w = idle.pop();
    const id = ++epId;
    const learnerSide = Math.random() < 0.5 ? 'A' : 'B';
    const phase1 = eps < PHASE1;
    const usePool = !phase1 && pool.length > 0 && Math.random() < 0.6;
    const oppParams = usePool ? pool[Math.floor(Math.random() * pool.length)] : policy.params;
    const msg = {
      type: 'ep', id, learnerSide, oppMode: phase1 ? 'legacy' : 'policy',
      learnerParams: policy.params.slice(),
      oppParams: oppParams.slice(),
      epLen: EP_LEN, temp: 1.0,
      priorW: priorWAt(eps),
      netGain: NET_GAIN,
    };
    inflight.set(id, res => {
      if (shuttingDown) return;
      const trajRaw = learnerSide === 'A' ? res.trajA : res.trajB;
      if (res.diverged || !trajRaw || trajRaw.length < 3) {
        eps++;
        return;
      }
      const traj = buildTrajectory(trajRaw, res.events, learnerSide);
      // update on the same prior-blended softmax the actions were sampled from
      policy.priorW = msg.priorW;
      policy.netGain = NET_GAIN;
      const { entropy } = policy.update(traj, { lr: 2e-3, entropy: ENT_BETA });

      const myKO = res.ko.includes(learnerSide);
      const oppSide = learnerSide === 'A' ? 'B' : 'A';
      sumRet += traj.reduce((a, s) => a + s.ret, 0) / traj.length;
      sumHits += res.hits[learnerSide];
      sumBlocks += res.blocks;
      sumEnt += entropy;
      if (myKO) { wins++; kos++; }

      if (++eps % 200 === 0) {
        pool.push(policy.params.slice());
        if (pool.length > POOL_MAX) pool.shift();
      }
      if (eps % 25 === 0) {
        const avg = sumRet / 25, hps = sumHits / 25;
        const el = ((Date.now() - tStart) / 60000).toFixed(1);
        console.log(`ep ${eps}${eps === PHASE1 ? ' [phase2: self-play]' : ''}  avgRet=${avg.toFixed(2)}  hits/ep=${hps.toFixed(1)}  ` +
          `blocks/ep=${(sumBlocks / 25).toFixed(1)}  H=${(sumEnt / 25).toFixed(3)}  ` +
          `KO-wins=${wins}/${kos}  pool=${pool.length}  ${el}min`);
        if (eps % 100 === 0 && avg > bestAvg) {
          bestAvg = avg;
          savePolicy('best');
        }
        sumRet = 0; sumHits = 0; sumBlocks = 0; sumEnt = 0;
      }
      if (MINUTES > 0 && (Date.now() - tStart) > MINUTES * 60000) shuttingDown = true;
    });
    w.postMessage(msg);
  }
}

function savePolicy(tag) {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  const json = policy.toJSON({
    trainedEpisodes: eps, tag, epLen: EP_LEN, savedAt: new Date().toISOString(),
    priorW: DEPLOY_PRIORW, netGain: NET_GAIN, // deploy weights baked in: browser == evaluated policy
  });
  fs.writeFileSync(OUT, JSON.stringify(json));
}

// tick scheduler
await new Promise(resolve => {
  const loop = setInterval(() => {
    dispatch();
    if ((eps >= EPISODES && inflight.size === 0) || (shuttingDown && inflight.size === 0)) {
      clearInterval(loop);
      resolve();
    }
    // refill idle workers that finished without a new dispatch (handled in callback)
    if (idle.length && eps + inflight.size < EPISODES) dispatch();
  }, 10);
});

savePolicy('final');
console.log(`done: ${eps} episodes, saved -> ${path.relative(root, OUT)}`);
for (const w of workers) await w.terminate();
process.exit(0);
