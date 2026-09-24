// Episode worker for tools/train_tactics.mjs — one headless scripted-mode
// MuJoCo env per worker thread. Runs one self-play episode per message and
// returns both fighters' tactic trajectories + the raw reward events.
import fs from 'node:fs';
import path from 'node:path';
import { parentPort, workerData } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';

const root = workerData.root;
const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const mujoco = await loadMujoco();

const vfs = new mujoco.MjVFS();
const assetsDir = path.join(root, 'models/unitree_g1/assets');
for (const f of fs.readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl')) {
    vfs.addBuffer('unitree_g1/assets/' + f, new Uint8Array(fs.readFileSync(path.join(assetsDir, f))));
  }
}
const xml = fs.readFileSync(path.join(root, 'models/scene_boxing_amo.xml'), 'utf8');
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

const OBJ = mujoco.mjtObj;
const id = (t, n) => mujoco.mj_name2id(model, t.value ?? t, n);
const { BoxingController } = await import('../src/boxing_ai.mjs');
const { TacticsPolicy } = await import('../src/tactics.mjs');

const ctl = new BoxingController(model, data, {
  mujoco,
  ids: {
    pelvisA: id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB: id(OBJ.mjOBJ_BODY, 'B_pelvis'),
    headA: id(OBJ.mjOBJ_SITE, 'A_head'), headB: id(OBJ.mjOBJ_SITE, 'B_head'),
    fistLA: id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA: id(OBJ.mjOBJ_SITE, 'A_right_fist_site'),
    fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site'),
  },
});
// AMO mode: the scripted fallback cannot stand in this scene (its gantry assist
// predates the AMO scene and cannot hold the robot up), and training must match
// deployment anyway — the tactic command semantics are identical in both modes.
// KO by damage/limp/reset gives the learner real fight dynamics.
const bin = fs.readFileSync(path.join(root, 'vendor/policy/amo.bin'));
const meta = JSON.parse(fs.readFileSync(path.join(root, 'vendor/policy/amo_meta.json'), 'utf8'));
const { AMONetwork } = await import('../src/rl_policy.mjs');
const amoNet = new AMONetwork(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
ctl.setAMO('A', amoNet);
ctl.setAMO('B', amoNet);
ctl.aggression = 1;

const polA = new TacticsPolicy();
const polB = new TacticsPolicy();
ctl.tactics = { A: polA, B: polB }; // per-side policies (opponent pool support)

function runEpisode({ learnerParams, oppParams, oppMode, learnerSide, epLen, temp, priorW, netGain }) {
  polA.params.set(learnerParams);
  polB.params.set(oppParams ?? learnerParams);
  polA.tacticsTemp = temp;
  polB.tacticsTemp = temp;
  // prior weight / net gain must match between sampling (here) and the REINFORCE
  // update (trainer side) — the log-prob of the sampled action is computed on
  // the same blended softmax, otherwise the gradient is importance-biased.
  polA.priorW = priorW ?? 0;
  polB.priorW = priorW ?? 0;
  polA.netGain = netGain ?? 1;
  polB.netGain = netGain ?? 1;

  // learner side always gets the policy; in 'legacy' mode the opponent side
  // has no policy entry and falls back to the hardcoded weighted-random AI —
  // phase 1 of the curriculum (learn to attack/exploit before self-play).
  ctl.tactics = oppMode === 'legacy'
    ? { [learnerSide === 'A' ? 'A' : 'B']: learnerSide === 'A' ? polA : polB }
    : { A: polA, B: polB };
  ctl.fighters.A.traj = [];
  ctl.fighters.B.traj = [];
  const t0 = ctl.time;
  const evStart = ctl.events.length;
  const hits0 = { A: ctl.fighters.A.scores.hits, B: ctl.fighters.B.scores.hits };
  const dt = model.opt.timestep;
  const steps = Math.round(epLen / dt);
  let diverged = false;
  const pelvisA = ctl.ids.pelvisA, pelvisB = ctl.ids.pelvisB;
  for (let i = 0; i < steps; i++) {
    ctl.update(dt);
    mujoco.mj_step(model, data);
    if (!Number.isFinite(data.xpos[3 * pelvisA + 2]) || !Number.isFinite(data.xpos[3 * pelvisB + 2])) {
      diverged = true;
      break;
    }
  }
  const t1 = ctl.time;
  const events = ctl.events.splice(evStart).filter(e => e.t >= t0 - 1e-9);
  return {
    trajA: ctl.fighters.A.traj,
    trajB: ctl.fighters.B.traj,
    events,
    t0, t1,
    diverged,
    hits: { A: ctl.fighters.A.scores.hits - hits0.A, B: ctl.fighters.B.scores.hits - hits0.B },
    blocks: events.filter(e => e.type === 'block').length,
    ko: events.filter(e => e.type === 'ko').map(e => e.winner ?? null),
    simTime: t1 - t0,
  };
}

parentPort.on('message', msg => {
  if (msg.type === 'ep') {
    try {
      const r = runEpisode(msg);
      parentPort.postMessage({ type: 'ep', id: msg.id, ...r });
    } catch (err) {
      parentPort.postMessage({ type: 'error', id: msg.id, error: String(err && err.stack || err) });
    }
  }
});
parentPort.postMessage({ type: 'ready' });
