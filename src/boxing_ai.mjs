// Boxing controller shared by the browser app (src/main.js) and the headless
// regression (tools/test_rl.mjs). Pure logic on top of the MuJoCo model/data.
//
// Scene: models/scene_boxing_amo.xml — two official AMO g1.xml robots (23 torque
// motors each, wrists fused) + glove/head/torso collision geoms. Because the
// actuators are TORQUE motors, every write to ctrl is a torque: both control
// modes compute PD torques in JS (kp/kd/torque limits from the AMO deployment
// recipe, see rl_policy.mjs).
//
// Modes per robot:
//  - AMO mode: the pretrained AMO whole-body policy balances and moves the
//    robot (50 Hz policy on the 500 Hz sim, decimation 10). The boxing AI only
//    emits commands (vx/vy/yaw/height/torso + 8 arm-joint targets). No gantry,
//    no scripted poses — the robot is kept upright solely by the policy, and a
//    KO simply removes the policy drive (torque -> 0) so it collapses naturally.
//  - Scripted fallback: guard-pose state machine (blended pose targets) + the
//    classic pelvis "gantry" assist, for when the AMO weights are not loaded.
//
// Hits are detected from real contacts: glove spheres vs opponent head/torso.

import { AMOFighter, POLICY_JOINTS, DEFAULT_POSE, KP, KD, TORQUE_LIM } from './rl_policy.mjs';
import { TrackingFighter } from './tracking_policy.mjs';
import { OBS_DIM, TACTIC_ACTIONS, ACTION_COMBOS } from './tactics.mjs';

// tactic decision -> tracking clip name (stage-4 motion layer). Footwork/idle
// states ride the looping guard clip; attacks trigger their punch clip.
const TACTIC_CLIP = {
  approach: 'guard', retreat: 'guard', strafe_left: 'guard', strafe_right: 'guard',
  wait: 'guard',
  jab: 'jab', double: 'jab', cross: 'cross', hook: 'hook',
};

// pelvis-to-pelvis distance window at which a punch clip may START. The punch
// motions carry forward drive (jab ~0.2m, cross/hook ~0.8-1.0m): firing them
// point-blank turns the lunge into a body-check tangle, firing them too far
// is a whiff-charge. Outside the window the scheduler plays guard instead.
const PUNCH_RANGE = {
  // 标定（拳-头最小距离 vs 起始间距）：单方面出拳够不着（0.49m@0.7m 间距），
  // 命中发生在双方面互刺、各自前倾 0.2-0.4m 的交换里——窗口收窄到接触带
  jab: [0.62, 1.05],
  cross: [0.62, 1.05],
  hook: [0.62, 1.05],
};

const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;

// AMO arm targets (8): [L_sp, L_sr, L_sy, L_elbow, R_sp, R_sr, R_sy, R_elbow].
// Guard = AMO default pose. Jab poses verified stable in the headless pipeline.
const ARM_GUARD = [...DEFAULT_POSE.subarray(15)];
const ARM_POSES = {
  guard: ARM_GUARD,
  jab_left:  [-0.6, -0.2, 0.2, -0.9,  0.5, 0, -0.2, 0.3],
  jab_right: [ 0.5, 0, 0.2, 0.3,  -0.6, 0.2, -0.2, -0.9],
  cross_left:  [-1.05, -0.25, 0.35, -0.95,  0.45, 0.05, -0.1, 0.55],
  cross_right: [ 0.45, 0.05, -0.1, 0.55,  -1.05, 0.25, -0.35, -0.95],
  hook_left:  [-0.35, -0.85, 0.35, -1.25,  0.5, 0, -0.2, 0.3],
  hook_right: [ 0.5, 0, 0.2, 0.3,  -0.35, 0.85, -0.35, -1.25],
};

const COMBOS = [
  { seq: ['jab_left'], recover: 0.35 },
  { seq: ['jab_right'], recover: 0.35 },
  { seq: ['jab_left', 'jab_right'], recover: 0.45 },
  { seq: ['cross_right'], recover: 0.5 },
  { seq: ['cross_left'], recover: 0.5 },
  { seq: ['hook_left'], recover: 0.45 },
  { seq: ['hook_right'], recover: 0.45 },
  { seq: ['jab_left', 'hook_right'], recover: 0.55 },
];
const COMBO_POOL = [
  ['jab', 3], ['double', 2], ['cross', 2], ['hook', 1.5],
];
const POOL_MAP = {
  jab: [0, 1],
  double: [2],
  cross: [3, 4],
  hook: [5, 6, 7],
};

function weightedPick(pool, rng) {
  let total = 0; for (const [, w] of pool) total += w;
  let r = rng() * total;
  for (const [name, w] of pool) { r -= w; if (r <= 0) return name; }
  return pool[0][0];
}

// ---- scripted fallback poses (23-joint, name-keyed, no wrists) ---------------
const CROUCH = { hip_pitch: 0.10, knee: 0.20, ankle_pitch: -0.10 };
const GUARD = {
  left_shoulder_pitch: -0.72, left_shoulder_roll: -0.10, left_shoulder_yaw: 0.0, left_elbow: 1.85,
  right_shoulder_pitch: -0.72, right_shoulder_roll: 0.10, right_shoulder_yaw: 0.0, right_elbow: 1.85,
  waist_yaw: 0.0, waist_pitch: 0.06, waist_roll: 0.0,
  left_hip_pitch: CROUCH.hip_pitch, right_hip_pitch: CROUCH.hip_pitch,
  left_knee: CROUCH.knee, right_knee: CROUCH.knee,
  left_ankle_pitch: CROUCH.ankle_pitch, right_ankle_pitch: CROUCH.ankle_pitch,
};
const SPOSES = {
  guard: {},
  jab_left: {
    left_shoulder_pitch: -1.42, left_shoulder_roll: -0.05, left_shoulder_yaw: -0.30,
    left_elbow: 0.25, waist_yaw: -0.42, waist_pitch: 0.10,
  },
  jab_right: {
    right_shoulder_pitch: -1.42, right_shoulder_roll: 0.05, right_shoulder_yaw: 0.30,
    right_elbow: 0.25, waist_yaw: 0.42, waist_pitch: 0.10,
  },
  stagger: {
    left_shoulder_pitch: -0.2, right_shoulder_pitch: -0.2,
    left_elbow: 0.9, right_elbow: 0.9, waist_pitch: -0.12,
  },
};
function resolvePose(pose) {
  const out = { ...GUARD };
  for (const [k, v] of Object.entries(pose)) {
    if (k.startsWith('left_') || k.startsWith('right_') || k.startsWith('waist_')) { out[k] = v; continue; }
    out['left_' + k] = v; out['right_' + k] = v;
  }
  return out;
}
const SRESOLVED = Object.fromEntries(Object.entries(SPOSES).map(([k, v]) => [k, resolvePose(v)]));

const smooth = t => t * t * (3 - 2 * t);
const snap = t => 1 - (1 - t) * (1 - t) * (1 - t);
function lerpMap(a, b, s) {
  const out = { ...a };
  for (const k of Object.keys(b)) out[k] = (a[k] ?? 0) + ((b[k] ?? 0) - (a[k] ?? 0)) * s;
  return out;
}
function addOffsets(base, offs, k) {
  const out = { ...base };
  for (const [jn, v] of Object.entries(offs)) {
    const names = jn.startsWith('left_') || jn.startsWith('right_') || jn.startsWith('waist_')
      ? [jn] : [`left_${jn}`, `right_${jn}`];
    for (const n of names) out[n] = (out[n] ?? 0) + v * k;
  }
  return out;
}

// ---- fighter ----------------------------------------------------------------
class Fighter {
  constructor(side, anchor0) {
    this.side = side;
    this.forward = side === 'A' ? 1 : -1;      // A walks +x, B walks -x
    this.faceYaw = side === 'A' ? 0 : Math.PI; // absolute yaw command toward opponent
    this.anchor0 = anchor0;
    this.anchor = [...anchor0];
    this.state = 'idle';
    this.stateT = 0;
    this.idleFor = 0.7 + Math.random() * 0.6;
    this.combo = null;
    this.seqIdx = 0;
    this.phaseT = 0;
    this.phaseDur = 0;
    this.staggerFor = 0;
    this.down = false;
    this.bobPhase = Math.random() * 6.28;
    this.scores = { hits: 0, points: 0 };
    this._ppos = null;
    this.opponent = null;
    this.amo = null;         // AMOFighter when AMO mode is on
    this.amoRestore = null;  // (unused now — torque model has no gains to restore)
    this.cmd = { vx: 0, vy: 0 };
    this.poseT = 0;
    this.strafeDir = 1;
    this.traj = null;        // set to [] by the trainer to record tactic decisions
    this._obsPrev = { dist: null, t: null, fistL: null, fistR: null };
    this.tracking = null;    // TrackingFighter when clip-tracking mode is on
    this.trackingClips = null; // {guard,jab,cross,hook} -> TrackingNetwork
    this.clip = null;        // currently playing clip
    this.desiredClip = null; // clip the tactics layer wants next
    this.clipSwaps = 0;
  }
}

// ---- controller -------------------------------------------------------------
export class BoxingController {
  constructor(model, data, { mujoco, ids, assist = true, aggression = 1 }) {
    this.model = model;
    this.data = data;
    this.mujoco = mujoco;
    this.ids = ids;
    this.assist = assist;
    this.aggression = aggression;
    this.fighters = {};
    for (const side of ['A', 'B']) {
      const b = side === 'A' ? ids.pelvisA : ids.pelvisB;
      this.fighters[side] = new Fighter(side, [data.xpos[3 * b], data.xpos[3 * b + 1]]);
    }
    this.fighters.A.opponent = this.fighters.B;
    this.fighters.B.opponent = this.fighters.A;

    // actuator ids in POLICY_JOINTS order, per side
    this.actIdx = {};
    for (const side of ['A', 'B']) {
      const arr = new Int32Array(23);
      for (let i = 0; i < 23; i++) {
        const aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, side + '_' + POLICY_JOINTS[i]);
        if (aid < 0) throw new Error('actuator not found: ' + side + '_' + POLICY_JOINTS[i]);
        arr[i] = aid;
      }
      this.actIdx[side] = arr;
    }
    this.freeDof = {};
    for (const side of ['A', 'B']) {
      const j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, side + '_pelvis');
      this.freeDof[side] = model.jnt_dofadr[j];
    }

    const gid = n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, n);
    this.fistGeoms = {
      A: new Set([gid('A_left_fist'), gid('A_right_fist')]),
      B: new Set([gid('B_left_fist'), gid('B_right_fist')]),
    };
    this.headGeoms = { A: new Set([gid('A_head_col')]), B: new Set([gid('B_head_col')]) };
    this.vulnerableGeoms = {
      A: new Set([gid('A_head_col'), gid('A_torso_col')]),
      B: new Set([gid('B_head_col'), gid('B_torso_col')]),
    };
    this.hitCooldown = { A: 0, B: 0 };
    this.damage = { A: 0, B: 0 };
    this.knockback = { A: null, B: null };
    this.events = [];
    this._pendingHits = 0;
    this._pendingBlocks = 0;
    this.koState = null;
    this.time = 0;
    this._lastSimTime = -1;
    this._divergences = 0;
    this._farT = 0;   // 追踪模式僵局计时：双机距离过远且无法拉近时重置回合
  }

  // ---- AMO mode ---------------------------------------------------------------
  // net: shared AMONetwork instance (built from vendor/policy/amo.bin + meta)
  setAMO(side, net) {
    this.clearAMO(side);
    const f = this.fighters[side];
    f.amo = new AMOFighter(this.mujoco, this.model, this.data, side, net);
    f.amo.reset();
    f.amoName = 'amo';
    return f.amo;
  }

  clearAMO(side) {
    const f = this.fighters[side];
    f.amo = null;
    f.amoName = null;
  }

  // ---- tracking mode (stage 4): per-side clip-policy library -----------------
  // nets: { guard, jab, cross, hook } -> TrackingNetwork instances. The fighter
  // plays its current clip; tactic decisions only set the DESIRED clip, and the
  // actual swap happens when the running clip reaches its end — every clip is
  // bookended with the guard stance, so the switch lands on a stable pose.
  setTracking(side, nets) {
    this.clearTracking(side);
    const f = this.fighters[side];
    f.trackingClips = nets;
    // B faces yaw π; the clips are baked at yaw 0 — mirror the reference
    // heading for B (the policy is yaw-invariant, see TrackingFighter)
    f.tracking = new TrackingFighter(this.mujoco, this.model, this.data, side, nets.guard,
      side === 'B' ? { yawOffset: Math.PI } : {});
    f.tracking.reset(0);
    f.clip = 'guard';
    f.desiredClip = 'guard';
    f.amoName = 'tracking';
    return f.tracking;
  }

  clearTracking(side) {
    const f = this.fighters[side];
    f.tracking = null;
    f.trackingClips = null;
    f.clip = null;
    f.desiredClip = null;
    if (f.amoName === 'tracking') f.amoName = null;
  }

  // ---- tactics policy (see src/tactics.mjs) ----------------------------------
  // When set, the idle-decision branch consults the learned policy instead of
  // the hardcoded distance thresholds + weighted-random combo pick.
  setTactics(policy, { temp = 0.8 } = {}) {
    this.tactics = policy;
    this.tacticsTemp = temp;
    policy.tacticsTemp = temp;
  }

  clearTactics() { this.tactics = null; }

  // 13-dim normalized observation, self-centered & mirrored (policy can play
  // either side). See OBS_DIM in tactics.mjs for the layout.
  buildTacticsObs(side) {
    const d = this.data;
    const f = this.fighters[side], op = f.opponent;
    const me = this.pelvisPos(side), om = this.pelvisPos(op.side);
    const dist = Math.hypot(om[0] - me[0], om[1] - me[1]);
    const lat = (om[1] - me[1]) * f.forward / 0.8; // >0 = opponent to body-frame left
    const P = d.site_xpos;
    const S = id => [P[3 * id], P[3 * id + 1], P[3 * id + 2]];
    const myHead = S(side === 'A' ? this.ids.headA : this.ids.headB);
    const opFistL = S(op.side === 'A' ? this.ids.fistLA : this.ids.fistLB);
    const opFistR = S(op.side === 'A' ? this.ids.fistRA : this.ids.fistRB);
    const myFistL = S(side === 'A' ? this.ids.fistLA : this.ids.fistLB);
    const myFistR = S(side === 'A' ? this.ids.fistRA : this.ids.fistRB);
    const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const threatL = d3(myHead, opFistL), threatR = d3(myHead, opFistR);
    const prev = f._obsPrev;
    const dtObs = prev.t === null ? 0 : this.time - prev.t;
    let fistSpeed = 0, closing = 0;
    if (prev.t !== null && dtObs > 1e-3 && dtObs < 1.5) {
      if (prev.fistL) fistSpeed = Math.max(d3(opFistL, prev.fistL), d3(opFistR, prev.fistR)) / dtObs;
      if (prev.dist !== null) closing = (prev.dist - dist) / dtObs;
    }
    f._obsPrev = { dist, t: this.time, fistL: opFistL, fistR: opFistR };
    const ext = 0.5 * (d3(myHead, myFistL) + d3(myHead, myFistR));
    const obs = new Float32Array(OBS_DIM);
    obs[0] = 1;
    obs[1] = Math.min(dist, 1.5) / 1.5;
    obs[2] = Math.max(-1.5, Math.min(1.5, closing)) / 1.5;
    obs[3] = Math.max(-2, Math.min(2, lat)) / 2;
    obs[4] = Math.min(threatL, 1.2) / 1.2;
    obs[5] = Math.min(threatR, 1.2) / 1.2;
    obs[6] = Math.min(fistSpeed, 3) / 3;
    obs[7] = (op.state === 'windup' || op.state === 'strike') ? 1 : 0;
    obs[8] = op.staggerFor > 0 ? 1 : 0;
    obs[9] = f.staggerFor > 0 ? 1 : 0;
    obs[10] = this.damage[side] / 7;
    obs[11] = this.damage[op.side] / 7;
    obs[12] = Math.min(ext, 0.9) / 0.9;
    return obs;
  }

  decideTactics(f, dist) {
    const pol = this.tactics[f.side] ?? this.tactics; // per-side policies OR shared
    if (!pol || typeof pol.act !== 'function') return false; // this side: fallback
    const obs = this.buildTacticsObs(f.side);
    const { action } = pol.act(obs, { sample: true, temp: pol.tacticsTemp ?? this.tacticsTemp });
    if (f.traj) f.traj.push({ t: this.time, obs: Array.from(obs), action });
    const name = TACTIC_ACTIONS[action];
    // tracking mode: the decision only queues the next clip — the motion layer
    // executes it at the current clip's guard boundary. A queued/playing punch
    // is never overwritten by later footwork decisions (queue lock), otherwise
    // the rapid non-attack decisions starve punches out before the boundary.
    if (f.tracking) {
      if (f.clip !== 'guard' || (f.desiredClip && f.desiredClip !== 'guard')) return true;
      f.desiredClip = TACTIC_CLIP[name] ?? 'guard';
      return true;
    }
    switch (name) {
      case 'approach':
        f.state = 'approach'; f.stateT = 0; f.phaseDur = 1.4; break;
      case 'retreat':
        f.state = 'retreat'; f.stateT = 0; f.phaseDur = 0.5; break;
      case 'strafe_left':
        f.state = 'strafe'; f.strafeDir = 1; f.stateT = 0; f.phaseDur = 0.6; break;
      case 'strafe_right':
        f.state = 'strafe'; f.strafeDir = -1; f.stateT = 0; f.phaseDur = 0.6; break;
      case 'wait':
        f.stateT = 0; f.idleFor = 0.25 + Math.random() * 0.35; break; // hold guard, re-decide soon
      default: { // attack family -> concrete combo (side randomized for variety)
        const opts = ACTION_COMBOS[name] ?? [0];
        f.combo = COMBOS[opts[Math.floor(Math.random() * opts.length)]];
        f.seqIdx = 0; f.phaseT = 0; f.state = 'windup'; f.phaseDur = 0.08;
      }
    }
    return true;
  }

  writeTorqueScripted(f) {
    // scripted mode: PD on pose targets (name-keyed goal -> 23-vector)
    const goal = f._goal ?? SRESOLVED.guard;
    const d = this.data;
    for (let i = 0; i < 23; i++) {
      const jn = POLICY_JOINTS[i].replace(/_joint$/, '');
      const target = goal[jn] ?? DEFAULT_POSE[i];
      const a = this.actIdx[f.side][i];
      const q = d.qpos[this.model.jnt_qposadr[this._jid(f.side, POLICY_JOINTS[i])]];
      const dq = d.qvel[this.model.jnt_dofadr[this._jid(f.side, POLICY_JOINTS[i])]];
      let t = (target - q) * KP[i] - dq * KD[i];
      d.ctrl[a] = clamp(t, -TORQUE_LIM[i], TORQUE_LIM[i]);
    }
  }
  _jidCache = {};
  _jid(side, jn) {
    const key = side + jn;
    if (this._jidCache[key] === undefined) {
      this._jidCache[key] = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_JOINT.value, key);
    }
    return this._jidCache[key];
  }

  writeLimp(side) {
    for (let i = 0; i < 23; i++) this.data.ctrl[this.actIdx[side][i]] = 0;
  }

  pelvisPos(side) {
    const b = side === 'A' ? this.ids.pelvisA : this.ids.pelvisB;
    return [this.data.xpos[3 * b], this.data.xpos[3 * b + 1], this.data.xpos[3 * b + 2]];
  }

  update(dt) {
    const d = this.data;
    dt = Number.isFinite(dt) ? dt : (this.model.opt?.timestep ?? 0.002);
    if (this._lastSimTime >= 0 && d.time < this._lastSimTime - 1e-9) {
      this._divergences++;
      for (const side of ['A', 'B']) {
        const f = this.fighters[side];
        f.state = 'idle'; f.stateT = 0; f.staggerFor = 0; f.combo = null;
        f._ppos = null; f.anchor = [...f.anchor0];
        if (f.amo) f.amo.reset();
        if (f.tracking) { f.tracking.reset(0); f.clip = 'guard'; f.desiredClip = 'guard'; }
      }
      this.knockback = { A: null, B: null };
      this.koState = null;
    }
    this._lastSimTime = d.time;
    this.time += dt;

    if (this.koState) {
      // limp both robots while the KO settle plays out, then reset the round
      this.writeLimp('A'); this.writeLimp('B');
      for (const side of ['A', 'B']) {
        const b = side === 'A' ? this.ids.pelvisA : this.ids.pelvisB;
        for (let k = 0; k < 6; k++) d.xfrc_applied[6 * b + k] = 0;
      }
      if (this.time - this.koState.t > 2.2) this.resetRound();
      return;
    }

    // ---------------- behaviour ----------------
    for (const side of ['A', 'B']) {
      const f = this.fighters[side];
      if (f.down) continue;

      // distance between pelvis roots
      const me = this.pelvisPos(side), op = this.pelvisPos(f.opponent.side);
      const dist = Math.hypot(op[0] - me[0], op[1] - me[1]);

      f.stateT += dt;
      let vx = 0, vy = 0;
      const punch = { arm: null, blend: 5, torsoYaw: 0 };

      if (f.state === 'idle') {
        // subtle sway while guarding
        vy = 0.05 * Math.sin(this.time * 1.7 + f.bobPhase);
        if (f.stateT >= f.idleFor) {
          if (this.tactics && this.decideTactics(f, dist)) {
            // learned tactic handled the decision
          } else if (dist > 0.85) { f.state = 'approach'; f.stateT = 0; f.phaseDur = 1.4; }
          else if (dist < 0.42) { f.state = 'retreat'; f.stateT = 0; f.phaseDur = 0.5; }
          else {
            const pick = weightedPick(COMBO_POOL, Math.random);
            const opts = POOL_MAP[pick];
            f.combo = COMBOS[opts[Math.floor(Math.random() * opts.length)]];
            f.seqIdx = 0; f.phaseT = 0; f.state = 'windup'; f.phaseDur = 0.08;
          }
        }
      } else if (f.state === 'approach') {
        vx = 0.35 * f.forward;
        if (dist < 0.62 || f.stateT >= f.phaseDur) { f.state = 'idle'; f.stateT = 0; f.idleFor = (0.25 + Math.random() * 0.5) / this.aggression; }
      } else if (f.state === 'retreat') {
        vx = -0.3 * f.forward;
        if (f.stateT >= f.phaseDur) { f.state = 'idle'; f.stateT = 0; f.idleFor = (0.2 + Math.random() * 0.4) / this.aggression; }
      } else if (f.state === 'strafe') {
        // lateral footwork (body-frame vy); the AMO policy executes it, the
        // scripted fallback only sways in place — training mostly ignores it
        vy = 0.3 * (f.strafeDir || 1);
        if (f.stateT >= f.phaseDur) { f.state = 'idle'; f.stateT = 0; f.idleFor = 0.2 + Math.random() * 0.3; }
      } else if (f.state === 'windup') {
        f.phaseT += dt;
        if (f.phaseT >= f.phaseDur) { f.state = 'strike'; f.phaseT = 0; f.phaseDur = 0.22; }
      } else if (f.state === 'strike') {
        f.phaseT += dt;
        const poseName = f.combo.seq[f.seqIdx];
        punch.arm = ARM_POSES[poseName];
        punch.blend = 4;
        punch.torsoYaw = poseName.includes('left') ? -0.45 : 0.45;
        vx = 0.12 * f.forward; // slight lunge
        if (f.phaseT >= f.phaseDur) {
          f.seqIdx += 1; f.phaseT = 0;
          if (f.seqIdx < f.combo.seq.length) { f.state = 'windup'; f.phaseDur = 0.06; }
          else {
            punch.arm = ARM_GUARD; punch.blend = 6; punch.torsoYaw = 0;
            f.state = 'recover'; f.phaseDur = f.combo.recover;
          }
        }
      } else if (f.state === 'recover') {
        f.phaseT += dt;
        if (f.phaseT >= f.phaseDur) {
          f.state = 'idle'; f.stateT = 0;
          f.idleFor = (0.4 + Math.random() * 0.8) / this.aggression;
        }
      }
      if (f.staggerFor > 0) { f.staggerFor -= dt; vx *= 0.2; vy = 0; }

      // ---- drive the robot ----
      if (f.tracking) {
        // stage-4 motion layer: play the current clip; swap at its guard boundary.
        // All clips bookend with the SAME guard stance, so the swap hands the
        // previous fighter's last action / PD target straight to the new one
        // (cross-fade-lite): zeroing them instead would snap the pose for one
        // tick right as the punch clips head into their marginal deep stance.
        if (f.tracking.clipDone) {
          const prev = f.tracking;
          if (f.clip !== 'guard' && f.desiredClip === f.clip) f.desiredClip = 'guard';
          let want = f.desiredClip && f.trackingClips[f.desiredClip] ? f.desiredClip : 'guard';
          if (want !== 'guard') {
            const r = PUNCH_RANGE[want];
            if (!(dist >= r[0] && dist <= r[1])) want = 'guard';  // 距离不合适：本轮不出拳
            // （允许同时出拳：互相前倾才够得着对方的头——见 PUNCH_RANGE 标定）
          }
          if (want !== f.clip || want !== 'guard') {
            const nf = new TrackingFighter(this.mujoco, this.model, this.data, side,
              f.trackingClips[want], side === 'B' ? { yawOffset: Math.PI } : {});
            nf.reset(0);
            nf.lastAction.set(prev.lastAction);
            nf.pdTarget.set(prev.pdTarget);
            f.tracking = nf;
            f.clip = want;
            f.clipSwaps++;
            f.desiredClip = 'guard';  // 队列已消费（无论打没打）
          } else {
            f.tracking.reset(0);
            f.tracking.lastAction.set(prev.lastAction);
            f.tracking.pdTarget.set(prev.pdTarget);
          }
        }
        f.tracking.physicsStep(4);
      } else if (f.amo) {
        f.amo.setCommand(
          vx * f.forward, vy, f.faceYaw,
          0,                  // height offset
          punch.torsoYaw, 0, 0,
        );
        if (punch.arm) f.amo.setArmTarget(punch.arm, punch.blend);
        f.amo.physicsStep();
      } else {
        // scripted fallback state machine (blended absolute pose targets)
        f.poseT += dt;
        let goal;
        if (f.staggerFor > 0) goal = SRESOLVED.stagger;
        else {
          const t = this.time * 2.2 + f.bobPhase;
          goal = addOffsets(SRESOLVED.guard, {
            hip_pitch: 0.04 + 0.04 * Math.sin(t),
            knee: 0.06 + 0.05 * Math.sin(t),
            waist_pitch: 0.02 + 0.04 * Math.sin(t),
            waist_yaw: 0.12 * Math.sin(t * 0.7),
            left_shoulder_pitch: 0.04 * Math.sin(t + 1),
            right_shoulder_pitch: 0.04 * Math.sin(t + 1.2),
          }, 1);
          if (f.state === 'strike' && f.combo) {
            const poseName = f.combo.seq[Math.min(f.seqIdx, f.combo.seq.length - 1)]
              .replace('cross_', 'jab_').replace('hook_', 'jab_');
            const tt = Math.min(1, f.phaseT / f.phaseDur);
            goal = lerpMap(goal, SRESOLVED[poseName] ?? SRESOLVED.guard, (f.phaseDur > 0.15 ? snap : smooth)(tt));
          }
        }
        f._goal = goal;
        this.writeTorqueScripted(f);
      }
    }

    // ---------------- balance assist (scripted only) + knockback -------------
    for (const side of ['A', 'B']) {
      const f = this.fighters[side];
      const b = side === 'A' ? this.ids.pelvisA : this.ids.pelvisB;
      const px = d.xpos[3 * b], py = d.xpos[3 * b + 1], pz = d.xpos[3 * b + 2];
      let fx = 0, fy = 0, fz = 0, tx = 0, ty = 0, tz = 0;
      if (this.assist && !f.amo && !f.down) {
        if (!f._ppos) f._ppos = [px, py, pz];
        const vxx = (px - f._ppos[0]) / dt, vyy = (py - f._ppos[1]) / dt, vzz = (pz - f._ppos[2]) / dt;
        f._ppos = [px, py, pz];
        if (f.tracking) {
          // tracking mode: no XY anchor (the clip owns planar motion), only a
          // soft vertical seat toward the reference pelvis height. The punch
          // clips' deep stance (0.487m) is the policies' least stable window —
          // this keeps a wobbling crouch from becoming a self-KO. ~30N typical,
          // capped well below body weight: tracking dynamics stay dominant.
          const refZ = f.tracking.net.refAt(f.tracking.timeStep).body_pos_w[2];
          fz = clamp(120 * (refZ - pz) - 25 * vzz, 0, 110);
          // 躯干扶正：深蹲段的侧向倾倒（垂直托力救不了）——小力矩扶正骨盆，
          // 幅度远小于脚本模式的吊架
          const R = d.xmat, o = 9 * b;
          const w = this.freeDof[side] + 3;
          const wx = d.qvel[w], wy = d.qvel[w + 1], wz = d.qvel[w + 2];
          tx = clamp(100 * R[o + 5] - 8 * wx, -55, 55);
          ty = clamp(-100 * R[o + 2] - 8 * wy, -55, 55);
          tz = clamp(-8 * wz, -15, 15);
          // 缠斗分离：punch 片段只拉近距离（cross 前冲 0.8m），守卫片段原地——
          // 没有东西能拉开距离，贴身后必然缠倒。距离 <0.55m 时给双方一个
          // 温和的分离力，等效裁判分开缠斗，让下一轮出拳在有效距离发起。
          const op = this.pelvisPos(f.opponent.side);
          const dx = px - op[0], dy = py - op[1];
          const dd = Math.hypot(dx, dy) || 1e-6;
          if (dd < 0.55 && !f.opponent.down) {
            const s = clamp(90 * (0.55 - dd), 0, 80);
            fx += s * dx / dd; fy += s * dy / dd;
          }
        } else {
        const hurt = f.staggerFor > 0;
        const fMax = hurt ? 100 : 70;
        fx = clamp(130 * (f.anchor[0] - px) - 70 * vxx, -fMax, fMax);
        fy = clamp(130 * (f.anchor[1] - py) - 70 * vyy, -fMax, fMax);
        fz = clamp(60 * (0.78 - pz) - 40 * vzz, -80, hurt ? 180 : 150);
        const R = d.xmat, o = 9 * b;
        const zbx = R[o + 2], zby = R[o + 5];
        const w = this.freeDof[side] + 3;
        const wx = d.qvel[w], wy = d.qvel[w + 1], wz = d.qvel[w + 2];
        const tiltK = hurt ? 130 : 100, tMax = hurt ? 90 : 60;
        tx = clamp(tiltK * zby - 8 * wx, -tMax, tMax);
        ty = clamp(-tiltK * zbx - 8 * wy, -tMax, tMax);
        tz = clamp(-8 * wz, -15, 15);
      }
      }
      const kb = this.knockback[side];
      if (kb) {
        if (this.time > kb.until) this.knockback[side] = null;
        else { fx += kb.f[0]; fy += kb.f[1]; }
      }
      const xfo = 6 * b;
      if (!Number.isFinite(fx + fy + fz + tx + ty + tz)) continue;
      d.xfrc_applied[xfo] = fx; d.xfrc_applied[xfo + 1] = fy; d.xfrc_applied[xfo + 2] = fz;
      d.xfrc_applied[xfo + 3] = tx; d.xfrc_applied[xfo + 4] = ty; d.xfrc_applied[xfo + 5] = tz;
    }

      if (this.fighters.A.tracking || this.fighters.B.tracking) {
        // 僵局检测：接触漂移把双机推远后没有任何片段能拉近距离——
        // 距离 >1.9m 持续 3s 就重开回合（回到标准 1.0m 站位）
        const far = Math.hypot(d.xpos[3 * this.ids.pelvisA] - d.xpos[3 * this.ids.pelvisB],
          d.xpos[3 * this.ids.pelvisA + 1] - d.xpos[3 * this.ids.pelvisB + 1]);
        if (far > 1.9) {
          this._farT += dt;
          if (this._farT > 3) {
            this._farT = 0;
            this.events.push({ type: 'round', t: this.time, stalemate: true });
            this.resetRound();
            return;
          }
        } else {
          this._farT = Math.max(0, this._farT - dt);
        }
      }

      // ---------------- hit detection + fall/KO ----------------
      this.detectHits();
    this.checkFall();
    for (const side of ['A', 'B']) {
      this.damage[side] = Math.max(0, this.damage[side] - 0.15 * dt);
      if (this.damage[side] >= 7) {
        const f = this.fighters[side];
        f.down = true;
        const winner = side === 'A' ? 'B' : 'A';
        this.koState = { side, t: this.time, winner, byDamage: true };
        this.events.push({ type: 'ko', down: side, winner, t: this.time });
        this.damage[side] = 0;
        return;
      }
    }
  }

  fistSiteId(side, hand) {
    return this.ids[side === 'A' ? (hand === 'left' ? 'fistLA' : 'fistRA')
                              : (hand === 'left' ? 'fistLB' : 'fistRB')];
  }
  headId(side) { return side === 'A' ? this.ids.headA : this.ids.headB; }

  siteDist(sa, sb) {
    const p = this.data.site_xpos;
    return Math.hypot(p[3 * sa] - p[3 * sb], p[3 * sa + 1] - p[3 * sb + 1], p[3 * sa + 2] - p[3 * sb + 2]);
  }

  detectHits() {
    const ncon = this.data.ncon;
    if (ncon === 0) return;
    const con = this.data.contact; // embind copies the vector — must delete() it
    try {
      for (let i = 0; i < ncon; i++) {
        const c = con.get(i);
        const pair = [c.geom1, c.geom2];
        c.delete();
        for (const side of ['A', 'B']) {
          const opp = side === 'A' ? 'B' : 'A';
          const mine = this.fistGeoms[side];
          if (!pair.some(g => mine.has(g))) continue;
          const other = mine.has(pair[0]) ? pair[1] : pair[0];
          if (this.vulnerableGeoms[opp].has(other)) { this.registerHit(side, opp, other); break; }
          if (this.fistGeoms[opp].has(other)) {
            if (this.hitCooldown[side] <= this.time && this.hitCooldown[opp] <= this.time) {
              this.events.push({ type: 'block', t: this.time });
              this._pendingBlocks++;
              this.hitCooldown[side] = this.time + 0.3;
            }
            break;
          }
        }
      }
    } finally {
      con.delete();
    }
  }

  registerHit(attacker, victim, vulGeom) {
    if (this.hitCooldown[attacker] > this.time) return;
    this.hitCooldown[attacker] = this.time + 0.45;
    const points = this.headGeoms[victim].has(vulGeom) ? 2 : 1;
    const kind = points === 2 ? 'head' : 'body';
    const f = this.fighters[attacker];
    f.scores.hits += 1;
    f.scores.points += points;
    this._pendingHits++;
    this.events.push({ type: 'hit', attacker, victim, points, kind, t: this.time });
    const av = this.fighters[victim];
    this.knockback[victim] = { f: [-av.forward * 40, 0, 0], until: this.time + 0.06 };
    av.staggerFor = 0.35 + Math.random() * 0.2 + (kind === 'head' ? 0.2 : 0);
    this.damage[victim] += kind === 'head' ? 1.5 : 1.0;
  }

  checkFall() {
    if (this.koState) return;
    for (const side of ['A', 'B']) {
      const b = side === 'A' ? this.ids.pelvisA : this.ids.pelvisB;
      // tracking mode: the punch clips' boxing stance dips to pelvis 0.487m —
      // a wobbling crouch must not read as a fall; 0.40 still catches real ones
      const zLine = this.fighters[side].tracking ? 0.40 : 0.45;
      if (this.data.xpos[3 * b + 2] < zLine) {
        this.fighters[side].down = true;
        const winner = side === 'A' ? 'B' : 'A';
        this.koState = { side, t: this.time, winner };
        this.events.push({ type: 'ko', down: side, winner, t: this.time });
        return;
      }
    }
  }

  resetRound() {
    this.mujoco.mj_resetDataKeyframe(this.model, this.data, 0);
    this.mujoco.mj_forward(this.model, this.data);
    for (const side of ['A', 'B']) {
      const f = this.fighters[side];
      f.down = false; f.state = 'idle'; f.stateT = 0;
      f.idleFor = 0.6 + Math.random() * 0.5;
      f.staggerFor = 0; f.combo = null;
      f.anchor = [...f.anchor0];
      f._ppos = null;
      f._goal = null;
      f._obsPrev = { dist: null, t: null, fistL: null, fistR: null };
      if (f.amo) f.amo.reset();
      if (f.tracking) {
        // new round: back to the guard clip at frame 0
        f.tracking = new TrackingFighter(this.mujoco, this.model, this.data, side, f.trackingClips.guard,
          side === 'B' ? { yawOffset: Math.PI } : {});
        f.tracking.reset(0);
        f.clip = 'guard';
        f.desiredClip = 'guard';
      }
    }
    this.knockback = { A: null, B: null };
    this.koState = null;
    this.damage = { A: 0, B: 0 };
    this._lastSimTime = -1;
    this.events.push({ type: 'round', t: this.time });
  }

  takeHitCount() { const n = this._pendingHits; this._pendingHits = 0; return n; }
  takeBlockCount() { const n = this._pendingBlocks; this._pendingBlocks = 0; return n; }
}
