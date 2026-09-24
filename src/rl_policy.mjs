// AMO (Adaptive Motion Optimization, RSS 2025, OpenTeleVision/AMO, Apache-2.0)
// whole-body RL policy runtime for the Unitree G1 — plain JS, no onnxruntime.
//
// The student policy maps a 2474-dim input to 15 leg+waist actions; arm-joint
// targets are tracked through an input adapter and a PD controller. The JS
// reimplementation below matches the torch reference to ~1.5e-7 (see
// .workbuddy/tmp/verify_amo.mjs) and the full sim recipe to the headless
// python regression (test_scene_dual.py / baseline_amo_official.py).
//
// Input layout (2474):
//   [0..2325)    extra history  25 frames x 93 (obs_prop, oldest -> newest)
//   [2325..2341) text_feat_merger( text_enc(last 4 obs_prop frames) )
//   [2341..2434) obs_prop (current, 93)
//   [2434..2451) obs_demo  (17)
//   [2451..2454) obs_priv  (3, zeros)
//   [2454..2474) history_encoder(10 obs_prop frames)  (20)
//
// obs_prop (93):
//   omega*0.25 (3) | rpy[:2] (2) | sin/cos(dyaw) (2) | q23-default (23)
//   | dq23*0.05 (23) | last_action (23) | gait sin (2) | adapter out (15)
//
// Control: 50 Hz policy on a 500 Hz sim (timestep 0.002, decimation 10).
//   pd_target = [action*0.25 + default[0:15], arm_target(8)]
//   torque    = clip((pd_target - q)*kp - dq*kd, +-torque_lim)   every sim step.
// Torque actuators come from the official AMO g1.xml (23 motors, wrists fused).

// Policy dof order = AMO training order (legs L/R, waist, arms L/R).
export const POLICY_JOINTS = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
  'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
  'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
];

// AMO default (guard-ish) pose, POLICY_JOINTS order.
export const DEFAULT_POSE = new Float32Array([
  -0.1, 0, 0, 0.3, -0.2, 0,
  -0.1, 0, 0, 0.3, -0.2, 0,
  0, 0, 0,
  0.5, 0, 0.2, 0.3,
  0.5, 0, -0.2, 0.3,
]);

// PD gains + torque limits, POLICY_JOINTS order (AMO deployment recipe).
export const KP = new Float32Array([
  150, 150, 150, 300, 80, 20, 150, 150, 150, 300, 80, 20,
  400, 400, 400,
  80, 80, 40, 60, 80, 80, 40, 60,
]);
export const KD = new Float32Array([
  2, 2, 2, 4, 2, 1, 2, 2, 2, 4, 2, 1,
  15, 15, 15,
  2, 2, 1, 1, 2, 2, 1, 1,
]);
export const TORQUE_LIM = new Float32Array([
  88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50,
  88, 50, 50,
  25, 25, 25, 25, 25, 25, 25, 25,
]);

export const POLICY_DT = 0.02;      // 50 Hz
export const DECIMATION = 10;       // sim 500 Hz -> policy 50 Hz
const ACTION_SCALE = 0.25;
const ANG_VEL_SCALE = 0.25;
const DOF_VEL_SCALE = 0.05;
const GAIT_FREQ = 1.3;

// ---------------------------------------------------------------------------
// tiny matmul kernels on preallocated scratch (weights are Float32 views into
// the shared amo.bin buffer; layout: W row-major [out,in], then bias [out])
// ---------------------------------------------------------------------------
function linear(x, L, out) {
  const { W, b, out: nOut, in: nIn } = L;
  for (let j = 0; j < nOut; j++) {
    let s = b[j];
    const row = j * nIn;
    for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
    out[j] = s;
  }
  return out;
}
function elu(x, out) {
  for (let i = 0; i < x.length; i++) { const v = x[i]; out[i] = v > 0 ? v : Math.expm1(v); }
  return out;
}
function lrelu(x, out) {
  for (let i = 0; i < x.length; i++) { const v = x[i]; out[i] = v > 0 ? v : 0.01 * v; }
  return out;
}
function bn(x, key, out) {
  const b = this.bn[key];
  const { gamma, beta, mean, var: variance, eps } = b;
  for (let i = 0; i < x.length; i++) out[i] = gamma[i] * (x[i] - mean[i]) / Math.sqrt(variance[i] + eps) + beta[i];
  return out;
}

// Parsed weights shared by every robot (the net itself is stateless).
export class AMONetwork {
  constructor(buffer, meta) {
    this.meta = meta;
    const f32 = new Float32Array(buffer);
    const off = meta.offsets;   // NOTE: byte offsets from the extractor
    const layer = (name, extra = 0) => {
      const l = meta.layers.find(x => x.name === name);
      const fo = off['W:' + name] / 4, bo = off['b:' + name] / 4;
      return {
        W: f32.subarray(fo, fo + l.out * l.in + extra),
        b: f32.subarray(bo, bo + l.out),
        out: l.out, in: l.in,
      };
    };
    this.s0 = layer('student.0'); this.s2 = layer('student.2');
    this.s4 = layer('student.4'); this.s6 = layer('student.6');
    this.tE0 = layer('text_enc.0'); this.tE2 = layer('text_enc.2'); this.tM = layer('merger.0');
    this.hEnc = layer('hist.enc0');
    this.hC0 = layer('hist.conv0', 20 * 30 * 3);   // conv W is [out,in,k] = 20*30*4
    this.hC2 = layer('hist.conv2', 10 * 20 * 1);   // [10,20,2]
    this.hLin = layer('hist.lin');
    this.ad0 = layer('adapter.0'); this.ad3 = layer('adapter.3');
    this.ad6 = layer('adapter.6'); this.ad9 = layer('adapter.9');
    this.bn = meta.bn;
    this.stats = meta.stats;
    this.c0shape = meta.hist_conv0_shape;  // [20,30,4]
    this.c2shape = meta.hist_conv2_shape;  // [10,20,2]

    // scratch (per network; robots run sequentially inside one JS thread tick)
    this._te = new Float32Array(128);
    this._te2 = new Float32Array(16);
    this._tf4 = new Float32Array(64);
    this._tm = new Float32Array(16);
    this._heF = new Float32Array(10 * 30);
    this._heS = new Float32Array(93);
    this._heP = new Float32Array(30 * 10);
    this._heC0 = new Float32Array(20 * 4);
    this._heC2 = new Float32Array(10 * 3);
    this._heOut = new Float32Array(20);
    this._in2474 = new Float32Array(2474);
    this._h1 = new Float32Array(1024);
    this._h2 = new Float32Array(1024);
    this._h3 = new Float32Array(512);
    this._act = new Float32Array(15);
    this._ad1 = new Float32Array(512);
    this._ad2 = new Float32Array(512);
    this._ad3 = new Float32Array(256);
    this._adIn = new Float32Array(12);
    this._adOut = new Float32Array(15);
    this._single93 = new Float32Array(93);
  }

  adapter(qArm8, dh, ty, tp, tr) {
    const s = this.stats;
    const x = this._adIn;
    x[0] = 0.75 + dh; x[1] = ty; x[2] = tp; x[3] = tr;
    for (let i = 0; i < 8; i++) x[4 + i] = qArm8[i];
    for (let i = 0; i < 12; i++) x[i] = (x[i] - s.input_mean[i]) / (s.input_std[i] + 1e-8);
    let h = linear(x, this.ad0, this._ad1);
    h = lrelu(bn.call(this, h, 'adapter.bn1', this._ad2), this._ad2);
    h = linear(h, this.ad3, this._ad1);
    h = lrelu(bn.call(this, h, 'adapter.bn4', this._ad2), this._ad2);
    h = linear(h, this.ad6, this._ad3);
    h = lrelu(bn.call(this, h, 'adapter.bn7', this._ad2), this._ad2);
    linear(h, this.ad9, this._adOut);
    const o = this._adOut, om = s.output_mean, os = s.output_std;
    for (let i = 0; i < 15; i++) o[i] = o[i] * os[i] + om[i];
    return o;
  }

  textFeat(last4 /* Float32Array 372 */) {
    for (let f = 0; f < 4; f++) {
      this._single93.set(last4.subarray(f * 93, f * 93 + 93));
      let h = elu(linear(this._single93, this.tE0, this._te), this._te);
      h = elu(linear(h, this.tE2, this._te2), this._te2);
      this._tf4.set(h, f * 16);
    }
    return elu(linear(this._tf4, this.tM, this._tm), this._tm);
  }

  historyEncoder(hist930) {
    return this._historyEncoderInner(hist930);
  }

  _historyEncoderInner(hist930) {
    const frames = this._heF; // 300 floats: frame f at f*30
    for (let f = 0; f < 10; f++) {
      this._single93.set(hist930.subarray(f * 93, f * 93 + 93));
      const e = elu(linear(this._single93, this.hEnc, this._heP.subarray(0, 30)), this._heP.subarray(0, 30));
      frames.set(e, f * 30);
    }
    // permute to [channel, time]: channel c, frame f -> perm[c*10+f]
    const perm = this._heP;
    for (let c = 0; c < 30; c++) for (let f = 0; f < 10; f++) perm[c * 10 + f] = frames[f * 30 + c];
    const r0 = this._conv1d(perm, 30, 10, this.hC0, this.c0shape, 2, 4, this._heC0);
    elu(r0.y, r0.y);
    const r1 = this._conv1d(r0.y, 20, r0.T_out, this.hC2, this.c2shape, 1, 2, this._heC2);
    elu(r1.y, r1.y);
    // r1.y is [10*3 = 30] channel-major; flatten matches torch .flatten()
    const lin = linear(r1.y, this.hLin, this._heOut);
    return elu(lin, this._heOut);
  }

  _conv1d(x, chIn, T, L, Wshape, stride, k, scratch) {
    const outCh = Wshape[0];
    const T_out = Math.floor((T - k) / stride) + 1;
    const y = new Float32Array(outCh * T_out);
    for (let o = 0; o < outCh; o++) {
      for (let t = 0; t < T_out; t++) {
        let s = L.b[o];
        for (let c = 0; c < chIn; c++) {
          const wRow = o * chIn * k + c * k;
          const xRow = c * T + t * stride;
          for (let kk = 0; kk < k; kk++) s += L.W[wRow + kk] * x[xRow + kk];
        }
        y[o * T_out + t] = s;
      }
    }
    return { y, T_out };
  }

  // obs1043: [obs_prop(93) | obs_demo(17) | obs_priv(3) | proprio hist(930)]
  // extra2325: extra history, oldest -> newest
  infer(obs1043, extra2325) {
    const tm = this.textFeat(obs1043.subarray(obs1043.length - 372));
    const he = this._historyEncoderInner(obs1043.subarray(obs1043.length - 930));
    const input = this._in2474;
    input.set(extra2325, 0);
    input.set(tm, 2325);
    input.set(obs1043.subarray(0, 93), 2341);
    input.set(obs1043.subarray(93, 110), 2434);
    input.set(obs1043.subarray(110, 113), 2451);
    input.set(he, 2454);
    let h = elu(linear(input, this.s0, this._h1), this._h1);
    h = elu(linear(h, this.s2, this._h2), this._h2);
    h = elu(linear(h, this.s4, this._h3), this._h3);
    linear(h, this.s6, this._act);
    return this._act;
  }
}

// ---------------------------------------------------------------------------
// per-robot runner: owns policy histories, commands and the PD write-out
// ---------------------------------------------------------------------------
const N_PROPRIO = 93;

export class AMOFighter {
  constructor(mujoco, model, data, side, net) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.side = side;           // 'A' | 'B'
    this.net = net;             // shared AMONetwork

    const jid = n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, side + '_' + n);
    const aid = n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, side + '_' + n);
    this.qposadr = new Int32Array(23);
    this.dofadr = new Int32Array(23);
    this.actId = new Int32Array(23);
    for (let i = 0; i < 23; i++) {
      const j = jid(POLICY_JOINTS[i]);
      if (j < 0) throw new Error('joint not found: ' + side + POLICY_JOINTS[i]);
      this.qposadr[i] = model.jnt_qposadr[j];
      this.dofadr[i] = model.jnt_dofadr[j];
      const a = aid(POLICY_JOINTS[i]);
      if (a < 0) throw new Error('actuator not found: ' + side + POLICY_JOINTS[i]);
      this.actId[i] = a;
    }
    const pj = jid('pelvis');   // freejoint, named A_pelvis / B_pelvis
    this.freeQpos = model.jnt_qposadr[pj];
    this.freeDof = model.jnt_dofadr[pj];

    this.proprioFrames = [];    // 10 x Float32Array(93), oldest -> newest
    this.extraFrames = [];      // 25 x Float32Array(93), oldest -> newest
    this.lastAction = new Float32Array(23);
    this.gaitCycle = [0.25, 0.25];
    this.inPlace = true;
    // command layout (AMO order): [vx, yaw, vy, dHeight, torsoYaw, torsoPitch, torsoRoll]
    this.cmd = new Float32Array(7);
    this.cmdTarget = new Float32Array(7);
    this.armTarget = DEFAULT_POSE.slice(15);      // 8, desired arm pose
    this.armPrev = DEFAULT_POSE.slice(15);
    this.armBlend = 1;                            // ticks to full blend
    this.pdTarget = DEFAULT_POSE.slice();         // 23, written by policy ticks
    this.stepCount = 0;
    this.healthy = true;
    this.reset();
  }

  reset() {
    this.proprioFrames = [];
    this.extraFrames = [];
    for (let i = 0; i < 10; i++) this.proprioFrames.push(new Float32Array(N_PROPRIO));
    for (let i = 0; i < 25; i++) this.extraFrames.push(new Float32Array(N_PROPRIO));
    this.lastAction.fill(0);
    this.gaitCycle = [0.25, 0.25];
    this.inPlace = true;
    this.cmd.fill(0); this.cmdTarget.fill(0);
    this.armTarget.set(DEFAULT_POSE.subarray(15));
    this.armPrev.set(DEFAULT_POSE.subarray(15));
    this.armBlend = 1;
    this.pdTarget.set(DEFAULT_POSE);
    this.stepCount = 0;
    this.healthy = true;
  }

  // AMO command surface. height: pelvis height offset from 0.75.
  setCommand(vx, vy, yaw, height, torsoYaw, torsoPitch, torsoRoll) {
    const t = this.cmdTarget;
    t[0] = vx; t[1] = yaw; t[2] = vy; t[3] = height;
    t[4] = torsoYaw; t[5] = torsoPitch; t[6] = torsoRoll;
  }

  setArmTarget(arr8, blendTicks = 6) {
    // armPrev carries the running blend state, so successive targets blend
    // continuously (a mid-blend retarget just changes the destination).
    this.armTarget.set(arr8);
    this.armBlend = Math.max(1, blendTicks);
  }

  // call every physics step (500 Hz). Runs the policy every DECIMATION steps,
  // writes PD torques into ctrl every step.
  physicsStep() {
    if (this.stepCount % DECIMATION === 0) this.policyTick();
    this.stepCount++;
    this.writeTorque();
  }

  writeTorque() {
    const d = this.data, pd = this.pdTarget;
    for (let i = 0; i < 23; i++) {
      const q = d.qpos[this.qposadr[i]], dq = d.qvel[this.dofadr[i]];
      let t = (pd[i] - q) * KP[i] - dq * KD[i];
      if (t > TORQUE_LIM[i]) t = TORQUE_LIM[i];
      else if (t < -TORQUE_LIM[i]) t = -TORQUE_LIM[i];
      d.ctrl[this.actId[i]] = t;
    }
  }

  quatToRpy(qw, qx, qy, qz) {
    return [
      Math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)),
      Math.asin(Math.max(-1, Math.min(1, 2 * (qw * qy - qz * qx)))),
      Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
    ];
  }

  policyTick() {
    const d = this.data;
    const q23 = new Float32Array(23), dq23 = new Float32Array(23);
    for (let i = 0; i < 23; i++) {
      q23[i] = d.qpos[this.qposadr[i]];
      dq23[i] = d.qvel[this.dofadr[i]];
    }
    // free-joint angular velocity is body-frame (matches the AMO gyro convention)
    const wx = d.qvel[this.freeDof + 3], wy = d.qvel[this.freeDof + 4], wz = d.qvel[this.freeDof + 5];
    // free-joint qpos = [x,y,z, qw,qx,qy,qz] — the quaternion starts at +3
    const qw = d.qpos[this.freeQpos + 3], qx = d.qpos[this.freeQpos + 4],
          qy = d.qpos[this.freeQpos + 5], qz = d.qpos[this.freeQpos + 6];
    const rpy = this.quatToRpy(qw, qx, qy, qz);

    // smooth commands toward targets (~0.3 s convergence at 50 Hz)
    const cmd = this.cmd, tgt = this.cmdTarget;
    for (let i = 0; i < 7; i++) cmd[i] += (tgt[i] - cmd[i]) * 0.12;

    const inPlace = Math.abs(cmd[0]) < 0.1 && Math.abs(cmd[2]) < 0.1;
    const dyaw = inPlace ? 0
      : (((rpy[2] - cmd[1] + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

    // adapter: arm-conditioning features
    const aOut = this.net.adapter(q23.subarray(15), cmd[3], cmd[4], cmd[5], cmd[6]);

    // ---- build obs_prop (93) ----
    const obsProp = new Float32Array(N_PROPRIO);
    let o = 0;
    obsProp[o++] = wx * ANG_VEL_SCALE; obsProp[o++] = wy * ANG_VEL_SCALE; obsProp[o++] = wz * ANG_VEL_SCALE;
    obsProp[o++] = rpy[0]; obsProp[o++] = rpy[1];
    obsProp[o++] = Math.sin(dyaw); obsProp[o++] = Math.cos(dyaw);
    for (let i = 0; i < 23; i++) obsProp[o++] = q23[i] - DEFAULT_POSE[i];
    for (let i = 0; i < 23; i++) {
      // ankle roll / waist roll+pitch velocities masked as in the deploy recipe
      const masked = (i === 4 || i === 5 || i === 10 || i === 11 || i === 13 || i === 14) ? 0 : dq23[i];
      obsProp[o++] = masked * DOF_VEL_SCALE;
    }
    for (let i = 0; i < 23; i++) obsProp[o++] = this.lastAction[i];
    obsProp[o++] = Math.sin(this.gaitCycle[0] * 2 * Math.PI);
    obsProp[o++] = Math.sin(this.gaitCycle[1] * 2 * Math.PI);
    for (let i = 0; i < 15; i++) obsProp[o++] = aOut[i];

    // ---- build obs_demo (17) ----
    const obsDemo = new Float32Array(17);
    for (let i = 0; i < 8; i++) obsDemo[i] = q23[15 + i];  // current arm joints
    obsDemo[8] = cmd[0];      // vx
    obsDemo[9] = cmd[2];      // vy
    obsDemo[10] = 0;          // (unused yaw-rate slot in the deploy layout)
    obsDemo[11] = cmd[4]; obsDemo[12] = cmd[5]; obsDemo[13] = cmd[6];
    obsDemo[14] = obsDemo[15] = obsDemo[16] = 0.75 + cmd[3];

    // push into histories (oldest -> newest), then assemble the 1043 obs
    this.proprioFrames.push(obsProp);
    if (this.proprioFrames.length > 10) this.proprioFrames.shift();
    this.extraFrames.push(obsProp);
    if (this.extraFrames.length > 25) this.extraFrames.shift();

    const obs = new Float32Array(113 + 930);
    obs.set(obsProp, 0);
    obs.set(obsDemo, 93);
    const hist = obs.subarray(113);
    for (let f = 0; f < 10; f++) hist.set(this.proprioFrames[f], f * 93);
    const extra = new Float32Array(25 * 93);
    for (let f = 0; f < 25; f++) extra.set(this.extraFrames[f], f * 93);

    const raw = this.net.infer(obs, extra);
    let finite = true;
    for (let i = 0; i < 15; i++) if (!Number.isFinite(raw[i])) finite = false;
    this.healthy = finite;
    for (let i = 0; i < 15; i++) raw[i] = Math.max(-40, Math.min(40, raw[i]));

    this.lastAction.set(raw);
    for (let i = 0; i < 8; i++) this.lastAction[15 + i] = (q23[15 + i] - DEFAULT_POSE[15 + i]) / ACTION_SCALE;

    // ---- pd_target = [action*0.25 + default[0:15], blended arm target] ----
    const pd = this.pdTarget;
    for (let i = 0; i < 15; i++) pd[i] = raw[i] * ACTION_SCALE + DEFAULT_POSE[i];
    // blend arms from armPrev toward armTarget over armBlend ticks
    const s = Math.min(1, 1 / this.armBlend);
    for (let i = 0; i < 8; i++) {
      this.armPrev[i] += (this.armTarget[i] - this.armPrev[i]) * s;
      pd[15 + i] = this.armPrev[i];
    }
    if (this.armBlend > 1) this.armBlend--;

    // ---- gait phase advance / snap (deploy recipe) ----
    this.inPlace = inPlace;
    this.gaitCycle[0] = (this.gaitCycle[0] + 0.02 * GAIT_FREQ) % 1;
    this.gaitCycle[1] = (this.gaitCycle[1] + 0.02 * GAIT_FREQ) % 1;
    const near = (v, x) => Math.abs(v - x) < 0.05;
    if (inPlace && (near(this.gaitCycle[0], 0.25) || near(this.gaitCycle[1], 0.25))) {
      this.gaitCycle = [0.25, 0.25];
    } else if (!inPlace && near(this.gaitCycle[0], 0.25) && near(this.gaitCycle[1], 0.25)) {
      this.gaitCycle = [0.25, 0.75];
    }
  }
}
