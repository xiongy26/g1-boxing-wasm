// BeyondMimic / mjlab motion-tracking policy runtime — plain JS, no onnxruntime.
//
// This is the milestone-B runtime for docs/plan-mjlab-gpu-training.md: the
// spinkick_safe.onnx (and, later, the boxing clip policies it validates the
// pipeline for) re-implemented on top of the extracted .bin/.json pair, aligned
// to the ONNX reference to ~1e-6 by tools/test_tracking.mjs.
//
// ONNX graph (mjlab whole_body_tracking export):
//   time_step -> clip(min 0, max motion_len-1) -> Gather motion tables
//   obs -> (obs - mean) / std -> Gemm/Elu x3 -> Gemm -> actions
//
// obs layout comes from the ONNX metadata `observation_names`, e.g. (29-DoF G1):
//   command      [ref_joint_pos(29) | ref_joint_vel(29)]      58
//   motion_anchor_ori_b  (2 columns of the relative rotmat)   6
//   base_ang_vel (body frame)                                 3
//   joint_pos    (q - default)                               29
//   joint_vel                                                29
//   actions      (previous *smoothed* action)                 29
// Deployment recipe (RoboJuDo BeyondmimicPolicy / motion_tracking_controller):
//   smoothed = last + beta * (raw - last)      (beta = 1 -> no EMA)
//   pd_target = smoothed * action_scale + default_joint_pos
//   torque = (pd_target - q) * kp - dq * kd
// The tracking kp/kd are far softer than the AMO recipe — they come from the
// training-side MJX config baked into the ONNX metadata.

export class TrackingNetwork {
  constructor(buffer, meta) {
    this.meta = meta;
    const f32 = new Float32Array(buffer);
    const view = (name) => {
      const o = meta.offsets[name] / 4, shape = meta.shapes[name];
      let n = 1;
      for (const d of shape) n *= d;
      return f32.subarray(o, o + n);
    };
    this.mean = view('mean');
    this.std = view('std');
    this.layers = [];
    for (let i = 0; i < 4; i++) {
      this.layers.push({ W: view('W:' + i), b: view('b:' + i), shape: meta.shapes['W:' + i] });
    }
    this.motion = {
      joint_pos: view('motion:joint_pos'),
      joint_vel: view('motion:joint_vel'),
      body_pos_w: view('motion:body_pos_w'),
      body_quat_w: view('motion:body_quat_w'),
      body_lin_vel_w: view('motion:body_lin_vel_w'),
      body_ang_vel_w: view('motion:body_ang_vel_w'),
    };
    this.nq = meta.num_joints;
    this.n = this.nq;   // alias used by TrackingFighter
    this.nBody = meta.num_bodies;
    this.motionLen = meta.motion.length;
    this.maxStep = meta.motion.max_step;
    this.defaultQ = Float32Array.from(meta.default_joint_pos);
    this.kp = Float32Array.from(meta.kp);
    this.kd = Float32Array.from(meta.kd);
    this.actionScale = Float32Array.from(meta.action_scale);

    const dim = meta.obs_dim;
    this._h1 = new Float32Array(this.layers[0].shape[0]);
    this._h2 = new Float32Array(this.layers[1].shape[0]);
    this._h3 = new Float32Array(this.layers[2].shape[0]);
    this._act = new Float32Array(this.layers[3].shape[0]);
    this._norm = new Float32Array(dim);
  }

  // reference motion frame at a policy tick (clamped at both ends), exact
  // replica of the ONNX Gather branch
  refAt(step) {
    const t = Math.max(0, Math.min(this.maxStep, step | 0));
    const nq = this.nq, nb = this.nBody;
    return {
      step: t,
      joint_pos: this.motion.joint_pos.subarray(t * nq, t * nq + nq),
      joint_vel: this.motion.joint_vel.subarray(t * nq, t * nq + nq),
      body_pos_w: this.motion.body_pos_w.subarray(t * nb * 3, t * nb * 3 + nb * 3),
      body_quat_w: this.motion.body_quat_w.subarray(t * nb * 4, t * nb * 4 + nb * 4),
      body_lin_vel_w: this.motion.body_lin_vel_w.subarray(t * nb * 3, t * nb * 3 + nb * 3),
      body_ang_vel_w: this.motion.body_ang_vel_w.subarray(t * nb * 3, t * nb * 3 + nb * 3),
    };
  }

  // exact replica of the actor branch: normalize -> Gemm/Elu x3 -> Gemm
  infer(obs) {
    const norm = this._norm;
    for (let i = 0; i < obs.length; i++) norm[i] = (obs[i] - this.mean[i]) / this.std[i];
    let x = norm;
    for (let l = 0; l < this.layers.length; l++) {
      const { W, b, shape } = this.layers[l];
      const out = l === 0 ? this._h1 : l === 1 ? this._h2 : l === 2 ? this._h3 : this._act;
      const nOut = shape[0], nIn = shape[1];
      for (let j = 0; j < nOut; j++) {
        let s = b[j];
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
        out[j] = l < 3 ? (s > 0 ? s : Math.expm1(s)) : s;   // Elu on hidden layers
      }
      x = out;
    }
    return this._act;
  }
}

// ---------------------------------------------------------------------------
// per-robot runner: builds the 154-dim observation, advances the clip clock,
// and writes PD torques — the tracking counterpart of AMOFighter.
//
// Rates mirror the training env exactly: 200 Hz sim (timestep 0.005),
// 50 Hz policy (decimation 4), PD torque written every sim step.
// obs order = meta.obs_names == ONNX observation_names:
//   command(58) anchor_ori_b(6) base_ang_vel(3) joint_pos(29) joint_vel(29)
//   actions(29)  [has_state_estimation=False export: no anchor_pos, no lin_vel]
// ---------------------------------------------------------------------------

// torque limits per joint (mjlab G1 actuator forcerange; model property shared
// by every clip policy, keyed like POLICY_JOINTS-style bare names)
const TRACKING_TORQUE_LIM = {
  hip_pitch: 88, hip_roll: 139, hip_yaw: 88, knee: 139,
  ankle_pitch: 50, ankle_roll: 50,
  waist_yaw: 88, waist_roll: 50, waist_pitch: 50,
  shoulder_pitch: 25, shoulder_roll: 25, shoulder_yaw: 25, elbow: 25,
  wrist_roll: 25, wrist_pitch: 5, wrist_yaw: 5,
};

const rotMatFromQuat = (qw, qx, qy, qz, m) => {
  // row-major 3x3, wxyz quaternion
  const xx = qx * qx, yy = qy * qy, zz = qz * qz;
  m[0] = 1 - 2 * (yy + zz); m[1] = 2 * (qx * qy - qz * qw); m[2] = 2 * (qx * qz + qy * qw);
  m[3] = 2 * (qx * qy + qz * qw); m[4] = 1 - 2 * (xx + zz); m[5] = 2 * (qy * qz - qx * qw);
  m[6] = 2 * (qx * qz - qy * qw); m[7] = 2 * (qy * qz + qx * qw); m[8] = 1 - 2 * (xx + yy);
};

export class TrackingFighter {
  constructor(mujoco, model, data, side, net, { yawOffset = 0 } = {}) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.side = side;
    this.net = net;
    this.beta = meta_beta(net.meta);          // deployment EMA (RoboJuDo default 1.0)
    // side B faces the opposite way from the clip's baked-in yaw. The policy
    // is yaw-invariant (all obs are body-frame), so rotating the REFERENCE
    // torso quat by `yawOffset` makes it track a world-rotated clip instead of
    // spinning 180° to chase the original heading.
    this.yawOffset = yawOffset;
    this._qz = yawOffset ? [Math.cos(yawOffset / 2), 0, 0, Math.sin(yawOffset / 2)] : null;

    const jid = n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, side + '_' + n);
    const aid = n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, side + '_' + n);
    const names = net.meta.joint_names;       // obs/action order (training order)
    const n = names.length;
    this.n = n;
    this.qposadr = new Int32Array(n);
    this.dofadr = new Int32Array(n);
    this.actId = new Int32Array(n);
    this.lim = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const bare = names[i];
      const j = jid(bare);
      if (j < 0) throw new Error('joint not found: ' + side + bare);
      this.qposadr[i] = model.jnt_qposadr[j];
      this.dofadr[i] = model.jnt_dofadr[j];
      const a = aid(bare);
      if (a < 0) throw new Error('actuator not found: ' + side + bare);
      this.actId[i] = a;
      const family = bare.replace(/^(left|right)_/, '').replace(/_joint$/, '');
      this.lim[i] = TRACKING_TORQUE_LIM[family] ?? 25;
    }
    // pelvis freejoint: mjlab names it "floating_base_joint", AMO names it
    // "pelvis" — resolve via the pelvis body's first joint instead
    const pbody = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, side + '_pelvis');
    const pj = model.body_jntadr[pbody];
    if (!(pj >= 0) || model.jnt_type[pj] !== mujoco.mjtJoint.mjJNT_FREE.value) {
      throw new Error('pelvis freejoint not found for side ' + side);
    }
    this.freeQpos = model.jnt_qposadr[pj];
    this.freeDof = model.jnt_dofadr[pj];
    const tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, side + '_torso_link');
    if (tb < 0) throw new Error('torso body not found: ' + side);
    this.torsoBody = tb;

    this.lastAction = new Float32Array(n);
    this.pdTarget = Float32Array.from(net.defaultQ);
    this.actionRaw = new Float32Array(n);
    this.timeStep = 0;
    this.stepCount = 0;
    this.clipDone = false;
    this.healthy = true;
  }

  reset(fromStep = 0) {
    this.lastAction.fill(0);
    this.pdTarget.set(this.net.defaultQ);
    this.timeStep = fromStep;
    this.stepCount = 0;
    this.clipDone = false;
    this.healthy = true;
  }

  // call every physics step (200 Hz); policy every DECIMATION steps
  physicsStep(decimation = 4) {
    if (this.stepCount % decimation === 0) this.policyTick();
    this.stepCount++;
    this.writeTorque();
  }

  writeTorque() {
    const d = this.data, pd = this.pdTarget, n = this.n;
    for (let i = 0; i < n; i++) {
      const q = d.qpos[this.qposadr[i]], dq = d.qvel[this.dofadr[i]];
      let t = (pd[i] - q) * this.net.kp[i] - dq * this.net.kd[i];
      const lim = this.lim[i];
      if (t > lim) t = lim; else if (t < -lim) t = -lim;
      d.ctrl[this.actId[i]] = t;
    }
  }

  policyTick() {
    const d = this.data, net = this.net, n = this.n;
    const ref = net.refAt(this.timeStep);

    // obs: reference command [joint_pos | joint_vel]
    const obs = new Float32Array(net.meta.obs_dim);
    obs.set(ref.joint_pos, 0);
    obs.set(ref.joint_vel, n);
    let o = 2 * n;

    // motion_anchor_ori_b: relative orientation of the reference torso in the
    // robot torso frame, first two rotmat columns (row-major)
    const tq = this.data.xquat;
    const rqw = tq[4 * this.torsoBody], rqx = tq[4 * this.torsoBody + 1],
          rqy = tq[4 * this.torsoBody + 2], rqz = tq[4 * this.torsoBody + 3];
    // q_rel = q_robot^-1 * q_ref  (wxyz Hamilton); anchor index from body_names
    const rj = ref.body_quat_w;
    const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
    let qw = rj[4 * ai], qx = rj[4 * ai + 1], qy = rj[4 * ai + 2], qz = rj[4 * ai + 3];
    if (this._qz) {
      // rotate the reference quat by yawOffset about world z (wxyz Hamilton)
      const [zw, zx, zy, zz] = this._qz;
      const rw = zw * qw - zz * qz;
      const rx = zw * qx + zz * qy;
      const ry = zw * qy - zz * qx;
      const rz = zw * qz + zz * qw;
      qw = rw; qx = rx; qy = ry; qz = rz;
    }
    const iw = rqw, ix = -rqx, iy = -rqy, iz = -rqz;  // conjugate: (w, -x, -y, -z)
    // (iw,ix,iy,iz)*(qw,qx,qy,qz)
    const cw = iw * qw - ix * qx - iy * qy - iz * qz;
    const cx = iw * qx + ix * qw + iy * qz - iz * qy;
    const cy = iw * qy - ix * qz + iy * qw + iz * qx;
    const cz = iw * qz + ix * qy - iy * qx + iz * qw;
    const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    rotMatFromQuat(cw, cx, cy, cz, M);
    obs[o++] = M[0]; obs[o++] = M[1];
    obs[o++] = M[3]; obs[o++] = M[4];
    obs[o++] = M[6]; obs[o++] = M[7];

    // base_ang_vel: pelvis gyro == free-joint body-frame angular velocity
    obs[o++] = d.qvel[this.freeDof + 3];
    obs[o++] = d.qvel[this.freeDof + 4];
    obs[o++] = d.qvel[this.freeDof + 5];

    // joint_pos_rel / joint_vel in the net's joint order
    for (let i = 0; i < n; i++) obs[o++] = d.qpos[this.qposadr[i]] - net.defaultQ[i];
    for (let i = 0; i < n; i++) obs[o++] = d.qvel[this.dofadr[i]];

    // last (smoothed) action
    obs.set(this.lastAction, o);

    // infer + EMA + PD target
    const raw = net.infer(obs);
    let finite = true;
    for (let i = 0; i < n; i++) if (!Number.isFinite(raw[i])) finite = false;
    this.healthy = finite;
    const beta = this.beta;
    for (let i = 0; i < n; i++) {
      const target = Math.max(-100, Math.min(100, raw[i]));
      this.actionRaw[i] = target;
      const sm = this.lastAction[i] + beta * (target - this.lastAction[i]);
      this.lastAction[i] = sm;
      this.pdTarget[i] = sm * net.actionScale[i] + net.defaultQ[i];
    }

    // advance the clip clock (refAt clamps; the scheduler decides what next)
    if (!this.clipDone && this.timeStep >= net.motionLen - 1) this.clipDone = true;
    this.timeStep++;
  }
}

function meta_beta(meta) {
  return meta.deployment?.action_beta ?? 1.0;
}
