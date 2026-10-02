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

// 统一的 yaw 提取（wxyz 四元数，ZYX 欧拉），结果归一化到 (−π,π]。调度器的重
// 锚定角必须走本函数：裸 2*atan2(z,w) 对后半圈朝向给出 (π,2π] 的值（如
// +198.2°），再与非归一化角相减会差出整 360°。
export function yawOfQuat(w, x, y, z) {
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

const rotMatFromQuat = (qw, qx, qy, qz, m) => {
  // row-major 3x3, wxyz quaternion
  const xx = qx * qx, yy = qy * qy, zz = qz * qz;
  m[0] = 1 - 2 * (yy + zz); m[1] = 2 * (qx * qy - qz * qw); m[2] = 2 * (qx * qz + qy * qw);
  m[3] = 2 * (qx * qy + qz * qw); m[4] = 1 - 2 * (xx + zz); m[5] = 2 * (qy * qz - qx * qw);
  m[6] = 2 * (qx * qz - qy * qw); m[7] = 2 * (qy * qz + qx * qw); m[8] = 1 - 2 * (xx + yy);
};

export class TrackingFighter {
  // opponent（W2，plan-fight-20260929）：对手 TrackingFighter 引用，仅 fight
  // ONNX（meta.obs_layout.opponent_state 存在）的对手段观测需要；可空——缺席
  // 时按 side 翻转的命名约定自解析（见 _resolveOpponentRefs）。
  // ctrlRange（P-A，终审第 3 轮）：逐关节 PD 目标位置夹紧窗（Float32Array
  // [lo,hi]×n，meta.joint_names 顺序），复刻训练侧 position actuator 的
  // ctrlrange 夹紧——训练 ctrl 目标被 mjlab 软窗夹住，浏览器 JS PD 原本无
  // 此约束；仅 fight 模式传入（见 boxing_ai.mjs FIGHT_CTRL_RANGE），默认
  // null = 不夹紧，guard/combo 路径字节不变。
  constructor(mujoco, model, data, side, net, { yawOffset = 0, faceoff = false, opponent = null, ctrlRange = null } = {}) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.side = side;
    this.net = net;
    this.opponent = opponent;
    this.ctrlRange = ctrlRange;
    this.beta = meta_beta(net.meta);          // deployment EMA (RoboJuDo default 1.0)
    // side B faces the opposite way from the clip's baked-in yaw. The policy
    // is yaw-invariant (all obs are body-frame), so rotating the REFERENCE
    // torso quat by `yawOffset` makes it track a world-rotated clip instead of
    // spinning 180° to chase the original heading.
    this.yawOffset = yawOffset;
    this._qz = yawOffset ? [Math.cos(yawOffset / 2), 0, 0, Math.sin(yawOffset / 2)] : null;
    // faceoff（Phase 2，2026-09-29）：true 时 _qz/rebaseToReference 使用正确的
    // Rz(δ) 前乘公式（ry = zw*qy + zz*qx）。历史代码两处同用 ry = zw*qy −
    // zz*qx——不是合法四元数合成（非同态），δ=0 时 zw=1/zz=0 两式恒等（已
    // 交付 delta=0 路径字节不变），δ≠0 且 ref0 带侧倾时状态侧与观测侧不相
    // 消，anchor_ori 观测泄漏实测 0.645（_probe_faceoff obsdiff）。默认
    // false 保持既有行为字节不变。
    this._faceoffFix = faceoff;

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
    this.pelvisBody = pbody;   // 对手段观测/对手侧引用用（fight 模式）
    // FIGHT 对手段引用（plan-fight-20260929 §5.2 W2）：仅当 ONNX meta 带
    // opponent_state 段（obs_dim 168）时解析；旧 154 维 ONNX 整块跳过，既有
    // 路径字节不变。一致性锚点（两侧审计点）：训练侧 anchor = MotionCommand
    // cfg 的 anchor_body_name = "torso_link"（config/g1/env_cfgs.py:42，
    // robot_anchor_pos_w/quat_w 即 torso_link 的 body pos/quat）↔ 浏览器侧
    // self 锚点 = side_torso_link（this.torsoBody，与 anchor_ori 观测同一
    // body）；对方 root = opp pelvis、双拳 = opp 左/右 wrist_yaw_link，与
    // sparring_mdp.opponent_state() 的 body 选择一一对应。
    this._oppRefs = null;
    if (net.meta.obs_layout?.opponent_state) this._resolveOpponentRefs(true);

    this.lastAction = new Float32Array(n);
    this.pdTarget = Float32Array.from(net.defaultQ);
    this.actionRaw = new Float32Array(n);
    this.timeStep = 0;
    this.stepCount = 0;
    this.clipDone = false;
    this.healthy = true;
    // PD 目标后处理钩子（fight 守卫混合用，2026-10-02）：每物理步在
    // policyTick 之后、writeTorque 之前调用 targetHook(this)，可对 pdTarget
    // 做最后修正（如非出拳手臂向守卫姿态混合）。默认 null —— guard/combo/
    // faceoff 路径零调用，行为字节不变；仅 boxing_ai 的 fight 分支赋值。
    this.targetHook = null;
    // 前馈力矩加性通道（第 4 轮 P0-2，与 targetHook 同族的钩子传参，2026-
    // 10-02）：非 null 时 writeTorque 逐关节在限幅前叠加 extraTorque[i]，
    // 供 fight 守卫重力前馈使用（分配/清理由 boxing_ai._attachFightGuard /
    // applyFightGuard 负责：per-fighter Float32Array(n)，每物理步先清后写，
    // 出拳/早退路径清零）。默认 null——guard/combo/faceoff 路径 writeTorque
    // 零分支，行为字节不变；不改任何状态机字段（stepCount/lastAction/pdTarget
    // 语义照旧）。
    this.extraTorque = null;
  }

  // 对手 body 引用解析（惰性缓存一次）：优先用注入对手 fighter 的 side 与
  // pelvis id，缺席（单机向量测试/构造顺序）时按 side 翻转的命名约定自解析
  // ——双机共享同一 model/data，body id 与 side 一一对应，两种来源等价；对手
  // fighter 事后被替换（swapTo 每次新建）也不失配。严格模式（构造期）缺
  // body/自由关节直接抛错，避免 fight 模式带病上线。
  _resolveOpponentRefs(strict = false) {
    if (this._oppRefs) return this._oppRefs;
    const opp = this.opponent && this.opponent.side !== this.side ? this.opponent : null;
    const oppSide = opp ? opp.side : (this.side === 'A' ? 'B' : 'A');
    const bid = n => this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_BODY.value, oppSide + '_' + n);
    const pelvis = opp && opp.pelvisBody >= 0 ? opp.pelvisBody : bid('pelvis');
    const fistL = bid('left_wrist_yaw_link'), fistR = bid('right_wrist_yaw_link');
    const pj = pelvis >= 0 ? this.model.body_jntadr[pelvis] : -1;
    const freeDof = pj >= 0 && this.model.jnt_type[pj] === this.mujoco.mjtJoint.mjJNT_FREE.value
      ? this.model.jnt_dofadr[pj] : -1;
    if (pelvis < 0 || fistL < 0 || fistR < 0 || freeDof < 0) {
      if (strict) {
        throw new Error('opponent_state bodies unresolved for side ' + oppSide +
          ` (pelvis=${pelvis} fistL=${fistL} fistR=${fistR} freeDof=${freeDof})`);
      }
      return null;
    }
    this._oppRefs = { pelvis, fistL, fistR, freeDof };
    return this._oppRefs;
  }

  reset(fromStep = 0) {
    this.lastAction.fill(0);
    this.pdTarget.set(this.net.defaultQ);
    this.timeStep = fromStep;
    this.stepCount = 0;
    this.clipDone = false;
    this.healthy = true;
  }

  // FIGHT 航向伺服（boxing_ai.mjs updateFightHeadingServo 调用，2026-10-01）：
  // 运行中更新 yaw 偏移 δ。δ 同时是观测侧（policyTick 的 _qz 参考旋转）与状
  // 态侧（下次 rebaseToReference 的 Rz(Δ)）的唯一基准——两侧共用同一 δ，观测
  // 与目标自洽。δ=0 时本方法构造的 _qz 是精确单位四元数（zw=1/zz=0 的乘加为
  // IEEE 精确运算），与构造器 δ=0 → _qz=null 的旧路径数值等价；本方法仅被
  // fight 伺服调用，guard/combo/faceoff 的一次性 delta 路径不经此处。
  setHeadingOffset(delta) {
    this.yawOffset = delta;
    this._qz = [Math.cos(delta / 2), 0, 0, Math.sin(delta / 2)];
  }

  // 原位重锚定 RSI：把机器人状态设为训练回合起点——29 关节 qpos/qvel 取参考
  // 帧位姿（startStep，默认 0 = 既有回合起点帧；fight 模式传非零相位偏移，
  // 即训练 RSI 随机起始帧的确定性等价物）；基座世界 x/y 保留（战斗站位
  // 不动）、z 取参考值、四元数 = Rz(yawDelta) ⊗ 参考帧基座四元数（wxyz
  // Hamilton 世界系前乘，与 policyTick 的 _qz 观测旋转同约定）。基座 6 维
  // 速度清零；lastAction 清零、pdTarget 回 defaultQ、时钟从 startStep 起
  // （reset(fromStep) 原生支持）。切换片段时以 yawDelta = 当前 torso yaw −
  // 参考起始帧 torso yaw 调用，anchor-ori 观测在重锚定瞬间归零。不调用
  // mj_forward：双机器人共用 model/data，由调用方统一刷一次。
  // startStep 默认 0：既有调用（guard/combo 全部路径）不传该参数，行为字节不变。
  rebaseToReference(yawDelta = 0, startStep = 0) {
    const d = this.data, ref = this.net.refAt(startStep);
    for (let i = 0; i < this.n; i++) {
      d.qpos[this.qposadr[i]] = ref.joint_pos[i];
      d.qvel[this.dofadr[i]] = ref.joint_vel[i];
    }
    // (qw,qx,qy,qz) ⊗ (zw,0,0,zz)：Hamilton 后乘，与 policyTick 的 _qz 观测
    // 旋转同式同 Δ——状态侧与观测侧代数相消，勿单侧修改（ref0 基座严格
    // 竖直、纯 yaw 时前后乘精确等价，但不要依赖这一点改乘序）。
    // 2026-09-29 修复：基座 qpos 写入必须用 freeQpos（jnt_qposadr，含 3 位置
    // +4 四元数布局），此前误用 freeDof（jnt_dofadr）——A 侧两者同为 0 未暴
    // 露，B 侧 qposadr=7/dofadr=6 时整段状态错位写穿（y 被高度覆盖、四元数
    // 串位），rebase 后必倒。
    const zw = Math.cos(yawDelta / 2), zz = Math.sin(yawDelta / 2);
    const qw = ref.body_quat_w[0], qx = ref.body_quat_w[1];
    const qy = ref.body_quat_w[2], qz = ref.body_quat_w[3];
    if (this._faceoffFix) {
      // 正确的 Rz(δ) ⊗ q（wxyz Hamilton 前乘=世界系旋转），与下方 policyTick
      // 修正式完全同式。遗留后乘是体轴旋转，参考基座相对四元数带俯仰时不与
      // Rz 交换（组合片段 q_rel 俯仰 +0.247），状态/观测两侧不相消。
      d.qpos[this.freeQpos + 3] = zw * qw - zz * qz;
      d.qpos[this.freeQpos + 4] = zw * qx - zz * qy;
      d.qpos[this.freeQpos + 5] = zw * qy + zz * qx;
      d.qpos[this.freeQpos + 6] = zw * qz + zz * qw;
    } else {
      // 遗留公式（δ=0 时与修正式恒等；保持既有路径字节不变）
      d.qpos[this.freeQpos + 3] = zw * qw - zz * qz;
      d.qpos[this.freeQpos + 4] = zw * qx + zz * qy;
      d.qpos[this.freeQpos + 5] = zw * qy - zz * qx;
      d.qpos[this.freeQpos + 6] = zw * qz + zz * qw;
    }
    d.qpos[this.freeQpos + 2] = ref.body_pos_w[2];
    for (let k = 0; k < 6; k++) d.qvel[this.freeDof + k] = 0;
    this.reset(startStep);   // lastAction/pdTarget 归零、时钟 = 参考起始帧
  }

  // call every physics step (200 Hz); policy every DECIMATION steps
  physicsStep(decimation = 4) {
    if (this.stepCount % decimation === 0) this.policyTick();
    // fight 守卫混合钩子（见构造器 targetHook 注释）：在 pdTarget 刚被
    // policyTick 重算之后、力矩写出之前逐物理步修正——混合在策略目标之上
    // 生效，不污染 lastAction（策略内部动作状态保持自洽）。
    if (this.targetHook) this.targetHook(this);
    this.stepCount++;
    this.writeTorque();
  }

  writeTorque() {
    const d = this.data, pd = this.pdTarget, n = this.n;
    const ff = this.extraTorque;   // fight 守卫重力前馈（null = 无，见构造器注释）
    for (let i = 0; i < n; i++) {
      const q = d.qpos[this.qposadr[i]], dq = d.qvel[this.dofadr[i]];
      let t = (pd[i] - q) * this.net.kp[i] - dq * this.net.kd[i];
      if (ff) t += ff[i];
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
      const rx = this._faceoffFix ? zw * qx - zz * qy : zw * qx + zz * qy;
      const ry = this._faceoffFix ? zw * qy + zz * qx : zw * qy - zz * qx;
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

    // ---- opponent_state（W2，fight ONNX 专属；旧 ONNX 无此段整块跳过，obs
    // 游标路径字节不变）。14 维对手段，全部在观察者自身 torso（anchor）系，
    // 与训练侧 sparring_mdp.opponent_state() 逐维对齐（.workbuddy/gpu/
    // g1dance_pipeline/third_party/unitree_rl_mjlab/src/tasks/tracking/mdp/
    // sparring_mdp.py）：
    //   [0:3]  R_anchor^T·(p_opp_pelvis − p_self_anchor)   opp_root_pos_rel
    //   [3:6]  R_anchor^T·v_opp_pelvis（世界系线速度）      opp_root_lin_vel_rel
    //   [6:9]  R_anchor^T·(p_opp_Lwrist − p_self_anchor)   opp_fist_L_rel
    //   [9:12] R_anchor^T·(p_opp_Rwrist − p_self_anchor)   opp_fist_R_rel
    //   [12]   sin(yaw_opp − yaw_self)   [13] cos(...)
    // yaw 口径照抄训练侧：yaw_self 取自锚点（torso_link）四元数、yaw_opp 取自
    // 对方 pelvis（root_link）四元数（euler_xyz 的 z 角，同 yawOfQuat 公式）。
    // 对方线速度 = 对方自由关节 qvel 前 3 维（MuJoCo 约定：全局系、body 原点
    // 速度）↔ 训练侧 root_link_lin_vel_w。
    const oppLayout = net.meta.obs_layout?.opponent_state;
    if (oppLayout) {
      const refs = this._resolveOpponentRefs();
      if (!refs) throw new Error('opponent_state refs unresolved in policyTick (side ' + this.side + ')');
      const base = oppLayout[0];   // 数据契约 §5.1：偏移 154、宽 14
      const xa = 3 * this.torsoBody;
      const apx = d.xpos[xa], apy = d.xpos[xa + 1], apz = d.xpos[xa + 2];
      const qo = 4 * this.torsoBody;
      const Ra = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      rotMatFromQuat(tq[qo], tq[qo + 1], tq[qo + 2], tq[qo + 3], Ra);
      // rel = R_anchor^T · Δp（row-major 转置乘）
      const put3 = (i, wx, wy, wz) => {
        const dx = wx - apx, dy = wy - apy, dz = wz - apz;
        obs[base + i]     = Ra[0] * dx + Ra[3] * dy + Ra[6] * dz;
        obs[base + i + 1] = Ra[1] * dx + Ra[4] * dy + Ra[7] * dz;
        obs[base + i + 2] = Ra[2] * dx + Ra[5] * dy + Ra[8] * dz;
      };
      const rp = 3 * refs.pelvis, rl = 3 * refs.fistL, rr = 3 * refs.fistR;
      put3(0, d.xpos[rp], d.xpos[rp + 1], d.xpos[rp + 2]);
      const fv = refs.freeDof;
      obs[base + 3] = Ra[0] * d.qvel[fv] + Ra[3] * d.qvel[fv + 1] + Ra[6] * d.qvel[fv + 2];
      obs[base + 4] = Ra[1] * d.qvel[fv] + Ra[4] * d.qvel[fv + 1] + Ra[7] * d.qvel[fv + 2];
      obs[base + 5] = Ra[2] * d.qvel[fv] + Ra[5] * d.qvel[fv + 1] + Ra[8] * d.qvel[fv + 2];
      put3(6, d.xpos[rl], d.xpos[rl + 1], d.xpos[rl + 2]);
      put3(9, d.xpos[rr], d.xpos[rr + 1], d.xpos[rr + 2]);
      const yawSelf = yawOfQuat(tq[qo], tq[qo + 1], tq[qo + 2], tq[qo + 3]);
      const yawOpp = yawOfQuat(tq[4 * refs.pelvis], tq[4 * refs.pelvis + 1],
        tq[4 * refs.pelvis + 2], tq[4 * refs.pelvis + 3]);
      const yawDiff = yawOpp - yawSelf;
      obs[base + 12] = Math.sin(yawDiff);
      obs[base + 13] = Math.cos(yawDiff);
    }

    // last (smoothed) action
    obs.set(this.lastAction, o);

    // infer + EMA + PD target
    const raw = net.infer(obs);
    let finite = true;
    for (let i = 0; i < n; i++) if (!Number.isFinite(raw[i])) finite = false;
    this.healthy = finite;
    const beta = this.beta;
    const cr = this.ctrlRange;
    for (let i = 0; i < n; i++) {
      const target = Math.max(-100, Math.min(100, raw[i]));
      this.actionRaw[i] = target;
      const sm = this.lastAction[i] + beta * (target - this.lastAction[i]);
      this.lastAction[i] = sm;
      // P-A：fight 模式按训练 ctrlrange 软窗夹紧 PD 目标位置（复刻训练侧
      // position actuator 的 ctrl 夹紧；力矩仍由 writeTorque 的
      // TRACKING_TORQUE_LIM 封顶，两窗语义不同勿混淆）。cr=null（guard/
      // combo）走原表达式，字节不变。
      this.pdTarget[i] = cr
        ? Math.max(cr[2 * i], Math.min(cr[2 * i + 1], sm * net.actionScale[i] + net.defaultQ[i]))
        : sm * net.actionScale[i] + net.defaultQ[i];
    }

    // advance the clip clock (refAt clamps; the scheduler decides what next)
    if (!this.clipDone && this.timeStep >= net.motionLen - 1) this.clipDone = true;
    this.timeStep++;
  }
}

function meta_beta(meta) {
  return meta.deployment?.action_beta ?? 1.0;
}
