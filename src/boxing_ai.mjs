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
import { TrackingFighter, yawOfQuat } from './tracking_policy.mjs';
import { OBS_DIM, TACTIC_ACTIONS, ACTION_COMBOS } from './tactics.mjs';

// tactic decision -> tracking clip name (stage-4 motion layer). Footwork/idle
// states ride the looping guard clip; attacks trigger their punch clip.
// §4.2: 追踪片段策略尚未训练步法指令，footwork 仍映射 guard；新增指令必须在
// 训练中出现后才允许传入运动层（plan-realistic-boxing §4.2）。
const TACTIC_CLIP = {
  approach: 'guard', retreat: 'guard', strafe_left: 'guard', strafe_right: 'guard',
  wait: 'guard',
  jab: 'jab', double: 'jab', cross: 'cross', hook: 'hook',
};

// 拳法片段启用门禁（临时）：现有拳法权重在真实切换中存活余量不足——jab/hook
// fresh-RSI 起步 minZ 仅 0.481，任何 1° 级基座残差都会翻倒（机制 B，见
// docs/stage4-tracking-notes.md 2026-09-28 节；hook 槽位即 jab 权重复制品）；
// cross 干净 RSI 虽有 0.578 余量，但活体接触下 2 次执行摔 1 次（seed 3 B@9.0
// 切换即倒），2026-09-28 队长决策一并禁用。全部拳法片段临时禁用，决策层
// 映射回 guard，待 v3 片段重训后逐个恢复——验收门槛：无辅助单机 RSI
// minZ≥0.55 + 双机三次 seed 回归无切换摔倒。
const CLIP_ENABLED = { guard: true, jab: false, cross: false, hook: false, combo: true };

// combo 槽位（plan-revamp-20260928 §5.2 D5-D8）：35s 连续拳击组合的单策略
// 常驻循环。COMBO_MODE 下跳过 tactics 采样/拳法门禁/KO 与回合系统；
// 非 combo 模式下该键不参与任何决策（TACTIC_CLIP 不产出 combo）。
const PUNCH_CLIP = { jab: true, double: true, cross: true, hook: true, combo: true };

// pelvis-to-pelvis distance window at which a punch clip may START. The punch
// motions carry forward drive (jab ~0.2m, cross/hook ~0.8-1.0m): firing them
// point-blank turns the lunge into a body-check tangle, firing them too far
// is a whiff-charge. Outside the window the scheduler plays guard instead.
// §4.2 标定：切片元数据携带实测 punch_range_m 时优先采用（punchRangeFor）。
const PUNCH_RANGE = {
  // 标定（拳-头最小距离 vs 起始间距）：单方面出拳够不着（0.49m@0.7m 间距），
  // 命中发生在双方面互刺、各自前倾 0.2-0.4m 的交换里——窗口收窄到接触带
  jab: [0.82, 1.08],
  cross: [0.85, 1.08],
  hook: [0.85, 1.08],
};

// §4.4 出拳意图的有效期：排队后在这个时间内没到合适距离就取消，防止一个
// 永远打不出去的拳法把步法决策饿死（"拳法请求长期阻塞移动"）。
const PUNCH_INTENT_TTL = 1.6;
// §4.4 出拳节奏：回合开始后的稳定期（策略从关键帧出生需要先站稳）+ 两次
// 出拳之间的最短间隔——防止出拳请求连发把节奏变成冲撞推挤。
const PUNCH_SETTLE_T = 2.5;
const PUNCH_COOLDOWN = 0.5;
// §4.4 允许从守卫循环提前切入的拳法：只放开低前冲的刺拳（前冲 ~0.2m，
// 试探拳）。cross/hook 前冲 0.4-1.0m，在出生间距就提前发动会变成冲撞顶人
// ——等阶段 A 重训片段携带实测 punch_range_m 后再逐拳放开（§4.2 标定）。
const PUNCH_EARLY_EXIT = { jab: true };
// §4.5 有效命中的拳速门槛 (m/s, ~60ms EMA)：真出拳 2-4 m/s，贴身挤压/漂移
// 接触 <0.5 m/s。低于门槛的接触记为 graze（不计分不伤害）。
const HIT_MIN_FIST_SPEED = 0.8;
// §4.2 朝向指令的转速上限 (rad/s)：避免朝向目标突然跳变。
const FACE_YAW_RATE = 2.5;
// FIGHT_MODE（Q6 裁决）：60s 单回合循环——超时即重开回合，无局数上限；KO
// 摔倒路径经 koState 结算后同样回到 resetRound。训练侧 episode 保持 10s 不变。
const FIGHT_ROUND_S = 60;

// FIGHT 反镜像（终审 P1 第 2 轮根因修复）：镜像出生 + 共享策略 + 对称观测 +
// 确定性物理 = 确定性镜像锁定（zA===zB 逐位相等、同步双倒、零命中）。训练
// 侧靠 obs 噪声 + RSI 位姿随机化 + push 事件打破对称——浏览器全确定性零打破。
// 以下两个确定性扰动表是 RSI 随机化的等价物（按回合序号循环取值，可复现）：
const FIGHT_PHASE_OFFSETS = [30, 55, 15, 40];   // B 机片段起始帧偏移（50fps → 0.3-1.1s），A 恒 0
// B 机出生位抖动（|yaw|≤0.02rad、|Δpos|≤0.02m，次级扰动与相位偏移共同作用）
const FIGHT_SPAWN_JITTER = [
  { dx: 0.000, dy: 0.000, yaw: 0.020 },
  { dx: 0.014, dy: 0.014, yaw: -0.020 },
  { dx: -0.020, dy: 0.000, yaw: 0.000 },
  { dx: 0.000, dy: -0.020, yaw: 0.015 },
];

// FIGHT 航向伺服（H1/H2 修复，2026-10-01）：δ=yawOffset 令"参考片段朝向+δ"
// 始终对准对手，同时作用于观测侧（policyTick 的 _qz 参考旋转）与状态侧
// （rebaseToReference 的 Rz 链路）——两侧共用同一 δ 保证自洽，见
// fightRebaseDelta / updateFightHeadingServo。默认开启；FIGHT_HEADING_SERVO=0
// 或构造项 headingServo=false 关闭（code-review 的 A/B 基线对比用）。τ 默认
// 0.3s（方案给定区间 [0.2,0.5] 的中值），FIGHT_HEADING_TAU 或构造项
// headingTau 可覆盖（夹紧 [0.05,2.0]s，覆盖参考片段自身 ±3 rad/s 的 yaw 摆
// 速带宽）。浏览器以原生 ES Module 加载本文件（无 process），env 读取带
// typeof 守卫。
const HEADING_SERVO_DEFAULT =
  typeof process === 'undefined' || !process.env || process.env.FIGHT_HEADING_SERVO === undefined
    ? true
    : process.env.FIGHT_HEADING_SERVO !== '0';
const HEADING_TAU_DEFAULT = (() => {
  const v = typeof process === 'undefined' || !process.env ? NaN : Number(process.env.FIGHT_HEADING_TAU);
  return Number.isFinite(v) && v > 0 ? Math.min(2.0, Math.max(0.05, v)) : 0.3;
})();

// ---- FIGHT 视觉形态修复（2026-10-02，路线 A 运行时层）----------------------
// 队长侦察证据（浏览器 4 回合截图）：贴脸缠抱、抱架丢失（双臂垂下）、双臂
// 对称前推、原地站桩。fight 参考动作自身即缺陷源（boxing_fight.bin 参考帧
// 大多双拳前伸 ~0.33m、末段垂手 el≈1.56——策略忠实跟踪即摆出摔跤/人偶形
// 态）。三项运行时对策，浏览器与 headless 共用本实现（无 harness 特判）：
//  1) 守卫混合：非出拳手臂的 PD 目标向高抱架目标混合（applyFightGuard 经
//     TrackingFighter.targetHook 逐物理步生效；第 2 轮 P0-2 修订为"覆写边界
//     按持久混合水平重混 + 窗口内指数微调"两级结构，见常量区与
//     applyFightGuard 注释）；第 4 轮 P0-2 追加臂关节重力前馈（随 guardMix
//     缩放，同一写入路径，见 FIGHT_GUARD_FEEDFORWARD）；出拳判定（该臂拳速
//     EMA 超阈）期间该臂完全交还策略——出拳→收拳自动回抱架。
//  2) 裁判分离力：躯干水平间距过近时对双方骨盆施加水平分离力（combo 模式
//     "缠斗分离"先例的 fight 版），按深度比例、限幅、计量（fightForceUsage）。
//  3) 距离保持力：间距过远时施加温和接近力，制造可感知的进退步法；过远
//     僵局重置（>1.9m/3s）保留兜底。
// 守卫关节目标选值依据（scene_boxing_tracking.xml 左臂链 FK，torso 系，第 4
// 轮 P0-2 el 加深后数值）：
//   sp=-0.88 sr=-0.05 sy=-0.10 el=-0.60 → 拳套 (0.195, 0.070, 0.399)，头 site
//   在 z=0.430 —— 拳套近头高（|拳-头z|=0.031）、高于肘 +0.26（探针两腿均
//   脱离边际，抱架形态）。右臂镜像（roll/yaw 反号，sp/el 同值）。el 为负是
//   屈臂方向（参考数据 el min=-0.02 从不屈臂，-0.60 越界 0.58rad，肘关节限位
//   -1.0472 内）。对照旧值 el=-0.35：拳套 (0.256, 0.052, 0.379)、高于肘
//   +0.24——第 3 轮终验诊断臂关节执行端跟踪亏差（肩 pitch 实际 p50 较指令低
//   0.15-0.27rad ≈ 重力矩/kp），加深屈肘直接抬拳套、缩短前臂重力臂，与重力
//   前馈（FIGHT_GUARD_FEEDFORWARD）同向合围。
// 部署的 boxing_guard 片段参考手臂即默认垂手站姿（不可用作守卫目标），故
// 目标为常量表而非片段采样。
// 出拳判定阈值同时是 tools/test_tracking_boxing.mjs 抱架保持率指标的
// "非出拳状态"定义（同一常量，指标与控制自洽）。判定只用拳速、不用前向
// 伸距：fight 参考帧双拳常驻前伸 0.29-0.42m，骑在 any 伸距阈值上会让门禁
// 抖振（且参考保持段伸距 0.38-0.42 会让门禁常开、抱架永远混不进去）。
// 第 2 轮 P0-2（2026-10-02 审查）根因 1——过触发：0.6 阈值下参考跟踪的常规
// 摆臂 EMA 也常越阈（实测"出拳中"占比 ~50%，而 60s 仅 9 次命中），守卫混合
// 一半时间被挂起。提到 1.2 m/s：真出拳 2-4 m/s（EMA τ~16ms，出手即远超阈），
// 参考跟踪/贴身挤压臂速 <1 m/s，单阈值恢复干净分离。指标分母口径随本常量
// 自动同源收紧——新口径：任一臂拳速 EMA >1.2 m/s 才算"出拳中"。
export const FIGHT_PUNCH_SPEED = 1.2;    // 拳速 EMA 超此值 (m/s) = 出拳中
const FIGHT_GUARD_TAU = 0.20;            // 覆写窗口内的守卫微调时间常数 (s)
// 第 2 轮 P0-2（2026-10-02 审查）根因 2——覆写重置：policyTick 每
// FIGHT_POLICY_DECIMATION 物理步整体重写 pdTarget，逐帧指数混合在覆写窗口
// 内累计仅 ~9.5%（α=1−e^(−0.005/0.2)≈0.025 × 4 步），持续 ≥0.5s 非出拳窗口
// 的抱架探针通过率仅 41-54%（仅调 FIGHT_GUARD_TAU 无效：每步 α 恒 0.025，
// 瓶颈在覆写把爬坡归零）。改为主从两级混合：
//  - 持久混合水平 guardMix（per-arm，存 Fighter）：非出拳每步向 MIX_MAX 指数
//    回升（τ=MIX_TAU），出拳瞬间清零——水平跨覆写窗口持久，不每窗口爬坡；
//  - 应用：覆写边界按当前水平一次性重混（稳态边界 = MIX_MAX 抱架——第 2 轮
//    取 70% 时审查建议 30-50% 边界混合取上沿并稳态保持；窗口内在边界结果上
//    继续 τ=FIGHT_GUARD_TAU 指数微调）。
// 第 3 轮 P0-2（2026-10-02 复验）：0.70 稳态混合实测仅把 guard-hold 托到 56%
// （<70% 门槛）。提到 0.8：scene_boxing_tracking.xml 左臂链 FK（校准复现第 1
// 轮注释值——纯守卫目标拳套 z≈0.38、高于肘≈0.23）估算混合稳态：最差策略姿态
// （参考末段垂手 el=1.56）下拳-肘 Δz +0.085→+0.147，|拳-头| 0.233→0.16m——
// 探针第二腿 |拳-头|≤0.25m 在低姿态才是紧约束，两项同时脱离边际。勿超 0.85
// （再高接近完全接管手臂，伤出拳真实感）。0.8 已把最差姿态托过探针线（Δz
// +0.147 > +0.03）；第 4 轮 P0-2 启用其时预留的剩余杠杆——el 加深至 -0.6
// （FIGHT_GUARD_JOINTS，最差混合姿态 |拳-头z| 0.166→0.125、高于肘
// +0.150→+0.190）+ 臂关节重力前馈（FIGHT_GUARD_FEEDFORWARD），不再动 MIX_MAX。
const FIGHT_GUARD_MIX_MAX = 0.8;
const FIGHT_GUARD_MIX_TAU = 0.10;        // 出拳结束后混合水平回升时间常数 (s)
//                              （第 3 轮 0.15→0.10：清零后 0.23s 回升到 90%
//                              水平、0.30s 到 95%，压缩出拳判定结束后守卫
//                              尚未回位的低混合尾窗——尾窗即保持率的 FAIL 段）
// policyTick 抽取倍率（tracking_policy.mjs physicsStep 的 decimation 实参，
// 200Hz 物理 / 50Hz 策略）。applyFightGuard 以 stepCount % 本值 === 0 判定
// "pdTarget 刚被 policyTick 覆写"（钩子在覆写之后、stepCount 自增之前调用，
// 见 physicsStep）；唯一调用点 update() 的 physicsStep(FIGHT_POLICY_DECIMATION)
// 与此同源，改抽取率只动这一处常量。
const FIGHT_POLICY_DECIMATION = 4;
const FIGHT_GUARD_JOINTS = {
  left_shoulder_pitch_joint: -0.88, left_shoulder_roll_joint: -0.05,
  left_shoulder_yaw_joint: -0.10, left_elbow_joint: -0.6,
  left_wrist_roll_joint: 0, left_wrist_pitch_joint: 0, left_wrist_yaw_joint: 0,
  right_shoulder_pitch_joint: -0.88, right_shoulder_roll_joint: 0.05,
  right_shoulder_yaw_joint: 0.10, right_elbow_joint: -0.6,
  right_wrist_roll_joint: 0, right_wrist_pitch_joint: 0, right_wrist_yaw_joint: 0,
};
// 第 4 轮 P0-2（2026-10-02 队长裁决）：臂关节重力前馈（guard-hold 决胜轮）。
// 第 3 轮终验诊断：混合强度已不是瓶颈（非出拳时段 guardMix≥0.7 占 79-87%），
// 根因是臂关节执行端跟踪亏差——肩 pitch 指令 p50 -0.91~-1.00 vs 实际
// -0.63~-0.76（亏 0.15-0.27rad × kp 14.25 ≈ 2.1-3.8 N·m），肘指令
// -0.27~-0.35 vs 实际 -0.15~-0.22。模型静态核验（qfrc_bias，v=0 即纯重力
// 广义力，guard 姿态/垂姿 p50 双点）与亏差×kp 相互印证：肩 pitch 3.82/4.11
// N·m（亏差主因即重力），肘 0.40/1.44（前臂越垂重力臂越长）。
// 幅值表（单位 N·m，可调常量；写入见 applyFightGuard，经 TrackingFighter
// .extraTorque 加性通道在 writeTorque 限幅前叠加）：
//   shoulder_pitch ±3.5 —— 重力矩 3.82-4.11 的上段；×guardMix 稳态(0.8)施加
//     2.8，残差 ≤(4.11-2.8)/14.25≈0.09rad（原亏差 0.15-0.27），配合 el 加深
//     合围；单项 < 4 N·m 护栏，远低 25 N·m 限幅。
//   elbow ±1.4 —— 垂姿重力矩 1.44（亏差×kp 1.57-1.85 的主部）；稳态施加
//     1.12，残差 ~0.02rad。守卫姿态重力矩仅 0.40，超额部分即回收期抗垂余量。
//   roll/yaw 0 —— 模型重力矩 |0.21-0.25| N·m（<限幅 1%），诊断亏差亦集中于
//     pitch/肘，不补。
// 方向：肩 pitch/肘负向 = 抬臂/屈肘，恰为重力下垂（正向）的反向。腕不进表
// （前馈只作用臂关节 8 项）。力矩 = k × guardMix 随混合缩放：出拳判定瞬间
// guardMix 清零 → 前馈同步为零（applyFightGuard 出拳分支显式清零该臂，钩子
// 每物理步先清后写、早退路径整表清零，无滞后残留），策略出拳动力学不受影响。
const FIGHT_GUARD_FEEDFORWARD = {
  left_shoulder_pitch_joint: -3.5, left_elbow_joint: -1.4,
  right_shoulder_pitch_joint: -3.5, right_elbow_joint: -1.4,
};
// 裁判分离力：间距 < SEP_DIST 起作用，力 = K·(SEP_DIST−d)，限幅 FMAX
const FIGHT_SEP_DIST = 0.42;
const FIGHT_SEP_K = 260;      // N/m
const FIGHT_SEP_FMAX = 80;    // N（与 combo 缠斗分离同上限）
// 距离保持力：间距 > APPR_DIST 起作用，力 = K·(d−APPR_DIST)，限幅 FMAX
// （<6% 体重，接触鲁棒训练分布内的温和扰动；仅制造进退，不推动平衡）
const FIGHT_APPR_DIST = 1.25;
const FIGHT_APPR_K = 70;      // N/m
const FIGHT_APPR_FMAX = 28;   // N


// FIGHT PD 目标位置夹紧窗（P-A，终审第 3 轮）：复刻训练侧 position actuator
// 的 ctrlrange 夹紧——训练 ctrl 目标被 mjlab 软窗夹住，浏览器 JS PD 原本无
// 此约束（力矩限幅是另一回事，勿与 TRACKING_TORQUE_LIM 混淆）。
// 来源：训练侧编译后模型真值 dump
// .workbuddy/gpu/logs/sparring_model_dump.json（Unitree-G1-Sparring-P2 play
// env → sim.mj_model，robot/ 侧 29 actuator；opponent/ 侧逐项一致已核对）。
// dump md5 = 21f3937166bee2dd0d48143e5f58aac1（2026-09-29 导出）。
// 顺序 = meta.joint_names（观测/动作序）。仅 fight 构造点传入，A/B 共用
// （两机同构、窗对称已含左右差异）。
const FIGHT_CTRL_RANGE = new Float32Array([
  -4.720885851964, 5.069985851964,   // left_hip_pitch_joint
  -1.926245865515, 4.369745865515,   // left_hip_roll_joint
  -4.947785851964, 4.947785851964,   // left_hip_yaw_joint
  -1.489912865515, 4.282445865515,   // left_knee_joint
  -2.626979255693, 2.277909255693,   // left_ankle_pitch_joint
  -2.016109255693, 2.016109255693,   // left_ankle_roll_joint
  -4.720885851964, 5.069985851964,   // right_hip_pitch_joint
  -4.369745865515, 1.926245865515,   // right_hip_roll_joint
  -4.947785851964, 4.947785851964,   // right_hip_yaw_joint
  -1.489912865515, 4.282445865515,   // right_knee_joint
  -2.626979255693, 2.277909255693,   // right_ankle_pitch_joint
  -2.016109255693, 2.016109255693,   // right_ankle_roll_joint
  -4.808185851964, 4.808185851964,   // waist_yaw_joint
  -2.274309255693, 2.274309255693,   // waist_roll_joint
  -2.274309255693, 2.274309255693,   // waist_pitch_joint
  -4.843509255693, 4.424709255693,   // left_shoulder_pitch_joint
  -3.342509255693, 4.005809255693,   // left_shoulder_roll_joint
  -4.372309255693, 4.372309255693,   // left_shoulder_yaw_joint
  -2.801509255693, 3.848709255693,   // left_elbow_joint
  -3.726529255693, 3.726529255693,   // left_wrist_roll_joint
  -1.912433481318, 1.912433481318,   // left_wrist_pitch_joint
  -1.912433481318, 1.912433481318,   // left_wrist_yaw_joint
  -4.843509255693, 4.424709255693,   // right_shoulder_pitch_joint
  -4.005809255693, 3.342509255693,   // right_shoulder_roll_joint
  -4.372309255693, 4.372309255693,   // right_shoulder_yaw_joint
  -2.801509255693, 3.848709255693,   // right_elbow_joint
  -3.726529255693, 3.726529255693,   // right_wrist_roll_joint
  -1.912433481318, 1.912433481318,   // right_wrist_pitch_joint
  -1.912433481318, 1.912433481318,   // right_wrist_yaw_joint
]);

// 可复现回归用的确定性 RNG（mulberry32）。测试通过 controller rng 选项注入，
// 默认 Math.random 保持旧行为。
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;

// 守卫片段的参考姿态是否回到与目标片段书端一致的站架：参考关节与目标片段
// 帧 0 的无穷范数 < 0.035 rad。§4.4 守卫提前切换的触发条件。对比目标片段
// 帧 0（而非网络默认姿态）使 v3 拳击架势书端的片段同样适用——切换条件就是
// "参考不会跳变"。
function nearGuardPose(net, step, targetNet) {
  const ref = net.refAt(step);
  const t0 = targetNet.refAt(0);
  for (let i = 0; i < net.nq; i++) {
    const d = ref.joint_pos[i] - t0.joint_pos[i];
    if (d > 0.035 || d < -0.035) return false;
  }
  return true;
}

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
  constructor(side, anchor0, rng = Math.random) {
    this.side = side;
    this.forward = side === 'A' ? 1 : -1;      // A walks +x, B walks -x
    this.faceYaw = side === 'A' ? 0 : Math.PI; // absolute yaw command toward opponent
    this.anchor0 = anchor0;
    this.anchor = [...anchor0];
    this.state = 'idle';
    this.stateT = 0;
    this.idleFor = 0.7 + rng() * 0.6;
    this.combo = null;
    this.seqIdx = 0;
    this.phaseT = 0;
    this.phaseDur = 0;
    this.staggerFor = 0;
    this.down = false;
    this.bobPhase = rng() * 6.28;
    this.rng = rng;
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
    this.clipMeta = null;    // {clip -> motion.clip 元数据}（有效出拳区间等，可空）
    this.clip = null;        // currently playing clip
    this.desiredClip = null; // clip the tactics layer wants next
    this.clipSwaps = 0;
    // §4.4 离散出拳请求：{clip, t}，与连续步法决策分开排队；带 TTL 防饿死
    this.punchIntent = null;
    // §4.5 出拳事件去重：每次出拳片段执行/组合步递增，一次出拳只计一分
    this.punchEventId = 0;
    this.lastScoredPunch = -1;
    this.lastPunchEnd = -1e9;   // 上一次出拳片段结束的时刻（出拳冷却用）
    this.noPunchUntil = -1e9;   // 出拳禁止窗口（回合稳定期 + 片段间冷却）
    this.fistSpeed = { left: 0, right: 0 };   // EMA 拳速 (m/s)，命中判定用
    this._fistPrev = { left: null, right: null, t: null };
    // FIGHT 守卫混合的 per-arm 持久水平（[0, FIGHT_GUARD_MIX_MAX]，第 2 轮
    // P0-2 的跨覆写窗口偏置；仅 applyFightGuard 读写）。resetRound/divergence
    // 自愈有意不清零：回合重置/回绕后立即以稳态强度回抱架。非 fight 模式
    // 恒 {left:0,right:0} 不被读，零影响。
    this.guardMix = { left: 0, right: 0 };
  }
}

// ---- controller -------------------------------------------------------------
export class BoxingController {
  constructor(model, data, { mujoco, ids, assist = true, aggression = 1, rng = Math.random, trackingAssist = false, comboMode = false, faceoff = false, fightMode = false, headingServo, headingTau }) {
    this.model = model;
    this.data = data;
    this.mujoco = mujoco;
    this.ids = ids;
    this.assist = assist;
    // FIGHT_MODE（plan-fight-20260929 §5.2 W3）：?scene=tracking&fight=1 真实
    // 对打——两台 G1 共享 168 维观测（含 opponent_state 14 维）的 fight 策略
    // 自主对打，KO/回合重置/记分全启用（不进下方 comboMode 的 KO 禁用分支，
    // checkFall tracking 阈 0.40 与训练侧 fall 终止同口径）。独立开关且优先
    // 于 combo（URL 契约 §5.1：同给时 fight 优先，此处归一化兜底）；faceoff
    // 是 combo 子选项，fight 下强制关。
    this.fightMode = fightMode;
    // COMBO_MODE（plan-revamp-20260928 §5.2）：两个拳手常驻循环 combo 片段，
    // 禁 KO 与回合重置（连续观察 N1）。默认 false，不影响既有路径。
    this.comboMode = comboMode && !fightMode;
    // faceoff（Phase 2，2026-09-29）：combo 子选项（?combo=1&faceoff=1）——B 侧
    // 按 rebaseYawDelta 重锚定（出生朝向 π − combo ref0 yaw −0.81 = −2.33 rad），
    // 180° 面向 A 成真对抗站位；A 侧保持 delta=0。依赖 TrackingFighter 的
    // faceoff 修正公式（世界系 Rz 前乘；遗留体轴后乘对带俯仰的 ref 不守恒，
    // anchor_ori 观测泄漏 0.645，见 tracking_policy.mjs）。默认 false：
    // delta=0 + 遗留公式，已交付路径字节不变。头less 验证（iter-6000 权重）：
    // 单机 B 30s PASS minZ=0.595、双机 30s PASS minZ=0.647、各 2 次回绕
    // （tools/_probe_faceoff.mjs obsdiff：全 154 维观测+动作与 delta=0 逐位相同）。
    this.faceoffMode = faceoff && !fightMode;
    // fight 60s 回合计时（Q6）：resetRound 归零
    this._roundT = 0;
    // FIGHT 反镜像（见 FIGHT_PHASE_OFFSETS）：回合序号与 B 机相位/出生抖动
    // 状态。boot = 第 0 回合（相位表[0]=30）；resetRound 递增循环取值。
    // 非 fight 模式这两个状态只被写不被读。
    this._roundNo = 0;
    this._fightPhase = { A: 0, B: FIGHT_PHASE_OFFSETS[0] };
    // FIGHT 航向伺服开关与时间常数（见 HEADING_SERVO_DEFAULT 注释）。构造项
    // 未给时才落 env/默认——显式传入 false 的 A/B 基线优先于 env。
    this.headingServo = headingServo ?? HEADING_SERVO_DEFAULT;
    this.headingTau = headingTau ?? HEADING_TAU_DEFAULT;
    // 伺服每物理步要用双方 torso 世界位姿，body id 一次性解析缓存（原
    // rebaseYawDelta 的逐次 name2id 语义不变，仅伺服路径走缓存）。
    this._torsoBody = {};
    for (const side of ['A', 'B']) {
      const tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, side + '_torso_link');
      if (tb < 0) throw new Error('torso body not found: ' + side);
      this._torsoBody[side] = tb;
    }
    // §4.6：追踪片段策略是无外力训练的，外部扶正/阻尼与片段动力学直接对抗
    // （实测真辅助让出拳段摔得更快）。默认关断，只对脚本回退模式生效；
    // 将来训练"带辅助课程"的策略后再开（trackingAssist=true）。
    this.trackingAssist = trackingAssist;
    this.aggression = aggression;
    this.rng = rng;
    this.fighters = {};
    for (const side of ['A', 'B']) {
      const b = side === 'A' ? ids.pelvisA : ids.pelvisB;
      this.fighters[side] = new Fighter(side, [data.xpos[3 * b], data.xpos[3 * b + 1]], rng);
    }
    this.fighters.A.opponent = this.fighters.B;
    this.fighters.B.opponent = this.fighters.A;
    for (const side of ['A', 'B']) this.fighters[side].noPunchUntil = PUNCH_SETTLE_T;

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
      // 通过骨盆 body 的第一个关节解析自由关节（2026-09-26 修复：原先按 AMO
      // 场景的关节名 side_pelvis 解析，追踪场景的自由关节叫
      // side_floating_base_joint → id=-1 → qvel 读出 undefined → NaN 污染
      // 辅助力输出，追踪模式的辅助自上线以来静默失效——辅助开/关轨迹完全
      // 相同即此症状）。
      const pbody = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, side + '_pelvis');
      const pj = pbody >= 0 ? model.body_jntadr[pbody] : -1;
      if (!(pj >= 0) || model.jnt_type[pj] !== mujoco.mjtJoint.mjJNT_FREE.value) {
        throw new Error('pelvis freejoint not found for side ' + side);
      }
      this.freeDof[side] = model.jnt_dofadr[pj];
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
    // §4.6 辅助用量计量：|力|(N·s) 与 |力矩|(N·m·s) 的积分，验收时报告
    // "外部辅助力与力矩的使用量"，避免辅助掩盖失败。
    this.assistUsage = {
      A: { fx: 0, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 },
      B: { fx: 0, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 },
    };
    this._lastAssistSample = { t: 0, A: null, B: null };
    // FIGHT 裁判分离力/距离保持力计量（|F|·dt 积分，N·s）：与平衡辅助
    // assistUsage 分开——fight 模式 assist 默认关，assistUsage 恒 0，这两项
    // 是 fight 形态整形力的透明账目（回归输出报告）。第 4 轮 P0-2 增 ff：
    // 守卫臂关节重力前馈用量（Σ|τ|·dt，N·m·s，applyFightGuard 写入处累加）。
    this.fightForceUsage = {
      A: { sx: 0, sy: 0, ax: 0, ay: 0, ff: 0 },
      B: { sx: 0, sy: 0, ax: 0, ay: 0, ff: 0 },
    };
    // FIGHT 守卫混合的一次性缓存（fight net meta 的关节序 → 守卫目标向量 +
    // 左右臂索引表），首次 _attachFightGuard 时构建（A/B 共用同一 fight net）。
    this._fightGuard = null;
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
    // 第二层闸门（审查建议）：fightMode 开着但本次挂载的栈没有 fight 槽
    // （回退后重开 / API 误用）——就地归一化回默认追踪语义，避免 fight
    // 短路分支撞上不存在的 guard 槽位形成混合语义。
    if (this.fightMode && !nets.fight) this.fightMode = false;
    f.trackingClips = nets;
    // 切片元数据（cut_boxing_clips.py 导出、extract_tracking_onnx.py 注入到
    // policy meta 的 motion.clip 字段）：有效出拳区间/出手窗口/来源。缺失时
    // 调度器回退到内建标定值。
    f.clipMeta = {};
    for (const [name, net] of Object.entries(nets)) {
      f.clipMeta[name] = net.meta?.motion?.clip ?? null;
    }
    // COMBO_MODE 且 combo 权重已部署：常驻循环从 combo 片段起步；FIGHT_MODE
    // 走单策略 fight 槽（168 维对打权重，boxing_ai 侧无片段库语义）；否则回退
    // 既有 guard 起步行为（combo=1 无权重时优雅降级，不破坏默认模式）。
    const startClip = this.fightMode && nets.fight ? 'fight'
      : this.comboMode && nets.combo ? 'combo' : 'guard';
    const startNet = nets[startClip];
    if (!startNet) throw new Error('tracking stack needs a ' + startClip + ' policy');
    // 重锚定角统一公式替换硬编码 π：初始时机器人 torso 在出生朝向（A≈0、
    // B≈π），guard ref0 yaw=0 → Δ_A≈0、Δ_B≈π，行为兼容。
    // COMBO 模式锁 delta=0（2026-09-29 实测裁决）：非零 yaw 重锚定对该策略
    // 致命（观测虽理论相消，实测 0.81 rad 偏移 1.5s 内倒地；delta=0 则 42s
    // 三回绕稳定）。默认 guard 路径仍用计算 delta（字节级行为不变）。
    // faceoff=1 时仅 B 侧例外（comboDelta：重锚定回出生朝向 π，配合
    // TrackingFighter faceoff 修正公式，_probe_faceoff 已验证 30s 站立）。
    // fight 走计算 delta（rebaseYawDelta）：出生朝向与 fight 参考起始帧 yaw
    // 对齐，启动/回绕/回合重置三处同一公式，anchor-ori 观测在重锚定瞬间归零。
    // 反镜像：B 机参考起始帧 = 本回合相位偏移（_fightPhase，A 恒 0）——训练
    // RSI 随机起始帧的确定性等价物，打破双机镜像锁定。
    const phase = startClip === 'fight' ? this._fightPhase[side] : 0;
    // fight + 伺服：启动 δ0 = bearing − 参考相位帧 yaw（出生位 ±0.6 面对面，
    // A bearing≈0 / B≈π，启动即面向对手）；伺服关闭回 rebaseYawDelta 旧基准
    // （当前朝向），A/B 基线路径字节不变。
    const delta = startClip === 'combo' ? this.comboDelta(side, startNet)
      : startClip === 'fight' && this.headingServo ? this.fightRebaseDelta(side, startNet, phase)
      : this.rebaseYawDelta(side, startNet, phase);
    // faceoff 修正公式（世界系 Rz 前乘）在 fight 下双侧启用：fight 重锚定角
    // 非零（B≈π、A=−ref0 yaw），且 fight ref0 基座可能带俯仰——遗留体轴后乘
    // 与 policyTick 观测旋转（Rz 前乘）不互逆，anchor-ori 观测会泄漏（
    // _probe_faceoff 实测 0.645 的同款机制）。δ=0 时两式恒等，旧路径不受影响。
    f.tracking = new TrackingFighter(this.mujoco, this.model, this.data, side, startNet,
      { yawOffset: delta, faceoff: this.fightMode || (this.faceoffMode && side === 'B'),
        opponent: f.opponent?.tracking ?? null,
        // P-A：fight 才带 ctrlrange 目标夹紧窗（guard/combo null = 不夹紧）
        ctrlRange: startClip === 'fight' ? FIGHT_CTRL_RANGE : null });
    f.tracking.reset(0);
    if (startClip === 'combo' || startClip === 'fight') {
      // 组合/对打片段帧 0 非场景关键帧书端：启动即原位重锚定到参考起始帧——
      // 训练回合起点的 RSI 等价物（sparring 训练 reset 同样由 RSI 摆位，见
      // sparring_env_cfgs.py：robot 出生由 MotionCommand RSI 覆盖）。
      // 默认 guard 路径保持原状（书端≈关键帧，字节级行为不变）。
      f.tracking.rebaseToReference(delta, phase);
      // FIGHT：B 机出生位抖动（次级反镜像扰动；内部守卫，combo/guard 空操作）
      this._applyFightSpawnJitter(side);
      // 出生间距隔离在 main.js 的 COMBO_MODE/FIGHT_MODE 场景加载替换中完成
      // （±1.2m / ±0.6m），setTracking 不再做额外推挤。
      this.mujoco.mj_forward(this.model, this.data);
    }
    // FIGHT 守卫混合钩子（非 fight / 回退片段栈不挂载，见 _attachFightGuard）
    this._attachFightGuard(f);
    f.clip = startClip;
    f.desiredClip = startClip;
    f.amoName = 'tracking';
    return f.tracking;
  }

  clearTracking(side) {
    const f = this.fighters[side];
    f.tracking = null;
    f.trackingClips = null;
    f.clipMeta = null;
    f.clip = null;
    f.desiredClip = null;
    f.punchIntent = null;
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
  // §4.2 距离标定：优先使用切片元数据的实测 punch_range_m（按拳法区分前冲
  // 距离），缺失时回退到内建标定窗口。
  punchRangeFor(f, clip) {
    const m = f.clipMeta?.[clip];
    if (m && Array.isArray(m.punch_range_m)) return m.punch_range_m;
    return PUNCH_RANGE[clip] ?? [0.82, 1.08];
  }

  // combo 槽位重锚定角（faceoff 子选项）：默认（含 faceoffMode=false）恒 0，
  // 与 2026-09-29 的"COMBO 锁 delta=0"裁决字节一致；faceoff=1 时仅 B 侧按
  // rebaseYawDelta 计算（出生朝向 π − combo ref0 yaw ≈ −2.33 rad，转身面向
  // A），A 侧保持 0。B 侧 fighter 同时携带 faceoff 标志走世界系 Rz 修正。
  comboDelta(side, net) {
    if (!this.faceoffMode || side !== 'B') return 0;
    return this.rebaseYawDelta(side, net);
  }

  // FIGHT 反镜像：按回合序号更新 B 机片段起始帧偏移（A 恒 0）。boot（构造
  // 后 _roundNo=0）与每次 resetRound 调用；表长 4 循环，确定性可复现。
  _assignFightPhase() {
    this._fightPhase.B = FIGHT_PHASE_OFFSETS[this._roundNo % FIGHT_PHASE_OFFSETS.length];
  }

  // FIGHT 反镜像次级扰动：B 机出生位抖动（±0.02m 平移 + ±0.02rad yaw，按回
  // 合序号循环）。必须在 rebaseToReference 之后、mj_forward 之前调用（rebase
  // 保留 keyframe 出生 x/y，抖动在其上叠加）。三重守卫：fightMode + fight 槽
  // + 仅 B 机——combo/default/A 侧零触碰。yaw 用世界系 Rz 前乘（wxyz
  // Hamilton，与 rebaseToReference 的 _faceoffFix 修正式同约定）。
  _applyFightSpawnJitter(side) {
    if (!this.fightMode || side !== 'B') return;
    const f = this.fighters[side];
    if (!f.tracking || !f.trackingClips?.fight) return;
    const j = FIGHT_SPAWN_JITTER[this._roundNo % FIGHT_SPAWN_JITTER.length];
    const q = f.tracking.freeQpos, d = this.data;
    d.qpos[q + 0] += j.dx;
    d.qpos[q + 1] += j.dy;
    if (j.yaw) {
      const zw = Math.cos(j.yaw / 2), zz = Math.sin(j.yaw / 2);
      const w = d.qpos[q + 3], x = d.qpos[q + 4], y = d.qpos[q + 5], z = d.qpos[q + 6];
      d.qpos[q + 3] = zw * w - zz * z;
      d.qpos[q + 4] = zw * x - zz * y;
      d.qpos[q + 5] = zw * y + zz * x;
      d.qpos[q + 6] = zw * z + zz * w;
    }
  }

  // 重锚定角 Δ = 当前机器人 torso 实际 yaw − 目标片段参考起始帧 torso yaw
  // （startStep 默认 0 = 既有 ref0 语义；fight 相位偏移传非零帧，使重锚定
  // 瞬间 anchor-ori 观测归零的基准帧与 rebaseToReference 的起始帧一致）。
  // 各片段烘焙的世界朝向不一致（guard 0° / cross −15.3° / jab、hook
  // −161.8°），调度器不能假设所有片段共享世界约定。Δ 同时用于旋转参考
  // （yawOffset）与原位重锚定（TrackingFighter.rebaseToReference）。yaw 必
  // 须走 yawOfQuat（wxyz 四元数，归一化 (−π,π]）；需在 mj_forward / 仿真步
  // 进之后调用（xquat 就绪）。
  rebaseYawDelta(side, net, startStep = 0) {
    const tb = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_BODY.value, side + '_torso_link');
    if (tb < 0) throw new Error('torso body not found: ' + side);
    const tq = this.data.xquat, o = 4 * tb;
    const cur = yawOfQuat(tq[o], tq[o + 1], tq[o + 2], tq[o + 3]);
    const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
    const r = net.refAt(startStep).body_quat_w, ro = 4 * ai;
    const ref = yawOfQuat(r[ro], r[ro + 1], r[ro + 2], r[ro + 3]);
    let delta = cur - ref;
    delta = ((delta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return delta;
  }

  // ---- FIGHT 航向伺服（H1/H2 修复，2026-10-01）-----------------------------
  // 缺陷：fight 参考片段 torso yaw 0.6s 内摆动 ±54°（峰峰 ~108°），机器人航向
  // 被 motion_anchor_ori_b 观测锁在"参考 yaw + 常数 δ"上——反镜像相位表让
  // B 机播错开相位的参考，双机各跟各的参考转身（按相位表开环计算互面误差平
  // 均 98-102°、~73% 时间 >90°）；且 fight 无"转向对手"环路，rebaseYawDelta
  // 以"当前朝向"为基准，朝向误差跨片段回绕永久保留（跟空气打）。
  //
  // 修复：维护平滑跟踪的 δ(t)，目标 δ*(t) = bearing(t) − refYaw(t)，即让
  // "参考当前帧 anchor 朝向 + δ"恒等于"指向对手的方位角"。一致性约束：δ 同
  // 时驱动 (a) 观测侧 policyTick 的 _qz 参考旋转（TrackingFighter.
  // setHeadingOffset）与 (b) 状态侧（re）锚定的 rebaseToReference Rz(Δ)——
  // 两侧共用同一 δ，观测与目标自洽（等价于把参考姿态整体绕机器人竖直轴旋转
  // δ，关节角不变）。方案中"参考基座位置 x/y 偏移同步旋转"一项经核为空操作：
  // 本 ONNX 导出无 anchor_pos 观测（has_state_estimation=False，obs 布局见
  // tracking_policy.mjs 头注），opponent_state 全部为 body-frame 相对量，
  // rebaseToReference 保留世界 x/y 不取参考位置——参考基座 x/y 从不进入任何
  // 观测，无需旋转。
  // δ 不做阶跃：回合内指数平滑 δ ← δ + (δ*−δ)·(1−e^(−dt/τ))，τ=headingTau
  // （默认 0.3s）；（re）锚定边界（setTracking/swapTo/resetRound）δ 直接初
  // 始化为 bearing − 参考相位帧 yaw，状态同刻被 RSI 到该朝向——边界时刻立即
  // 面向对手。伺服关闭（A/B 基线）时所有路径回 rebaseYawDelta 旧语义，字节
  // 不变。AMO 的 bearing 环路（update 的 f.amo 分支）与单人 tracking 模式不
  // 经过本组方法。

  // 方位角：self torso → 对手 torso 的水平 atan2（方案 §A：用双方 torso 世界
  // 位置，非骨盆出生锚）。body id 走构造器缓存 _torsoBody。
  fightBearing(side) {
    const m = this.data.xpos;
    const a = 3 * this._torsoBody[side];
    const b = 3 * this._torsoBody[this.fighters[side].opponent.side];
    return Math.atan2(m[b + 1] - m[a + 1], m[b] - m[a]);
  }

  // 锚定边界 δ0 = bearing − 参考起始帧 anchor torso yaw，归一化 (−π,π]。
  // 与 rebaseYawDelta 同形，仅"当前 torso 实际 yaw"换成"指向对手的 bearing"
  // ——时序约定相同：需在 mj_forward / 仿真步进之后调用（xpos/xquat 就绪）。
  fightRebaseDelta(side, net, startStep = 0) {
    const bearing = this.fightBearing(side);
    const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
    const r = net.refAt(startStep).body_quat_w, ro = 4 * ai;
    const ref = yawOfQuat(r[ro], r[ro + 1], r[ro + 2], r[ro + 3]);
    let delta = bearing - ref;
    delta = ((delta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return delta;
  }

  // 回合内伺服目标 δ* = bearing − 参考当前帧（tracking.timeStep = 下一次
  // policyTick 将消费的参考帧，与观测同帧自洽）anchor yaw。refAt 对越界帧
  // 夹紧，与 policyTick 看到的帧一致。
  fightServoTarget(side) {
    const f = this.fighters[side];
    const net = f.tracking.net;
    const bearing = this.fightBearing(side);
    const ai = net.meta.body_names.indexOf(net.meta.anchor_body_name);
    const r = net.refAt(f.tracking.timeStep).body_quat_w, ro = 4 * ai;
    const ref = yawOfQuat(r[ro], r[ro + 1], r[ro + 2], r[ro + 3]);
    let delta = bearing - ref;
    delta = ((delta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return delta;
  }

  // 每物理步伺服推进：δ* 先归一化，再取与当前 δ 最近的代表（bearing 与
  // refYaw 各自在 ±π 回绕时 δ* 可能跳 2π，不取近代表会让机器人反向转一整
  // 圈）；δ 本身不归一化——四元数对幅值不敏感，保连续性。平滑只作用回合
  // 内，不做角度阶跃（参考片段自身 yaw 变化率 ±3 rad/s，τ=0.3s 平滑覆盖该
  // 带宽且滞后有限）。
  updateFightHeadingServo(f, dt) {
    const tf = f.tracking;
    const target = this.fightServoTarget(f.side);
    let err = target - tf.yawOffset;
    err = ((err + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    const alpha = 1 - Math.exp(-dt / this.headingTau);
    tf.setHeadingOffset(tf.yawOffset + err * alpha);
  }

  // ---- FIGHT 守卫混合（抱架保持，2026-10-02）--------------------------------
  // 消除"双臂垂下/双臂对称前推"：非出拳手臂的 PD 目标向高抱架目标混合
  // （第 2 轮 P0-2 起为两级结构——覆写边界按持久水平重混 + 窗口内指数微调，
  // 见 applyFightGuard 注释与常量区），出拳瞬间该臂交还策略。挂在
  // TrackingFighter.targetHook 上（policyTick 之后、writeTorque 之前，见
  // tracking_policy.mjs），guard/combo/非 fight 路径不挂载、字节不变。
  // setTracking/swapTo/resetRound 三处 TrackingFighter 重建点后各调一次
  // （divergence 自愈路径复用同一对象，钩子随对象存活，无需重挂）。
  _attachFightGuard(f) {
    if (!this.fightMode || !f.tracking || !f.trackingClips?.fight) return;
    if (f.tracking.targetHook) return;
    if (!this._fightGuard) {
      const names = f.tracking.net.meta.joint_names;
      const targets = new Float32Array(names.length);
      const armIdx = { left: [], right: [] };
      const ffIdx = { left: [], right: [] };
      const ffVal = new Float32Array(names.length);
      names.forEach((nm, i) => {
        const g = FIGHT_GUARD_JOINTS[nm];
        if (g !== undefined) {
          targets[i] = g;
          armIdx[nm.startsWith('left_') ? 'left' : 'right'].push(i);
        }
        const k = FIGHT_GUARD_FEEDFORWARD[nm];
        if (k) {
          ffVal[i] = k;
          ffIdx[nm.startsWith('left_') ? 'left' : 'right'].push(i);
        }
      });
      this._fightGuard = { targets, armIdx, ffIdx, ffVal };
    }
    f.tracking.targetHook = () => this.applyFightGuard(f);
    // 前馈加性通道（见 FIGHT_GUARD_FEEDFORWARD 与 tracking_policy.mjs
    // extraTorque 注释）：per-fighter 分配，索引与 pdTarget/writeTorque 同序。
    if (!f.tracking.extraTorque) f.tracking.extraTorque = new Float32Array(f.tracking.n);
  }

  // 逐物理步守卫混合（见 _attachFightGuard 注释、文件头 FIGHT 视觉形态修复
  // 小节与常量区第 2/4 轮 P0-2 说明）。两级结构：
  //  - 持久混合水平：非出拳臂每步向 FIGHT_GUARD_MIX_MAX 指数回升
  //    （τ=FIGHT_GUARD_MIX_TAU）；出拳判定（该臂拳速 EMA > FIGHT_PUNCH_SPEED，
  //    真出拳 2-4 m/s，出手瞬间即放开门禁）时清零——该臂完全交还策略，收拳
  //    后 ~0.23s 水平回升到 90%（τ=0.10，第 3 轮），把拳套拉回抱架。
  //  - 应用：policyTick 覆写边界（stepCount % FIGHT_POLICY_DECIMATION === 0，
  //    钩子在覆写后、stepCount 自增前被调用，见 physicsStep）按当前水平一次性
  //    重混——水平不随覆写窗口归零，破"每窗口从头爬坡"瓶颈；窗口内在边界
  //    结果上继续 τ=FIGHT_GUARD_TAU 的指数微调。
  //  - 重力前馈（第 4 轮 P0-2，FIGHT_GUARD_FEEDFORWARD）：臂关节 8 项加性
  //    力矩经 extraTorque 通道在 writeTorque 限幅前叠加，τ = k × guardMix
  //    与混合水平同源缩放、无独立滞后。本钩子每物理步先清后写（出拳分支
  //    清零该臂、早退路径整表清零），writeTorque 只消费本步刚写的值——
  //    出拳瞬间前馈严格为零，无残留。用量入 fightForceUsage.ff。
  // 倒地（骨盆 z<0.42）与策略不健康（NaN）时不混合，不干扰摔倒/自愈过程。
  applyFightGuard(f) {
    const tf = f.tracking;
    const ff = tf && tf.extraTorque;
    if (!tf || !tf.healthy) { if (ff) ff.fill(0); return; }
    const d = this.data;
    const pb = f.side === 'A' ? this.ids.pelvisA : this.ids.pelvisB;
    if (d.xpos[3 * pb + 2] < 0.42) { if (ff) ff.fill(0); return; }
    const g = this._fightGuard;
    if (!g) return;   // 未挂载时 extraTorque 亦为 null，无残留
    const pd = tf.pdTarget;
    const atTick = tf.stepCount % FIGHT_POLICY_DECIMATION === 0;
    const alpha = 1 - Math.exp(-this.model.opt.timestep / FIGHT_GUARD_TAU);
    const rise = 1 - Math.exp(-this.model.opt.timestep / FIGHT_GUARD_MIX_TAU);
    const u = this.fightForceUsage[f.side];
    const dt = this.model.opt.timestep;
    for (const hand of ['left', 'right']) {
      for (const i of g.armIdx[hand]) ff[i] = 0;   // 先清整臂（含腕），再按比例写
      if (f.fistSpeed[hand] > FIGHT_PUNCH_SPEED) {   // 出拳中：交还策略并清偏置
        f.guardMix[hand] = 0;
        continue;                                    // 前馈已随上方清零
      }
      const m = (f.guardMix[hand] += (FIGHT_GUARD_MIX_MAX - f.guardMix[hand]) * rise);
      if (atTick) {
        for (const i of g.armIdx[hand]) pd[i] += m * (g.targets[i] - pd[i]);
      } else if (m > 0) {
        for (const i of g.armIdx[hand]) pd[i] += alpha * (g.targets[i] - pd[i]);
      }
      if (m > 0) {
        for (const i of g.ffIdx[hand]) {
          ff[i] = g.ffVal[i] * m;
          u.ff += Math.abs(ff[i]) * dt;
        }
      }
    }
  }

  // §4.4/§4.5 追踪模式下该拳手当前是否处于有效出拳阶段：优先用元数据的
  // 出手窗口（秒，从片段起始计），缺失时只要有非守卫片段在播就算。
  punchActive(f) {
    // FIGHT_MODE：策略自主出拳，无片段/出手窗口语义——拳套接触一律进入有效
    // 命中判定（真实出拳门槛 HIT_MIN_FIST_SPEED 仍在 registerHit 把关，避免
    // 贴身挤压漂移接触刷分）。
    if (this.fightMode) return !!f.tracking;
    if (!f.tracking || f.clip === 'guard' || !PUNCH_CLIP[f.clip]) return false;
    const m = f.clipMeta?.[f.clip];
    if (m && Array.isArray(m.strike_window_s)) {
      const t = f.tracking.timeStep / (f.tracking.net.meta.motion.fps || 50);
      return t >= m.strike_window_s[0] && t <= m.strike_window_s[1];
    }
    return true;
  }

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

    // ---- tracking mode: 决策只排队下一个片段，运动层在守卫边界执行 ----
    // §4.4 队列纪律：
    //  - 片段播放中：队列锁死，不打断（出拳发力段不允许强行中断）；
    //  - 出拳意图挂起：直接返回（不重新采样、不覆盖），由 update() 在距离
    //    合适时执行；超过 TTL 仍未执行则取消，防止拳法请求长期阻塞移动；
    //  - 空闲：采样一次决策，重置决策计时（修复：原先到达空闲阈值后每个
    //    物理步都重新采样）。
    if (f.tracking) {
      // FIGHT_MODE：对打无脚本编排（需求 F-7）——出拳/逼近由策略权重自主
      // 决策，决策层只维持常驻 fight 片段，不采样战术、不排队拳法意图。
      if (this.fightMode) {
        f.desiredClip = f.clip;
        return true;
      }
      // COMBO_MODE：组合循环常驻——跳过 tactics 采样与拳法门禁，desiredClip
      // 恒为 combo（运动层在片段回绕处消费；无 combo 权重时守卫回退）。
      if (this.comboMode) {
        f.desiredClip = f.trackingClips.combo ? 'combo' : 'guard';
        f.punchIntent = f.desiredClip === 'combo' ? { clip: 'combo', t: this.time } : null;
        return true;
      }
      if (f.clip !== 'guard') return true;
      if (f.desiredClip && f.desiredClip !== 'guard') {
        if (f.punchIntent && this.time - f.punchIntent.t > PUNCH_INTENT_TTL) {
          f.desiredClip = 'guard';
          f.punchIntent = null;    // 意图过期：取消（§4.4）
        }
        return true;
      }
      const obs = this.buildTacticsObs(f.side);
      const { action } = pol.act(obs, { sample: true, temp: pol.tacticsTemp ?? this.tacticsTemp, rng: f.rng });
      if (f.traj) f.traj.push({ t: this.time, obs: Array.from(obs), action });
      const name = TACTIC_ACTIONS[action];
      let clip = TACTIC_CLIP[name] ?? 'guard';
      // 追踪模式过渡语义（§4.2 缺位期）：步法决策全部映射原地守卫片段，
      // "接近"没有可执行语义——在 jab 距离带内按先验本意（band 边缘用刺拳
      // 试探）转为出拳，避免步法决策把出拳饿死。步法片段训练落地后移除。
      const jr = this.punchRangeFor(f, 'jab');
      if (!PUNCH_CLIP[clip] && (name === 'approach' || name === 'wait') &&
          dist >= jr[0] && dist <= jr[1]) {
        clip = 'jab';
      }
      // 拳法门禁（临时）：未启用片段映射回 guard，意图不入队
      if (!CLIP_ENABLED[clip]) clip = 'guard';
      if (PUNCH_CLIP[clip]) {
        f.desiredClip = clip;
        f.punchIntent = { clip, t: this.time };
      } else {
        f.desiredClip = 'guard';   // 步法决策不携带出拳请求
        f.punchIntent = null;
      }
      f.stateT = 0;
      f.idleFor = (0.25 + f.rng() * 0.3) / this.aggression;  // 限频：~2-3 决策/秒
      return true;
    }

    const obs = this.buildTacticsObs(f.side);
    const { action } = pol.act(obs, { sample: true, temp: pol.tacticsTemp ?? this.tacticsTemp, rng: f.rng });
    if (f.traj) f.traj.push({ t: this.time, obs: Array.from(obs), action });
    const name = TACTIC_ACTIONS[action];
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
        f.stateT = 0; f.idleFor = 0.25 + f.rng() * 0.35; break; // hold guard, re-decide soon
      default: { // attack family -> concrete combo (side randomized for variety)
        const opts = ACTION_COMBOS[name] ?? [0];
        f.combo = COMBOS[opts[Math.floor(f.rng() * opts.length)]];
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
    // §4.6 KO 断力矩必须覆盖该侧全部执行器：追踪场景 29 自由度含每臂 3 个
    // 腕部执行器（共 6 个），只清 POLICY_JOINTS 的 23 个会让手腕保留最后的
    // PD 力矩，KO 塌倒姿态不自然且可能撑住身体。
    const f = this.fighters[side];
    const list = f.tracking ? f.tracking.actId : this.actIdx[side];
    for (let i = 0; i < list.length; i++) this.data.ctrl[list[i]] = 0;
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
        f.punchIntent = null; f.punchEventId = 0; f.lastScoredPunch = -1;
        f.noPunchUntil = this.time + PUNCH_SETTLE_T;  // 回合稳定期
        f.fistSpeed = { left: 0, right: 0 };
        f._fistPrev = { left: null, right: null, t: null };
        if (f.amo) f.amo.reset();
        if (f.tracking) {
          // COMBO_MODE：引擎时间回退的自愈重置也保持 combo 常驻片段；
          // FIGHT_MODE 同理保持 fight 常驻片段（时钟回本回合相位帧，不回 0
          // ——否则自愈一次就与 A 重新镜像同步）
          const clip = this.fightMode && f.trackingClips.fight ? 'fight'
            : this.comboMode && f.trackingClips.combo ? 'combo' : 'guard';
          f.tracking.reset(clip === 'fight' ? this._fightPhase[side] : 0);
          f.clip = clip; f.desiredClip = clip;
        }
      }
      this.knockback = { A: null, B: null };
      this.koState = null;
      this._roundT = 0;   // fight 60s 回合计时一并归零（引擎时间已回退，重开回合）
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

    // FIGHT_MODE（Q6）：60s 单回合循环——超时即重开（无局数上限，比分累计
    // 在 scores）。KO 摔倒路径经上方 koState 结算后同样回到 resetRound；
    // _roundT 在 resetRound 归零。非 fight 模式不进此分支（零影响）。
    if (this.fightMode) {
      this._roundT += dt;
      if (this._roundT >= FIGHT_ROUND_S) {
        this.resetRound();
        return;
      }
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
            // learned tactic handled the decision (tracking: queue + reset timer;
            // AMO/scripted: state transition)
          } else if (f.tracking) {
            // tracking mode without a tactics policy: weighted-random fallback
            // queues clips directly (previously the tracking robot never punched)
            if (this.fightMode) {
              // fight 短路：无战术权重时也维持常驻 fight 片段（同 decideTactics
              // 的 fight 分支——对打无脚本编排，不排队拳法）
              f.desiredClip = f.clip;
              f.stateT = 0;
            } else if (this.comboMode) {
              // COMBO_MODE 短路：不采样，直接常驻 combo（同 decideTactics）
              f.desiredClip = f.trackingClips.combo ? 'combo' : 'guard';
              f.punchIntent = f.desiredClip === 'combo' ? { clip: 'combo', t: this.time } : null;
              f.stateT = 0;
              f.idleFor = (0.25 + f.rng() * 0.3) / this.aggression;
            } else {
              const pick = weightedPick(COMBO_POOL, f.rng);
              let clip = TACTIC_CLIP[pick] ?? 'guard';
              if (!CLIP_ENABLED[clip]) clip = 'guard';   // 拳法门禁（临时）
              f.desiredClip = clip;
              f.punchIntent = PUNCH_CLIP[clip] ? { clip, t: this.time } : null;
              f.stateT = 0;
              f.idleFor = (0.25 + f.rng() * 0.3) / this.aggression;
            }
          } else if (dist > 0.85) { f.state = 'approach'; f.stateT = 0; f.phaseDur = 1.4; }
          else if (dist < 0.42) { f.state = 'retreat'; f.stateT = 0; f.phaseDur = 0.5; }
          else {
            const pick = weightedPick(COMBO_POOL, f.rng);
            const opts = POOL_MAP[pick];
            f.combo = COMBOS[opts[Math.floor(f.rng() * opts.length)]];
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
        if (f.phaseT >= f.phaseDur) { f.state = 'strike'; f.phaseT = 0; f.phaseDur = 0.22; f.punchEventId++; }
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

      // §4.5 拳速跟踪（短 EMA，时间常数 ~30ms）：有效命中判定用——真出拳
      // 2-4 m/s，贴身挤压/漂移接触远低于此。
      {
        const P = d.site_xpos;
        const fL = side === 'A' ? this.ids.fistLA : this.ids.fistLB;
        const fR = side === 'A' ? this.ids.fistRA : this.ids.fistRB;
        for (const [hand, sid] of [['left', fL], ['right', fR]]) {
          const p = [P[3 * sid], P[3 * sid + 1], P[3 * sid + 2]];
          const prev = f._fistPrev[hand];
          const pdt = f._fistPrev.t === null ? 0 : this.time - f._fistPrev.t;
          if (prev && pdt > 1e-5 && pdt < 0.2) {
            const v = Math.hypot(p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]) / pdt;
            f.fistSpeed[hand] = 0.7 * f.fistSpeed[hand] + 0.3 * v;
          }
          f._fistPrev[hand] = p;
        }
        f._fistPrev.t = this.time;
      }

      // §4.2 持续面向对手：方位角指令平滑跟踪对手实际方位（限转速），替换
      // 固定 0/π——接触漂移把双机推离初始连线后，旧指令会让机器人不再面向
      // 对手。追踪模式不喂朝向指令（片段参考拥有航向，新指令必须先训练）。
      if (f.amo) {
        const bearing = Math.atan2(op[1] - me[1], op[0] - me[0]);
        let dyaw = bearing - f.faceYaw;
        dyaw = ((dyaw + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        const maxStep = FACE_YAW_RATE * dt;
        f.faceYaw += clamp(dyaw, -maxStep, maxStep);
      }

      // ---- drive the robot ----
      if (f.tracking) {
        // stage-4 motion layer: play the current clip; swap at its guard
        // boundary. All clips bookend with the SAME guard stance. Switches use
        // in-place rebase RSI (swapTo): joint/base state snaps to the new
        // clip's frame-0 reference re-anchored at the robot's current heading —
        // the exact state a training episode starts from. Clips bake different
        // world yaws (guard 0° / cross −15.3° / jab、hook −161.8°)，不重锚定的
        // 话 anchor-ori 观测在切换瞬间带恒定 162-198° 误差，策略分布外摔倒。
        const swapTo = (want) => {
          // Δ 同时驱动参考旋转（yawOffset）与机器人状态重锚定，两者相消使
          // 切换瞬间 anchor-ori 观测归零。lastAction/pdTarget 归零而非接续
          // prev：接续会把前一片段末端动作带进分布外起点，与训练回合起点
          // 不一致（守卫循环内的 guard→guard 延续走 carryOver，不经此处）。
          // COMBO 模式锁 delta=0（同 setTracking，实测非零 delta 致倒）。
          // faceoff=1 时仅 B 侧例外（comboDelta），新 fighter 同步携带 faceoff
          // 修正标志，回绕重锚定与观测旋转保持世界系 Rz 同约定。
          // fight 回绕（伺服关闭）：rebaseYawDelta（当前朝向−参考起始帧 yaw），
          // 起始帧 = 本回合相位偏移——回绕不重置相位，否则与 A 重新镜像同步。
          // 伺服开启时改为 fightRebaseDelta（bearing−参考相位帧 yaw），见下。
          const phase = want === 'fight' ? this._fightPhase[side] : 0;
          // fight + 伺服：回绕重锚定 δ0 = bearing − 参考相位帧 yaw——回绕边
          // 界立即面向对手（H2：旧 rebaseYawDelta 以当前朝向为基准，朝向误差
          // 跨回绕永久保留）。伺服关闭回旧路径。
          const delta = want === 'fight' && this.headingServo
            ? this.fightRebaseDelta(side, f.trackingClips[want], phase)
            : this.comboMode ? this.comboDelta(side, f.trackingClips[want]) : this.rebaseYawDelta(side, f.trackingClips[want], phase);
          const nf = new TrackingFighter(this.mujoco, this.model, this.data, side,
            f.trackingClips[want], { yawOffset: delta,
              faceoff: this.fightMode || (this.faceoffMode && side === 'B'),
              opponent: f.opponent?.tracking ?? null,
              ctrlRange: want === 'fight' ? FIGHT_CTRL_RANGE : null });
          nf.rebaseToReference(delta, phase);
          this.mujoco.mj_forward(this.model, this.data);
          f.tracking = nf;
          this._attachFightGuard(f);   // fight 回绕重建后重挂守卫混合钩子
          f.clip = want;
          f.clipSwaps++;
          f.desiredClip = 'guard';  // 队列已消费（无论打没打）
          if (PUNCH_CLIP[want]) f.punchEventId++;
          else f.noPunchUntil = this.time + PUNCH_COOLDOWN;  // 出拳后冷却（§4.4 节奏）
          // （fight 的 'fight' 槽不在 PUNCH_CLIP：走 else 分支设 noPunchUntil，
          // 该计时在 fight 路径无消费者——回绕分支与 punchActive 均不看它，
          // 命中控频由 registerHit 的 hitCooldown 承担。）
          f.punchIntent = null;
        };
        const carryOver = () => {
          // guard→guard 循环：原地重启参考时钟（动作状态本就延续，无需拷贝；
          // 历史版本的 prev 自拷贝 set 对同一对象是无操作，已删）
          f.tracking.reset(0);
        };
        if (f.tracking.clipDone) {
          // FIGHT_MODE：单策略常驻循环（训练 10s episode，参考播完即回绕）。
          // 回绕 = 原位重锚定 RSI（swapTo 内 rebaseYawDelta：当前朝向−ref0
          // yaw，重锚定瞬间 anchor-ori 观测归零，与守卫路径切换同语义）；
          // 不走拳法距离门禁——对打出拳由策略权重自主驱动，无脚本编排。
          if (this.fightMode && f.trackingClips.fight) {
            swapTo('fight');
            f.desiredClip = 'fight';   // 常驻循环：队列不被消费
          } else if (this.comboMode) {
            // COMBO_MODE：combo 播到 max_step 后回绕同片段——swapTo 内部走
            // rebaseToReference（重锚定回片段第 0 帧，消除累积漂移，回绕瞬间
            // 允许 1 帧内姿态微跳），跳过拳法距离门禁与冷却。
            const want = f.trackingClips.combo ? 'combo' : 'guard';
            swapTo(want);
            if (want === 'combo') f.desiredClip = 'combo';  // 常驻循环：队列不被消费
          } else {
          if (f.clip !== 'guard' && f.desiredClip === f.clip) f.desiredClip = 'guard';
          let want = f.desiredClip && f.trackingClips[f.desiredClip] ? f.desiredClip : 'guard';
          if (want !== 'guard') {
            const r = this.punchRangeFor(f, want);
            // 距离不合适或处于稳定期/冷却：本轮不出拳（§4.2/§4.4）
            if (!(dist >= r[0] && dist <= r[1]) || this.time < f.noPunchUntil) want = 'guard';
            // （允许同时出拳：互相前倾才够得着对方的头——见 PUNCH_RANGE 标定）
          }
          if (want === 'guard' && f.clip === 'guard') carryOver();
          else swapTo(want);
          }
        } else if (f.clip === 'guard' && f.desiredClip && f.desiredClip !== 'guard' &&
                   f.trackingClips[f.desiredClip] && f.punchIntent &&
                   PUNCH_EARLY_EXIT[f.desiredClip]) {
          // §4.4 缩短守卫等待：出拳意图挂起且距离合适时，参考姿态一回到共享
          // 守卫站架（书端静态帧/弹跳过零点，误差 ≤0.035 rad）就提前切入，
          // 不必等整个守卫循环播完（守卫片段 ~9s，旧逻辑最坏 9s 延迟）。
          // 出拳片段自带的守卫书端 + 过渡段就是"准备"阶段，衔接天然连续。
          const r = this.punchRangeFor(f, f.desiredClip);
          if (dist >= r[0] && dist <= r[1] && this.time >= f.noPunchUntil &&
              nearGuardPose(f.tracking.net, f.tracking.timeStep, f.trackingClips[f.desiredClip])) {
            swapTo(f.desiredClip);
          }
        }
        // FIGHT 航向伺服（H1/H2 修复）：回合内 δ 指数平滑跟踪"参考帧朝向+δ
        // = 指向对手"目标，观测侧（_qz 参考旋转）与状态侧（下次重锚定 δ0）
        // 共用同一 δ——见 updateFightHeadingServo 注释。仅 fight 常驻片段生
        // 效；AMO bearing 环路（上方 f.amo 分支）与单人 tracking 模式不经此处。
        if (this.fightMode && this.headingServo && f.clip === 'fight') {
          this.updateFightHeadingServo(f, dt);
        }
        f.tracking.physicsStep(FIGHT_POLICY_DECIMATION);
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
      if (this.assist && !f.amo && !f.down &&
          ((f.tracking && this.comboMode) || this.trackingAssist || !f.tracking)) {
        if (!f._ppos) f._ppos = [px, py, pz];
        const vxx = (px - f._ppos[0]) / dt, vyy = (py - f._ppos[1]) / dt, vzz = (pz - f._ppos[2]) / dt;
        f._ppos = [px, py, pz];
        if (f.tracking) {
          if (!this.comboMode) {
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
            tx = clamp(150 * R[o + 5] - 10 * wx, -70, 70);
            ty = clamp(-150 * R[o + 2] - 10 * wy, -70, 70);
            tz = clamp(-8 * wz, -15, 15);
            // 水平阻尼：互刺时上身碰撞的冲量是缠倒主因，阻尼抵消部分推挤
            if (f.clip !== 'guard') {
              fx += clamp(-90 * vxx, -55, 55);
              fy += clamp(-90 * vyy, -55, 55);
            }
          }
          // 缠斗分离：punch 片段只拉近距离（cross 前冲 0.8m），守卫片段原地——
          // 没有东西能拉开距离，贴身后必然缠倒。距离 <0.55m 时给双方一个
          // 温和的分离力，等效裁判分开缠斗，让下一轮出拳在有效距离发起。
          // plan-revamp v2c（2026-09-29 队长预授权）：COMBO_MODE 下对 tracking
          // 机器人仅启用此分离项（双机对练 1m 间距互殴是双机闸门唯一死因，
          // 单机全段已 PASS），其余辅助仍关断。
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
      const xfo = 6 * b;
      // FIGHT 裁判分离力 + 距离保持力（2026-10-02，见文件头 FIGHT 视觉形态
      // 修复小节）：与平衡辅助无关（assist 关闭仍生效）——分离力消除"贴脸
      // 缠抱"（等效裁判分开缠斗），距离保持力在过远时温和拉近，制造可感知
      // 的进退步法。仅 fight 模式、双方站立（骨盆 z≥0.40 与 KO 阈同口径）
      // 时生效；方向 = 双方骨盆水平连线。
      if (this.fightMode && !f.down) {
        const opb = f.opponent.side === 'A' ? this.ids.pelvisA : this.ids.pelvisB;
        const oz = d.xpos[3 * opb + 2];
        if (pz >= 0.40 && oz >= 0.40) {
          const dx = px - d.xpos[3 * opb], dy = py - d.xpos[3 * opb + 1];
          const dd = Math.hypot(dx, dy) || 1e-6;
          let ffx = 0, ffy = 0;
          if (dd < FIGHT_SEP_DIST) {
            const s = clamp(FIGHT_SEP_K * (FIGHT_SEP_DIST - dd), 0, FIGHT_SEP_FMAX);
            ffx = s * dx / dd; ffy = s * dy / dd;
            const u = this.fightForceUsage[side];
            u.sx += Math.abs(ffx) * dt; u.sy += Math.abs(ffy) * dt;
          } else if (dd > FIGHT_APPR_DIST) {
            const s = clamp(FIGHT_APPR_K * (dd - FIGHT_APPR_DIST), 0, FIGHT_APPR_FMAX);
            ffx = -s * dx / dd; ffy = -s * dy / dd;
            const u = this.fightForceUsage[side];
            u.ax += Math.abs(ffx) * dt; u.ay += Math.abs(ffy) * dt;
          }
          fx += ffx; fy += ffy;
        }
      }
      if (!Number.isFinite(fx + fy + fz + tx + ty + tz)) continue;
      // §4.6 辅助用量计量：对 |F|、|τ| 做时间积分（N·s / N·m·s）。放在有限性
      // 检查之后（NaN 不得污染计量），击退力在此之后才叠加，不计入——那是
      // 展示效果，不是平衡辅助。
      {
        const u = this.assistUsage[side];
        u.fx += Math.abs(fx) * dt; u.fy += Math.abs(fy) * dt; u.fz += Math.abs(fz) * dt;
        u.tx += Math.abs(tx) * dt; u.ty += Math.abs(ty) * dt; u.tz += Math.abs(tz) * dt;
      }
      const kb = this.knockback[side];
      if (kb) {
        if (this.time > kb.until) this.knockback[side] = null;
        else { fx += kb.f[0]; fy += kb.f[1]; }
      }
      d.xfrc_applied[xfo] = fx; d.xfrc_applied[xfo + 1] = fy; d.xfrc_applied[xfo + 2] = fz;
      d.xfrc_applied[xfo + 3] = tx; d.xfrc_applied[xfo + 4] = ty; d.xfrc_applied[xfo + 5] = tz;
    }

      if ((this.fighters.A.tracking || this.fighters.B.tracking) && !this.comboMode) {
        // 僵局检测：接触漂移把双机推远后没有任何片段能拉近距离——
        // 距离 >1.9m 持续 3s 就重开回合（回到标准 1.0m 站位）。
        // COMBO_MODE 下禁用（D8：无回合重置，N1 连续观察）。
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
      // COMBO_MODE（D8）：禁用伤害 KO 注入——组合循环连续观察不重置回合
      if (!this.comboMode && this.damage[side] >= 7) {
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
    const f = this.fighters[attacker];
    const av = this.fighters[victim];
    // §4.5 命中判定：出拳手段处于发力阶段 + 拳速达标才算有效命中；否则记为
    // 轻触（graze，不计分/不伤害/不击退）——区分轻触、贴身挤压与有效出拳。
    const attacking = f.tracking ? this.punchActive(f)
      : (f.state === 'strike' || f.state === 'windup');
    const speed = Math.max(f.fistSpeed.left, f.fistSpeed.right);
    if (!attacking || speed < HIT_MIN_FIST_SPEED) {
      this.hitCooldown[attacker] = this.time + 0.15;  // 抑制 graze 连发刷屏
      this.events.push({ type: 'graze', attacker, victim, t: this.time });
      return;
    }
    // 以出拳事件为单位去重：同一次出拳（片段执行/组合步）只计一分，
    // 贴身持续接触无法靠冷却窗口反复得分。
    // （fight 模式例外：无出拳事件语义——punchEventId 在常驻 fight 片段里
    // 不递增，按事件去重会一场只记一分；控频交给上一行的 0.45s hitCooldown。）
    if (!this.fightMode && f.lastScoredPunch === f.punchEventId) return;
    this.hitCooldown[attacker] = this.time + 0.45;
    f.lastScoredPunch = f.punchEventId;
    const points = this.headGeoms[victim].has(vulGeom) ? 2 : 1;
    const kind = points === 2 ? 'head' : 'body';
    f.scores.hits += 1;
    f.scores.points += points;
    this._pendingHits++;
    this.events.push({ type: 'hit', attacker, victim, points, kind, t: this.time });
    // §4.5 击退：方向取实际接触方向（拳→受击者骨盆的水平向量），限制幅度
    // 与时长；物理碰撞已产生冲量，展示击退不再额外放大。
    // FIGHT（终审修复）：击退是训练分布外的外注入（xfrc 30N×60ms；终审已证
    // 非双倒主因，但训练自博弈中不存在该力）——fight 模式下不注入，命中
    // 效果只走 stagger/伤害/记分，与训练侧命中语义（奖励+接触力）对齐。
    if (!this.fightMode) {
      const d = this.data;
      const fid = this.fistSiteId(attacker, speed >= f.fistSpeed.right ? 'left' : 'right');
      const vb = victim === 'A' ? this.ids.pelvisA : this.ids.pelvisB;
      let kdx = d.xpos[3 * vb] - d.site_xpos[3 * fid];
      let kdy = d.xpos[3 * vb + 1] - d.site_xpos[3 * fid + 1];
      const kn = Math.hypot(kdx, kdy) || 1e-6;
      this.knockback[victim] = { f: [(kdx / kn) * 30, (kdy / kn) * 30, 0], until: this.time + 0.06 };
    }
    av.staggerFor = 0.35 + f.rng() * 0.2 + (kind === 'head' ? 0.2 : 0);
    this.damage[victim] += kind === 'head' ? 1.5 : 1.0;
  }

  checkFall() {
    if (this.koState) return;
    // COMBO_MODE（D8）：禁用摔倒 KO——摔倒判读交给回归断言（minZ≥0.40），
    // 组合循环内不触发 KO/回合系统。
    // FIGHT_MODE 走本函数正常路径（KO 启用）：tracking 阈 0.40 与训练侧
    // sparring_mdp.FALL_HEIGHT_THRESHOLD 同口径（数据契约 §5.1）。
    if (this.comboMode) return;
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
    // FIGHT 反镜像：回合序号推进并刷新 B 机相位（30→55→15→40→…），A 恒 0。
    // 非 fight 模式仅计数不生效。
    this._roundNo++;
    this._assignFightPhase();
    for (const side of ['A', 'B']) {
      const f = this.fighters[side];
      f.down = false; f.state = 'idle'; f.stateT = 0;
      f.idleFor = 0.6 + this.rng() * 0.5;
      f.faceYaw = side === 'A' ? 0 : Math.PI;  // 回合重置：回到出生朝向
      f.staggerFor = 0; f.combo = null;
      f.punchIntent = null; f.punchEventId = 0; f.lastScoredPunch = -1;
      f.noPunchUntil = this.time + PUNCH_SETTLE_T;  // 回合稳定期
      f.fistSpeed = { left: 0, right: 0 };
      f._fistPrev = { left: null, right: null, t: null };
      f.anchor = [...f.anchor0];
      f._ppos = null;
      f._goal = null;
      f._obsPrev = { dist: null, t: null, fistL: null, fistR: null };
      if (f.amo) f.amo.reset();
      if (f.tracking) {
        // new round: back to the guard clip at frame 0（重锚定角统一公式替换
        // 硬编码 π；keyframe 重置 + mj_forward 已在前，xquat 就绪）。
        // FIGHT_MODE：回 fight 常驻片段，delta = 出生朝向 − ref0 yaw（KO/
        // 超时回合重置后与 setTracking 启动同状态）。
        // COMBO_MODE：手动回合重置也回 combo 常驻片段。
        // COMBO 模式锁 delta=0（同 setTracking/swapTo，实测非零 delta 致倒）。
        const clip = this.fightMode && f.trackingClips.fight ? 'fight'
          : this.comboMode && f.trackingClips.combo ? 'combo' : 'guard';
        // FIGHT 反镜像：参考起始帧 = 本回合相位偏移（_assignFightPhase 已刷新），
        // delta 基准帧与之对齐，重锚定瞬间 anchor-ori 观测归零（训练 RSI 起点）。
        const phase = clip === 'fight' ? this._fightPhase[side] : 0;
        // fight + 伺服：回合重置 δ0 = bearing − 参考相位帧 yaw（keyframe 出
        // 生位面对面，与 setTracking 启动同状态，KO/超时重置后立即面向对手）；
        // 伺服关闭回 rebaseYawDelta 旧基准（当前朝向=出生朝向）。
        const delta = clip === 'fight' && this.headingServo
          ? this.fightRebaseDelta(side, f.trackingClips[clip], phase)
          : clip === 'fight' ? this.rebaseYawDelta(side, f.trackingClips[clip], phase)
          : this.comboMode && clip === 'combo' ? 0
          : this.rebaseYawDelta(side, f.trackingClips[clip]);
        f.tracking = new TrackingFighter(this.mujoco, this.model, this.data, side, f.trackingClips[clip],
          { yawOffset: delta,
            // fight 下修正公式双侧启用（同 setTracking：非零 delta 需与
            // policyTick 观测旋转互逆）
            faceoff: this.fightMode || (this.faceoffMode && side === 'B'),
            opponent: f.opponent?.tracking ?? null,
            ctrlRange: clip === 'fight' ? FIGHT_CTRL_RANGE : null });
        f.tracking.reset(0);
        if (clip === 'combo' || clip === 'fight') {
          // 同 setTracking：combo/fight 起始帧非关键帧书端，重置后原位重锚定
          // 到参考起始帧；fight 再叠加 B 机出生位抖动（次级反镜像扰动）
          f.tracking.rebaseToReference(delta, phase);
          this._applyFightSpawnJitter(side);
          this.mujoco.mj_forward(this.model, this.data);
        }
        f.clip = clip;
        f.desiredClip = clip;
        // FIGHT：回合重置重建后重挂守卫混合钩子（非 fight 路径空操作）
        this._attachFightGuard(f);
      }
    }
    this.knockback = { A: null, B: null };
    this.koState = null;
    this.damage = { A: 0, B: 0 };
    this._lastSimTime = -1;
    this._roundT = 0;   // fight 60s 回合计时归零（Q6）
    this.events.push({ type: 'round', t: this.time });
  }

  takeHitCount() { const n = this._pendingHits; this._pendingHits = 0; return n; }
  takeBlockCount() { const n = this._pendingBlocks; this._pendingBlocks = 0; return n; }
}
