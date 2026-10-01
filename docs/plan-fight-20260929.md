# G1 双机真实对打开发方案（plan-fight-20260929）

- 日期：2026-09-29
- 作者：architecture-design（只出方案，不实现）
- 需求来源：队长 Mika 派发（含侦察结论 A、开源调研结论 B、队长预设决策）
- 交付对象：coding / code-review（实现方）
- 状态：待队长确认后派发
- 行号基准：本文所有 `文件:行号` 均已在本地仓库实地核验（训练侧根目录 `<TRAIN>` = `.workbuddy/gpu/g1dance_pipeline/third_party/unitree_rl_mjlab/src`；浏览器侧根目录 `<WEB>` = 仓库根）

---

## 1. 背景与目标

### 1.1 背景

g1-boxing-wasm 是浏览器端 MuJoCo WASM 双 G1 拳击演示，已有成熟单人拳击跟踪管线：策略 iter-5000 部署于 `?combo=1`（mjlab 1.2.0 + unitree_rl_mjlab + rsl_rl 5.0.1，训练副本在 `.workbuddy/gpu/g1dance_pipeline/`）。现有 combo 模式是"两台各自跟踪同一段参考"的伪对打，KO 与回合重置被禁用（`<WEB>/src/boxing_ai.mjs:254-256`）。

本地证据（侦察结论 A4）：t4_combo_3000_v3/v4.log 双机贴身时 ncon 4-16，2 秒内双双摔倒（minZ 0.06）。现策略训练分布零对抗接触，**对手观测 + 接触鲁棒训练是真实对打的前置条件**。

### 1.2 目标

两台 G1 在浏览器中**真实对打**：互相逼近、出拳命中、格挡、可摔倒，无脚本编排。

分层目标（与阶段对应）：

- G1（耐打）：我方在被脚本对手连续击打下不倒，跟踪质量保持，命中记分照常 —— 可独立交付。
- G2（自主）：两机共享一个对称观测策略，自博弈训练后自主进入交战距离并发生真实命中与摔倒。
- G3（部署）：`?fight=1` 新模式：KO/回合重置/记分启用，默认模式与 combo 模式字节级不回归。

硬件前提：RTX 4060 Ti 16GB（空闲）、16 核、磁盘余 29G。无硬截止，按小时级节奏推进。

---

## 2. 需求拆解与验收标准

### 2.1 功能点拆解

| 编号 | 功能点 | 说明 |
|---|---|---|
| F-1 | 训练场景双实体 | 对手 G1 实体加入训练 scene，跨实体碰撞开启，出生位姿可配 |
| F-2 | 脚本影子拳手 | 程序化出拳驱动器（逼近-出拳-撤步状态机），力/频率课程化；可选 xsens_boxing_v2.npz 回放变体 |
| F-3 | 接触鲁棒微调（阶段 1） | 不加对手观测，被打不倒；复用 push 事件 + 接触课程 |
| F-4 | 对手观测（阶段 2） | actor obs 增 14 维对手段，双机对称构造 |
| F-5 | 交战奖励与自博弈 | engage 门控 + 接触验证命中 + 快照池自博弈 |
| F-6 | 独立任务注册 | 新任务名 `Unitree-G1-Sparring-P1/P2`；旧任务与 iter-5000 复现链路零改动 |
| F-7 | 浏览器 fight 模式 | `?fight=1`：新 ONNX、obs 构造加对手段、KO/回合/记分启用、出生对峙 |
| F-8 | 回归保障 | 默认模式与 combo 模式行为字节级不回归 |

### 2.2 验收标准（量化）

- **A1（G1 耐打闸门 D1）**：微调后跟踪质量 torsoErr 相对 iter-5000 基线劣化 ≤ 15%；脚本对手连续击打 **3 分钟仿真时**存活率 ≥ 90%（"3 分钟/90%"为本方案提案值，见待确认清单 Q1）。
- **A2（G2 自博弈闸门 D2，3k iters 中检）**：交战率（处于 d∈[0.5,1.2]m 且 cosθ>0.5 的时间占比）≥ 15% 且命中事件 ≥ 0.05 次/s 且双双躺平回合占比 ≤ 30%；终检（5-8k iters）：交战率 ≥ 30%、命中 ≥ 0.1 次/s 交战、双双躺平 ≤ 20%。
- **A3（浏览器端到端）**：`?fight=1` 下两机 60s 内自主进入交战距离（≤1.2m），发生 ≥ 5 次真实命中（接触判定与浏览器 detectHits 同源），持续 ≥ 60s 或至 KO。
- **A4（KO/回合）**：摔倒（root 高度 < 0.40m，与 `checkFall` 阈值一致）触发 KO + 回合重置，连续 ≥ 3 回合正常。
- **A5（回归）**：默认模式与 `?combo=1` 代码路径 diff = 0、场景 XML 字节不变、既有 headless 回归断言（如 minZ≥0.40）全绿。

---

## 3. 现状与前提假设

### 3.1 已核验现状（实地勘察，行号为实际项目文件）

训练侧（副本，原项目 g1-dance 零写入）：

- `<TRAIN>/tasks/tracking/config/g1/env_cfgs.py:23`：`cfg.scene.entities = {"robot": get_g1_robot_cfg()}`，单实体；`:25-34` 已有 `ContactSensorCfg` + `ContactMatch` 先例（primary/secondary 支持 entity 字段，可跨实体）；`:65-71` per-robot 终止 body 列表（v2 已移除双腕 ee 终止）。
- `<TRAIN>/tasks/tracking/config/g1/__init__.py:7-21`：注册 `Unitree-G1-Tracking` 与 `Unitree-G1-Tracking-No-State-Estimation` 两个任务。
- `<TRAIN>/tasks/tracking/tracking_env_cfg.py`：`actor_terms :49-82`（8 项，合计 154 维）；`push_robot :168-173`（interval 1-3s，x/y±0.5 z±0.2，roll/pitch±0.52 yaw±0.78，play 模式移除 `env_cfgs.py:94`）；奖励 9 项 `:211-253`（全为 motion tracking exp 项 + 正则）；终止 4 项 `:259-281`；`SimulationCfg(nconmax=35, njmax=250) :304-306`；`episode_length_s=10.0 :314`，`decimation=4`，timestep 0.005。
- `<TRAIN>/tasks/tracking/mdp/observations.py:18-69`：观测函数单实体绑定 MotionCommand。
- `<TRAIN>/tasks/tracking/mdp/commands.py:59+`：`MotionCommand` 单 entity 绑定，`MotionLoader` 单 npz 假设——对手若走 MotionCommand 需二次实例化（本方案避开，见 4.2）。
- `<TRAIN>/assets/robots/unitree_g1/g1_constants.py`：`KNEES_BENT_KEYFRAME :207-220`（pos z=0.78）；`get_g1_robot_cfg() :273-284` 每次返回全新 `EntityCfg` 实例（注释明确防共享变异），`FULL_COLLISION :229-234` 全碰撞含自碰撞。
- mjlab Scene 原生多实体：`SceneCfg.entities` dict + 逐实体 attach，跨实体默认全碰撞（侦察结论 A2）。

浏览器侧：

- `<WEB>/src/tracking_policy.mjs:276-345`：`policyTick` 按 `net.meta.obs_dim` 分配 obs（:281），插入点在 lastAction 之前（:326 前）；154 = command58 + anchor_ori6 + ang_vel3 + jpos29 + jvel29 + act29。
- `<WEB>/src/boxing_ai.mjs`：`fighters.A/B.opponent` 互指 `:277-278`；`f.tracking` 挂载 `:228`；`new TrackingFighter` 共 3 处 `:383-384、:800、:1129-1130`；`comboMode :256`（禁 KO/重置）、`faceoffMode :265`；`detectHits :1016-1044`（拳套 geom vs 对方要害，拳-拳=格挡）、`registerHit :1046-1082`（head 2 分/body 1 分）、`checkFall :1084-1102`（tracking 阈 0.40，:1093）、`resetRound :1104-1146`。
- `<WEB>/src/main.js:100-107`：COMBO_MODE 出生隔离 = keyframe 字符串替换 ±0.5→±1.2m，场景 XML 字节不动。
- `<WEB>/tools/extract_tracking_onnx.py`：`obs_dim` 从权重推导 `:159`；size 表 `:184-187`；`assert cursor==obs_dim :190` 兜底；obs_layout 按 ONNX meta `observation_names` 累加 `:179-190`。现部署 meta：`command[0,58]、motion_anchor_ori_b[58,6]、base_ang_vel[64,3]、joint_pos[67,29]、joint_vel[96,29]、actions[125,29]`。
- 记分/KO/回合系统全部可复用，无需重写。

数据：

- LAFAN fight CSV（g1_fight1_subject2/3/5、g1_fightAndSports1_subject1，各 7347 帧@30fps）**不是 duet**：两两根距 mean 2.6-5.2m、<0.6m 占比 ≤7%、关节角相关仅 0.07-0.14——**不可回放双人对打**；单人片段继续作单人技能/脚本对手回放源。
- `xsens_boxing_v2.npz`（12.5s 拳击，iter-5000 训练数据）可作对手影子拳手回放。

### 3.2 关键假设（未证实项，允许实现中被推翻）

| 编号 | 假设 | 依据 / 验证方式 |
|---|---|---|
| H1 | 双 articulation 在 4060 Ti 16GB 上 500-2048 envs 可训，步时约单机 1.5-2 倍 | 调研 B5 类比估计；D0 冒烟 30 分钟实测 |
| H2 | 对手用"环境内冻结快照池策略"即可产生足够对抗压力，无需 rsl_rl 框架级自博弈 | RoboStriker LS-NFSP 本质是冻结对手池轮换（调研 B1）；D2 中检验证 |
| H3 | 跳过 RoboStriker 的 32 维潜空间蒸馏，直接 iter-5000 初始化 + 全 obs 微调可在预算内收敛 | iter-5000 已是跟踪先验（RoboStriker 消融：无先验胜率 0%，故先验必须有；蒸馏属工程优化非必要条件）；D2 不收敛则回退启用蒸馏（备选路径） |
| H4 | 训练 episode 10s 内可完成"逼近-交战-命中"闭环（约 1-2s 逼近 1-2m） | 物理常识 + D2 指标验证 |
| H5 | 浏览器 WASM 侧新 obs 段计算成本可忽略（14 维向量运算） | 现有 policyTick 同量级运算 |
| H6 | 训练侧"要害 geom"可与浏览器 `vulnerableGeoms` 对齐（G1 无头部连杆，用 torso 上身链近似"头部"命中） | 冒烟时列出 geom 清单裁决（Q2） |

---

## 4. 方案设计

### 4.1 总体架构（分层与模块）

三阶段递进，每阶段独立可验收、独立可交付：

- **阶段 1（P1，接触鲁棒微调）**：训练环境加对手实体 + 脚本影子拳手，不加对手观测。产物 = 耐打策略（obs 仍 154 维，与现部署格式兼容）+ 可交付的"脚本对手耐打版对打演示"。
- **阶段 2（P2，对手观测 + 交战自博弈）**：obs 加 14 维对手段、奖励重构为 0.8 task / 0.2 style、快照池自博弈。产物 = 对称共享策略（obs 168 维）。
- **阶段 3（D，浏览器部署）**：`?fight=1` 新模式 + 部署链路适配 + 回归。

模块边界：训练侧所有新逻辑进**新文件**，既有文件只允许两类改动——`config/g1/__init__.py` 追加注册块、`mdp/__init__.py` 追加导出行；`tracking_env_cfg.py`、`commands.py`、旧 `env_cfgs.py` 零改动（保证 iter-5000 复现链路字节不变）。

### 4.2 核心流程（文字描述）

阶段 2 训练单步流程：

1. reset：双机按 `KNEES_BENT_KEYFRAME` 出生于 ±0.6m 对峙位（A 朝 yaw=0、B 朝 yaw=π），RSI 随机化沿用现有 `motion_cmd.pose_range`；快照池按当前课程选出冻结对手 checkpoint。
2. 每控制步（decimation=4）：
   - 构造 actor obs（168 维）= 现有 154 维 + 对手段 14 维（在自身 torso 系中表达对方：相对位置/速度/双拳/朝向）。
   - actor 前向得本机动作；对手 obs（同样 168 维、对称构造）经冻结快照策略前向得对手动作。
   - 双机动作分别施加到各自 articulation，mj_step ×4。
3. 奖励：task 组（engage 门控接近 + 接触验证命中 + 存活 - 摔倒罚 + 降权跟踪项）×0.8 + style 组（现有 motion tracking exp 项）×0.2。
4. 终止：任一机 root 高度 < 0.40m（摔倒判负，对方存活奖励）、超时、跟踪 anchor 误差超阈（沿用现有终止防姿态崩塌）。
5. 每 K iters（默认 200）：当前 actor 快照入池；池容量 4，先进先出淘汰最旧。

阶段 1 与阶段 2 差异：无对手段 obs（154 维）、对手为 `OpponentScriptDriver` 程序化驱动、奖励只有"存活 - 摔倒罚 + 原跟踪项（不降权）+ push/接触课程"。

浏览器 fight 模式流程：

1. URL 含 `fight=1` → 加载 fight 场景（出生 ±0.6m 对峙，faceoff 机制已有）+ fight ONNX（obs 168）。
2. 双 `TrackingFighter` 共享同一 ONNX，每 tick 各自在对称自系构造 obs（含对方 14 维；对手段引用复用 `fighters.A.opponent=B` 现有互指）。
3. 命中记分 `detectHits`/`registerHit` 与 KO `checkFall`（0.40）**不禁用**（fight 模式不受 comboMode 的 KO 禁用分支影响），摔倒触发 `resetRound`。

### 4.3 对手驱动器设计（阶段 1 关键件）

`OpponentScriptDriver`（新文件，纯环境侧，不经 MotionCommand）：

- 状态机：`approach`（朝我方平移至目标距离 d_target，默认 1.2m，课程范围 0.8-1.6m）→ `strike`（随机左/右直拳，肩肘关节目标位置插值，出拳时长 0.3-0.5s，随机相位）→ `recover`（回拳架，0.5-1.0s）→ 循环。
- 实现方式：直接写对手 articulation 的关节位置目标（与 `JointPositionAction` 同语义；mjlab Entity 运行期 API确切名称冒烟时确认）+ 根部速度设定推进；不引入第二个 MotionCommand（规避 `MotionLoader` 单 npz 假设改造）。
- 课程参数（cfg 可配）：出拳频率（0.2→1.0 Hz）、拳速、d_target、是否随机撤步。
- 变体开关：`driver="script" | "replay"`；replay 模式回放 `xsens_boxing_v2.npz` 关节序列（仅作对照实验，非主线）。
- 课程叠加：现有 `push_robot` 事件保持开启（play 移除逻辑不变），另加 `contact_curriculum` 事件项（阶段 1 后半程：对手拳接触瞬间对我方 root 施加脉冲扰动，模拟重击）。

### 4.4 快照池自博弈设计（阶段 2 关键件，选型见 §6.3）

- learner 只训 actor 策略（rsl_rl 正常 PPO 流程，框架零改动）。
- 对手策略 = 冻结 checkpoint（快照池），在 env 内以前向推理产生动作：env 持有 `torch` 模块副本，独立构造对手 obs 后前向（批内并行，无梯度）。
- 轮换：每 K iters 将当前 actor 权重拷贝入池；池满淘汰最旧；课程前 500 iters 池中只有 iter-5000 初始化副本（等价 RoboStriker 的"固定对手 warmup"，其消融显示无 warmup 直接自博弈 ηhit 仅 0.050）。
- 好处：完全避开 rsl_rl 5.0.1 参数共享自博弈的 runner 兼容性风险（调研 B5 未验证项）；坏处：无协同梯度，探索压力全在 actor 侧——以快照池多样性 + 课程补偿，D2 中检验证。

### 4.5 奖励表（阶段 2；阶段 1 为其子集）

风格框架采用 RoboStriker 的 0.8 task + 0.2 style（调研 B1）。不引入 AMP 判别器，style 组直接沿用现有 motion tracking exp 项（理由见 §6.4）。

| 组 | 项 | 权重（起点） | 定义 | 来源 |
|---|---|---|---|---|
| task(0.8) | engage | +0.5 | d∈[0.5,1.2]m 且 facing cosθ>0.9 时给正奖励；课程起步 cosθ>0.5、d∈[0.4,1.6]，逐段收紧 | RoboStriker 交战判定 |
| task | approach | +0.3 | 距离变化率负奖励的对偶（相互逼近时奖励，clamp 防冲刺） | RoboStriker 接近速度 |
| task | hit | +2.0 | 拳套 geom 与对方要害接触且接触力 ≥ 10N，单次命中奖励 + 0.5s 冷却防刷分 | RoboStriker 力阈值 10N |
| task | alive | +0.1 | 每步存活 | 惯例 |
| task | fall | -5.0 | 本机 root < 0.40m 触发并终止本机 | RoboStriker 摔倒判负 + 阈值对齐浏览器 checkFall |
| task | stalemate | -0.2 | 回合后半段仍 d>2.0m 的惰距惩罚 | 防"绕圈不打" |
| task | tracking(降权) | 现有权重 ×0.3 起，随交战率上升退坡至 ×0.1 | 沿用 anchor/body pos/ori exp 项 | 保跟踪先验不崩 |
| task | 正则 | 不变 | action_rate_l2、joint_limit、self_collisions 沿用 | — |
| style(0.2) | motion_body_pos/ori/vel exp | 现有权重归一化后 ×0.2 | 沿用现有项 | RoboStriker R=0.8task+0.2style |

阶段 1 奖励 = 现有全部项不动 + alive(+0.1) + fall(-5.0) + 接触课程事件；无 engage/hit/approach/stalemate。

### 4.6 观测表

阶段 1：obs 不变（154 维），策略与现部署格式兼容（这是"耐打版可直接回滚部署"的前提）。

阶段 2：新增 `opponent_state` 14 维，全部在**观察者自身 torso 系**中表达（对称性保证双机共享同一策略）：

| 段 | 维度 | 定义 |
|---|---|---|
| opp_root_pos_rel | 3 | 对方 root（pelvis）位置，自 torso 系 |
| opp_root_lin_vel_rel | 3 | 对方 root 线速度，旋转到自 torso 系 |
| opp_fist_L_rel | 3 | 对方左腕（wrist_yaw_link）位置，自 torso 系 |
| opp_fist_R_rel | 3 | 对方右腕位置，自 torso 系 |
| opp_heading_sin / cos | 2 | 对方朝向相对自机朝向的 yaw 差 sin/cos |
| 合计 | 14 | obs_dim 154 → 168 |

对标 RoboStriker"仅两组向量、不编对手关节、不要未来预测"的结论（调研 B1），14 维取中庸（队长预设 10-16 维区间内）；critic 组同步加同 14 维（保持 actor/critic 同构，避免非对称复杂度）。

实现注：obs 新段加噪声课程——训练前 200 iters 对手段以概率 p 随机置零（p: 0.5→0），缓解新观测分布突变对 iter-5000 先验的冲击。

### 4.7 分阶段任务分解

#### 阶段 1：接触鲁棒微调（P1）

- 训练配置：任务 `Unitree-G1-Sparring-P1`；双实体（robot + opponent）；obs 154 维不变；episode 10s；envs 数取 D0 冒烟结论（预期 2048）；初始化自 iter-5000 checkpoint；`nconmax/njmax` 上调（预期 35→60 / 250→500，以冒烟实测 ncon 峰值 2 倍裕量定值）；训练 iters 1-2k。
- 改动清单：见 §5.2 训练侧 T1-T5。
- 验收判据：A1（torsoErr 劣化 ≤15% + 3 分钟击打存活率 ≥90%）。
- 回退：一轮调参重训（课程放缓一档 / fall 罚减半 / iters +1k，预算 +2h）；再失败 → 触发 F1 fallback（§8.2）。

#### 阶段 2：对手观测 + 交战自博弈（P2）

- 训练配置：任务 `Unitree-G1-Sparring-P2`；初始化自 P1 最优 checkpoint；obs 168 维；快照池自博弈（K=200 iters，池深 4，池初值 = iter-5000 与 P1 快照）；训练 iters 3k 中检 + 5-8k 终检（8k ≈ 4h 封顶，Q5）。
- 改动清单：训练侧 T6-T9 + rl_cfg。
- 验收判据：A2（D2 中检 + 终检量化指标）。
- 回退梯度：a) engage 阈值再放宽 + hit 权重上调（+1h 微调）；b) 启用 H3 备选：潜空间蒸馏压缩先验后再训（+0.5 天，需队长批准）；c) 停止 P2，交付 P1 产物（F1）。

#### 阶段 3：浏览器部署（D）

- 模式：`?fight=1`。出生间距 ±0.6m（与训练一致，Q3）；KO/回合/记分启用；单 ONNX 双机共享。
- 改动清单：见 §5.2 部署侧 W1-W5。
- 验收判据：A3、A4、A5。
- 回退：部署侧 bug 只修部署侧，不动训练产物；若 fight ONNX 行为与训练 eval 不符，先查 obs 构造一致性（契约 §5.1）。

---

## 5. 接口与数据契约

### 5.1 数据契约

obs 布局（fight ONNX meta，obs_dim=168）：

| 段 | 偏移 | 维度 |
|---|---|---|
| command | 0 | 58 |
| motion_anchor_ori_b | 58 | 6 |
| base_ang_vel | 64 | 3 |
| joint_pos | 67 | 29 |
| joint_vel | 96 | 29 |
| actions | 125 | 29 |
| **opponent_state** | **154** | **14** |

契约约束：

- ONNX meta `observation_names` 必须含 `opponent_state`；`extract_tracking_onnx.py:190` 的 `assert cursor==obs_dim` 为一致性兜底，若导出管道未自动写入该名，导出脚本需显式补写（实现注意点）。
- `opponent_state` 各子段顺序固定为 §4.6 表序；训练与浏览器两侧各自实现构造函数，以**同一份注释契约**对齐（浏览器侧实现在 `tracking_policy.mjs` 注释中锚定训练侧函数名）。
- 摔倒阈值三方一致：训练 fall 终止 0.40m = 浏览器 `checkFall` tracking 阈 0.40（`boxing_ai.mjs:1093`）= 回归断言 minZ≥0.40。
- 命中判定：训练侧力阈值 10N（接触传感器）；浏览器侧沿用 `detectHits` 几何 + 拳速判定（WASM 侧无力读数时可用 `cfrc_ext`，实现时评估；两者判定差异列入 Q2）。
- 任务注册名：`Unitree-G1-Sparring-P1`、`Unitree-G1-Sparring-P2`；旧任务 `Unitree-G1-Tracking(-No-State-Estimation)` 注册块零改动。
- URL 契约：`?fight=1` 独立开关（不与 combo 叠加）；未带 fight=1 的所有 URL 行为字节级不变。

### 5.2 模块与接口边界（文件:行号级改动清单）

训练侧（`<TRAIN>` 前缀，新增文件优先，既有文件仅注册/导出行）：

| 编号 | 文件 | 改动 |
|---|---|---|
| T1 | `tasks/tracking/config/g1/sparring_env_cfgs.py`（新增） | `make_sparring_env_cfg(phase, play=False)`：基于 `make_tracking_env_cfg()` 返回值覆写——`cfg.scene.entities["opponent"] = get_g1_robot_cfg()`（改 init pos=(±0.6,0,0.78)、rot 朝向对方）；`cfg.scene.sensors` 追加 `fight_contact`（primary=robot 拳套 geom，secondary=opponent torso 链，fields 含 force）；`cfg.sim.nconmax/njmax` 上调；phase 开关控制观测/奖励/课程组 |
| T2 | `tasks/tracking/mdp/opponent_driver.py`（新增） | `OpponentScriptDriver` 状态机（§4.3）+ 可选 replay 变体；env 侧每 step 调用 |
| T3 | `tasks/tracking/mdp/observations.py`（追加函数） | `opponent_state(env)` 14 维（读 `env.scene["opponent"]`）；既有 4 函数 :18-69 不动 |
| T4 | `tasks/tracking/mdp/rewards.py`（追加函数） | `engage_reward` / `approach_reward` / `hit_reward`（读 fight_contact sensor，力阈值 10N + 冷却）/ `sparring_alive` |
| T5 | `tasks/tracking/mdp/terminations.py`（追加函数） | `opponent_fall` / `robot_fall`（root 高度 0.40）；现有 4 项终止沿用且 body 列表按机各配 |
| T6 | `tasks/tracking/mdp/__init__.py`（导出行） | 追加 T3-T5 新函数导出 |
| T7 | `tasks/tracking/config/g1/sparring_rl_cfg.py`（新增） | P2 runner cfg（含快照池参数 K、池深、噪声课程） |
| T8 | `tasks/tracking/config/g1/__init__.py`（追加注册块，:7-21 不动） | `register_mjlab_task("Unitree-G1-Sparring-P1", ...)`、`(...P2...)`，runner_cls 沿用 `MotionTrackingOnPolicyRunner` |
| T9 | 导出脚本（pipeline scripts 内，新增/扩展） | P2 checkpoint → ONNX：meta 写入 `observation_names`（含 opponent_state）、`default_joint_pos` 等，复用现有导出约定 |

禁改清单（训练侧）：`tracking_env_cfg.py`、`mdp/commands.py`、`config/g1/env_cfgs.py`、`g1_constants.py`；g1-dance 原项目全部。

部署侧（`<WEB>` 前缀）：

| 编号 | 文件:行号 | 改动 |
|---|---|---|
| W1 | `tools/extract_tracking_onnx.py:184-187` | size 表加 `"opponent_state": 14`；:190 assert 自动兜底；obs_dim 仍从权重推导 :159 无需改 |
| W2 | `src/tracking_policy.mjs:276-345` | `policyTick` 在 lastAction 写入前（:326 前）插入 `opponent_state` 填充：按 `meta.obs_layout["opponent_state"]` 存在性开关（旧 ONNX 无此段时跳过，天然兼容）；构造函数注入对手 mjData 引用；新增 14 维相对量计算（自 torso 系，与训练侧契约一致） |
| W3 | `src/boxing_ai.mjs:383-384、:800、:1129-1130` | 三处 `new TrackingFighter` 注入对手引用；新增 `fightMode` 分支：不进入 comboMode 的 KO 禁用（:256）分支，`checkFall :1084-1102`、`resetRound :1104-1146`、`detectHits/registerHit :1016-1082` 全启用；faceoff 站位启用 |
| W4 | `src/main.js:100-107` | `FIGHT_MODE` 场景加载分支：keyframe 出生位替换 ±0.6m（替换逻辑与 COMBO_MODE 同构、参数不同）；默认与 combo 路径字节不动 |
| W5 | `src/main.js`（URL 解析处，COMBO_MODE 解析先例旁） | 解析 `?fight=1`；`index.html` 如有模式说明文案一并更新 |

---

## 6. 技术选型与权衡（引用调研结论）

### 6.1 总体路线：跟踪先验 → 接触鲁棒 → 对手观测 → 自博弈（采纳 RoboStriker 配方）

RoboStriker（arXiv 2601.22517 / 2608.16195，双 G1 自主拳击，唯一直接同任务；Isaac Lab + 4090）配方为：跟踪先验 → 32 维有界潜空间蒸馏 → 固定对手 warmup → LS-NFSP 自博弈。其消融数据是本方案阶段化的直接依据：无先验从零自博弈胜率 0%（ηhit 0.142）、无 warmup 直接自博弈 ηhit 仅 0.050。因此本方案坚持：迭代-5000 先验必用（P1/P2 均热启动）、P2 前 500 iters 固定对手 warmup、之后再入快照池自博弈。OP3 足球（arXiv 2304.13653）的分层课程思想同构支撑"P1→P2"顺序。

**偏离声明（H3）**：跳过潜空间蒸馏，直接全 obs 微调。理由：蒸馏是工程优化而非消融证明的必要条件；16GB 单卡 + 小时级预算下省去一段训练管线。风险与回退见 §8。此偏离为可推翻项（Q4 关联）。

### 6.2 对手驱动：程序化脚本 > MotionCommand 二次实例化

`MotionCommand` 单实体绑定（`commands.py:59+`）+ `MotionLoader` 单 npz 假设，二次实例化需框架级改造；而 LAFAN fight CSV 已证非 duet、无可回放双人对打数据（侦察 A1），回放路线天然受限。程序化脚本对手出拳节奏/力度可控可课程化，且阶段 2 后对手本就换成策略快照，脚本只是阶段 1 脚手架。xsens_boxing_v2.npz 回放仅留作对照变体。

### 6.3 自博弈实现：环境内快照池（选 A）> rsl_rl 批翻倍参数共享（选 B）

- 选 A（本方案默认）：对手 = env 内冻结快照策略前向，rsl_rl 零改动。与队长预设"快照池自博弈（旧 checkpoint 冻结轮换）"一致；与 RoboStriker LS-NFSP 的 reservoir 冻结对手池思想同源（配比差异：不实现混合 η=0.1 的 RL buffer，纯池轮换，简化）。
- 选 B（备选）：双机对称 obs、batch 维翻倍单次前向、动作拆分回写，需改 runner/rollout 层。调研 B5 明确标注 rsl_rl 5.0.1 兼容性未验证。仅当 A 在 D2 被证"对手压力不足导致策略退化坍缩"时，经队长批准后以 0.5 天 spike 试做。
- 明确不采纳（调研 B6）：从零自博弈（消融已证 0%）；引入独立 MARL 框架（改造面与维护成本不可控）。

### 6.4 风格项：沿用 tracking exp 项，不引入 AMP 判别器

RoboStriker 用 AMP 风格判别器组成 0.2 style。本方案以现有 motion tracking exp 项充当 style 组：iter-5000 已证明这些项足以产生稳定拳击风格，判别器会引入额外训练不稳定面与调参成本。代价：风格约束弱于 AMP，摔倒姿态可能更"难看"——不影响功能验收（A1-A5），列入已知取舍。

### 6.5 攻击强度课程（借鉴 SA2RT，arXiv 2507.08303 无代码）

软预算思想落地为：hit 奖励随训练进度分段解锁（前 1k iters 0.5×，之后全额），避免早期为刷 hit 学出扑击倒地。push 事件（`tracking_env_cfg.py:168-173`）在 P1 直接复用叠加。

### 6.6 复用与自研边界

| 类别 | 内容 |
|---|---|
| 直接复用（零改） | 现有 154 维 obs 管线与 iter-5000 兼容部署、`detectHits/registerHit/checkFall/resetRound` 记分 KO 系统、faceoff 机制、push 事件、`MotionTrackingOnPolicyRunner`、g1 常量与 FULL_COLLISION、LAFAN/xsens 数据资产 |
| 借鉴配方（自研实现） | RoboStriker 奖励框架/交战判定/力阈值/warmup-自博弈顺序、OP3 课程思想、SA2RT 攻击课程、快照池自博弈机制 |
| 全自研 | `OpponentScriptDriver`、`opponent_state` 观测构造（训练+浏览器双实现）、`fight_contact` 传感器配置、sparring env cfg 组、浏览器 fight 模式分支与 obs 扩展 |
| 不做 | AMP 判别器、潜空间蒸馏（首版）、MotionCommand 双实例化、MARL 框架 |

---

## 7. 分阶段实施计划（小时粒度）

依赖顺序：D0 冒烟 → P1 实现 → P1 短冒烟 → P1 训练 → D1 闸门 → P2 实现 → P2 训练（含 D2 中检）→ 导出部署 → 回归。每个决策点（D0-D3）停下报队长后再继续。

| 时段 | 任务 | 产物 / 闸门 |
|---|---|---|
| T0+0.0–0.5 | D0 冒烟 spike：sparring env 骨架（双实体 + 跨实体接触传感器 + 500 envs）实测吞吐/显存/ncon 峰值 | 决策 envs 数（2048 vs 1024）、nconmax/njmax 取值；失败 → 降档重测或停 |
| T0+0.5–1.5 | P1 实现：T1-T8（sparring cfg + OpponentScriptDriver + 奖励/终止 + 注册） | 代码 + 单 env 手跑录像 |
| T0+1.5–2.0 | P1 短冒烟：200 iters，验证不发散、torsoErr 量级正常、对手课程生效 | 冒烟报告 |
| T0+2.0–4.0 | P1 正式训练 1-2k iters（估 1.1-1.5s/iter） | checkpoint 序列（每 500 iter 存，保留最近 3 + best） |
| T0+4.0–4.5 | D1 闸门：eval torsoErr 对照基线 + 3 分钟击打存活率统计 | A1 判定；通过 → P2；失败 → 1 轮调参重训（+2h） |
| T0+4.5–5.5 | P2 实现：obs 对手段（T3/T6）+ 奖励重构 + 快照池自博弈 + spike（冻结前向与 obs packing 正确性 0.5h 内验证） | 代码 + spike 证据 |
| T0+5.5–9.5 | P2 训练：0-3k warmup+课程，3k 处 **D2 中检**（A2 中检指标）；通过续训至 5-8k；每小时查一次指标曲线 | 中检报告 → 队长决策 |
| T0+9.5–10.5 | ONNX 导出（T9/W1）+ 部署改造（W2-W5） | fight ONNX + fight 模式代码 |
| T0+10.5–11.0 | 回归：A3/A4 端到端 + A5 默认/combo 字节级回归 | 验收报告 D3 |
| T0+11.0–12.0 | 缓冲池（重训/修复预算来源）+ `DELIVERY_PROVENANCE.md` 增补 §9（禁清理资产清单延续 + 新增产物清单） | 文档 |

里程碑：M1 = D1 通过（耐打策略成型，F1 可交付点）；M2 = D2 终检通过（自主对打策略成型）；M3 = D3 回归全绿（完整交付）。

磁盘护栏：全程新增产物（checkpoints + logs + ONNX）预算 < 5GB，29G 余量充足；checkpoint 保留策略强制（每 500 iter 1 个、滑动保留 3 + best）；禁清理资产清单（xsens npz、LAFAN CSV、iter-5000 ckpt、现部署 ONNX、models/）写入 DELIVERY_PROVENANCE.md §9。

---

## 8. 风险与缓解

### 8.1 风险表

| 编号 | 风险 | 概率/影响 | 缓解 |
|---|---|---|---|
| R1 | 双 articulation 显存/吞吐超预期（16GB） | 中/高 | D0 冒烟前置量化；降 envs 2048→1024→512；仍不可行则停，报队长（H1 证伪） |
| R2 | 接触分布外：贴身接触仍摔倒（阶段 1 不收敛） | 中/高 | 课程化击打力度/频率；push + 接触脉冲叠加；一轮调参预算；F1 fallback 兜底 |
| R3 | obs 新段冲击先验（P2 初期跟踪崩塌） | 中/中 | P1 热启动 + obs 置零课程（§4.6）+ tracking 项降权而非移除；anchor 终止沿用兜底 |
| R4 | 奖励 hacking：僵持不打 / 扑击刷分 | 中/中 | 距离门控 + 力阈值 10N + hit 冷却 + stalemate 罚 + SA2RT 分段解锁 |
| R5 | 快照池自博弈对手压力不足，策略退化为"互不接近" | 中/高 | D2 中检量化（交战率趋势）；恶化路径：池扩容/轮换加快 → 选 B spike（Q4） |
| R6 | 训练-部署观测不一致（系、顺序、阈值） | 中/高 | §5.1 契约表 + :190 assert + A5 回归；不一致先查 obs 构造 |
| R7 | 磁盘/长训中断 | 低/中 | 5GB 预算 + 滑动保留；每 500 iter checkpoint 可续 |
| R8 | rsl_rl 5.0.1 导出 meta 不含新 obs 名 | 中/低 | T9 导出脚本显式补写 + W1 assert 兜底 |

### 8.2 Fallback 树（量化判据，文字描述）

1. **D0 失败**（500 envs 双机 OOM 或 step 时 > 2× 单人且降档后仍不可行）→ 停止，报队长重新评估硬件或缩减范围。不进入训练。
2. **D1 失败 → 一轮调参重训（+2h）→ 再失败** → 交付 **F1 = 耐打版对打演示**：保留 iter-5000 现网行为，部署侧交付"脚本对手 + 耐打策略"的 `?fight=1&opp=script` 形态（若耐打策略本身失败，则 F1 退化为"脚本对手 + 现有 combo 策略 + KO/记分启用"，命中记分照常，KO 仅对脚本对手生效）+ 问题报告。旧 combo 演示永不受损（A5 保障）。
3. **D2 中检不达标但趋势向上** → 延长训练至 8k 封顶（Q5）再判。
4. **D2 终检不达标**（预算内不收敛）→ 交付 F1（P1 产物），P2 产出物归档 + 不收敛分析报告（含奖励曲线、交战率曲线、失败模式录像），供后续立项。
5. **部署回归失败**（A5 破坏）→ 只回滚部署侧改动（fight 分支独立提交），训练产物不受影响；修复后重验。

---

## 9. 待确认清单

| 编号 | 问题 | 本方案临时取值（可改） |
|---|---|---|
| Q1 | 阶段 1 存活率验收的时长与阈值（队长预设为"X 分钟"未定值） | 3 分钟仿真时 / 存活率 ≥ 90% |
| Q2 | 训练侧"要害 geom"集合（G1 无头部连杆）与浏览器 `vulnerableGeoms` 对齐清单；训练力阈值判定与浏览器几何+拳速判定的差异是否接受 | torso 上身链近似；差异记录不强行抹平 |
| Q3 | 双机出生间距与朝向终值（训练 init 与浏览器 faceoff 必须一致） | ±0.6m 对峙（A yaw 0 / B yaw π），D0 冒烟后可调 |
| Q4 | 自博弈路线终选：默认 A（env 内快照池）；是否接受"仅当 D2 证伪 A 后"再批 B（rsl_rl 批翻倍 spike 0.5 天） | A |
| Q5 | 阶段 2 训练预算上限 | 8k iters ≈ 4h 封顶 |
| Q6 | 比赛规则细节：回合时长（浏览器建议 60s，训练 episode 保持 10s 不动）、局数、平局判定 | 60s 单回合循环，无局数上限 |
| Q7 | H3 偏离（跳过 RoboStriker 潜空间蒸馏）是否接受为初版决策（不收敛时蒸馏作为 b 级回退，需再批 0.5 天） | 接受偏离 |
| Q8 | 本方案无对"双人对打数据回放"路线的遗留依赖（LAFAN fight CSV 非 duet 已证）——是否需正式同步需求方关闭该路线 | 视为已关闭 |
| Q9 | 缺开源参考调研补充项：RoboStriker 匿名审稿库为 Isaac Lab 实现，仅"配方借鉴"无代码迁移；是否接受此复用边界 | 接受 |

---

## 10. Out of scope

- 摔倒后起身（getup）恢复——OP3 分层课程中的 getup 阶段明确不做，摔倒即回合终结。
- 真机部署与 sim-to-real；本文全部为仿真与浏览器侧。
- 独立 MARL 框架引入、从零自博弈、AMP 判别器、潜空间蒸馏（首版）。
- 腿法/踢击/摔投等扩大动作集；每机超出双拳的接触语义。
- 多于两台机器人、联网多人对战、观众/回放系统。
- LAFAN fight CSV 的 duet 化改造或任何双人对打数据回放管线。
- 训练侧 g1-dance 原项目任何写入；旧任务 `Unitree-G1-Tracking(-No-State-Estimation)` 及 iter-5000 复现链路的任何改动。
- 浏览器默认模式与 combo 模式的任何行为变更（A5 字节级回归保障）。

---

## 交接声明

本方案为架构设计交付，不含任何实现。方案完成后交接给实现方（coding 按 §4.7/§5.2 执行，code-review 按 §2.2 验收），阶段决策点 D0-D3 由队长主持放行。

---

## 附录 A：验收判据补录（coding 补录，队长 2026-09-29 批准留档）

### A.1 D0 裁决记录（队长 2026-09-29 批准）

| 项 | 终值 | 依据 |
|---|---|---|
| envs | 1024 | 实测 500/1024/2048 三档吞吐 0.88/1.46/2.63 s/iter，1024 留 1.6x 余量 |
| nconmax | 128 | 双机每 world ncon 实测 peak 57 / median 51-56，取 ~2.3x 裕量 |
| njmax | 768 | 与 nconmax 同步上调 |
| 出生间距 spawn_dist | 1.2m | Q3 落定；对手逼近至 d_target=0.8m 拳距再出拳（§4.3 课程参数，实测 1.2m 处拳不可达，命中 17→829 次/3min） |

### A.2 D1 闸门判据（A1 的三项联合修正，替代原单一存活率口径）

背景：iter-5000 基线实测显示"存活率假象"——anchor 跟踪终止（0.25m 阈值）先于摔倒触发，保护性重置掩盖跟踪崩塌，基线存活率也为 100%，单一存活率指标无区分度。

| # | 判据 | 阈值 | 基线（iter-5000，32 envs 3 分钟连续击打） |
|---|---|---|---|
| 1 | 3 分钟存活率（无 robot_fall） | ≥ 90% | 100%（假象，见上） |
| 2 | anchor 跟踪终止率 | ≤ 5.7 次/env/10s 等效（基线下降 ≥60%） | 14.17 次/env/10s |
| 3 | torsoErr（单机 sim 全段，rolling_gate2 同链路） | ≤ 0.231（基线 0.201 × 1.15） | 0.201 |

三项同时达标方可进入 P2。基线日志：`logs/spar_d1_baseline_3min.log`（d_target=0.8 口径）。
