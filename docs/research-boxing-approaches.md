# 调研报告：双机器人拳击的更优实现方案

> 调研日期：2026-09-21 · 针对项目：g1-boxing-wasm（AMO 底座 + 脚本拳击 AI）

## 0. 结论摘要（TL;DR）

**你的判断是对的**：当前"AMO 策略管腿 + 脚本 8 关节姿态管拳 + 加权随机挑招式"的方式，是这个领域里最弱的架构。调研下来，业界已经跑通了完整的"人形机器人自主拳击"，而且**代码和拳击动捕数据都已开源**。

三条可行路线，按投入排序：

| 路线 | 一句话 | 训练成本 | 拟真/对抗水平 |
|---|---|---|---|
| **A. 本地自博弈（快赢）** | 不换底座，用现有无头仿真把"加权随机"换成自博弈学出来的战术策略 | 无需 GPU，node 跑几小时 | ★★ |
| **B. BeyondMimic/mjlab 动作跟踪（主推）** | 用重定向的拳击动捕数据训跟踪策略，替换脚本出拳 | 单张 4090，1~2 天 | ★★★★ |
| **C. RoboStriker 全流程（终极）** | 隐空间自博弈拳击，两台 G1 真机已验证 | 4090，数天 | ★★★★★ |

**关键发现**：RoboStriker（arXiv:2608.16195，2026-08）代码匿名开源在
`https://anonymous.4open.science/r/robostriker-B25B`，**仓库里自带拳击动捕数据
`fight1_subject2.npz`，且 stage1/2 就是 BeyondMimic 的 `whole_body_tracking` 跟踪训练**——
意味着走路线 B 时，拳击动作数据可以直接从它仓库里拿，不用自己找动捕源。

---

## 1. 现状痛点分析

当前架构（AMO 50Hz 全身策略 + 脚本拳击 AI）的"不合理"具体在四处：

1. **出拳没有动力学**。刺拳/交叉拳/勾拳是 8 个手臂关节角的静态目标姿态切换，
   拳速由插值速度决定，没有重心转移、蹬地发力、转腰的整链动力——真人拳击的
   力量恰恰来自腿部和躯干，而这部分被 AMO 锁死在"平衡行走"模式。
2. **战术层是伪智能**。招式选择 = `weightedPick(['jab':3, 'double':2, ...])` 加权随机，
   不看对手状态、不学防守反击。
3. **踉跄/击退是假的**。击退靠外力冲量（40N×0.06s）硬推，踉跄靠脚本姿态；
   真正的受击反应应该是策略在扰动下的自然恢复。
4. **两台机器人不互动**。各自的脚本互不感知（只在命中判定时间接耦合），
   没有攻防博弈，看起来像两台各自表演的机器。

---

## 2. 候选方案全景（GitHub 实证）

### 路线 C（终极）：RoboStriker — 隐空间自博弈拳击 ★ 业界当前最优

- **论文**：arXiv:2608.16195（2026-08-18），沪/上交大等；项目页
  `https://sites.google.com/view/robo-striker`
- **代码**：`https://anonymous.4open.science/r/robostriker-B25B`（148 文件，
  含 source/ 与 scripts/，匿名仓库，发表后应会转正式 GitHub）
- **方法**（三阶段，正好对应当前架构的三个缺陷）：
  1. **Stage 1 运动跟踪器**：在拳击动捕（`fight1_subject2.npz`，50Hz）上用
     BeyondMimic 式 PPO 训练 29 维全身跟踪策略——拳法带真实发力链
  2. **Stage 2 隐空间蒸馏**：把跟踪技能蒸馏到 32 维隐码（论文消融：8/16 维重建
     误差 128/86mm，64 维搜索空间过大落点率反降，32 维最优）
  3. **Stage 3 自博弈**：LS-NFSP（隐空间神经虚拟自博弈），最佳响应策略进化进攻、
     平均策略锚定稳定，η=0.1——机器人学出真正的攻防战术
- **成绩**：与 8 个基线交叉对战全胜；进攻落点率 0.685（原始关节空间自博弈的
  4.8 倍）；交战率 0.824（不消极避战）；关键消融——去掉打靶预热落点率只剩 0.05，
  去掉 AMP 风格约束落点率跌 28%
- **工程性**：单张 RTX 4090 训练（Isaac Lab 4096 并行环境、物理 200Hz、策略 50Hz）；
  真机部署是 ONNX Runtime 50Hz 跑在机载 CPU——**50Hz 小策略，
  和本项目现有 AMO student 的移植形态完全一致**
- **风险**：匿名仓库无预训练权重，需自己跑完三阶段训练；许可未明（匿名期）

### 路线 B（主推）：BeyondMimic / mjlab 动作跟踪 + 拳击动作库

即"只做 RoboStriker 的 Stage 1，战术层先用脚本顶上"。

- **BeyondMimic**：`https://github.com/HybridRobotics/whole_body_tracking`
  （UC Berkeley/Stanford，MIT 许可，2.2k star，论文 arXiv:2508.08241）
  - 零调参训练 LAFAN1 任意动作；自适应采样 + RSI；MIT 许可可商用
  - 另有 mjlab 后端（MuJoCo-Warp GPU 训练）——**物理引擎与本项目运行时同源，
    sim2sim 陷阱最少**
- **现成范例**：`https://github.com/mujocolab/g1_spinkick_example`
  （mjlab 官方示例，G1 双旋踢：数据转换脚本 + 完整训练配置 + **预训练 ONNX
  checkpoint** + RoboJuDo 部署）——把"旋踢动作"换成"拳击动作"就是我们的路线
- **数据管线**（从视频到机器人）：GVHMR（单目视频→SMPLX 全局运动）→
  GMR（人体→G1 重定向）→ csv/npz；GhostTrial 数据集（HuggingFace
  `DaveRc/ghosttrial-g1-scorpion-motion`，CC BY-4.0）示范了整条管线实操
- **部署**：RoboJuDo（`robojudo`，轻量 C++/Rust 部署框架）与
  motion_tracking_controller 均支持 ONNX，移植回浏览器 JS 与 AMO 同路数
- **局限**：跟踪策略只会"放"一段动作，攻防切换、对手感知仍需上层战术——
  所以战术层可先沿用路线 A 的自博弈

### 路线 A（快赢）：现有底座上的自博弈战术层

不需要任何新框架：现有无头仿真（`tools/test_amo.mjs` 那套，node 直接跑）本身
就是一个完美的自博弈训练场。

- 把 `boxing_ai.mjs` 的决策点（何时接近/后撤/出哪招/何时摇闪）抽象成观测：
  双方距离、朝向差、对方拳套位置与速度、自身伤害、冷却状态（~20 维）
- 动作空间保持现有的招式池 + 步法指令（离散选择）
- 训练算法：对手池 + PPO/简单交叉熵策略梯度，在 node 无头环境跑
  （物理 500Hz、决策 5~10Hz，一个回合 ~20s 仿真时间，node 单核每秒可跑
  数百回合，几小时可达数十万回合）
- 上限：战术变聪明，但出拳动力学仍是脚本——**这是当前架构天花板的补丁，
  不是解法**，可与路线 B 叠加

### 其他看过并排除/降级的

| 项目 | 结论 |
|---|---|
| **PBHC / KungfuBot2 (VMS)** `TeleHuman/PBHC` | NeurIPS 2025 + ICRA 2026，多动作自适应跟踪很强；但 CC BY-NC 4.0 禁商用，且本项目上月移植 v1 checkpoint 失败（main 分支与权重版本不符）。KungfuBot2 的 VMS 统一了代码库，可观察社区复现情况后再试 |
| **HOMIE** `xiangmei-chen/OpenHomie` | 纯 MLP 最易移植（已在本项目备选清单），但 12 腿 RL + 上肢直控，无躯干动力学，出拳拟真度与现状同阶，不解决问题 |
| **AMO 本身** | 保留。它的指令接口（vx/vy/yaw + 躯干 + 手臂目标）依然是路线 B 战术层的执行底座 |
| **Unity/pygame 自博弈格斗**（NeuroStrike、StickFight 等） | 关键帧动画 + 命中盒，非物理真实，仅方法论参考（零和奖励设计可借鉴到路线 A） |
| **unitree_rl_gym** | 12dof 行走，与动作跟踪无关（此前已弃） |

---

## 3. 浏览器移植约束（所有路线共同）

本项目部署端是 MuJoCo WASM + 纯 JS 推理，因此任何新策略必须满足：

1. **策略规模**：MLP/CNN 且 ≤5M 参数（AMO student 17.8MB 已是上限量级）。
   BeyondMimic 的扩散策略（Transformer 20M 参数 × 20 步去噪）**不可移植**，
   只用它的动作跟踪策略（小型 MLP）。
2. **50Hz 推理 / 500Hz PD**：与 RoboStriker、BeyondMimic 的部署频率一致，
   无需改节奏。
3. **导出路径**：训练侧 ONNX → 手写 JS 前向（本项目的
   `rl_policy.mjs` 已验证过 AMO 的 Conv1d+MLP 全流程，误差 ~1e-7，管线现成）。
4. **双机对战** = 同一策略实例化两份，观测里加对方状态（RoboStriker 就是这么做的）。

---

## 4. 推荐实施顺序

```
第 1 步（无需 GPU，1~2 天）   路线 A：自博弈战术层
  └ 现有无头仿真上训练上层决策策略，直接替换 weightedPick

第 2 步（需 4090，3~7 天）    路线 B：拳击动作跟踪策略
  └ 拿 RoboStriker 仓库的 fight1_subject2.npz（或 GVHMR+GMR 从拳击视频自提）
  └ mjlab（MuJoCo-Warp）训练跟踪策略 → ONNX → JS 移植
  └ 拳击 AI 从"8 关节姿态切换"升级为"动作片段触发"（AMO 指令接口照旧）
  └ 参考范本：mujocolab/g1_spinkick_example（有现成 ONNX 可先跑通管线）

第 3 步（需 4090 + 数天，可选）路线 C：RoboStriker 完整复现
  └ Stage2 隐空间蒸馏 + Stage3 LS-NFSP 自博弈
  └ 两台机器人获得真正的攻防博弈；许可与正式仓库发布后再投入
```

第 1、2 步可以并行推进且互不阻塞：战术策略的观测/动作接口与动作跟踪策略的
触发接口是独立的，最后在浏览器里拼装。

---

## 附：核心链接

- RoboStriker 论文 https://arxiv.org/abs/2608.16195 · 项目页 https://sites.google.com/view/robo-striker · 代码 https://anonymous.4open.science/r/robostriker-B25B
- BeyondMimic 训练 https://github.com/HybridRobotics/whole_body_tracking · 部署 https://github.com/HybridRobotics/motion_tracking_controller
- mjlab 旋踢范例 https://github.com/mujocolab/g1_spinkick_example
- MimicKit（Xue Bin Peng，DeepMimic/AMP/ASE 统一框架）https://github.com/xbpeng/MimicKit
- PBHC/KungfuBot https://github.com/TeleHuman/PBHC （CC BY-NC 4.0）
- HOMIE https://github.com/xiangmei-chen/OpenHomie （禁商用）
- 数据管线：GVHMR（视频→SMPLX）+ GMR（重定向）+ GhostTrial 数据集示例
  https://huggingface.co/datasets/DaveRc/ghosttrial-g1-scorpion-motion
