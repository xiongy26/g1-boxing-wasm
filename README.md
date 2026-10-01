# G1 机器人拳击 · MuJoCo WASM + AMO 强化学习策略

在浏览器里让 **两台 Unitree G1（23 自由度，官方 AMO 模型）** 互相拳击的实时物理仿真。
物理引擎是 Google DeepMind 官方的 **MuJoCo WebAssembly 构建**（`@mujoco/mujoco` 3.13.0），
渲染用 Three.js，全部计算都在浏览器本地完成，无需任何服务端。

> **项目总计划与进度总览见 [`docs/ROADMAP.md`](docs/ROADMAP.md)**；
> GPU 训练执行手册见 [`docs/plan-mjlab-gpu-training.md`](docs/plan-mjlab-gpu-training.md)。

![截图](docs/screenshot_amo.jpg)

## 运行

```bash
node server.mjs 8080
# 打开 http://localhost:8080/
```

推荐使用 Chrome / Edge。首次加载需要下载约 21MB 的 WASM 引擎、51 个 STL 网格，
启用 AMO 模式时还会再下载 18MB 策略权重。

## AMO 强化学习模式（核心）

- 策略：**AMO (Adaptive Motion Optimization)**，RSS 2025，UT Austin
  （[OpenTeleVision/AMO](https://github.com/OpenTeleVision/AMO)，Apache-2.0），
  全身 RL 控制器，输出 15 个腿部+腰部关节动作，手臂通过输入适配器以 8 关节目标跟踪
- **默认开启 RL**：打开页面后两台机器人即挂载 AMO 策略（权重 18MB 在加载页与
  引擎/网格并行下载，不增加额外等待）；面板 **A: AMO 策略 / B: AMO 策略**
  可把任一台切回脚本模式或重新开启：
  - **AMO 模式**（默认）：策略以 50Hz 推理（纯 JS 重实现，与 torch 参考误差 ~1.5e-7），
    500Hz PD 扭矩输出，机器人**自主平衡、行走、摆臂**——没有任何吊架辅助，
    被击中时靠策略抗扰，KO 时移除策略让机器人自然瘫倒
  - **脚本模式**：姿态库状态机 + 可关闭的骨盆"吊架"辅助（旧的展示模式）
- 拳击 AI 在两种模式下共用：试探步、接近/后撤、刺拳/交叉拳/勾拳/连击、
  腰部转体出拳、命中后踉跄
- **战术 AI（自博弈训练）**：`src/tactics.mjs` 是一个 13 观测 → 10 动作的
  纯 JS 小策略网络，在无头 MuJoCo 里通过 REINFORCE 自博弈训练
  （`node tools/train_tactics.mjs`，详见下文），替代旧版
  "距离阈值 + 加权随机挑招式"。`vendor/tactics/tactics.json` 存在时浏览器
  自动加载并在面板提供开关；文件不存在时回退旧逻辑，完全可选

## 玩法与界面

- **击中头部 +2 分 / 击中躯干 +1 分**，双拳相碰判为格挡
- 伤害值累积（头部 1.5 / 躯干 1.0，随时间恢复）到阈值即 **KO 倒地**，2 秒后自动开下一回合
- 控制面板：仿真速度（0.5×/1×/2×）、攻击性、AMO 策略开关（A/B）、暂停、重置比赛、
  平衡辅助（仅脚本模式）、接触点可视化、镜头环绕
- 鼠标拖拽旋转视角，滚轮缩放

## 实现要点

```
src/rl_policy.mjs    AMO 策略运行时（纯 JS，浏览器/无头共用）
                     · student 网络 2474→1024→1024→512→15（ELU）
                     · 输入适配器 12→15（BatchNorm+LeakyReLU，手臂条件输入）
                     · 10 帧本体感知历史（Conv1d 编码）+ 25 帧额外历史
                     · 50Hz 策略 tick / 500Hz PD 扭矩（decimation 10）
src/tracking_policy.mjs  阶段4 追踪策略运行时（纯 JS，BeyondMimic/mjlab 导出）
                     · TrackingNetwork：154→512→256→128→29（ELU）+ 动作表 Gather
                     · TrackingFighter：50Hz 观测 / 200Hz PD，与 ONNX ~1e-6 对齐
                     · B 侧 yawOffset：参考姿态旋转 π（策略体坐标系不变性）
src/boxing_ai.mjs    拳击控制器（无头调参与浏览器共用）
                     · AMO 指令接口：vx/vy/朝向/躯干 + 8 臂关节目标
                     · 战术决策：tactics.json 策略优先，缺省回退
                       距离阈值 + 加权随机（行为与旧版一致）
                     · 追踪模式：战术决策只排队片段，切换发生在守卫边界
                     · 脚本回退模式：姿态库 + 吊架辅助
                     · 命中判定：真实接触对（拳套 × 头/躯干）+ 冷却 + 击退冲量
                     · 伤害值 KO（AMO 下即断扭矩软瘫倒地）/ 回合重置
src/tactics.mjs      战术策略网络（纯 JS MLP + REINFORCE 更新 + Adam）
src/main.js          浏览器主程序：WASM 初始化、VFS 喂入 STL、
                     mjv_updateScene 抽象场景 → Three.js 网格、HUD/计分/特效
tools/gen_amo_scene.py      生成 AMO 双机器人场景 models/scene_boxing_amo.xml
tools/gen_tracking_scene.py 生成 29-DoF 追踪双机器人场景
                            models/scene_boxing_tracking.xml（物理与 mjlab 训练端
                            逐项对齐：200Hz/armature/脚 priority 摩擦 0.6）
tools/cut_boxing_clips.py   LAFAN1 fight CSV → 拳击技能片段（守卫过渡段）
tools/extract_tracking_onnx.py  追踪 ONNX → bin/meta/测试向量（两种导出变体）
tools/deploy_tracking_clips.sh  训练 run 目录 → vendor/policy/boxing_*
tools/test_amo.mjs          无头回归：node tools/test_amo.mjs 22
tools/test_tracking.mjs     追踪策略 JS↔ONNX 对齐
tools/test_tracking_sim.mjs 追踪策略单机 sim2sim（WASM）
tools/test_tracking_boxing.mjs  追踪模式双机无头回归
vendor/policy/       amo.bin（18MB）+ spinkick.bin（管线保险）+ boxing_*（阶段4片段）
vendor/mujoco/       @mujoco/mujoco 3.13.0（Apache-2.0）
vendor/three/        three.js r170（MIT）
models/unitree_g1/   官方 G1 网格 assets（AMO 与追踪模型共用）
models/tracking_g1/  mjlab G1 29-DoF 基模型（阶段4）
```

### 追踪模式（阶段 4，`?scene=tracking` 或面板"追踪模式"）

`vendor/policy/boxing_{guard,jab,cross,hook}.*` 齐备时自动启用：
29-DoF G1 由动作片段策略驱动（战术决策排队片段，在守卫段首尾切换）。
当前为 v0 预览（guard 策略四槽位，训练收敛后用
`tools/deploy_tracking_clips.sh` 替换）。详见 `docs/stage4-tracking-notes.md`。

> **图例（2026-09-29）**：默认模式当前为 **guard-only**（守卫循环），完整拳法
> 组合循环见 `?scene=tracking&combo=1`（双机 12.5s 组合拳常驻循环）。
> 追加 `&faceoff=1` 开启面对面对抗子选项（B 机 180° 转身面向 A 机）。
> 调试类脚本（`_probe_*`/`_diag_*`）已归档至 `tools/_archive/`。

### 场景模型的坑（务必知道）

- AMO 策略训练于 **AMO 仓库的官方 g1.xml**（23 扭矩电机、腕部融合）。
  mujoco_menagerie 的模型（29 位置执行器）与策略不兼容，站立都会倒
- 官方 xml 所有 mesh geom 均为纯视觉（contype=0），**机器人只靠每只脚 4 个 5mm
  小球与地面接触**；拳套/头部球/躯干胶囊是本 demo 添加的碰撞体
  （躯干胶囊 density=0 不改变动力学）
- 官方 `<default>` 给所有铰链关节 `damping=2 frictionloss=0.2`——生成场景时
  丢失这两项会让策略"能站不能走"
- 头部碰撞球若放在官方头网格锚点（z=-0.044）会与腰部链接碾磨，行走必摔；
  必须放在真实头部中心（torso 系 z≈+0.406，质量 1.036kg 与官方 head_link 一致）

### 关键的 WASM 绑定注意点（踩坑记录）

- `data.qpos` / `data.ctrl` / `model.mesh_vert` 等是 **WASM 堆的实时视图**（Float64Array/Float32Array），直接读写即可
- **自由关节 qpos 布局 `[x,y,z, qw,qx,qy,qz]`——四元数从 qposadr+3 开始**；
  `qvel` 的角速度部分是**体坐标系**
- `data.contact` 每次访问会在堆上**分配一份副本**，用完必须 `delete()`，否则内存泄漏直至 `memory access out of bounds`
- `mjvScene.geoms`、`mjv_geoms.get(i)` 返回的包装对象同理需要 `delete()`
- timestep 在 `model.opt.timestep`（顶层没有 `model.timestep`）
- 从字符串加载带网格的模型：`MjModel.from_xml_string(xml, vfs)` + `vfs.addBuffer('unitree_g1/assets/x.STL', bytes)`（键要与 xml 的 meshdir 对上）
- **`mjvGeom.dataid` 在本 WASM 构建里是坏的**：对 mesh geom 返回的是真实 mesh id 的 **2 倍**，
  直接拿来取网格会导致机器人"散架"（每个零件装错 STL）。正确做法是用 `mjvGeom.objid`
  （geom id）反查 `model.geom_dataid[objid]` 拿真实 mesh id，见 `src/main.js` 的 WORKAROUND 注释

## 回归测试

```bash
node tools/test_amo.mjs 22   # AMO 双机器人 + KO 注入 + 回合恢复 → PASS
python .workbuddy/tmp/test_scene_dual.py   # python 侧 10s 双机器人回归
# 阶段 4 追踪模式：
node tools/test_tracking.mjs vendor/policy/boxing_guard   # JS↔ONNX 对齐
node tools/test_tracking_sim.mjs boxing_guard             # 单机 sim2sim
node tools/test_tracking_boxing.mjs 22 boxing             # 双机无头回归
```

## 战术 AI 自博弈训练

```bash
node tools/train_tactics.mjs --episodes 3000 --workers 6 --ep-len 24
node tools/eval_tactics.mjs 20 24   # 策略 vs 旧版加权随机的命中数对比
```

- 训练在脚本模式（无 AMO 权重）下进行：快且稳定，战术指令语义两种模式共用，
  学到的策略直接用于 AMO 模式
- REINFORCE + 移动平均 baseline + 熵正则（Adam），对手池取历史快照防止
  策略互相追尾；奖励 = 命中 +（头 3/躯干 1.5）、被击中 -2、格挡 +0.3、KO ±8
- 产物 `vendor/tactics/tactics.json`（约 15KB），浏览器加载后自动启用
- 调研报告（更进一步的方案：mjlab 动作跟踪 / RoboStriker 自博弈）见
  `docs/research-boxing-approaches.md`

## 版权

- MuJoCo：Apache-2.0（Google DeepMind）
- AMO 策略权重：Apache-2.0（OpenTeleVision/AMO, RSS 2025）
- Unitree G1 网格：Unitree 官方（经 AMO 仓库分发）
- three.js：MIT
