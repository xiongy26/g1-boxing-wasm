# G1 Boxing WASM — 双人形机器人浏览器拳击

在浏览器里让 **两台 Unitree G1 人形机器人** 实时物理对打：
物理引擎是 Google DeepMind 官方的 **MuJoCo WebAssembly 构建**（`@mujoco/mujoco` 3.13.0），
机器人由强化学习策略驱动，Three.js 渲染。
全部计算在浏览器本地完成，**无需任何服务端、无需 GPU**。

![截图](docs/screenshot_amo.jpg)

## 特性

- **真实全身动力学**：两台 23/29 自由度 G1，无吊架辅助，自主平衡、行走、抗扰
- **RL 策略驱动**：AMO（RSS 2025）全身控制器 50Hz 推理 / 500Hz PD 扭矩，纯 JS 重实现，
  与 torch 参考输出误差 ~1.5e-7；对打策略为 168 维观测（含对手状态）的自博弈训练产物
- **完整比赛规则**：击中头部 +2 / 躯干 +1，双拳相碰判格挡，伤害累积 KO，
  自动回合制，胜方举拳庆祝
- **战术 AI**：13 观测 → 10 动作的小型策略网络，仓库内即可用 CPU 完成自博弈训练
- **一键视频录制**：3D 画面 + 计分 HUD 合成录制，自动下载 MP4 / WebM
- **零后端**：`node server.mjs` 只是一个静态文件服务器

## 快速开始

```bash
node server.mjs 8080   # 端口参数可选，默认 8080；被占用时换一个
# 打开 http://localhost:8080/
```

任意较新版本的 Node.js 即可（仅用于起静态服务）。推荐 Chrome / Edge。
首次加载需下载约 21MB 的 WASM 引擎与 51 个 STL 网格，AMO 模式另需 18MB 策略权重
（与引擎并行下载，不增加等待）。

## 模式

四种玩法模式推荐在页面左下面板的**模式中心**直接点击卡片切换（当前模式金色高亮），
URL 参数与卡片一一对应：

| URL | 模式 | 场景 |
|---|---|---|
| 无参数 | AMO 自由模式 | 默认场景：23-DoF，两台机器人由 AMO 策略驱动拳击 |
| `?scene=tracking` | 守卫练习 | 29-DoF 动作片段追踪场景（缺权重时自动回退 AMO） |
| `?scene=tracking&combo=1` | 组合拳表演 | 双机组合拳常驻循环；`&faceoff=1` 让 B 机 180° 面向 A |
| `?scene=tracking&fight=1` | **真实对打**（推荐） | 两台 G1 由同一 fight 策略自主对打，KO / 回合 / 记分全自动 |

调试参数 `&policy=<名字>.bin` 可加载 `vendor/policy/` 下的其他 fight 权重
（如 `&policy=boxing_fight_dev.bin`）；非法名或权重缺失时自动回退默认行为。

## 玩法与界面

- 打开页面即可观看完整比赛，比分、回合与 KO **全自动进行**，无需任何操作
- 首次访问会弹出欢迎引导浮层；面板右上"？"可随时重看
- **机器人策略 A / B**：每台机器人可独立切换 追踪/fight 策略 · AMO 策略 · 脚本模式，
  例如让一台打策略、另一台当脚本木桩陪练
- 主区控件：仿真速度（0.25×–2×）、攻击性、暂停、重置比赛、镜头环绕、录制视频
- 调试分组（默认收起）：平衡辅助、接触点可视化、评估视角、拳套轨迹
- 所有按钮均有中文悬浮说明；鼠标拖拽旋转视角，滚轮缩放

## 视频录制

面板「● 录制视频」按钮一键录制：再次点击停止，自动下载
`g1-boxing-YYYYMMDD-HHMMSS.<ext>`。

- **所见即所得**：3D 对局画面 + 比分/血条/回合横幅/命中浮字 合成进视频；
  受击时的屏幕边缘闪光不进入视频
- **格式自适应**：MP4 直录优先 H.264 High 档（需 Chrome 126+），不支持时按
  `webm/h264 → webm/vp9 → webm` 逐级回退；码率按画面高度 6–24 Mbps 自适应，
  录制分辨率跟随窗口大小 × 设备像素比（上限 2×）
- **已知限制**：录制期间切换模式或刷新会中断录制；标签页切后台后录制画面冻结
- 自动化入口：控制台句柄 `window.__rec`（`start` / `stop` / `state()` / `lastBlob`）

## 技术实现

```
src/
  rl_policy.mjs        AMO 策略运行时（纯 JS，浏览器/无头共用）
                       student 网络 + 输入适配器 + 本体感知历史编码，
                       50Hz 策略 tick / 500Hz PD 扭矩
  tracking_policy.mjs  29-DoF 动作片段追踪策略运行时（BeyondMimic/mjlab 导出）
                       50Hz 观测 / 200Hz PD，与 ONNX 输出 ~1e-6 对齐
  boxing_ai.mjs        拳击控制器（无头调参与浏览器共用）
                       接近/后撤、刺拳/交叉拳/勾拳/连击、命中判定、伤害 KO、
                       fight 模式的航向伺服与守卫形态维持
  tactics.mjs          战术策略网络（纯 JS MLP + REINFORCE + Adam）
  main.js              浏览器主程序：WASM 初始化、STL 喂入 VFS、
                       MuJoCo 抽象场景 → Three.js 网格、HUD/计分/特效
models/
  unitree_g1/          官方 G1 网格 assets（AMO 模型）
  tracking_g1/         mjlab G1 29-DoF 基模型
  scene_*.xml          三种场景 XML（AMO / 追踪）
vendor/
  mujoco/              @mujoco/mujoco 3.13.0（Apache-2.0）
  three/               three.js r170（MIT）
  policy/              部署权重：amo.bin（18MB）、boxing_fight.*（对打）、
                       boxing_{guard,jab,cross,hook,combo}.*（动作片段），
                       以及开发期中间产物（.bin + _meta.json + _test_vectors.json 三件套）
  tactics/             tactics.json（战术网络权重，约 15KB，可选）
tools/
  train_tactics.mjs          战术 AI 自博弈训练（CPU）
  eval_tactics.mjs           战术策略 vs 旧版启发式的对比评估
  extract_tracking_onnx.py   ONNX 策略 → bin/meta/测试向量 三件套
  test_tracking.mjs          策略 JS↔ONNX 数值对齐
  test_tracking_sim.mjs      单机 sim2sim（WASM）
  test_tracking_boxing.mjs   双机无头回归（含 fight 量化评估）
  check_fight_meta.mjs       fight 观测布局契约检查
```

### 场景模型要点

- AMO 策略训练于 AMO 仓库的官方 g1.xml（23 扭矩电机、腕部融合）；mujoco_menagerie
  的模型（29 位置执行器）与该策略不兼容
- 官方 xml 所有 mesh geom 均为纯视觉（contype=0），机器人只靠每只脚 4 个 5mm 小球
  与地面接触；拳套/头部球/躯干胶囊是本 demo 添加的碰撞体（躯干胶囊 density=0，
  不改变动力学）
- 官方 `<default>` 给所有铰链关节 `damping=2 frictionloss=0.2`，生成场景时丢失
  这两项会让策略"能站不能走"
- 头部碰撞球必须放在真实头部中心（torso 系 z≈+0.406），放在官方头网格锚点
  （z=-0.044）会与腰部链接碾磨，行走必摔

### MuJoCo WASM 绑定注意点

- `data.qpos` / `data.ctrl` / `model.mesh_vert` 等是 WASM 堆的实时视图，直接读写即可
- 自由关节 qpos 布局为 `[x,y,z, qw,qx,qy,qz]`，四元数从 qposadr+3 开始；
  qvel 的角速度部分是体坐标系
- `data.contact` 每次访问会在堆上分配副本，用完必须 `delete()`，否则内存泄漏；
  `mjvScene.geoms`、`mjv_geoms.get(i)` 返回的包装对象同理
- timestep 在 `model.opt.timestep`（顶层没有 `model.timestep`）
- 从字符串加载带网格的模型：`MjModel.from_xml_string(xml, vfs)` + 
  `vfs.addBuffer('unitree_g1/assets/x.STL', bytes)`（键要与 xml 的 meshdir 对上）
- **本 WASM 构建的 `mjvGeom.dataid` 是坏的**：对 mesh geom 返回真实 mesh id 的
  2 倍，直接拿来取网格会让机器人"散架"。正确做法是用 `mjvGeom.objid` 反查
  `model.geom_dataid[objid]`，见 `src/main.js` 的 WORKAROUND 注释

## 回归测试

```bash
node tools/test_tracking.mjs vendor/policy/boxing_guard   # 策略 JS↔ONNX 对齐
node tools/test_tracking_sim.mjs boxing_guard             # 单机 sim2sim
node tools/test_tracking_boxing.mjs 22 boxing             # 双机追踪回归（22s）
node tools/test_tracking_boxing.mjs 60 --fight --seed 1   # 双机对打 60s 量化
node tools/check_fight_meta.mjs                           # fight 观测布局契约检查
node tools/test_tactics_grad.mjs                          # 战术网络梯度检查
```

`60 --fight` 会输出互面误差（median/p90、>90° 占比）、交战率、命中数、KO 与
双机站立占比等指标；验收门槛的定义与历史实测见
[`docs/plan-fight-selfplay-v3.md`](docs/plan-fight-selfplay-v3.md)。

已知问题：`node tools/test_tracking_boxing.mjs`（默认 16s 无参数）当前会因
A 机早期摔倒触发 minZ 门槛而失败，属航向伺服引入前的历史遗留，与 fight 模式无关。

## 策略训练

本仓库为**纯推理**仓库：对打与动作片段策略的训练在独立的 GPU 训练管线中完成
（基于 BeyondMimic/mjlab + rsl_rl，未随本仓库分发），训练产物经
`tools/extract_tracking_onnx.py` 转换为本仓库的权重三件套后部署：

```bash
uv run --no-project --with onnx --with onnxruntime python \
    tools/extract_tracking_onnx.py <run目录>/<run名>.onnx vendor/policy/<slot>
# 之后依次回归：test_tracking.mjs → test_tracking_sim.mjs → test_tracking_boxing.mjs
```

**文件名必须匹配加载端硬编码槽位**：动作片段 `boxing_{guard,jab,cross,hook,combo}`
（`src/main.js` 的 `TRACKING_CLIPS`），对打策略 `boxing_fight`（同文件）。

唯一的例外是**战术 AI**，它在本仓库内训练（纯 node、CPU 即可）：

```bash
node tools/train_tactics.mjs --episodes 3000 --workers 6 --ep-len 24
node tools/eval_tactics.mjs 20 24   # vs 旧版加权随机的命中数对比
```

- REINFORCE + 移动平均 baseline + 熵正则（Adam），对手池取历史快照防止策略互相追尾；
  奖励 = 命中（头 3/躯干 1.5）、被击中 -2、格挡 +0.3、KO ±8
- 训练在脚本模式下进行（无 AMO 权重），快且稳定；战术指令语义两种模式共用
- 产物 `vendor/tactics/tactics.json` 约 15KB，存在时浏览器自动加载并在面板提供开关，
  不存在时回退启发式逻辑，完全可选
- 方案调研见 [`docs/research-boxing-approaches.md`](docs/research-boxing-approaches.md)、
  [`docs/tactics-selfplay-notes.md`](docs/tactics-selfplay-notes.md)

## 更多文档

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — 项目总计划、阶段划分与关键结论
- [`docs/plan-fight-selfplay-v3.md`](docs/plan-fight-selfplay-v3.md) — 对打策略自博弈
  v3 方案与各训练轮评估记录
- [`docs/stage4-tracking-notes.md`](docs/stage4-tracking-notes.md) — 动作片段追踪阶段笔记
- [`docs/research-boxing-approaches.md`](docs/research-boxing-approaches.md) — 机器人拳击
  技术方案调研

## 许可证

本项目自身代码（`src/`、`tools/`、`server.mjs`、`models/scene_*.xml` 等）
以 [MIT License](LICENSE) 发布。包含的第三方组件：

| 组件 | 许可证 | 来源 |
|---|---|---|
| MuJoCo（`@mujoco/mujoco` 3.13.0） | Apache-2.0 | Google DeepMind |
| AMO 策略权重 | Apache-2.0 | [OpenTeleVision/AMO](https://github.com/OpenTeleVision/AMO)（RSS 2025） |
| Unitree G1 网格 | 见 [models/unitree_g1/LICENSE](models/unitree_g1/LICENSE) | Unitree 官方（经 AMO 仓库分发） |
| three.js r170 | MIT | three.js |
