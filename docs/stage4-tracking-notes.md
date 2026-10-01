# 阶段 4 执行记录：拳击动作跟踪（mjlab GPU 管线）

> 2026-09-24。本文档是 `docs/plan-mjlab-gpu-training.md` 的执行记录：
> 本机（RTX 4060 Ti 16GB）实际跑通了计划书的整条管线，并修复了四个
> sim2sim 移植 bug。训练仍在后台进行时的状态见文末。

## 一、管线总览（已全部打通）

```
LAFAN1 fight CSV (HF lvhaidong/LAFAN1_Retargeting_Dataset, 30fps)
  → tools/cut_boxing_clips.py            切片段 + 守卫过渡段（0.5s ease + 1s hold）
  → mjlab csv_to_npz                     50Hz 重采样 + FK 补全 + wandb artifact
  → g1_boxing_task (mjlab 1.1.0, MuJoCo-Warp)  4096/2048 envs PPO 训练
  → 训练 run 目录自动导出 ONNX（随 checkpoint 刷新）
  → tools/extract_tracking_onnx.py       ONNX → vendor/policy/<name>.bin + meta + 测试向量
  → src/tracking_policy.mjs              纯 JS TrackingNetwork + TrackingFighter
  → models/scene_boxing_tracking.xml     29-DoF 双机器人 WASM 场景
  → tools/test_tracking*.mjs             逐向量对齐 / 单机 sim2sim / 双机回归
```

里程碑状态（对应计划书 §6）：

| 里程碑 | 状态 | 产出 |
|---|---|---|
| A mjlab 环境 | ✅ | `.workbuddy/gpu/g1_{spinkick_example,boxing_task}`，uv.lock 锁 warp-lang==1.12.0.dev20260126 |
| B 旋踢 ONNX → JS 对齐 | ✅ | `tools/test_tracking.mjs`：16 向量，动作误差 2.1e-6（相对 3.5e-7，float64 vs float32 求和差异），动作表 0 误差 |
| C 拳击片段数据 | ✅ | jab/cross/hook（从 fight1_subject2 延伸事件切片）+ guard_idle（合成弹跳）；渲染抽帧肉眼验收通过；wandb artifact `bear_robot/csv_to_npz/boxing_*` |
| D jab 训练 | 🔄 后台运行（4096 envs, 20k iters, ~3.5s/iter） | `logs/rsl_rl/g1_boxing/2026-09-24_00-34-05/` |
| E 其余片段 | 🔄 guard_idle 并行训练中；cross/hook 待训 | `.../2026-09-24_00-35-17/` |
| F 浏览器端移植 | ✅ | `src/tracking_policy.mjs` + sim2sim PASS（见下） |
| G 片段调度器 | ✅ | `boxing_ai.mjs` TACTIC_CLIP 调度 + `tools/test_tracking_boxing.mjs` PASS（v0 策略） |
| H 浏览器 E2E | ✅（v0 策略） | `.workbuddy/gpu/e2e_tracking.cjs`：58FPS、双机站立、片段切换 |

## 二、sim2sim 移植的四个坑（按发现顺序）

1. **力矩限幅的关节族提取**：关节名 `left_wrist_pitch_joint` 去掉 `_joint` 后
   再 `split('_').pop()` 得到 `pitch` 而不是 `wrist_pitch`，髋/膝全部落到 25Nm
   兜底值（应为 88/139）→ 机器人腿软跪倒、自撞爆炸。限幅表按完整族名
   （`hip_pitch`/`wrist_pitch`/...）索引。
2. **armature（转子反射惯量）缺失**：mjlab 通过 BuiltinPositionActuatorCfg 注入
   per-joint armature（hip_roll/knee 0.0251、ankle 0.0072、wrist 0.0036~0.0043），
   编译出的模型里有但 XML 里没有。显式 Euler 200Hz 下低惯量自由度没有 armature
   必然数值爆炸（踝关节一步到 200rad/s）。已把数值烤进
   `tools/gen_tracking_scene.py` 的 ARMATURE 表。
3. **四元数共轭写错**：anchor ori 观测需要 conj(q_robot)⊗q_ref，共轭是
   `(w,-x,-y,-z)`——把 w 也翻了等于没共轭（q ≡ -q），策略看到的是
   q_robot⊗q_ref，方向反了。特征：m01/m10 符号翻转。务必用
   `iw=rqw, ix=-rqx, iy=-rqy, iz=-rqz`。
4. **出生朝向 vs 片段朝向**：追踪策略对 anchor 位置误差是盲的
   （has_state_estimation=False 导出没有 anchor_pos 项），参考朝向偏差只会
   通过 anchor ori 观测呈现。B 机器人出生朝向 π 而片段烤死在朝向 0 → 策略
   试图转身 180° 而不是跟踪。解法：利用策略完全工作在体坐标系的事实，给 B 的
   参考四元数乘 Rz(π)（`TrackingFighter yawOffset`），等效于跟踪一段旋转过的
   片段，物理 yaw 不变性保证这一步是严格的。

另：训练端与部署端的 ONNX 导出**命名不同**（`policy.mlp.N.*` vs `actor.N.*`、
动作表是初始化器 vs Constant 节点），`extract_tracking_onnx.py` 两种都认。

## 三、追踪场景的物理对齐清单（scene_boxing_tracking.xml）

与 mjlab 训练环境逐项对齐，缺一项就会复现上节的爆炸：

- timestep 0.005（200Hz 仿真、50Hz 策略、decimation 4）、Euler、Newton×100
- 关节 damping=0 / frictionloss=0（与 AMO 场景的 damping=2/frictionloss=0.2 **不同**！）
- per-joint armature（见上）
- 脚碰撞体 condim=3、priority=1、friction 0.6（mjlab FULL_COLLISION 配置；
  priority 使脚的摩擦覆盖地面摩擦）；其余自碰撞 condim=1
- 地面摩擦 1/0.005/0.0001；拳套为 density=0 无质量球（追踪策略在分布内）
- PD：kp/kd/动作缩放/默认姿态全部来自 ONNX 元数据（与 g1_constants 的
  STIFFNESS_5020/7520/4010 家族一致）；力矩限幅按关节族（88/139/50/25/5）

## 四、JS 侧运行时（src/tracking_policy.mjs）

- `TrackingNetwork`：refAt(step)（动作表 Gather，含 min/max 钳位）+ infer(obs)
  （归一化 → Gemm/Elu×3 → Gemm），与 ONNX 逐向量对齐（tools/test_tracking.mjs）
- `TrackingFighter`：50Hz 观测构建（命令 58 + anchor ori 6 + 体角速 3 +
  关节位置/速度 58 + 上帧动作 29 = 154）→ 200Hz PD 写出（decimation 4），
  与训练端 `tracking_env_cfg.py`/`observations.py` 逐项核对过
- 观测基准对照：mjlab env 关噪声后 dump 的 154 维观测 vs JS 同状态重建，
  最大误差 1.5e-8（`/tmp/mjlab_obs2.json` 流程，见执行记录）

## 五、部署新训练好的策略

```bash
tools/deploy_tracking_clips.sh <guard_run_dir> [jab_run_dir] [cross_run_dir] [hook_run_dir]
node tools/test_tracking_sim.mjs boxing_guard
node tools/test_tracking_boxing.mjs 22 boxing
```

`vendor/policy/boxing_{guard,jab,cross,hook}.{bin,meta.json}` 齐了之后，
浏览器 `?scene=tracking`（面板"追踪模式"开关）自动进入追踪模式；
缺失的拳法片段回退 guard 策略。当前部署的是 guard_idle ~1k 迭代的早期
checkpoint（四槽位同策略，v0 预览：双机能弹跳对峙、KO/回合系统全通）。

## 六、待办（训练收敛后）

- [ ] jab 到 20k 迭代（ETA ~19h 自启动起）→ play 验收 → `deploy_tracking_clips.sh`
- [ ] guard_idle / cross / hook 依次补齐（GPU 串行 ~19h/片段@4096envs，
      或降 2048 envs 减显存压力）
- [ ] deploy 后重跑 `test_tracking_sim.mjs` / `test_tracking_boxing.mjs`
      （此刻双机无头回归才有真实拳法命中）
- [ ] 浏览器肉眼验收蹬地-转腰发力链（计划书验收 E）
- [ ] clip 时长与战术层节奏匹配微调（当前战术决策间隔 ~0.5s，片段含过渡段
      ~4.8s：决策只是排队，切换发生在守卫边界）

## 七、2026-09-26 真实感优化会话（docs/plan-realistic-boxing.md 落地）

按《G1 真人拳击效果优化方案》落地的一批运行时/数据/部署修改，以及
过程中发现的三个存量 bug。

### 7.1 发现并修复的存量 bug

1. **战术层梯度符号反转**（`src/tactics.mjs update()`）：dLogits 原先算的是
   目标函数 J = adv·logp + ent·H 的上升方向，却交给做 `params -= lr·g` 的
   Adam——整个 REINFORCE 信号反着训。此前三连纯 REINFORCE 崩到互避对峙、
   靠 ring prior 兜底，与此吻合。已修复；`vendor/tactics/tactics.json` 是
   带病梯度训出的权重，仍可加载，但建议重训。
2. **netGain 链式因子缺失**（同函数）：forward 的最终 logits =
   netGain·net + priorW·prior，反向传播没乘 netGain——netGain<1 时梯度被
   放大 1/netGain（部署权重 netGain=0.1 ⇒ 梯度×10 噪声）。已修复。
   两处均由新增的数值梯度检查 `tools/test_tactics_grad.mjs` 抓出
   （解析 vs 有限差分，三个工作点 + softmax 混合一致性，全 PASS）。
3. **追踪模式辅助自上线以来静默失效**（`boxing_ai.mjs`）：`freeDof` 按 AMO
   场景的关节名 `side_pelvis` 解析自由关节，追踪场景的自由关节叫
   `side_floating_base_joint` → id=-1 → `qvel` 读 undefined → NaN 污染全部
   辅助输出，`xfrc_applied` 恒为 0——垂直托力/扶正力矩/水平阻尼/分离力
   从未生效，辅助开/关的追踪轨迹完全相同（此前所有"追踪模式 + 辅助"的
   观察结果实为裸奔）。已改为经骨盆 body 的第一个关节解析（与
   TrackingFighter 一致）并加自由关节校验。
4. **修复后的辅助实测有害，默认关断**：辅助真正生效后出拳段摔得更快——
   追踪策略是无外力训练的，扶正力矩（出拳前倾时 -30~-45 N·m）与水平阻尼
   直接对抗片段动力学。新增 `trackingAssist` 开关（默认 false）：辅助只对
   脚本回退模式生效，追踪策略按无外力裸奔评估（符合方案阶段 A"无辅助
   出拳收拳"的验收口径）；未来训练"带辅助课程"的策略后再开。
5. **KO 断力矩只清 23 个执行器**（`boxing_ai.writeLimp`）：追踪场景 29 DoF
   含每臂 3 个腕部执行器，KO 后手腕残留 PD 力矩。已按该侧执行器表全清。

### 7.2 运行时（src/boxing_ai.mjs，方案 §4.2/§4.4/§4.5/§4.6）

- **决策纪律**：追踪模式到达空闲阈值后不再每个物理步重采样（原先 ~400 次/秒
  ——旧代码里 3% 概率的出拳决策靠这个 bug 变成必中，掩盖了战术先验在
  追踪模式距离带内几乎不出拳的事实）；出拳意图带 1.6s TTL，超时自动取消；
  回合稳定期 2.5s + 出拳冷却 0.5s；cross/hook 前冲大暂不允许提前切出，
  jab（前冲 ~0.2m）在距离合适且参考回到守卫静态帧（≤0.035 rad）时提前
  切入——守卫片段 ~9s 的最坏等待消除。
- **步法缺位的过渡语义**：tactics 步法决策在追踪模式全部映射原地守卫片段，
  "接近"不可执行、距离永远收不进 jab 带——出拳距离带内的"接近/等待"
  决策按先验本意（band 边缘刺拳试探）转为 jab。步法片段训练落地（§4.2）
  后移除。
- **面向对手**（AMO 模式）：朝向指令从固定 0/π 改为平滑跟踪对手实际方位
  （限速 2.5 rad/s）。追踪模式仍由片段参考拥有航向——新指令必须先进训练。
- **命中规则**：有效命中要求出拳处于发力阶段（元数据出手窗或 strike 态）
  且拳速 EMA ≥0.8 m/s（实测有效命中 2.0-4.9 m/s，挤压/余摆接触 0.1-0.8）；
  以出拳事件为单位去重（同一次出拳只计一分）；新增 graze 事件（不计分）
  ——双机回归里此前会计分的贴身接触现在被正确分类为 graze；
  击退改用实际接触方向、限幅 30N×60ms（原为固定 -forward×40N）。
- **辅助计量**：assistUsage 对 |F|、|τ| 时间积分（排除击退、NaN 不污染），
  stats 栏与无头回归报告；测试支持 `--seed`（确定性 RNG mulberry32）与
  `--assist`（§7.4 辅助开关分别报告；追踪模式下辅助关断，开关不改变轨迹）。
- 无战术权重时追踪模式回退为加权随机排队（原先永不主动出拳）。

### 7.3 数据与检查（方案 §4.1）

- `tools/cut_boxing_clips.py` v3：书端换成 FK 验证过的前后脚拳击架势
  （拳套距头 0.14-0.16m、下颌高度、收肘、L 前脚 0.26m/横距 0.29m，肘留
  限位裕量）；过渡 pad 1.0→0.4s、blend 0.5→0.4s（出拳片段 4.0→2.6s）；
  manifest schema v2：动作阶段秒标 + 50Hz 帧索引 + 左右手/支撑脚/来源。
  `--stance v2` 可复现旧部署训练数据。产物在 `data/clips_v3/`（v2 保留）。
- `tools/check_clip.mjs`：FK 回放检查（护头距离/骨盆高度/滑脚/关节范围/
  出手窗检测/前冲标定），`--enrich` 把实测值回写 manifest。
  **发现**：现有 LAFAN 重定向出拳窗口脚底滑移 0.22-0.71m（重定向伪影，
  见 manifest measured.foot_slide_m）——重训前需换窗或修重定向。
- `tools/extract_tracking_onnx.py --clip-meta <manifest>`：片段元数据注入
  策略 meta 的 `motion.clip`（调度器按元数据取出手窗/出拳距离窗）。
- `tools/deploy_tracking_clips.sh`：部署时注入元数据、把训练 run 写回清单
  policy_run 字段、**四槽位权重哈希去重检查**（本次实测 boxing_jab ==
  boxing_hook 同哈希，方案 §2 记录的权重复用即此；重复部署直接 FAIL）。

### 7.4 回归状态（2026-09-26）

| 检查 | 结果 |
|---|---|
| `test_amo.mjs 22`（AMO 双机，新命中/朝向/调度/freeDof 解析） | PASS |
| `test_tracking.mjs` × guard/jab/cross/hook/spinkick | 全 PASS |
| `test_tracking_sim.mjs` ×4 | 全 PASS（jab 与 hook 输出一致 = 权重相同，见上） |
| `test_tracking_boxing.mjs 22 boxing --seed N` | FAIL——与改动前基线同类：v2 拳击策略在双机接触中出拳必摔（单机 sim2sim 通过 minZ 0.48，双机带接触即倒，基线复测同样 FAIL）。调度/命中分类按预期工作（贴身接触分类为 graze 不再假计分）。根治 = §4.1 重训数据 + §4.6 抗扰训练 |
| `test_tactics_grad.mjs`（新增） | PASS |
| `check_clip.mjs clips_v3 --enrich`（新增） | 守卫 PASS；出拳片段 FAIL 于滑移（数据问题，如实暴露） |

浏览器新增：评估视角固定机位、0.25× 慢放、拳套轨迹（四拳套最近 1.5s）、
stats 栏辅助用量/命中数（验收报告引用）。

### 7.5 下一步（按方案 §8 优先级）

1. 修滑移（换窗/重定向修正）→ 用 v3 架势片段重训 guard+jab → 无辅助单机验收
   （阶段 A 门槛）。
2. 修复了梯度的战术层重训（现有 tactics.json 建议弃用重训）。
3. 面向目标步法训练（§4.2）——需要先给运动层训练加相对目标观测（§4.3 课程）。

## 八、2026-09-28 追踪模式切换摔倒诊断与临时修复

### 8.1 双机制结论

- **机制 A：片段烘焙朝向不一致**。以各片段策略 meta 的 anchor body
  （torso_link，body_names[7]）ref0 世界四元数计：guard yaw 0°、cross
  −15.3°、jab+hook +198.2°（归一化 (−π,π] 后即 −161.8°）。切换片段时不
  校正朝向，anchor-ori 观测在切换瞬间出现 ~162-198° 的恒定误差——而训练
  回合起点（RSI）该观测恒为 0，策略直接分布外，表现为切换即倒。
- **机制 B：jab/hook 权重存活余量不足**。对照实验：干净 keyframe 重置 +
  RSI 到烘焙位姿可以存活整段，但 minZ 仅 0.481（骨盆余量 8cm）——任何
  ~1° 级基座残差都会在深蹲发力段（片段内 ~1.8s）被放大到翻倒；cross
  （20k 迭代）全部切换变体存活，minZ 0.576-0.581。本次复测：
  `test_tracking_sim.mjs` guard 0.749 / jab 0.481 / cross 0.578 / hook
  0.481（hook≡jab：boxing_hook.bin 与 boxing_jab.bin 字节一致，dcc37ab
  的权重复用即此）。

### 8.2 修复内容（src/tracking_policy.mjs + src/boxing_ai.mjs）

- **重锚定角统一公式**：Δ = 当前机器人 torso 实际 yaw − 目标片段 ref0
  torso yaw，yaw 提取统一走 `yawOfQuat()`（wxyz 四元数、ZYX 欧拉、归一化
  (−π,π]；裸 2*atan2(z,w) 在后半圈给出 (π,2π] 会差出整 360°）。A/B 侧
  统一处理，替换 B 侧硬编码 yawOffset=π（`rebaseYawDelta()`）。
- **swapTo 原位重锚定 RSI**：`TrackingFighter.rebaseToReference(Δ)` 把
  机器人状态设为训练回合起点——29 关节 qpos/qvel 取参考第 0 帧、基座
  quat = ref0 ⊗ Rz(Δ)（wxyz Hamilton **后乘**，与 policyTick 的 _qz 观测
  旋转同一约定；当前 ref0 基座严格竖直即纯 yaw，前后乘精确等价，但两处
  实现须保持同式同 Δ、勿单侧修改）、z 取参考值、x/y 保留（战斗站位
  不动）、基座 6 维速度清零、lastAction/pdTarget 归零。Δ 同时驱动参考
  旋转（yawOffset）与状态重锚定，两侧代数相消使切换瞬间 anchor-ori 观测
  归零。
- **拳法门禁（临时，2026-09-28 队长决策：全部禁用）**：
  `CLIP_ENABLED = { guard: true, jab: false, cross: false, hook: false }`
  ——全部拳法片段在决策层（decideTactics 追踪分支与加权随机回退分支）
  映射回 guard、意图不入队，待 v3 重训后逐个恢复（验收门槛：无辅助单机
  RSI minZ≥0.55 + 双机三次 seed 回归无切换摔倒）。hook 槽位本就是 jab
  权重复制品（见 dcc37ab）。

### 8.3 重训要求（恢复 jab/hook 的前置条件）

1. v3 片段统一把根节点烘焙朝向归一到 yaw 0（消除机制 A 的来源）；
2. 修脚底滑移（`check_clip.mjs` 已测出出拳窗口 0.22-0.71m，重定向伪影，
   见 §7.3）；
3. 训练加 RSI 噪声课程：带噪 RSI 起步 + 片段中段起步（消除机制 B 的
   8cm 存活余量）；
4. 部署时 `tools/extract_tracking_onnx.py --clip-meta` 注入片段元数据
   （出手窗/出拳距离窗，调度器依赖）。

### 8.4 验证结果（2026-09-28，无辅助、tactics 权重加载）

| 检查 | 结果 |
|---|---|
| `test_tracking_sim.mjs` ×4 片段 | 全 PASS：guard minZ 0.749 / jab 0.481 / cross 0.578 / hook 0.481（hook≡jab 同权重，maxCtrl/误差也一致）；全部禁用后复跑 guard 仍 PASS |
| `test_tracking_boxing.mjs 22 boxing --seed 1` | PASS：minZ A=0.749 B=0.750，swaps=0/0，hits=0/0，grazes=0，ko=1[A@13.2 注入]（门禁期预期：零切换零出拳） |
| `--seed 2` | PASS：同上，swaps=0/0，ko=1[A@13.2 注入] |
| `--seed 3` | PASS：同上，swaps=0/0，ko=1[A@13.2 注入]——三 seed 均无切换摔倒 |
| `test_amo.mjs 22` | PASS：minZ 0.733/0.733，KO 注入塌倒+回合重置正常，divergences=0 |
| `test_tactics_grad.mjs` | PASS（6 项梯度/概率一致性检查） |

测试脚本同步调整（`tools/test_tracking_boxing.mjs`）：swaps≥1 硬断言改为
纯观察项 crossQueued/crossPlayed（门禁期 swaps=0、零出拳即预期行为）；
RESULT 行新增 minZ/swaps/hits/ko 时间/cross 观察摘要。minZ≥0.40、KO 注入
触发、回合重置恢复三条硬断言未动。

### 8.5 观察与决策记录

- 门禁期（全部拳法禁用）追踪模式为守卫循环对峙：零切换、零出拳是预期
  行为而非缺陷（swaps=0、hits=0）；守卫片段 bookend 循环经 carryOver
  原地重启参考时钟（机器人状态不动、策略侧回到回合起点），不产生切换
  跳变。
- **队长决策（2026-09-28）：cross 一并临时禁用**。依据：全禁前套件中
  cross 活体执行共 2 次、摔 1 次（seed 1 存活 minZ 0.576；seed 3 B@9.0
  切到 cross 瞬间摔倒触发 KO，早于 13.2s 注入点）——摔倒率 1/2 不可接
  受；干净 RSI 余量 0.578-0.40≈0.18m 会被接触扰动吃掉（注：minZ 采样在
  koState 置位即停，摔倒本身不进 minZ 统计，经 ko 事件时间确认）。
- 恢复路径：按 8.3 重训要求完成 v3 片段后逐个片段过验收门槛——无辅助
  单机 RSI minZ≥0.55 + 双机三次 seed 回归无切换摔倒——再放开
  CLIP_ENABLED。
