# G1 拳击项目总计划（ROADMAP）

> 最后更新：2026-09-23。本文档是项目的**唯一进度总览**，各专项细节见文末链接的子文档。
> 项目目标：在浏览器（MuJoCo WASM + 纯 JS 推理，无服务器）里让两台 Unitree G1
> 进行物理真实的自主拳击对战——出拳有真实发力链、战术有攻防博弈。

---

## 一、技术架构总览

```
┌─ 渲染层  three.js 网格同步 + HUD            [已完成]
├─ 物理层  MuJoCo WASM，500Hz，双机器人场景     [已完成]
├─ 运动层  全身控制策略（50Hz 推理 / 500Hz PD） [见下方阶段]
├─ 战术层  决策策略：何时接近/出拳/防守         [第 1 步已完成]
└─ 裁判层  命中判定/伤害/KO/回合               [已完成]
```

分层解耦，各层可独立替换。运动层是当前唯一还在演进的层。

---

## 二、进度总表

| # | 阶段 | 内容 | 状态 | 完成日期 | 关键产出 |
|---|---|---|---|---|---|
| 0 | 物理底座 | 双机器人场景 XML、接触/命中判定、伤害值 KO（断扭矩软瘫）、回合重置、击退冲量 | ✅ | 2026-09 初 | `models/scene_boxing_amo.xml`、`src/boxing_ai.mjs` |
| 1 | AMO 全身策略 | AMO (RSS 2025) ONNX 权重移植浏览器：student MLP 2474→…→15，50Hz/500Hz，与 PyTorch 参考 ~1e-7 对齐 | ✅ | 2026-09-20 | `src/rl_policy.mjs`、`vendor/policy/amo.bin` |
| 2 | 方案调研 | GitHub 全景调研，确定三步走路线 | ✅ | 2026-09-21 | `docs/research-boxing-approaches.md` |
| 3 | 战术层自博弈 | REINFORCE 自博弈 + 距离带先验（prior-first）+ netGain 残差限幅；评估 1.13 hits 比值反超旧版 | ✅ | 2026-09-21 | `src/tactics.mjs`、`vendor/tactics/tactics.json`、`docs/tactics-selfplay-notes.md` |
| 4 | **拳击动作跟踪（当前阶段）** | mjlab/MuJoCo-Warp GPU 训练拳击片段跟踪策略，替换脚本手臂 | 🔄 管线全通（A/B/C/F/G/H ✅，D/E 训练中）；执行记录见 `docs/stage4-tracking-notes.md` | — | `src/tracking_policy.mjs`、`models/scene_boxing_tracking.xml`、`tools/test_tracking*.mjs`、`vendor/policy/boxing_*`（v0） |
| 5 | 隐空间自博弈（可选） | RoboStriker Stage2/3：隐空间蒸馏 + LS-NFSP 攻防自博弈 | ⏸ 未启动 | — | — |

当前对战形态：AMO 管平衡与步法 + 战术 AI（先验主导）管决策 + 脚本 8 关节姿态管出拳。
**天花板**：出拳无蹬地转腰发力链——这正是阶段 4 要解决的。
阶段 4 追踪模式（`?scene=tracking`）已可在浏览器运行（v0：guard 策略四槽位），
双机器人以 29-DoF 追踪策略自主平衡+守卫弹跳，拳法片段训练收敛后替换即得发力链。

---

## 三、已完成阶段的关键结论（防回退备忘）

1. **AMO 移植**：`mjvGeom.dataid` 对 mesh geom 返回 2 倍 id，必须用 `objid` 反查；
   场景三坑（joint damping=2/frictionloss=0.2、头碰撞球位置、官方 mesh contype=0）。
2. **战术自博弈**：小样本 REINFORCE 必然塌缩到互避 standoff（七轮实验实证）；
   唯一可行解 = 距离带先验硬性拥有"是否交战"决策 + `netGain=0.1` 限幅网络残差。
   部署权重 `priorW=1.0 / netGain=0.1` 烤进 JSON，浏览器行为 = 评估行为。
3. **训练环境**：脚本回退模式在 AMO 场景站不住（吊架 < 整机重力），
   任何训练都必须挂 AMO 权重；无头 node 管线（6 worker，~1.4×实时/worker）
   是现成的 sim2sim 回归器。

---

## 四、阶段 4：拳击动作跟踪（当前任务）

> 2026-09-24 更新：本机检出 RTX 4060 Ti 16GB + wandb 凭据，整条管线已在**本机**
> 执行（原计划假设"本机无 GPU"）。里程碑 A/B/C/F/G/H 已完成并全部有自动化验收，
> D/E（jab 与 guard_idle 的 20k 迭代 PPO）正在后台训练，cross/hook 排队。
> 执行细节、四个 sim2sim 移植 bug 与部署步骤：**`docs/stage4-tracking-notes.md`**。

**执行地点**：本机（RTX 4060 Ti 16GB）。执行手册：`docs/plan-mjlab-gpu-training.md`
（含逐步命令、验收标准、风险表——此处只列里程碑）。

| 里程碑 | 内容 | 预估 |
|---|---|---|
| A | mjlab 环境跑通（锁 v1.1.0，跟随旋踢范例 uv.lock） | 0.5 天 |
| B | **旋踢预训练 ONNX 在本仓库 JS 前向对齐**（训练前的移植链路保险） | 0.5 天 |
| C | 拳击片段数据：RoboStriker npz / LAFAN1 重定向 → 转换 + 守卫过渡段 + registry | 1 天 |
| D | `jab` 片段训练收敛 → play 验收 → ONNX 导出 | 1~2 天 |
| E | 其余片段（guard_idle / cross / hook / bob_weave）训完 | 1~2 天 |
| F | 浏览器端 `TrackingNetwork` 移植 + 测试向量对齐 | 1 天 |
| G | `boxing_ai.mjs` 片段调度器 + cross-fade + 无头回归 PASS | 1 天 |
| H | 浏览器 E2E PASS，肉眼确认蹬地-转腰发力链 | 0.5 天 |

**集成方式**：运动层从"单 AMO 策略"变为"片段策略库"——
战术 AI 的决策输出（jab/cross/hook/步法）不再映射到臂关节姿态，
而是触发对应跟踪策略，在片段首尾的守卫站姿过渡段做切换。
裁判层/战术层/渲染层全部不动。

---

## 五、阶段 5 与后续方向（可选，按价值排序）

1. **RoboStriker Stage2/3**（需 4090 + 数天）：隐空间蒸馏 + LS-NFSP 自博弈，
   两台机器人获得真正的攻防博弈；等匿名仓库转正式发布、许可明确后启动。
2. **单策略多技能**：conditioned-on-clip-id 的统一跟踪策略，替代多 ONNX 切换
   （若阶段 4 的片段切换出现抖动则提前）。
3. **从真实比赛视频自提动作**：GVHMR + GMR 管线，扩充拳法库（组合拳、闪避反击）。
4. **对抗性战术再学习**：动作跟踪策略上线后，战术层（tactics.mjs）在新的
   动作空间上重训——观测/动作接口已解耦，训练基建（worker 池、评估协议）现成。

---

## 六、验证体系（贯穿所有阶段）

- `node tools/test_amo.mjs 22` — 无头回归：双机对战 + KO 注入 + 回合恢复（每次改动必跑）
- `node tools/eval_tactics.mjs 16 14` — 战术层 vs 旧版量化对比
- `node tools/test_tracking.mjs vendor/policy/<name>` — 追踪策略 JS 前向 vs ONNX 逐向量对齐（阶段 4）
- `node tools/test_tracking_sim.mjs <name>` — 追踪策略单机 sim2sim（WASM 场景整段跟踪不倒，阶段 4）
- `node tools/test_tracking_boxing.mjs 22 boxing` — 追踪模式双机无头回归（调度/KO/回合，阶段 4）
- `.workbuddy/gpu/e2e_tracking.cjs` — 浏览器 E2E（追踪模式；`.workbuddy/tmp/e2e_default_rl.cjs` 为 AMO 模式）
- python↔JS 逐 tick 对齐 — 任何新策略移植的验收标准（~1e-6 误差）

## 相关文档

- 方案调研全景：`docs/research-boxing-approaches.md`
- 战术层自博弈结题：`docs/tactics-selfplay-notes.md`
- GPU 训练执行手册：`docs/plan-mjlab-gpu-training.md`
- 项目结构速览：`README.md`
