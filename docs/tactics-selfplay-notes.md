# 战术层自博弈：实现记录与结论（2026-09-21）

对应调研报告（research-boxing-approaches.md）三步走中的**第 1 步：无 GPU 快赢**。
目标：把 `boxing_ai.mjs` 里加权随机挑招式（`weightedPick`）升级为学习出来的战术决策。

## 交付物

| 文件 | 作用 |
|---|---|
| `src/tactics.mjs` | 纯 JS 战术策略：13 维观测 → MLP(48,48) → 10 动作；REINFORCE+Adam；**环形先验 `ringPriorLogits`**（距离带 + 反应式拳手常识）；`priorW`/`netGain` 两个部署超参随 JSON 存档 |
| `src/boxing_ai.mjs` | `decideTactics()`：按侧挂载策略，无策略侧回退旧版加权随机 |
| `tools/train_worker.mjs` | 无头 MuJoCo worker（AMO 模式），每消息跑一回合，返回双方决策轨迹+事件 |
| `tools/train_tactics.mjs` | 多 worker 训练器：旧版对手课程（phase1）→ 自博弈（phase2）+ 对手池，REINFORCE 更新 |
| `tools/eval_tactics.mjs` | 策略 vs 旧版对比评估（半程换边消除场地偏差） |
| `vendor/tactics/tactics.json` | 部署权重（`priorW=1.0, netGain=0.1`，main.js 启动时自动加载） |

动作空间 10 个：`approach / retreat / strafe_left / strafe_right / jab / double / cross / hook / jab_hook / wait`。

## 评估结果（16 回合 × 14s，半程换边）

```
tactics policy : hits/ep=1.06  pts/ep=1.13
legacy random  : hits/ep=0.94  pts/ep=0.94（KO=0）
offense ratio  : 1.13（策略方反超）
```

回归：`node tools/test_amo.mjs 22` → PASS；浏览器 E2E → PASS（56FPS、真实命中、无 NaN）。

## 七轮训练的教训（这是本文档的核心价值）

**纯 REINFORCE 自博弈在小样本下必然塌缩到"互相回避"（mutual standoff）均衡。**

1. 纯自博弈（hit 1.5×/挨打 -2）→ 50 回合内 hits 归零。
2. 击中奖励提到 3×points + 旧版对手课程 → 仍归零，avgRet 钉在 -0.20（= 纯 standoff 的塑造值）。
3. 强距离压力（远处 -0.40/决策）→ 更快塌缩，熵坍缩到 0.5（策略确定性躲）。
4. 先验引导 priorW=1.0 退火到 0.3 → 网络主动学大 logits 抵消先验（H≈2.0 却 standoff）。
5. priorW 恒 0.8、挨打 -0.5 → 依旧归零。
6. priorW=2.0 → 依旧归零（Adam 下网络 logits 无界，±3.2 的先验也能抵消）。
7. **`netGain=0.1`（网络输出增益限幅）→ 结构上杜绝 standoff**，先验硬性拥有"是否交战"决策，网络只能带内微调。

根因分析：**"接近"这个动作的信用分配天然被污染**——接近路径上吃到的拳会记在接近决策的回报上（负），而命中奖励落在后面的出拳决策（正）。即使在"对攻总体为正"的奖励结构下，梯度仍然系统性地学出"别接近"。这是 sparse-event + 单回合 batch REINFORCE 的结构性问题，与 RoboStriker 用 4096 并行环境 × 上万迭代才能解决的是同一个问题。

**最终架构：prior-first（先验主导）+ bounded residual（有界残差）**。这不是妥协，而是这个规模下的正确工程决策：
- "是否交战/何时出拳"由可解释的距离带先验硬编码（远=逼近 2.2、pocket=出拳 2.0、贴身=拉开、对手出拳=回避加成、对手踉跄=追击加成）；
- 网络残差 `netGain=0.1` 只能在每个距离带内部弯曲概率（选哪种拳、什么时机）；
- 部署超参烤进 JSON，浏览器行为 = 评估行为。

## 运行方式

```bash
node tools/train_tactics.mjs --episodes 4000 --workers 6 --ep-len 18 --minutes 15 --phase1 80
node tools/eval_tactics.mjs 16 14        # vs 旧版对比
node tools/test_amo.mjs 22               # 回归
```

删除 `vendor/tactics/tactics.json` 即回退旧版行为（main.js 加载失败时静默回退）。

## 下一步（对应调研报告第 2 步）

本基建直接服务 mjlab/BeyondMimic 路线：奖励设计的均衡分析、评估方法、worker 池都可复用。
真正的战术学习需要 GPU 大规模并行训练，本机 REINFORCE 已被证明不可行（见上）。
