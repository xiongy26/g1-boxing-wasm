# fight 自博弈策略 v3 优化方案（plan-fight-selfplay-v3）

> **实施结果注记（2026-10-03，T9 落盘）**：T1-T7 完成；T8 部署评估——best ckpt
> `model_2000.pt`（d3 复合分 seed0 0.9624 / seed7 0.9627，v2 基线同口径
> 0.8168 / 0.7926），训练侧闸门全过；浏览器 60s 双 seed 门槛**未过**
> （命中 2 ≪ 40，形态指标全过），按 §4.5 **未替换**，正式
> `vendor/policy/boxing_fight.*` 保持 v2（回滚后 seed 1 复测 PASS）。候选三件套
> `boxing_fight_v3cand.*`、v2 备份 `boxing_fight_v2.bak/` 均已保留。实测表与
> 归因见 README"双机对打"节"v3 自博弈训练轮记录"。
>
> **V3-A2 重训启动（2026-10-03）**：按 §7 止损第 1 档执行——形态权重减半
> （guard_hold 0.15→0.075、punch_extension 0.20→0.10，heading_align 0.10 不动），
> init 仍为 v2 model_3999、3000 iters（训练由队长启动）。
>
> **V3-B2：命中口径修复（拳 vs 头/躯干 geom）+ 重训，用户已批准（2026-10-03）**：
> fight_contact/fight_contact_rev 的 secondary 由 torso_link subtree（含对方整条
> 手臂与拳套，训练侧 10 个 geom）改为 torso_link body 精确匹配（= 头+躯干 2 个
> geom，与浏览器 vulnerableGeoms 逐一对齐），修 A1/A2 部署失败根因；探针
> pipeline/probe_contact_scope.py 19/19 过（解析 + 行为双验证），probe_mdp_v3/d0
> 回归零破坏。重训由队长启动。
>
> **V3-B2 结果（2026-10-03 补记）**：口径修复生效——d3 扫描 best =
> `model_2500.pt`（run `2026-10-03_16-30-36`）复合分 0.8885（v2 同口径
> 基线 0.6605，+34.5%），新口径下 B2-best 头/躯干接触率 0.307 次/s/env =
> v2 的 16 倍；浏览器 60s `--fight` 命中 2→14（A1/A2 曾为 2/3；v2=29、
> 门槛 ≥40），grazes 出现 10 次、与格挡正常分流——命中确为真实头/躯干
> 打击，其余指标全面优于 v2（站立 100% vs 89.4%、抱架 97.0%/98.5% vs
> 88.9%/94.4%、median 15.9° vs 21.4°、>90° 0% vs 4.3%）。主指标仍未达
> 门槛，按 §4.5 **未替换**，正式 `vendor/policy/boxing_fight.*` 保持 v2
> （回滚已 md5 验证）。候选三件套 `boxing_fight_v3cand`（A1）/
> `v3a2cand`（A2）/`v3b2cand`（B2）保留。后续可选（未承诺）：从 B2
> model_2500 续训 +3000 iters（命中曲线尚在爬升）或上调 hit/engage 权重
> 再训；fight 模式下双 seed 协议退化为单轨（harness 种子不影响 fight
> 轨迹），真双 seed 验收需给 harness 加出生抖动 RNG 注入点。实测表与
> 归因见 README"双机对打"节"v3 自博弈训练轮记录"。

## 1. 背景与目标

- 项目：`/home/xy/zcode/g1-boxing-wasm`（浏览器双机对打）。训练链在 `.workbuddy/gpu/g1dance_pipeline/`（本项目内部副本，与已清理的 `/home/xy/zcode/g1-dance` 仓库无关；conda env `g1dance` 为机器公共基础设施，本方案继续复用——"不用 g1-dance 训练"约束针对的是外部仓库及其产物链，不针对同名 conda 环境本身，此为显式声明）。
- 基线 v2：run `2026-10-01_05-09-50` model_3999（已部署 `vendor/policy/boxing_fight`）。训练侧 32envs/2min：交战率 91.3%、命中 542:507、零摔倒；浏览器 60s `--seed 1`：命中 29、站立 89.4%、抱架 88.9%/94.4%、median 21.4°、>90° 4.3%，现有门槛全过。
- 目标：在 v2 基础上提高对打质量，重点消除"补丁鸿沟"（守卫/形态靠 `src/boxing_ai.mjs` 运行时补丁兜底）、补齐训练中评估择优、加厚对手池、改善摔倒不对称（4.2:1）。

## 2. 需求拆解与验收标准

| # | 功能点 | 验收标准（量化） |
|---|---|---|
| F1 | 形态塑形奖励（护手/出拳/朝向） | 新 reward 项可单独开关；训练侧护手保持率 ≥60%；不触发 reward-hacking 自检（见 §4.2） |
| F2 | 对手池升级（锚+滚动） | 新驱动类；push/采样有日志；池构成 = 2 锚 + 4 FIFO |
| F3 | 摔倒处理（op3 式慑止） | 倒地层罚项生效；robot:opp 摔倒比 ≤2:1（v2 为 4.2:1） |
| F4 | 停机扫描择优 d3 | 扫 run 目录全部 ckpt → `verdict.json`（四件套+复合分）→ best.pt；v2 基线同口径复测入表 |
| F5 | 训练中 eval hook | 每 500 iters、16 envs、固定种子，写 tensorboard + `eval_hist.json`；总开销 <5% |
| F6 | P1 脚本对手早退 bug | 以新类修复（逐 env spawn + skip 掩码，同 snapshot_pool.py:457-473 已修模式），旧类零改动 |
| F7 | 部署与回归 | ONNX→extract→check_fight_meta 14 项过→浏览器双 seed 过 v3 门槛→才替换 boxing_fight.bin |

**v3 浏览器门槛（60s `--fight`，`--seed 1` 与 `--seed 7` 均须过）**：命中 ≥40（v2=29，+38%，主指标——89% 时间站立说明对抗强度仍有空间）、双机站立 ≥90%（v2 89.4%）、抱架保持 A/B 各 ≥85% 且均值 ≥90%、median ≤25°、>90° ≤8%、贴身 ≤12%（README 硬门 15% 收紧）。理由：v2 已高位，取"命中显著提升、其余不劣化"的保守组合，避免指标互斥（命中↑与站立/贴身天然张力）。

**训练侧闸门（d3 口径 = RoboStriker 四件套 + d2 判据）**：交战率 ≥80%（v2 91.3%，容许 -11pt 换形态；d2 终检线 30% 保底）、双向命中和 ≥0.2 次/s/env 且 robot:opp 命中比 ∈[0.5,2.0]、双躺 ≤20%、摔倒比 ≤2:1、护手保持率 ≥60%（首设观察项转硬门）。"更好"定义：best ckpt 复合分 ≥ v2 复测基线分 ×1.05，且浏览器双 seed 过上表。

## 3. 现状与前提假设

现状核实（与派发材料一致）：任务注册 `$MJ/src/tasks/tracking/config/g1/__init__.py:27-52`（P1/P2/P2S）；奖励布局 `sparring_env_cfgs.py`（engage 0.5/approach 0.3/hit 2.0/stalemate -0.2/upright 0.3/fall -15，tracking×0.3）；`HitReward`（10N+0.5s 冷却）在 `sparring_mdp.py:122-149`；池机制 `mdp/snapshot_pool.py`（深 4 FIFO、warmup 500、push 200，参数权威表 `sparring_rl_cfg.py:19-25`）；训练入口 `pipeline/train_p2.py`（learn_with_selfplay :188-283，push 钩子 :266-274，滑动保留 :167-185）；停机评估 `pipeline/d2_selfplay_eval.py`（脚本式，无扫描/复合分）；run 目录现存 ckpt：model_0/2000/2500/3000/3500/3999。
假设：① RTX 4060 Ti 16GB 独占，3.0s/iter（1024 envs）；② 热启动源 = run `2026-10-01_05-09-50` model_3999；③ 本轮不改浏览器 JS（`src/boxing_ai.mjs` 补丁全保留）；④ 进程管理沿用 `.workbuddy/gpu/sparring_pids.txt` 显式 PID 约定；⑤ rsl_rl 5.0.1 无内置周期评估 hook（调研已核），需自研。

## 4. 方案设计

### 4.1 范围裁定与文件布局

第一轮包含：F1 形态奖励 + F2 池升级 + F3 摔倒慑止 + F4/F5 评估择优 + F6 driver bug 新类修复。AMP 风格奖励、NFSP reservoir、exploiter、latent 蒸馏全部排除（理由见 §6）。

硬约束沿用："新逻辑只进新文件/新注册名"。新增文件清单（全部在本副本内）：

| 文件 | 内容 |
|---|---|
| `$MJ/src/tasks/tracking/mdp/sparring_mdp_v3.py` | guard_hold_reward / punch_extension_reward / heading_align_reward / down_penalty |
| `$MJ/src/tasks/tracking/mdp/selfplay_pool_v3.py` | `AnchoredSnapshotDriver(SnapshotOpponentDriver)`：锚槽+FIFO，接口同名 push_snapshot/set_pool_from_paths |
| `$MJ/src/tasks/tracking/mdp/opponent_script_driver_v2.py` | P1 脚本对手修复版新类（F6） |
| `$MJ/src/tasks/tracking/config/g1/sparring_env_cfgs_v3.py` | `make_sparring_v3_env_cfg()`：P2 布局 + v3 奖励组 |
| `$MJ/src/tasks/tracking/config/g1/sparring_rl_cfg_v3.py` | `SPARRING_SELFPLAY_V3` 参数表 + runner cfg（超参同 v2） |
| `pipeline/train_p3.py` / `train_p3.sh` | 复制改造 train_p2：v3 参数、eval hook、PID 写 sparring_pids.txt |
| `pipeline/d3_scan_eval.py` | 扫描择优（§4.4），自包含（d2 为脚本式不 import，口径复制并注释同源） |

唯一触碰的既有源文件：`$MJ/.../config/g1/__init__.py` **追加**注册 `Unitree-G1-Sparring-P3`（与 P1/P2/P2S 先例同模式，append-only，不动既有条目）。历史 run 可复现性：旧任务注册、旧 cfg、旧 ckpt 全部零改动。

### 4.2 形态奖励设计（F1，只作用于 learner）

对手为冻结快照不收奖励；因双机同权重、对手随快照轮换间接获得同形态，无需对称设计。所有项门控 upright（root_z ≥0.55）防躺地刷分。

| 项 | 定义草案 | 权重初值 |
|---|---|---|
| guard_hold | 指示项：双腕均满足"腕 z > 肘 z +0.03m 且 \|腕 z − 头 z\| ≤0.25m"（与浏览器抱架口径逐字一致，验收直接对齐）。出拳窗口（任一腕前向速度 >1.2 m/s，同浏览器 1.2 阈）不计入 | +0.15 |
| punch_extension | 连续项：交战窗口（d∈[0.5,1.2] 且 cosθ>0.9）内，出拳腕相对同侧肩的前向伸展 = clamp(前向距离/0.55m, 0, 1)；仅出拳窗口计分 | +0.20 |
| heading_align | 连续项：交战窗口内 cos(yaw_error)（engage 已有阈值指示 0.5，此项补连续梯度，专削 >90° 尾部） | +0.10 |
| down_penalty | 摔倒慑止（op3 思想，F3）：root_z <0.50 时每步 −1.0（比 upright 软地板 z_ref 0.55 更硬）；robot_fall 0.40 终止保留。op3"倒地步正奖励清零"严格实现需改 RewardManager，零改动约束下不可行——用大额倒地层罚+摔倒即终止近似等效（终止后正奖励自然归零），列为已知近似 | −1.0 |

防 hacking 设计：guard 与 extension 窗口互斥（出拳不计护手）；extension 除以固定臂展 0.55m 并 clamp 上限（不可堆收益）；heading_align 乘交战门控（不鼓励原地转向刷分）；hit 2.0 仍是最大单项，形态项总和上限 0.45 不改对抗收益结构。调参顺序：先 guard，若交战率掉 >10pt 先降 extension 减半；若命中掉再降 guard 减半；每次只动一项。

### 4.3 对手池与课程（F2）

- 池组成：`AnchoredSnapshotDriver` 池深 6 = 2 个永久锚槽 + 4 个 FIFO 滚动槽。锚槽：`iter-5000` 干净 tracking（154 维异构兼容，既有机制支持）+ v2 `model_3999`（强快照，防对手整体退化）。依据 op3-soccer 结论（快照池保留早期弱对手使对手变强慢于本体→训练显著更稳），并修正 v2"全近期快照"缺陷。
- 采样：每 env 回合出生时均匀抽槽（含锚槽），沿用父类机制，不做新鲜度加权（均匀更简单、op3 即均匀；加权列为备选参数 `sample="uniform"|"fresh"` 预留）。
- 周期：push 间隔 200 iters 不变；warmup 200 iters（v2 的 500 是为 154→168 扩维适应设的，v3 热启动 168 维恒等、仅奖励分布变化，缩短到 200 让 learner 先适应新奖励再换手）。obs 置零课程取消（zeroout_iters=0，opponent_state 已学过，直接用全知版）。

### 4.4 评估与择优体系（F4/F5）

- **d3_scan_eval.py**：输入 run 目录 + 可选 `--baseline`（v2 model_3999）；对每个 ckpt：32 envs、2min、对称互打（同 ckpt 入池，复用 set_pool_from_paths）、play 模式无扰动；统计 交战率 / H_robot / H_opp（10N+0.5s 冷却）/ robot_fall / opp_fall / 双躺率 / 护手保持率（训练侧估计：门控窗口同 guard_hold 定义）。复合分 = 0.35·min(交战率/0.90,1) + 0.30·min((H_r+H_o)/0.40,1) + 0.20·max(0,1−双躺/0.20) + 0.15·护手率。输出 `verdict.json`（逐 ckpt 指标+复合分+闸门 PASS/FAIL）+ `best.pt`（**复制**另存，不改原文件）。成本：rollout-only ≈8200 env-steps/s → 每 ckpt ≈1min，一个 run ≤10 个 ckpt ≈10min，停机跑可接受。
- **训练中 hook**：train_p3.py 的 learn_with_selfplay 内每 500 iters 插入 16 envs、60s、固定种子（seed=cfg.agent.seed+1000）迷你对称评估，仅 rollout + 指标，写 tensorboard 与 `eval_hist.json`。单次 ≈15-30s（48k env-steps ≈6s + 加载/复位开销），5000 iters 共 10 次 ≈+5min（<5%）。ckpt 加载用 `runner.load(..., load_cfg={"actor": True})` 临时副本，评完恢复训练态——实现时注意保存/恢复 obs_normalizer 与 rollout 缓冲状态。

### 4.5 部署与验收（F7）

1. best.pt → `pipeline/export_ckpt_onnx.py` → `tools/extract_tracking_onnx.py`（14 宽 opponent_state 已支持）→ `vendor/policy/boxing_fight_v3/` 三件套。
2. `node tools/check_fight_meta.mjs` 14 项契约全过。
3. 浏览器 60s 双回归：`node tools/test_tracking_boxing.mjs 60 --fight --seed 1` 与 `--seed 7`，对照 §2 门槛表，并与 v2 数字同表记录。
4. 全过才替换：`vendor/policy/boxing_fight/` 三件套先备份到 `vendor/policy/boxing_fight_v2.bak/`，再拷入新件；回滚 = 目录拷回（一条 cp）。任一 seed 不过 → 不替换，保留 v2，按 §7 止损。
5. 浏览器 JS 本轮零改动；补丁减弱（守卫混合 0.8→lower 等）列为形态达标后的后续独立轮次。

### 4.6 工程化补强

- 路径去硬编码（最小做法）：v3 新文件内默认路径一律由 `Path(__file__).resolve()` 相对推导副本根（`.../g1dance_pipeline/`），CLI 可覆写；旧文件不动（历史可复现）。
- train_p3.sh 头注释显式声明 conda env `g1dance` 复用及理由（§1 同文）；启动时 `echo $$ > .workbuddy/gpu/sparring_pids.txt`。
- README 更新点：训练链小节注明"fight 训练在本项目 `.workbuddy/gpu/g1dance_pipeline/`（v3 方案 `docs/plan-fight-selfplay-v3.md`）"，g1-dance 仓库不再承载训练。

## 5. 接口与数据契约

- **任务注册**：`Unitree-G1-Sparring-P3`，obs 168 维不变（与 v2/ONNX 布局逐维同构，部署链零改）；动作 23 关节、50Hz、num_steps_per_env 24 不变。
- **池驱动接口**：`AnchoredSnapshotDriver` 保持父类公开接口 `push_snapshot(state_dict)` / `set_pool_from_paths(paths)` / step 事件签名；新增构造参数 `anchors: tuple[str,...]`、`fifo_depth=4`、`sample="uniform"`。
- **v3 参数表** `SPARRING_SELFPLAY_V3 = {snapshot_interval_iters:200, fifo_depth:4, warmup_iters:200, obs_zeroout_iters:0, init_ckpt:"", anchors:[iter-5000, v2 model_3999]}`（路径运行时推导）。
- **verdict.json 契约**：`{run_dir, baseline:{...}, results:[{ckpt, it, engagement_rate, hit_rate_robot, hit_rate_opp, fall_robot, fall_opp, double_down_rate, guard_hold_rate, composite, gate_pass}], best:{ckpt, composite}}`。
- **eval_hist.json**：`[{it, engagement_rate, hit_total, double_down_rate}]` 追加写。

## 6. 技术选型与权衡（引用调研结论）

- **RoboStriker（arXiv:2608.16195）**：唯一公开 G1 拳击同类。采纳其指标四件套（win/η_hit/engagement/BOS）作为 d3 评估口径；其两阶段 latent 蒸馏工程量大，**不采纳**。
- **op3-soccer（arXiv:2304.13653）**：直接抄"池保留早期快照 + 对手变强慢于本体"→ §4.3 锚槽设计；倒地慑止思想 → down_penalty（清零正奖励以层罚近似，§4.2 已声明偏差）；其"起身为独立预训练技能"不采纳（本项目机器人摔倒即终止，无起身需求）。
- **AMP_mjlab（github.com/ccrpRepo/AMP_mjlab）**：同栈 AMP 实现可参考恢复窗口处理，但 License 未确认 + 需动捕数据管线，**本轮排除**，列为 P1 备选（采纳前必须核对 License）。
- **形态标量奖励**：调研确认无公开先例，自研成本低（3 个标量项，§4.2），优先于 AMP 判别器方案——正解"补丁鸿沟"的最短路径。
- **rsl_rl v5.x 无周期评估 hook**（调研已核源码）：生态两法中选 (b) 训练中迷你评估（F5，早发现问题）+ (a) 停机扫描择优（F4，最终裁决），两者互补。
- **league/PBT**：算力不适用（单卡 4060 Ti），排除；NFSP reservoir 列为池机制二级备选（本轮锚+FIFO 已覆盖其"保留历史"核心收益）。

## 7. 分阶段实施计划

依赖顺序 T1→T2→T3→T4→（T5 并行）→T6→T7→T8→T9；每步可独立验收。

| 阶段 | 内容 | 依赖 | 验收 | 墙钟估算 |
|---|---|---|---|---|
| T1 | sparring_mdp_v3.py 四个 reward 项 + CPU 数值探针自检（构造假 obs 验证门控/互斥/clamp） | - | 探针断言全过 | 0.5h |
| T2 | selfplay_pool_v3.py + sparring_env_cfgs_v3.py + sparring_rl_cfg_v3.py + __init__.py 注册 P3 + script_driver_v2.py（F5/F6） | T1 | d0 冒烟：P3 task 可构建、池构成日志 2 锚+4 FIFO、obs_dim=168 | 1h |
| T3 | train_p3.py/sh（含 eval hook、PID 落盘） | T2 | 20 iters 试跑通过，eval_hist.json 有 1 条 | 1h |
| T4 | d3_scan_eval.py（含 v2 基线复测入表） | T2 | 对 v2 run 扫描出 verdict.json，基线分与 §2 数字同量级 | 1.5h |
| T5 | V3-A 训练 3000 iters（init=v2 model_3999，热启动，lr/超参不变） | T3,T4 | 训练侧闸门全过（§2）；不过→权重减半重跑一次 | 2.5h+10min hook |
| T6 | V3-B 续训 2000 iters（同配置继续，池持续轮换） | T5 过闸 | 同上；若 A 阶段已饱和（复合分 plateau <2%/千 iters）可裁掉 | 1.7h |
| T7 | d3 扫描 run 目录 → best.pt（含 A/B 全部 ckpt + v2 基线对比） | T5/T6 | best 复合分 ≥ v2 基线 ×1.05 且闸门 PASS | 15min |
| T8 | 导出→契约检查→浏览器双 seed 回归→过则替换 boxing_fight.bin（备份+回滚路径 §4.5） | T7 | §2 浏览器门槛双 seed 全过；否则不替换并回 §7 止损 | 0.5h |
| T9 | README/docs 更新、交付汇总（结果/验证证据/剩余问题三要素） | T8 | 文档落盘 | 0.5h |

总计 ≈5h 独占 GPU + ≈4h 人工/编码，一天内可完成含一轮重试。

**止损动作**：T5 闸门不过（第 1 次）→ 形态权重减半重跑 3000 iters；第 2 次不过 → 降级为"仅 down_penalty + 池升级"（零形态项）重跑；仍不过 → 保留 v2 部署，输出总结。任一阶段 GPU 被占/限流 → 退避 15-30min 再续（训练支持 model_* 断点续训）。

## 8. 风险与缓解

| 风险 | 概率/影响 | 缓解 |
|---|---|---|
| 形态奖励压制对抗性（命中/交战下滑） | 中/高 | 权重总和 0.45 ≪ hit 2.0；闸门交战率 ≥80%；减半重跑预案；d3 复合分择优保证只有更好才部署 |
| 热启动遗忘（tracking 先验退化） | 低/中 | init=v2 终点、lr 2.5e-4 不变、tracking×0.3 不动；tensorboard 盯 torsoErr，恶化 >10% 即降 lr 至 1.5e-4 |
| 池改动致对手过弱/过强 | 中/中 | 锚槽含 v2 强快照托底；d2 分段统计（30s bucket）看趋势；FIFO 4 保留新鲜压力；异常时调 anchors 配比（纯参数，无代码改） |
| 摔倒层罚引发龟缩（不敢近身） | 中/中 | down_penalty 只在 z<0.50 触发，engage/hit 权重不动；stalemate -0.2 反龟缩已有；闸门交战率兜底 |
| eval hook 污染训练状态 | 低/高 | 实现要求：评估在 inference_mode + 独立 runner 副本/恢复 obs_normalizer 与 episode 计数；T3 验收含 20 iters 前后 loss 对比无异常 |
| 双 seed 浏览器只过一个 | 中/中 | 不替换、保留 v2；best 次优 ckpt 重试一轮（择优体系天然支持）；仍不过按止损 |

## 9. 待确认清单

1. 浏览器 v3 门槛数值（命中 ≥40、站立 ≥90%、抱架 ≥85%、median ≤25°、>90° ≤8%、贴身 ≤12%）为架构师裁定值，请队长/用户确认或上调。
2. 第二验收 seed 取 7（v2 只有 seed 1 基线，另一 seed 无 v2 对照，只按门槛判）——可换。
3. 锚槽组成（iter-5000 干净 154 维 + v2 model_3999）：是否改用 S 阶段 model_2999 替代 iter-5000（更接近当前分布，但少了"早期弱对手"多样性）。
4. V3-B 2000 iters 是否保留（若 A 阶段复合分饱和可裁，省 1.7h）。
5. AMP 风格奖励列为 P1 备选：采纳前需核 AMP_mjlab License 并建动捕片段管线，本轮不做——确认排除。
6. P1 脚本对手 bug 以新类修复（旧类保留）：若希望 P1 历史任务本身行为修复，需放宽"新逻辑新文件"约束，默认不放宽。

## 10. Out of scope

- 浏览器端任何 JS 改动（含守卫混合系数减弱、航向伺服调参）——本轮补丁全保留，后续独立轮次评估减弱。
- AMP 判别器 / 动捕数据管线、NFSP reservoir、exploiter、league/PBT、latent 蒸馏。
- 任何对既有源文件逻辑的修改（`__init__.py` 仅 append 注册条目）；g1-dance 外部仓库相关一切操作。
- 多卡/多机训练、obs 维度或动作空间变更、ONNX/部署链格式变更。
- P2S 站立课程、P1 任务本身的训练。

——方案交付完毕，交接给 coding 实施。实施产物以 T1-T9 顺序派发，每阶段验收标准见 §7。
