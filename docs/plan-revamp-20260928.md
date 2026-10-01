# G1 拳击动作重做方案（plan-revamp-20260928）

- 作者：architecture-design；交付对象：coding / code-review
- 硬性截止：2026-09-29 08:00（本文写作时约 23:30，剩约 8.5 小时）
- 一句话目标：用 g1-dance 训练管线把 35 秒真实动捕拳击序列训成单一 G1 跟踪策略，经既有 ONNX 部署链在浏览器 MuJoCo WASM 中连续循环打出可辨认的直拳+摆拳组合。
- 本文所有路径已实地核实（脚本签名、观测维度、ONNX 图契约均为 2026-09-28 夜间实读代码结论，非转述）。

---

## 1. 背景与目标

- 现状（侦察结论 A）：g1-boxing-wasm 的拳法 4 槽位全禁用（`src/boxing_ai.mjs:43` `CLIP_ENABLED={guard:true,jab:false,cross:false,hook:false}`），jab/hook 权重互为字节复制品、干净 RSI 存活余量仅 8cm，四槽位切换是结构性摔倒源；数据管线（LAFAN 重定向 clips_v3）滑移 0.217-0.715m，且旧训练仓库 v3 重训 run 全部早夭。
- 参考项目 g1-dance（侦察结论 B）：已验证的"重定向→mjlab/MuJoCo-Warp GPU 训练→ONNX"管线，其 `motions/npz/xsens_boxing_test.npz` 是 35.4s（1770 帧@50Hz，29 关节）真实 Xsens 动捕拳击重定向产物，从未参与任何训练，适配拳击改动量 0 行代码。
- 核心决策：训练端整体切到 g1-dance 管线、单一 35s 连续拳击动作训单一策略，从结构上消灭多槽位切换摔倒问题；部署端保留 g1-boxing-wasm 浏览器链路，新增第 5 个 clip 槽位 `combo` 承载该策略。

## 2. 需求拆解与验收标准

| # | 需求点 | 验收标准（可独立验收） |
|---|---|---|
| N1 | 浏览器仿真中机器人打出可辨认拳击组合 | `?scene=tracking`（combo 模式）下，单机器人连续打出组合拳，至少含直拳（jab/cross）与摆拳（hook）两类；30 秒不摔倒、无 NaN、无引擎自动重置 |
| N2 | 训练效果可量化 | 无头回归 `tools/test_tracking_boxing.mjs` 全部硬断言通过（无 NaN、divergences=0、minZ≥0.40、KO 注入触发、回合重置） |
| N3 | 动作不畸形 | eval_ckpt.py 无头评测截图 + 浏览器 e2e 截图与 `show_motion.sh` 参考动作对照，四肢运动次序与参考一致、无明显畸变（肘腕反关节、滑步离地） |
| N4 | 双机对战兼容（软目标） | 既有双机 tracking 模式行为不回退：两 TrackingFighter 各自循环同一 combo 动作，命中/KO/回合重置逻辑不改动且回归通过 |
| N5 | 保底可见 | AMO 默认模式脚本摆臂不动（保留现状即达标，不验收） |

验收证据形式：无头回归终端输出（PASS 行）+ e2e 浏览器截图 4-6 张 + 录屏（如时间允许）+ eval_ckpt 截图对照。

## 3. 关键假设（已验证事实 vs 待确认假设）

### 3.1 已实地核实的事实（方案依据，coding 无需重复验证）

1. `xsens_boxing_test.npz`：fps=50.0，1770 帧 = 35.4 秒，joint_pos(1770,29)/joint_vel/body_pos_w(1770,30,3)/body_quat_w/body_lin_vel_w/body_ang_vel_w，字段与 mjlab csv_to_npz 产物一致，可直接作 `--motion-file`。
2. `pipeline/train.sh <npz> [task] [num_envs] [iters]`，默认任务 `Unitree-G1-Tracking-No-State-Estimation`（29DoF），默认 2048 envs/8000 iters，日志 tensorboard，TMPDIR 已指向 `/home/xy/.g1dance_tmp`（不占 /tmp 小分区）。
3. **观测同源性确认**：g1-dance 的无状态估计 actor 观测组（`third_party/unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py` 过滤掉 `motion_anchor_pos_b` 与 `base_lin_vel`）= command(58=参考 joint_pos29+vel29) + motion_anchor_ori_b(6) + base_ang_vel(3) + joint_pos(29) + joint_vel(29) + actions(29) = 154 维，顺序与浏览器已部署 `vendor/policy/boxing_jab_meta.json` 的 `obs_layout`（command[0,58]/ori[58,6]/ang_vel[64,3]/joint_pos[67,29]/joint_vel[96,29]/actions[125,29]）**逐项一致**。浏览器 `TrackingFighter` 的 154 维构造无需改动。
4. **ONNX 图契约确认**：`src/tasks/tracking/rl/runner.py` 的 `MotionTrackingOnPolicyRunner.export_motion_policy_to_onnx()` 导出（每次 save 自动执行，文件名为 run 目录名.onnx）：输入 `obs[1,154]`+`time_step[1,1]`，输出 `actions[1,29]`+joint_pos/joint_vel/body_pos_w[1,14,3]/body_quat_w[1,14,4]/body_lin_vel_w/body_ang_vel_w —— 与 `tools/extract_tracking_onnx.py` 文档头声明的部署图（time_step→clip→motion 表 Gather；obs→normalize→Gemm/Elu×3→Gemm）完全一致。层命名正则 `(?:policy\.mlp|actor)\.(\d+)\.(weight|bias)$` 覆盖 mjlab 导出命名。
5. meta 所需字段（joint_names/default_joint_pos/kp/kd/action_scale/anchor_body_name=torso_link/body_names 14）由 `attach_metadata_to_onnx(get_base_metadata(...))` 写入 ONNX custom_metadata_map，extract 脚本自动解析。
6. 奖励函数 `motion_relative_body_position_error_exp` 等已支持 `body_names` 子集参数（`mdp/rewards.py:47-77`），且 g1 的 14 个跟踪 body 本身已含双肘（elbow_link）双腕（wrist_yaw_link）——v1 训练 0 配置改动即覆盖手臂跟踪。
7. 终止条件 `ee_body_pos` = 末端（双踝+双腕）z 高度误差 >0.25m（z-only，`terminations.py: bad_motion_body_pos_z_only`），body 列表在 `config/g1/env_cfgs.py` 中以 tuple 配置，可纯配置放宽。
8. `pipeline/convert.sh <csv> [robot] [name] [input_fps]`（30fps 36 列 CSV→50Hz npz）、`show_motion.sh <npz>`（无显示器自动 viser）、`eval_ckpt.py <ckpt.pt> [motion.npz]`（训练中可并行，约 1.5GB 显存）签名均已核实。
9. 部署链工具齐备：`tools/extract_tracking_onnx.py`、`tools/test_tracking.mjs`（JS 前向 vs ONNX 1e-6）、`tools/test_tracking_boxing.mjs`（断言行 144-154：NaN/divergences/minZ 0.40/KO）、`.workbuddy/gpu/e2e_tracking.cjs`（浏览器无头截图）。
10. 浏览器槽位机制：`src/main.js:43` `TRACKING_CLIPS=['guard','jab','cross','hook']` + `loadTrackingNets()`（任一 fetch 失败整体抛错，**不**是逐 clip 容错）；`src/boxing_ai.mjs:343 setTracking`、`:401-411` 切片 rebase（yawOfQuat）、`:504`/`:636` 拳法门禁两处。

### 3.2 假设（未验证，按假设推进，触发条件见风险节）

- A1：35s Xsens 动作中同时含直拳与摆拳两类拳法（T0 质检时看截图确认；若缺摆拳走 fallback a）。
- A2：xsens 重定向质量可训（从未质检；T0 用 show_motion 确认，失败走 fallback 树）。
- A3：rsl_rl 支持从 checkpoint 续训（崩溃恢复用；现场 15 分钟内确认 resume 参数，不可用则直接重训，见风险 R5）。
- A4：4060 Ti 上 35s 动作 2048 envs 迭代耗时 1.5-4.9s/iter（介于"短动作 1.2s"与"18s 舞蹈 4.9s"之间，按 1770 帧≈2.2 倍于 18s 动作保守取上限）。

## 4. 技术选型与权衡（引用调研/侦察结论）

| 选项 | 结论 | 理由 |
|---|---|---|
| **选定：g1-dance 管线 + xsens 35s 单动作 + 单策略** | 采用 | 队长倾向约束 1 + 侦察结论 B：数据现成（0 行代码换 npz 即训）、adaptive 采样（结论 C：BeyondMimic 官方证明不用自适应采样多数动作训不出）、实测墙钟可控、单一策略结构上消灭四槽位切换必摔 |
| PBHC/KungfuBot 数据 + IsaacGym | 否决 | 结论 C：pkl 格式与 CSV/npz 不兼容需转换脚本，框架迁移风险大于收益，时间不够 |
| 旧 g1_boxing_task（mjlab 1.1.0）继续修 v3 数据重训 | 否决 | 侦察结论 A：v3 滑移伪影未除、重训 run 早夭原因未查明、guard 的 v3 npz 缺失、每 clip 20k iters≈19h 远超预算 |
| BeyondMimic 官方 whole_body_tracking 仓库 | 借鉴不引入 | 结论 C：其配方（phase 观测+自适应采样）已由 unitree_rl_mjlab 继承（sampling_entropy/adaptive_* 已在 `commands.py` 核实），无需新引依赖 |
| 部署端观测对齐策略：浏览器适配层 vs 训练配置降级 | **都不需要** | §3.1 事实 3/4：观测维度、顺序、ONNX 图契约三重一致，1.1.0→1.2.0 无实际差异。仅保留 extract 阶段 test_vectors 1e-6 校验作最终门禁 |
| 浏览器展示：新增第 5 槽位 `combo` 单策略 35s 循环 | 采用（队长约束 4） | 复用既有槽位机制，改动最小；双机=两个 TrackingFighter 各自循环同一动作；AMO 脚本摆臂不动保底 |
| 训练超参 | 2048 envs / 4000 iters 上限 | 队长约束 3 预算 1.7-3.5h；结论 C：站桩出拳类是最易训动作类型，简单动作几千迭代可收敛；先出效果再谈精度的截止语境 |

## 5. 模块与接口边界

### 5.1 训练侧改动清单（g1-dance）

v1 主线 **0 行代码改动**，全部用现成脚本：

| 步骤 | 命令（在 /home/xy/zcode/g1-dance 下） | 产物 |
|---|---|---|
| T0 质检 | `./pipeline/show_motion.sh motions/npz/xsens_boxing_test.npz`（无显示器自动 viser，截图 6-8 张存 `eval_out/boxing_ref/`） | 参考动作质检证据 |
| T1 冒烟 | `./pipeline/train.sh motions/npz/xsens_boxing_test.npz Unitree-G1-Tracking-No-State-Estimation 2048 300` | 冒烟 run 目录（跑完即删，防磁盘膨胀） |
| T2 正式 | 同上，iters=4000 | `third_party/unitree_rl_mjlab/logs/rsl_rl/g1_tracking/<时间戳>/`（model_*.pt 每 500 迭代 6.5MB + `<时间戳>.onnx` 全量图） |
| 快评 | `MUJOCO_GL=egl "$MJLAB_PY" pipeline/eval_ckpt.py <最新 model_*.pt> motions/npz/xsens_boxing_test.npz --out eval_out/boxing_eval_N --steps 1500 --num-envs 8` | 指标 + 关键帧截图 |
| Fallback-a 数据转换 | `./pipeline/convert.sh /home/xy/zcode/g1-boxing-wasm/.workbuddy/gpu/data/g1_fight1_subject2.csv g1 boxing_lafan_f1 30` | `motions/npz/boxing_lafan_f1.npz` |

可选 v2 迭代（仅 D3 决策点批准后执行，均为纯配置改动）：
- 腕/肘加权：`third_party/unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py` rewards 字典新增 `motion_body_pos_arms` 项（`RewardTermCfg(func=mdp.motion_relative_body_position_error_exp, weight=0.5, std=0.3, params={command_name, body_names=("left_elbow_link","left_wrist_yaw_link","right_elbow_link","right_wrist_yaw_link")})`；dict 新 key，不改函数）。
- 腕部终止放宽：`config/g1/env_cfgs.py` 中 `cfg.terminations["ee_body_pos"].params["body_names"]` 由含双腕的 4 元组改为仅双踝 2 元组。

禁止改动：env.sh、runner.py、commands.py、observations.py、rewards.py 函数体、convert 脚本。

### 5.2 部署侧改动清单（g1-boxing-wasm）

导出与部署（不改 Python 脚本本体）：

| # | 文件 | 函数/位置 | 改动 |
|---|---|---|---|
| D1 | `tools/extract_tracking_onnx.py` | 无改动，直接调用 | `uv run --no-project --with onnx --with onnxruntime python tools/extract_tracking_onnx.py <训练run>/<时间戳>.onnx vendor/policy/boxing_combo`。**不传** `--clip-meta/--clip-name`（clips_v3 manifest 无此条目，省略即回退内建标定） |
| D2 | `src/main.js` | `TRACKING_CLIPS`（:43） | 加入 `'combo'` → `['guard','jab','cross','hook','combo']` |
| D3 | `src/main.js` | `loadTrackingNets()`（:46-64） | 改为逐 clip 容错：单 clip 404/解析失败时该槽位置 null 并跳过，全部失败才回退 AMO。防止 combo 未部署时破坏现有 4 槽位加载（改动约 5-8 行） |
| D4 | `src/main.js` | `TRACKING_REQUESTED` 附近（:57-70） | 新增 URL 参数 `combo=1` 解析为 `COMBO_MODE` 常量，传入 BoxingController 构造（默认 false，不影响既有路径） |
| D5 | `src/boxing_ai.mjs` | `CLIP_ENABLED`（:43） | 增加 `combo: true` |
| D6 | `src/boxing_ai.mjs` | `decideTactics`（约 :483-520）与 idle 加权随机分支（约 :627-641） | `COMBO_MODE` 时两处均短路：`f.desiredClip='combo'`、`f.punchIntent` 置常驻，跳过 tactics 采样与门禁（约 10 行） |
| D7 | `src/boxing_ai.mjs` | clip 播放结束/切片切换处理（:401-411 rebase 与 :735 `rebaseToReference` 调用处所在的控制流） | `COMBO_MODE` 下 clip='combo' 播到 `meta.motion.max_step` 后回绕 time_step=0 继续同 clip，回绕瞬间照常走 `rebaseToReference`（消除累积漂移；允许 1 帧内姿态微跳） |
| D8 | `src/boxing_ai.mjs` | KO/回合系统入口 | `COMBO_MODE` 下禁用 KO 注入与回合重置（N1 要求连续 30s 观察）；双机默认模式不变 |
| D9 | `tools/test_tracking_boxing.mjs` | main 流程 | 支持 `BOXING_COMBO=1` 环境变量：组合模式跑 ≥35s（覆盖至少一个完整循环+回绕），断言改为 无 NaN、divergences=0、minZ≥0.40（KO 注入断言仅默认模式执行） |
| D10 | `.workbuddy/gpu/e2e_tracking.cjs` | 追加用例 | 打开 `http://localhost:8080/?scene=tracking&combo=1`，间隔截图 ≥6 张覆盖 35s，肉眼比对拳形 |

行为契约（coding 实现必须满足）：
- 默认路径（无 `combo=1`）字节级行为不变；`test_tracking_boxing.mjs` 默认模式必须仍 PASS（回归红线）。
- combo 槽位部署物：`vendor/policy/boxing_combo.bin` + `boxing_combo_meta.json` + `boxing_combo_test_vectors.json`；meta 校验点：obs_dim=154、obs_names 六项与 §3.1 事实 3 同名同序、motion.fps=50、anchor_body_name=torso_link、joint_names 29。
- 双机 combo 模式：双方加载同一 boxing_combo，各自独立时钟循环（不要求帧级同步）。

### 5.3 复用与自研边界

- 原样复用（0 改动）：train.sh / show_motion.sh / eval_ckpt.py / convert.sh / env.sh；浏览器 TrackingNetwork、TrackingFighter（154 维观测构造）、rebaseToReference/yawOfQuat；extract_tracking_onnx.py；server.mjs。
- 小改复用（≤30 行总量）：§5.2 D2-D10 七处。
- 全新自研：无（不新建任何模块/脚本）。
- 明确不自研/不引入：IsaacGym、PBHC 转换器、GVHMR 视频重采、tactics 策略重训、步法训练。

## 6. 数据 Fallback 决策树（文字版，按序执行， gates 均在 T0/T1 完成判定）

1. **G0（T0，xsens 质检）**：show_motion.sh 播放 xsens_boxing_test.npz。通过条件（全部满足）：(a) 形态无穿地/自交/反关节等明显畸形；(b) 根位移小、无明显脚底滑移（站桩出拳预期）；(c) 动作含直拳类与摆拳类两类拳形 → 走主线训练。
2. G0 因"缺摆拳"失败（形态可训）→ **fallback a**：对 `.workbuddy/gpu/data/g1_fight1_subject{2,3,5}.csv` 逐个 show_motion/convert 质检，选含摆拳且滑移最小的一条走 convert.sh → 训练（仍单策略单动作，先 convert subject2）。
3. G0 因"数据畸形不可训"失败 → **fallback a'**：同第 2 条（LAFAN fight CSV 换源），跳过 xsens。
4. fallback a 的 convert/质检也失败 → **fallback b**：直接用 g1-boxing-wasm `.workbuddy/gpu/data/clips_v3/*.npz`（短 clip）在 g1-dance 管线训练最短的一段（jab），验收降级为"单一直拳可辨认"，由队长知悉（部分验收）。
5. **fallback c（视频重采，约 11 分钟/条 + 重定向 + 训练）超出剩余时间窗，本方案不排期，仅作记录。**
6. 训练阶段失败（T1 冒烟不过/D2 中期决策失败）→ 见 §9 风险 R4/R5 处置，不做数据侧二次探索（时间不够，宁降级交付）。

## 7. 倒排时间表（半小时粒度；现在约 23:30，硬截止 08:00）

| 时段 | 事项 | Gate/产物 |
|---|---|---|
| 23:30-00:00 | **T0**：xsens npz 质检（show_motion，截图存证，判定 §6-G0）。**并行**：coding 完成部署侧 D2-D8 骨架（无需权重，编译+AMO 回归过） | G0 判定：主线 or fallback a |
| 00:00-00:30 | **T1 冒烟**：train.sh 300 iters（预计 10-25 分钟）。检查：无 NaN、reward 上升、无 ee_body_pos 终止风暴、实测 iters 耗时 | G1：通过→T2；终止风暴→先打腕部终止放宽补丁再冒烟一次（+15 分钟上限）；>3.5s/it→num_envs 降 1024 重估 |
| 00:30-01:00 | 删除冒烟 run 目录；df 确认磁盘余量≥25G（必要时清 g1_boxing_task/logs 旧 run 与旧 eval_out）；**T2 正式训练启动**（4000 iters，tmux 后台） | **D1 决策点：正式启动最晚 01:00，硬闸 01:30**（过闸仍未启动→直接用冒烟 checkpoint 续训并压 iters） |
| 01:00-03:00 | 训练窗口。每 30 分钟用 eval_ckpt.py 快评最新 model_*.pt（约 1.5GB 显存可并行）；coding 同步完成 D9/D10 改造，并用旧 boxing_jab.onnx 走一遍 extract→临时 boxing_combo 占位→test_tracking.mjs 校验部署工具链 | 工具链 dry-run 通过 |
| 03:00-04:00 | **D2 中期决策**（看 episode 长度/终止占比与 end_time_ratio 趋势，不看 mean_reward）：趋势健康→继续训至 3500-4000；已平台且拳形可辨→提前停；明显不可收敛→启动 §6-fallback b（jab 单段 2000 iters≈1h）保底 | D2 判定落盘 |
| 04:00-04:45 | **T3 导出部署**：取最优 checkpoint 对应的 `<时间戳>.onnx` → extract_tracking_onnx.py → `vendor/policy/boxing_combo.*` → test_tracking.mjs（1e-6）→ test_tracking_sim.mjs 单机整段 | meta 校验点全绿 |
| 04:45-05:30 | **T4 回归**：test_tracking_boxing.mjs 默认模式 + BOXING_COMBO=1 模式；e2e_tracking.cjs 截图/录屏；对照 eval_ckpt 与 show_motion 参考做 N3 判定 | 全部 PASS |
| 05:30-06:30 | **D3 二次迭代窗口**（按需二选一，时间只够一次）：(a) 部署侧行为修缺陷（循环回绕/rebase/防摔参数）；(b) v2 重训（腕肘加权或终止放宽，3000 iters≈1.5-2h）——仅在 (a) 无法解释问题时选 | 修复后重跑 T4 |
| 06:30-07:30 | 最终回归（默认+combo 双模式全绿）+ 验收材料打包（截图 4-6 张、回归输出、指标摘要、对照图） | N1/N2/N3 证据齐 |
| 07:30-08:00 | 交接汇报（结果、验证证据、剩余问题） | 交付 |

磁盘护栏（全程生效）：checkpoint 只保留最新 2 个 run 目录；eval_out 单次 ≤100MB；tensorboard 日志不动 wandb；部署 bin 每个约 1-2MB 无压力；禁止任何产物写 /tmp。

## 8. 验证方式与验收标准汇总

- 功能（N1）：浏览器 `?scene=tracking&combo=1` 无头 e2e 截图序列 + 录屏，30s 连续直拳+摆拳、不摔、无 NaN。
- 回归（N2）：`node tools/test_tracking_boxing.mjs`（默认模式）与 `BOXING_COMBO=1 node tools/test_tracking_boxing.mjs` 双双输出 `RESULT: PASS`。
- 动作质量（N3）：`eval_ckpt.py` 关键帧拼图 vs `show_motion.sh` 参考帧对照 + 浏览器截图对照，拳路次序一致无畸变。
- 权重对齐门禁：`tools/test_tracking.mjs` JS 前向 vs ONNX 误差 ≤1e-6（16 测试向量）。
- 回归红线：默认双机模式行为不回退（D3 改动全部走 `COMBO_MODE` 分支）。

## 9. 风险与缓解

| # | 风险 | 缓解（已编入时间表） |
|---|---|---|
| R1 | xsens 数据从未质检，可能畸形或缺摆拳（A1/A2） | T0 G0 门禁 + §6 fallback a/b 决策树，23:50 前出结论 |
| R2 | 快拳致腕部 z 误差>0.25m 触发 ee_body_pos 终止风暴 | T1 冒烟监控终止计数；补丁为纯配置（env_cfgs.py body_names 改仅双踝），预留 15 分钟 |
| R3 | 35s 动作迭代耗时超预算（A4 上限 4.9s/it → 4000 it 5.4h 超窗） | T1 实测耗时后按表折算；超 3.5s/it 降 num_envs 1024；D2 允许 3500 it 提前停；收敛看 end_time_ratio/episode length |
| R4 | ONNX 导出与 extract 脚本层命名不匹配 | 图契约已三重核实（§3.1 事实 4）；若仍失配，只允许改 extract 正则（≤5 行，15 分钟上限）；01:00-03:00 工具链 dry-run 提前排雷 |
| R5 | 训练进程早夭（旧管线有前科） | tmux 后台 + 500-it checkpoint 粒度；崩溃后 15 分钟内确认 rsl_rl resume 参数（A3），可用则续训、不可用则重跑 4000（02:00 前崩溃仍来得及，之后走 fallback b） |
| R6 | 磁盘 95% 满 | TMPDIR 已指 /home；训练前后 df 检查；旧 run/eval_out 清理编入 00:30 与全程护栏；增量估算 <300MB |
| R7 | 35s 循环回绕姿态跳变 | 回绕点 rebaseToReference；残余 1 帧微跳可接受；D3 窗口兜底 |
| R8 | 双机兼容回归被破坏 | 所有部署改动走 COMBO_MODE 分支，默认路径红线回归（§8） |
| R9 | 单卡争用（eval_ckpt 与训练并行） | eval_ckpt 固定 --num-envs 8（约 1.5GB）；严禁同时起两个训练 |

## 10. Out of scope

- 步法/接近-撤退类追踪片段训练；tactics.json 策略重训（带病权重维持现状）。
- PBHC 数据集迁移、GVHMR 视频重采、任何新动捕采集。
- 真机部署、IsaacGym 任何形式的引入。
- AMO 默认模式的行为改动（含脚本摆臂）。
- 双机对战玩法逻辑改动（命中判定、KO 规则、回合系统）。
- 多 clip 混合/过渡优化（blend 权重打磨）。
- 旧 4 槽位（guard/jab/cross/hook）权重与门禁现状的清理或重训。

## 11. 待确认清单（不阻塞开工）

1. 35s xsens 动作是否含摆拳（T0 确认；缺则按 §6 走 fallback a，验收①不降级）。
2. rsl_rl checkpoint 续训参数可用性（A3，崩溃时 15 分钟内现场确认）。
3. 验收观看形式：截图是否足够、是否必须录屏 mp4（默认两者都出）。
4. 双机 combo 模式两机器人是否要求动作帧级同步（默认不同步，各自循环）。
5. 用户是否接受 fallback b 的降级验收（仅直拳）作为最坏情形的交付口径（默认接受，届时上报）。

## 12. 交接声明

本方案为架构产物，不含任何实现代码。方案完成后交接给 coding（按 §7 时间表执行，T0/T1 gate 后向队长回报）、code-review（T4 独立回归与验收对照）。方案级决策已全部固化于 §4/§5/§6/§7，coding 无需再做方案级选择；执行中发现与事实不符时，停手上报队长，不擅自改方案。

---

## 13. 交付记录修订（2026-09-29 06:05，应 code-review M1 溯源更正）

**实际部署权重 = 滚动门 iter-6000 快照**，唯一来源为 run 目录全量 ONNX：
`​.workbuddy/gpu/g1dance_pipeline/third_party/unitree_rl_mjlab/logs/rsl_rl/g1_tracking/2026-09-29_03-18-57/2026-09-29_03-18-57.onnx`（md5 `cee55d7ad98a226be966a4df69651f14`）；部署物 `vendor/policy/boxing_combo.bin` md5 `49c6fb54f0d1ff317880fd7cba7fdd44`，与该 ONNX 现取提取逐字节一致（06:03 实测）。

更正说明：早前报告所写"部署=model_3000.pt 重导出"不准确——model_3000 重导出版本曾于 04:39 过滚动门单机闸（12.88s / minZ 0.650），但已被后续滚动替换覆盖；其提取 md5 `6b85d0d8…` ≠ 部署物，留档于 `.workbuddy/gpu/eval_out/md5_probe/`。

门控行史：iter 3000（单机闸过）之后 3500/4000/4500/5000/5500 连续 5 档 FAIL（门控震荡），iter 6000 快照过全段闸成为部署物。**禁令：iter3000-5500 之间任何 checkpoint / ONNX 快照不可互换或互相重导出；部署一致性必须以 md5 对照验证。**

资产保护（禁清理清单）与已发生的清理损失记录（model_6000.pt 于 05:52 护栏清理中误删、全量 ONNX 幸存）：见 `.workbuddy/gpu/logs/DELIVERY_PROVENANCE.md`（溯源唯一权威文件）。最终 T4/e2e 全绿结果（combo 双机 42s PASS、默认 16s+42s PASS、向量 PASS、e2e PASS）均运行于该部署物之上，结论不变。
