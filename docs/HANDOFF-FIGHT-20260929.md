# 交接文档：双机真实对打功能（2026-09-29 23:15）

> 本文是完整交接说明。接手人从这里继续，无需前文上下文。
> 前置成果（单人拳击重构）已交付验收，不在本交接范围内，但关键约束延续（见 §5）。

## 1. 项目背景与目标

- 项目：`/home/xy/zcode/g1-boxing-wasm`——宇树 G1 双机器人拳击演示，浏览器端 MuJoCo WASM + Three.js，策略 ONNX 由 JS 推理。
- 已交付：`?scene=tracking&combo=1` 两台 G1 循环播放单人拳击动捕（组合拳，36s 零摔倒）；`&faceoff=1` 面对面站位。
- **当前目标（用户新需求）**：两台 G1 **真实对打**——互相逼近、出拳命中、格挡、可摔倒（区别于现状的"各打各的影子拳"）。
- 路线结论（已裁决）：双人动捕回放路线已被数据证伪（见 §3.1），走 **对抗自博弈路线**，三阶段递进（P1 耐打 → P2 交战 → P3 上线）。

## 2. 权威文件（接手人必读，按序）

1. `docs/plan-fight-20260929.md`（382 行）——本功能的架构方案：三阶段任务分解、奖励/观测设计、D0-D3 决策点、fallback 树、验收标准。**附录 A** 记录了 D0/D1 全部裁决与判据（含"存活率假象"背景）。
2. `.workbuddy/gpu/logs/DELIVERY_PROVENANCE.md`——权重溯源与禁清理资产清单（§1-§8 为单人拳击阶段，§9 预留给对打功能）。
3. `.workbuddy/gpu/sparring_pids.txt`——对打训练进程登记（显式 PID 规则）。
4. 训练代码（均已实现并冒烟通过，位于本项目副本内，**g1-dance 原项目绝不写入**）：
   - `.workbuddy/gpu/g1dance_pipeline/third_party/unitree_rl_mjlab/src/tasks/tracking/config/g1/sparring_env_cfgs.py`（双实体 env：robot + opponent）
   - `.../src/tasks/tracking/mdp/opponent_driver.py`(脚本对手状态机：逼近→出拳→撤步，SA2RT 软预算三档课程)
   - `.../src/tasks/tracking/mdp/sparring_mdp.py`（P1 奖励/终止 + P2 观测/奖励占位）
   - `.../src/tasks/tracking/config/g1/sparring_rl_cfg.py`（lr 2.5e-4 / desired_kl 0.004）
   - `.../src/tasks/tracking/config/g1/__init__.py`（注册 Unitree-G1-Sparring-P1/P2；旧 tracking 任务零改动）
   - `.../pipeline/train_sparring.sh`、`pipeline/d0_smoke_sparring.py`、`pipeline/d1_survival_sparring.py`
5. 日志：`.workbuddy/gpu/logs/spar_*.log`（D0 冒烟、D1 基线、两轮 P1 训练、D1 评估）。

## 3. 关键事实（已验证，直接采信）

### 3.1 数据定性
- LAFAN 双人格斗 CSV（`g1_fight1_subject2/3/5`、`g1_fightAndSports1_subject1`，本项目 `.workbuddy/gpu/data/`）**不是双人 duet**：两两根距 mean 2.6-5.2m / max 8-10m，<0.6m 时间占比 ≤7%，关节角相关仅 0.07-0.14，近距时不互指。**不可回放双人对打**，只能作单人技能源。
- 这批 CSV 四元数是 xyzw（与 csv_to_npz 直通）；wxyz 的坑只在 GMR 导出的 xsens 文件（已修，`xsens_boxing_v2.csv` 可直用）。
- `xsens_boxing_v2.npz`（12.5s 单人拳击，20.5-33.0s 窗口）是现役单人策略的数据，可作脚本对手回放源。

### 3.2 现役单人策略（回归基线）
- 部署物 `vendor/policy/boxing_combo.bin` md5 `cff79874`，来源 = 副本 `logs/rsl_rl/g1_tracking/2026-09-29_08-10-21/model_5000.pt`（v3 续训 run）。
- 单机全段 torsoErr 基线 **0.201**（test_tracking_sim.mjs 口径）；被击打基线：存活 100% 但 anchor 跟踪终止 **14.17 次/env/10s**（"存活率假象"：anchor 0.25m 保护性重置掩盖跟踪崩塌），命中 829 次/3min @0.8m 拳距。
- 训练管线：mjlab 1.2.0 + unitree_rl_mjlab + rsl_rl 5.0.1，conda env `g1dance`，1024 envs 实测 1.5-1.9s/iter（4060 Ti 独占）。

### 3.3 接触致命性
现策略零接触训练分布：双机贴身（0.5m 出生）实测 2 秒内双双倒地（t4_combo_3000_v3/v4.log，minZ≈0.06）。**对手观测 + 接触鲁棒训练是对打的前置条件**（P1 的意义）。

### 3.4 P1 已完成（本轮最新进展）
- 第一轮止损：lr 1e-3 下 600 iters，权重漂移 5-8%、torsoErr 0.469→0.509（判据 0.231）——"学耐打"变"忘跟踪"，显式 PID 止损。
- 第二轮（lr 2.5e-4 / KL 0.004 / 对手课程放缓）跑满 2000 iters（16:47-17:52），产物：`logs/rsl_rl/g1_sparring/2026-09-29_17-00-28/`（model_5000→6999.pt，全量图 `2026-09-29_17-00-28.onnx`，policy.onnx）。
- **D1 评估（model_6999，队长代跑，`spar_d1_model6999.log`）**：3 分钟 32 envs 连续击打（0.8m 拳距），**存活率 100%** ✓（判据 ≥90%）、anchor 跟踪终止 **0.62 次/env/10s** ✓（基线 14.17，降 96%；判据 ≤5.7）、有效击打 1580 次。

## 4. 待办（接手人按序执行）

### 任务 0：补完 D1 第三判据（半小时内）
- 从 `2026-09-29_17-00-28.onnx` 用 `tools/extract_tracking_onnx.py` 提取到**探针位** `vendor/policy/boxing_cand`（绝不直接覆写现部署），跑 `tools/test_tracking_sim.mjs` 测 torsoErr。
- 判据 ≤0.231（基线 0.201）。达标→P1 闸门通过；不达标→做"耐打换跟踪"的成本分析报负责人裁决（不必然否决）。

### 任务 1：P2 自博弈（核心，预计 3-5h 训练）
按方案 §P2 与已批裁决执行：
- **观测 154→168**：新增 14 维对手段（对手 root 相对位置 3 + 相对朝向 3 + 相对线速度 3 + 双拳位置 6，全部在己方 anchor 系）。新 obs term 写在 sparring_mdp.py，只在 sparring cfg 挂载，主 tracking cfg 不动。
- **快照池自博弈（已批选 A）**：env 内冻结快照池，每 200 iters 轮换、池深 4、初值池 = {单人 iter-5000 权重, model_6999}；对手 driver 从脚本切换到池内策略驱动（脚本模式保留作开关）。rsl_rl 零改动。
- **奖励**：0.8·task（跟踪降权正则 + engage 门控：距离 0.5-1.2m 且面对 cosθ>0.9 + 命中需接触力≥10N）+ 0.2·style（现有 tracking exp 项，不引入 AMP 判别器）；摔倒（身高<0.40m）判负终止。
- **预算**：8k iters 封顶 / 墙钟 5h 封顶，超时取最优 checkpoint。
- **D2 中检（~2k iters，停下报负责人）**：判据 = 自主进入 0.8-1.2m 交战带时间占比上升 + 出现无脚本编排的自主命中。退化（互撞躺平/躲太远）时启用三保险（力阈值/判负终止/快照池），仍不收敛报负责人再议（备选：参数共享批翻倍 spike 或 F1 耐打版交付）。
- 双机 obs 168 后先 500 envs 冒烟再上 1024。

### 任务 2：P3 浏览器 `?fight=1` 部署
- 新模式 `?scene=tracking&fight=1`：出生间距=训练终值（D0 定 1.2m，若 P2 交战带变化则对齐）、KO/回合重置/记分全套启用（复用 boxing_ai.mjs 的 detectHits:1016 / registerHit:1046 / checkFall:1084（KO 阈 0.40）/ resetRound:1104）。
- 浏览器 obs 加对手段：`src/tracking_policy.mjs` policyTick（:326 之后续写，obs 按 meta.obs_dim 分配）；对手引用注入 `src/boxing_ai.mjs:383-384` 与 `:1129-1130` 两处 `new TrackingFighter`（`f.opponent` 互指已存在 :277-278）。
- `tools/extract_tracking_onnx.py` 的 size 表（:184-187）加 `opponent_state: 14`；训练导出端 observation_names 同步；meta 校验更新。
- **红线**：默认模式与 `?combo=1`（含 faceoff）字节级不回归。最终验证 = T4 全套（BOXING_COMBO=1 42s 严格 + 默认 16s/42s 红线 + JS-vs-ONNX 向量 + meta 校验）+ e2e 三模式（combo / combo&faceoff / fight）全绿 + 截图。
- 部署流程：先探针位验证 → 替换部署 → 更新 `DELIVERY_PROVENANCE.md` §9（来源 checkpoint、md5 链、判据数字、三模式验证索引）。

### 任务 3：收尾
- 独立 code-review 复核全部证据 → 负责人验收 → 磁盘按禁清理清单收尾 → 报告。

## 5. 硬约束（违反即事故）

1. **`/home/xy/zcode/g1-dance` 绝对只读**（另一会话的 fosbury 任务在那工作）。一切训练在副本 `.workbuddy/gpu/g1dance_pipeline/` 内进行。
2. **import 坑**：`import src.tasks` 会被 site-packages 的 editable finder 抢占解析到原仓库（撞 fosbury 坏扩展）。启动训练前必须 `PYTHONPATH=<副本>/third_party/unitree_rl_mjlab` 前置 + `python -c "import src.tasks; print(src.tasks.__file__)"` 验证指向副本。
3. **进程终止仅限显式 PID 清单**（模式匹配 kill 曾误杀其他会话训练，立过队规）。登记文件：`sparring_pids.txt`。
4. **磁盘**：仅余 ~29G。训练目录按 DELIVERY_PROVENANCE 禁清理清单收尾（宁留勿删）。
5. **部署纪律**：探针位（boxing_cand）先行验证，全套 T4+e2e 通过才可替换线上；每次替换更新溯源文档（md5 链 + 判据数字）。
6. 新训练逻辑只进新文件/新任务注册名；既有 tracking 任务文件保持字节不动。
7. 回归红线：`?combo=1`（含 faceoff）与默认模式行为不回退，T4 是放行依据。

## 6. 已知风险与规避

- naive 自博弈震荡/退化 → 快照池 + 判负终止 + 命中力阈值三保险；D2 不收敛则报负责人决策（备选 F1 耐打版交付：脚本对手 + 耐打我方 + 记分启用，本身已是可演示形态）。
- 双 articulation 训练吞吐：D0 实测 1.5-1.9s/iter（1024 envs）；P2 加对手观测后先 500 envs 冒烟。
- 对手被大力击飞（dist 峰值 40-60m）：opponent_fall 终止正确触发，属正常；课程节奏可调（driver 三档，预授权范围）。
- rsl_rl 5.0.1 双机自博弈兼容性：快照池方案（选 A）规避了框架改动；若 D2 后需要批翻倍方案 B，先报批。

## 7. 参考配方（调研已核实）

- **RoboStriker**（双 G1 自主拳击，唯一直接先例）：arXiv 2601.22517 / 2608.16195，配方 = 跟踪先验微调（消融：从零自博弈胜率 0%）→ warmup → 快照池自博弈（LS-NFSP 思想）；奖励 0.8 task + 0.2 style；命中力阈值 10N；交战 d∈[0.5,1.2]m 且 cosθ>0.9；摔倒<0.4m 判负；对手观测仅两组相对向量。
- OP3 足球（arXiv 2304.13653，代码未开源）：分层课程先例。
- 本地 push 域随机化（tracking_env_cfg.py push_robot）：接触鲁棒的既有基座。
- 约定：只借鉴配方/超参/奖励结构，不拷贝 RoboStriker 代码（匿名审稿库 license 不明）。

## 8. 环境速查

- 训练：`cd .workbuddy/gpu/g1dance_pipeline && PYTHONPATH=$PWD/third_party/unitree_rl_mjlab PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /home/xy/miniconda3/envs/g1dance/bin/python scripts/train.py Unitree-G1-Sparring-P1 --motion-file .../data/xsens_boxing_v2.npz --env.scene.num-envs 1024 ...`（或用 `pipeline/train_sparring.sh`）
- 演示：`node server.mjs 8081` → `http://localhost:8081/?scene=tracking&combo=1`（&faceoff=1；fight=1 为待交付新模式）
- 浏览器回归：`BOXING_COMBO=1 node tools/test_tracking_boxing.mjs boxing`、`node tools/test_tracking_boxing.mjs {16,42} boxing`、`node tools/test_tracking.mjs vendor/policy/boxing_combo`、`node .workbuddy/gpu/e2e_tracking.cjs "<url>"`
- GPU：RTX 4060 Ti 16GB；注意另一会话可能在 g1-dance 跑训练（共享时吞吐降 ~50%，只接受不干预）。
