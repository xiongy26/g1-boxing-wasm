# G1 拳击动作跟踪训练 — GPU 机器执行手册

> ⚠️ 本文档所述 .workbuddy/gpu 本机训练链已于 2026-10 停用；动作训练移交
> /home/xy/zcode/g1-dance（见其 docs/ACTION_WORKFLOW.md）。本文仅作历史记录。

> 写于 2026-09-23。本文档面向**在一台有 NVIDIA GPU 的机器上执行训练**的开发者，
> 与本仓库（g1-boxing-wasm，浏览器端 MuJoCo WASM 项目）对接。
> 路线依据：`docs/research-boxing-approaches.md` 的"路线 B"，范本
> `mujocolab/g1_spinkick_example`（mjlab 官方示例，流程已验证）。
> 产出目标：能出拳击动作的 50Hz 全身跟踪策略（ONNX）→ 移植回本仓库浏览器端。

---

## 0. 总览：要做什么、产出什么

```
拳击动捕数据 → 重定向 CSV → npz(FK补全) → wandb registry
     → mjlab GPU 训练(4096 并行环境, PPO, ~1-2天/片段)
     → ONNX(自动导出) → 手写 JS 前向移植回本仓库
     → 拳击 AI 从"姿态切换"升级为"片段触发"
```

**成功标准**：机器人能用整条动力链（蹬地-转腰-送肩）打出刺拳/交叉拳/勾拳，
在 MuJoCo WASM 浏览器场景里稳定站立、命中判定正常、无 NaN。

**明确不做**：BeyondMimic 的扩散控制器（20M 参数，浏览器不可移植，只取跟踪 MLP）；
RoboStriker Stage2/3（隐空间蒸馏+自博弈，另行立项）。

---

## 1. 硬件与软件要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，≥24GB 显存最好（4090 参考配置）；mjlab 用 MuJoCo-Warp，CUDA 必须 |
| OS | Linux 优先（WSL2 可试但 EGL 渲染易踩坑），macOS Apple Silicon 未验证 |
| Python | 由 `uv` 管理（仓库带 `.python-version`，跟随即可） |
| 包管理 | [uv](https://docs.astral.sh/uv/) |
| 账号 | wandb（动作托管 + 训练日志都走它，流程里绕不开） |

---

## 2. 环境搭建（第 1 天上午）

```bash
# 1) 装 mjlab 并跑通官方旋踢范例 —— 这是整条管线的"烟雾测试"
git clone https://github.com/mujocolab/g1_spinkick_example.git && cd g1_spinkick_example
uv sync
```

**验收 A**：`uv run python -c "import mjlab; print(mjlab.__version__)"`
不报错，且 `pip list | grep mujoco-warp` 能看到。**锁版本**：旋踢范例当前锁
mjlab v1.1.0（2026-02），mjlab API 迭代快，一切以范例能跑通的版本为准，
`uv.lock` 整个保留。

```bash
# 2) 直接用仓库自带的预训练 ONNX 先验证部署链路（训练还没开始就先确认终点能走通）
#    spinkick_safe.onnx 已在仓库根目录
```

**验收 B（重要，别跳过）**：把这个 `spinkick_safe.onnx` 放进本仓库
（`vendor/policy/spinkick.bin` + 写一份 meta json），用 `src/rl_policy.mjs`
的手写 JS 前向跑通逐 tick 对齐——**在投入 1-2 天训练之前，先证明"训练端 ONNX →
浏览器 JS"这条路是通的**。AMO 当年就是这么验证的（误差 ~1e-7）。
若 obs 布局对不上，读 ONNX 元数据里的 `observation history lengths`
（whole_body_tracking 仓库 2025-10 的 commit "Add observation history lengths
to ONNX metadata" 就是干这个的）。

---

## 3. 数据获取与转换（第 1~2 天）

### 3.1 数据来源（按优先级）

1. **RoboStriker 匿名仓库** `https://anonymous.4open.science/r/robostriker-B25B`
   自带 `fight1_subject2.npz`（拳击动捕，G1 广义坐标，50Hz）。
   注意：匿名仓库许可未明，**仅作数据来源参考/对照**，方法引用论文
   arXiv:2608.16195。
2. **首选（许可干净）**：HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset`
   —— 宇树官方把 LAFAN1 动捕库重定向成 G1 CSV（广义坐标），
   正是 BeyondMimic/mjlab 要的输入格式。LAFAN1 的 fight 系列片段
   （RoboStriker 的文件名 `fight1_subject2` 与其命名习惯一致）优先找；
   找不到就用 fight/shadowboxing 相近片段。
3. **进阶（可选，管线更长）**：GVHMR（单目视频→SMPLX）+ GMR（人体→G1）
   从真实拳击视频自提；GhostTrial 数据集
   （`DaveRc/ghosttrial-g1-scorpion-motion`）是整条管线的实操示范。

### 3.2 转换流程（照抄旋踢范例，替换数据文件）

mjlab 要求输入 CSV（Unitree 重定向数据集的约定格式）→ npz：

```bash
# csv → npz（内部做 forward kinematics 补全最大坐标：body pose/vel/acc）
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
    --input-file boxing_fight1.csv \
    --output-name boxing_fight1 \
    --input-fps 30 \
    --output-fps 50 \
    --render True      # 生成参考动作视频，务必肉眼检查
```

`--render True` 会把动作上传 wandb registry 并产出参考视频。

### 3.3 片段切分（本项目特有的关键决策）

**不要把整段对打当一个动作训**。切成独立技能片段，每个片段：

- **首尾加"守卫站姿"过渡段**：0.5s 平滑过渡 + 1.0s 驻留
  （旋踢范例的 `--add-start-transition --add-end-transition
  --transition-duration 0.5 --pad-duration 1.0` 就是干这个的；如果用
  csv_to_npz 直转，需要自己写过渡段生成——参考旋踢仓库 `pkl_to_csv.py`
  的实现，它含 Handle dict-based pickle 的逻辑可抄）。
  **这是后面策略间无缝切换的物理基础，宁可多花半天也要做对。**
- 建议片段清单（按训练顺序）：

| 片段 | 内容 | 用途 |
|---|---|---|
| `guard_idle` | 守卫弹跳步循环（可 cycle 重复到 ~8s） | 默认待机/步法 |
| `jab` | 刺拳 1~2 发 + 回守卫 | 快速验证 |
| `cross` | 后手直拳 | 第二个验证 |
| `hook` | 勾拳 | 发力链最丰富，最后训 |
| `bob_weave` | 摇闪（可选） | 防守 |

- 循环片段（guard_idle）像旋踢那样 repeat 到目标时长。

**验收 C**：wandb registry 里每个片段能用 `replay` 脚本/视频回放，
起始和结束都是稳定的守卫站姿，无滑步无跳变。

---

## 4. 训练（第 3~4 天）

mjlab 内置 BeyondMimic tracking task，**零调参设计**：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run train \
    Mjlab-Boxing-Unitree-G1 \
    --registry-name {你的组织}/{registry名}/boxing_jab \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 20_000
```

要点：

- **顺序**：先训 `jab`（最短）验证全流程，成功后再并行/排队训其他片段。
  `guard_idle` 相对简单也早训。
- **超参考点**（旋踢范例 wandb report 有完整记录，链接见其 README）：
  2.65s 动作 2 万迭代收敛；拳击片段更长，预算 1~2 天/片段（4090）。
- 训练机制（均为 BeyondMimic 默认，知道原理即可）：
  - 观测 = 参考动作相对误差（command）+ 本体感知 + 动作历史，50Hz
  - 奖励 = DeepMimic 式（关节姿态/速度 + 身体位姿/速度/加速度跟踪 + 平滑项）
  - RSI 随机初始状态 + 自适应采样（失败多的片段段被优先抽中）
  - 动作空间 = G1 全 23 关节（12 腿 + 3 腰 + 8 臂）——
    **这就是出拳能带蹬地转腰发力链的原因**，浏览器端脚本手臂方案就此退役
- `play` 评估：

```bash
uv run play Mjlab-Boxing-Unitree-G1 \
    --wandb-run-path {org}/{project}/{run-id} \
    --num-envs 8
```

**验收 D**：play 里连续 10 次以上完成片段不摔倒；出拳动作与参考视频
逐帧对比形态合理；把 wandb run 的 ONNX artifact 下载下来。

---

## 5. 移植回本仓库（第 5~6 天）

在**本仓库**（浏览器端）做，不依赖 GPU：

1. 每个片段一份 `vendor/policy/boxing_<clip>.bin` + `boxing_<clip>_meta.json`
   （meta 格式照 `amo_meta.json` 的经验：字节偏移、obs 布局、历史长度）。
2. 在 `src/rl_policy.mjs` 里仿照 `AMONetwork` 写 `TrackingNetwork`：
   - 先用 python 导出若干 (obs → action) 测试向量随文件带来，
     JS 实现逐 tick 对齐到 ~1e-6（AMO 的验收标准，沿用）
   - 50Hz 推理 / 500Hz PD 节奏不变；PD kp/kd 用训练端 `robots/` 配置算出的值
     （**sim2sim 校准点**：训练端 MuJoCo-Warp vs 浏览器 WASM 的接触参数差异
     就在这里暴露）
3. `boxing_ai.mjs` 改造：拳击 AI 从"8 臂关节姿态切换"升级为**片段调度器**——
   - `tactics.mjs` 的战术决策（10 动作）输出不变，但执行层映射变为
     "触发哪个片段策略 + 何时切换"
   - 切换点选在片段首尾的站姿过渡段（0.5s cross-fade，
     参考动作切换的混合权重或直接在过渡段切策略）
4. 回归保障：`tools/test_amo.mjs 22` 的无头管线继续当 sim2sim 回归器
   （KO 注入 + 回合恢复 + 稳定站立 + 命中）。改造后所有回归必须 PASS。

**验收 E（最终）**：浏览器 E2E（`.workbuddy/tmp/e2e_default_rl.cjs` 模式）
PASS——双机对战、真实命中、KO 后 2s 回合重置、≥50FPS、无 NaN；
出拳动作肉眼可见重心转移与转腰。

---

## 6. 里程碑清单（按序打勾）

- [ ] A. mjlab 环境跑通，版本锁定 v1.1.0（或范例 uv.lock 实际版本）
- [ ] B. 旋踢预训练 ONNX 在本仓库浏览器端 JS 前向逐 tick 对齐（训练前的保险）
- [ ] C. 拳击片段数据：转换 + 过渡段 + registry 回放验证
- [ ] D. jab 片段训练收敛 → play 验收 → ONNX 下载
- [ ] E. 其余片段（guard_idle/cross/hook/bob_weave）训完
- [ ] F. 浏览器端 TrackingNetwork 移植 + 测试向量对齐
- [ ] G. boxing_ai 片段调度器 + cross-fade + 无头回归 PASS
- [ ] H. 浏览器 E2E PASS，视觉确认发力链

---

## 7. 已知风险与对策

| 风险 | 对策 |
|---|---|
| LAFAN1 里没有现成拳击片段 | 降级顺序：RoboStriker npz → GVHMR+GMR 从视频自提 → 先用 LAFAN1 近似片段（出拳类 upper-body 动作）打通管线 |
| mjlab API 与本文档记录不符 | 一切以 g1_spinkick_example 能跑通的 commit + uv.lock 为准 |
| 训练端与 WASM 物理差异导致策略失稳 | 验收 B 提前暴露；差异大时在训练配置里对齐 timestep/PD/接触参数，或在本仓库无头环境做 domain 微调 |
| 多片段切换抖动 | 过渡段 + cross-fade；仍抖则合并为 conditioned-on-clip-id 单策略（自己实现，参考 PBHC 思路但注意其 CC BY-NC 禁商用，不能抄代码） |
| 4096 并行环境显存不足 | `--env.scene.num-envs` 降到 2048/1024，训练时间相应拉长 |
| wandb 上传失败/网络问题 | csv_to_npz 支持本地路径回放（replay 脚本），可先本地验证再补传 |

---

## 8. 参考链接

- 旋踢范本（流程母本）https://github.com/mujocolab/g1_spinkick_example
- mjlab（MuJoCo-Warp RL 框架）https://github.com/mujocolab/mjlab
  （tracking task 源码：`src/mjlab/tasks/tracking/tracking_env_cfg.py`）
- BeyondMimic 论文/原版训练 https://arxiv.org/abs/2508.08241
  https://github.com/HybridRobotics/whole_body_tracking
- 部署参考 https://github.com/HybridRobotics/motion_tracking_controller
  与 RoboJuDo https://github.com/HansZ8/RoboJuDo（BeyondmimicPolicy）
- LAFAN1 重定向数据 https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
- RoboStriker（拳击自博弈终极路线）https://arxiv.org/abs/2608.16195
  代码 https://anonymous.4open.science/r/robostriker-B25B
- 本仓库背景文档 `docs/research-boxing-approaches.md`（路线全景）
  `docs/tactics-selfplay-notes.md`（战术层现状：prior-first 已上线，
  其决策输出正是片段调度器的输入）
