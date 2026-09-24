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
