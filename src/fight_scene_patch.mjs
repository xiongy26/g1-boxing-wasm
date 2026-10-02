// FIGHT_MODE 场景 XML 运行时补丁（2026-10-01 自 src/main.js 内联逻辑提取为
// 共享函数）：浏览器（src/main.js）与 headless 回归
// （tools/test_tracking_boxing.mjs）必须走同一份补丁——航向伺服验收要求两条
// 路径逐字节同源，杜绝 harness 特判（plan：浏览器路径与 harness 共用同一套
// 实现）。场景文件字节不动，全部为字符串运行时替换。
//
// 补丁内容与依据（原文迁移自 main.js FIGHT_MODE 分支，逐条注释见彼处）：
//  1) 出生 ±0.5 → ±0.6（间距 1.2m = 训练侧 SPARRING_SPAWN_DIST，A yaw 0 /
//     B yaw π 面对面）；
//  2) 求解器对齐：iterations 100 → 10 + ls_iterations=20（训练侧
//     tracking_env_cfg.py SimulationCfg.MujocoCfg 同值）；
//  3) 身体碰撞 hull condim 3 → 1（训练侧 FULL_COLLISION 除脚外全部
//     condim=1；condim=3 切向摩擦让贴身肢体互相"咬合"是同步双倒主因）；
//  4) 脚类 geom 显式 condim=3 + friction 0.6 + priority=1（抵消 3) 的类
//     继承，脚底保持摩擦防滑倒）；
//  5) 拳头 geom 去 priority/friction 改 condim=1（训练侧 hand_collision）。
// 锚串任一失配（上游 XML 改版）时对应替换静默不生效——两条调用路径各自的
// 失配报警（第 2 轮审查 P2-2 核实后的如实描述）：浏览器 main.js 加载后逐项
// includes 校验并 console.warn（出生/求解器/body condim/脚 condim/拳 geom）；
// headless harness（tools/test_tracking_boxing.mjs）核对锚串失配直接
// console.error + exit(1)，不静默带病运行。
export function applyFightScenePatch(xml0) {
  return xml0
    .split('-0.50 0 0.761').join('-0.60 0 0.761')
    .split(' 0.50 0 0.761').join(' 0.60 0 0.761')
    .split('solver="Newton" iterations="100"')
      .join('solver="Newton" iterations="10" ls_iterations="20"')
    .split('<geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1" />')
      .join('<geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1" condim="1" />')
    .split('<geom type="capsule" size="0.01" />')
      .join('<geom type="capsule" size="0.01" condim="3" friction="0.6 0.005 0.0001" priority="1" />')
    .split('contype="1" conaffinity="1" friction="0.8 0.005 0.0001" priority="1"')
      .join('contype="1" conaffinity="1" condim="1"');
}
