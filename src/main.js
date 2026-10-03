// G1 robot boxing — browser app: MuJoCo WASM physics + Three.js rendering.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BoxingController } from './boxing_ai.mjs';
import { AMONetwork } from './rl_policy.mjs';
import { TrackingNetwork } from './tracking_policy.mjs';
import { TacticsPolicy } from './tactics.mjs';
import { applyFightScenePatch } from './fight_scene_patch.mjs';
import { createBoxingGlove } from './boxing_gloves.js';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;

const $ = id => document.getElementById(id);
const loadbar = $('loadbar'), loadmsg = $('loadmsg');
const setProgress = (pct, msg) => { loadbar.style.width = pct + '%'; if (msg) loadmsg.textContent = msg; };

// ------------------------------------------------------------------ AMO weights
// The 18 MB policy download starts immediately and overlaps the engine/mesh
// boot below — RL mode is ON by default once the app opens. Shared promise:
// boot, the per-robot toggles and retries all reuse the same download.
let amoNetPromise = null;
function loadAMONetwork() {
  if (!amoNetPromise) {
    amoNetPromise = Promise.all([
      fetch('./vendor/policy/amo.bin').then(r => {
        if (!r.ok) throw new Error('amo.bin HTTP ' + r.status);
        return r.arrayBuffer();
      }),
      fetch('./vendor/policy/amo_meta.json').then(r => {
        if (!r.ok) throw new Error('amo_meta.json HTTP ' + r.status);
        return r.json();
      }),
    ]).then(([buf, meta]) => new AMONetwork(buf, meta))
      .catch(err => { amoNetPromise = null; throw err; }); // allow a later retry
  }
  return amoNetPromise;
}
loadAMONetwork();

// ------------------------------------------------------------------ mujoco
setProgress(5, '正在加载 MuJoCo WebAssembly 引擎…');
const mujoco = await loadMujoco({
  locateFile: f => new URL(`../vendor/mujoco/${f}`, import.meta.url).href,
});

// stage-4 clip policies (boxing_{guard,jab,cross,hook,combo}); present only
// after the GPU training pipeline has been run — per-clip tolerant (a missing
// slot is skipped; combo ships later than the 4 base clips).
const TRACKING_CLIPS = ['guard', 'jab', 'cross', 'hook', 'combo'];
let trackingNetsPromise = null;
function loadTrackingNets() {
  if (!trackingNetsPromise) {
    trackingNetsPromise = Promise.all(TRACKING_CLIPS.map(name => Promise.all([
      fetch(`./vendor/policy/boxing_${name}.bin`).then(r => {
        if (!r.ok) throw new Error(`boxing_${name}.bin HTTP ` + r.status);
        return r.arrayBuffer();
      }),
      fetch(`./vendor/policy/boxing_${name}_meta.json`).then(r => {
        if (!r.ok) throw new Error(`boxing_${name}_meta.json HTTP ` + r.status);
        return r.json();
      }),
    ]).then(([buf, meta]) => [name, new TrackingNetwork(buf, meta)])
      .catch(err => {
        // 逐 clip 容错（plan-revamp-20260928 §5.2 D3）：单槽位 404/解析失败
        // 只跳过该槽位，combo 未部署不破坏既有 4 槽位加载；全部失败才回退 AMO。
        console.warn(`tracking clip ${name} unavailable — slot skipped:`, err.message);
        return [name, null];
      })))
      .then(entries => {
        const nets = Object.fromEntries(entries.filter(([, net]) => net));
        if (!Object.keys(nets).length) throw new Error('no tracking clip policies deployed');
        return nets;
      })
      .catch(err => { trackingNetsPromise = null; throw err; });
  }
  return trackingNetsPromise;
}

// fight 单策略（FIGHT_MODE 专用，W4）：boxing_fight.bin（168 维观测）或
// &policy= 白名单指定的探针位权重。与 loadTrackingNets 同一容错形状——失败
// 时 promise 复位，调用方决定回退路径。
let fightNetPromise = null;
function loadFightNet() {
  if (!fightNetPromise) {
    fightNetPromise = Promise.all([
      fetch(`./vendor/policy/${FIGHT_BIN}.bin`).then(r => {
        if (!r.ok) throw new Error(`${FIGHT_BIN}.bin HTTP ` + r.status);
        return r.arrayBuffer();
      }),
      fetch(`./vendor/policy/${FIGHT_BIN}_meta.json`).then(r => {
        if (!r.ok) throw new Error(`${FIGHT_BIN}_meta.json HTTP ` + r.status);
        return r.json();
      }),
    ]).then(([buf, meta]) => new TrackingNetwork(buf, meta))
      .catch(err => { fightNetPromise = null; throw err; });
  }
  return fightNetPromise;
}

setProgress(15, '正在下载场景与机器人网格…');
// scene selection: default is the AMO scene; `?scene=tracking` runs the
// stage-4 29-DoF tracking scene — but only if the trained clip policies are
// actually deployed; otherwise fall back to AMO with a note.
const TRACKING_REQUESTED = new URLSearchParams(location.search).get('scene') === 'tracking';
// COMBO_MODE（plan-revamp-20260928 §5.2 D4）：?combo=1 时两拳手常驻循环
// combo 片段（35s 连续组合拳），禁 KO 与回合重置；默认 false 不影响既有路径。
// FIGHT_MODE（plan-fight-20260929 §5.2 W4）：?scene=tracking&fight=1 真实对打
// ——两台 G1 共享 168 维观测（含 opponent_state 14 维）的 fight 策略自主对打，
// KO/回合重置/记分全启用。独立开关（只在 tracking 场景生效）且优先于 combo
// （URL 契约 §5.1：同给时 fight 优先，故 COMBO_MODE 追加 && !FIGHT_REQUESTED）；
// 未带 fight=1 的 URL 两个布尔与旧版完全一致。
const FIGHT_REQUESTED = new URLSearchParams(location.search).get('scene') === 'tracking' &&
  new URLSearchParams(location.search).get('fight') === '1';
// &policy=<basename>.bin（fight 调试参数）：探针位权重加载，限 vendor/policy/
// 目录下的 .bin。白名单 = 字符集 [A-Za-z0-9._-] 且不含 ".."、以 .bin 结尾
// （防路径穿越，meta 路径由同一 basename 派生）；文件不存在时 fetch 404 回退
// 默认 fight 权重。不带该参数时行为与现在完全一致。
const FIGHT_BIN_DEFAULT = 'boxing_fight';
const FIGHT_BIN = (() => {
  const p = new URLSearchParams(location.search).get('policy');
  if (!p) return FIGHT_BIN_DEFAULT;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\.bin$/.test(p) || p.includes('..')) {
    console.warn('policy param rejected (vendor/policy whitelist):', p);
    return FIGHT_BIN_DEFAULT;
  }
  return p.slice(0, -'.bin'.length);
})();
const COMBO_MODE = new URLSearchParams(location.search).get('combo') === '1' && !FIGHT_REQUESTED;
// FACEOFF（Phase 2，2026-09-29）：?combo=1&faceoff=1 的子选项——B 侧重锚定
// 回出生朝向 π（180° 面向 A，真对抗站位）。默认关闭：已交付 delta=0 演示
// 路径字节不变。头less 验证见 tools/_probe_faceoff.mjs。
const FACEOFF_MODE = COMBO_MODE && new URLSearchParams(location.search).get('faceoff') === '1';
let trackingAvailable = false;
if (TRACKING_REQUESTED) {
  trackingAvailable = await fetch('./vendor/policy/boxing_guard_meta.json')
    .then(r => r.ok).catch(() => false);
  if (!trackingAvailable) {
    setProgress(16, '追踪片段策略尚未就绪，回退 AMO 模式…');
  }
}
const TRACKING = TRACKING_REQUESTED && trackingAvailable;
// fight 权重部署探测（与上方 guard meta 探测同构）：meta 缺失 = fight 尚未
// 部署 → FIGHT_MODE 不生效，场景 XML 不打 ±0.6 补丁，页面走既有 tracking
// 行为（控制台留痕）；meta 在而 bin 损坏属异常部署，启动加载路径兜底回退。
let fightAvailable = false;
if (FIGHT_REQUESTED) {
  fightAvailable = await fetch(`./vendor/policy/${FIGHT_BIN}_meta.json`)
    .then(r => r.ok).catch(() => false);
  if (!fightAvailable) {
    setProgress(16, `对打策略（${FIGHT_BIN}）尚未就绪，回退追踪模式…`);
    console.warn(`fight policy meta missing: vendor/policy/${FIGHT_BIN}_meta.json — fight=1 ignored`);
  }
}
// FIGHT_MODE 生效条件：fight=1 请求 + 追踪场景可用 + fight 权重已部署
// （fight 权重是 29-DoF tracking 场景策略，AMO 场景无法挂载）。
const FIGHT_MODE = FIGHT_REQUESTED && TRACKING && fightAvailable;
const SCENE_FILE = TRACKING ? './models/scene_boxing_tracking.xml' : './models/scene_boxing_amo.xml';
const xml0 = await fetch(SCENE_FILE).then(r => r.text());
// COMBO 模式出生隔离（plan-revamp 终局裁决 2026-09-29）：双机互看同一段
// 全接触拳击参考，±0.5m 出生间距下首拳必达对手引发连环倒（单机闸门已证
// 策略全段稳定）。加载时把 keyframe 出生位 ±0.5 → ±1.2（间距 2.4m，超出
// 拳距），tracking 场景文件字节不动；默认 AMO 场景不受影响。
// FIGHT 模式出生对峙（W4）：±0.5 → ±0.6（间距 1.2m，与训练侧
// SPARRING_SPAWN_DIST=1.2 一致，A yaw 0 / B yaw π 面对面），同构替换、
// 场景文件字节不动；fight 优先于 combo（FIGHT_MODE 蕴含 !COMBO_MODE）。
// FIGHT 物理对齐补丁（终审 P1）：出生 ±0.6 + 求解器/condim 对齐——2026-10-01
// 提取为 src/fight_scene_patch.mjs（浏览器与 headless 回归共用的同一函数，
// 航向伺服验收要求两条路径逐字节同源）；补丁逐条依据的原文注释随迁到该模块。
const xml = FIGHT_MODE
  ? applyFightScenePatch(xml0)
  : COMBO_MODE
    ? xml0.split('-0.50 0 0.761').join('-1.20 0 0.761').split(' 0.50 0 0.761').join(' 1.20 0 0.761')
    : xml0;
if (FIGHT_MODE && xml === xml0) console.warn('fight spawn replacement did not apply');
// 物理对齐补丁生效性校验：四个锚串任一失配（上游 XML 改版）都只丢对应项，
// 逐项报警避免静默带病运行。
if (FIGHT_MODE) {
  if (!xml.includes('ls_iterations="20"')) console.warn('fight patch: solver alignment did not apply');
  if (!xml.includes('<geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1" condim="1" />')) console.warn('fight patch: body collision condim did not apply');
  if (!xml.includes('size="0.01" condim="3"')) console.warn('fight patch: foot condim/friction did not apply');
  if (xml.includes('friction="0.8 0.005 0.0001" priority="1"')) console.warn('fight patch: fist geom alignment did not apply');
}
if (COMBO_MODE && xml === xml0) console.warn('combo spawn isolation did not apply');
// mesh file names come straight from the scene's asset list
const stlNames = [...new Set([...xml.matchAll(/file="([^"]+\.STL)"/g)].map(m => m[1]))];

const vfs = new mujoco.MjVFS();
let done = 0;
await Promise.all(stlNames.map(async name => {
  const buf = new Uint8Array(await (await fetch('./models/unitree_g1/assets/' + name)).arrayBuffer());
  vfs.addBuffer('unitree_g1/assets/' + name, buf);
  done++;
  setProgress(15 + 70 * done / stlNames.length, `正在加载网格 ${done}/${stlNames.length}…`);
}));

setProgress(90, '正在编译模型…');
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

const OBJ = mujoco.mjtObj;
const id = (t, n) => mujoco.mj_name2id(model, t.value ?? t, n);
const ctl = new BoxingController(model, data, {
  mujoco,
  comboMode: COMBO_MODE,
  faceoff: FACEOFF_MODE,
  fightMode: FIGHT_MODE,
  ids: {
    pelvisA: id(OBJ.mjOBJ_BODY, 'A_pelvis'), pelvisB: id(OBJ.mjOBJ_BODY, 'B_pelvis'),
    headA: id(OBJ.mjOBJ_SITE, 'A_head'), headB: id(OBJ.mjOBJ_SITE, 'B_head'),
    fistLA: id(OBJ.mjOBJ_SITE, 'A_left_fist_site'), fistRA: id(OBJ.mjOBJ_SITE, 'A_right_fist_site'),
    fistLB: id(OBJ.mjOBJ_SITE, 'B_left_fist_site'), fistRB: id(OBJ.mjOBJ_SITE, 'B_right_fist_site'),
  },
});
const DT = model.opt.timestep;

// RL on by default: attach policies to both robots before the first frame so
// they box from the moment the loading overlay disappears. AMO scene → the
// AMO whole-body policy; tracking scene → the clip-policy library.
// fightActive（P1-1，审查修复）：运行时 fight 生效标志——loadFightNet 成功挂载
// 后才置位。编译期 const FIGHT_MODE 只反映"meta 探测通过"；bin 损坏回退后
// UI/toggle 一律改读本标志，不再读 FIGHT_MODE。
let fightActive = false;
if (TRACKING) {
  if (FIGHT_MODE) {
    // fight 模式：单策略挂双侧（168 维 obs，双机对称自系构造见
    // tracking_policy.mjs policyTick 的 opponent_state 段）。
    // bin 缺失/解析失败的回退（P1-1）：关掉 ctl.fightMode 与 fightActive，
    // 挂既有片段栈——行为逻辑回到默认 tracking（KO/回合系统本就启用）。
    // 残留差异（catch 内无法撤销）：场景 model 已按 ±0.6 出生位编译（XML
    // 补丁在权重加载前打入），回合重置的 keyframe 仍是 1.2m 对峙位；需要
    // 精确默认行为的场景应去掉 fight=1 刷新页面。
    setProgress(94, '正在加载对打策略…');
    try {
      const fightNet = await loadFightNet();
      ctl.setTracking('A', { fight: fightNet });
      ctl.setTracking('B', { fight: fightNet });
      fightActive = true;
      console.log(`fight policy loaded: ${FIGHT_BIN} obs_dim=${fightNet.meta.obs_dim}` +
        ` opponent_state@${JSON.stringify(fightNet.meta.obs_layout?.opponent_state ?? null)}`);
    } catch (err) {
      console.error(`fight policy (${FIGHT_BIN}) failed to load — falling back to clip stack`, err);
      ctl.fightMode = false;
      fightActive = false;
      try {
        const nets = await loadTrackingNets();
        ctl.setTracking('A', nets);
        ctl.setTracking('B', nets);
      } catch (err2) {
        console.error('tracking policies failed to load at boot — scripted fallback', err2);
      }
    }
  } else {
    setProgress(94, '正在加载拳击片段策略…');
    try {
      const nets = await loadTrackingNets();
      ctl.setTracking('A', nets);
      ctl.setTracking('B', nets);
      console.log('tracking clip policies loaded:', Object.keys(nets).join(', '));
    } catch (err) {
      console.error('tracking policies failed to load at boot — scripted fallback', err);
    }
  }
} else {
  setProgress(94, '正在加载 AMO 策略权重…');
  try {
    const amoNet = await loadAMONetwork();
    ctl.setAMO('A', amoNet);
    ctl.setAMO('B', amoNet);
  } catch (err) {
    console.error('AMO policy failed to load at boot — using scripted fallback', err);
  }
}

setProgress(100, '完成！');
$('loading').style.opacity = '0';
setTimeout(() => $('loading').remove(), 600);

// ------------------------------------------------------------------ three.js
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0b0e14, 9, 22);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.up.set(0, 0, 1); // MuJoCo is z-up
camera.position.set(3.1, -2.4, 1.7);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.95);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI / 2 - 0.03;
controls.minDistance = 1.2;
controls.maxDistance = 9;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.7;
controls.addEventListener('start', () => setOrbit(false));

scene.add(new THREE.HemisphereLight(0x9db4ff, 0x4a4038, 0.85));
const fill = new THREE.AmbientLight(0xffffff, 0.25);
scene.add(fill);
const key = new THREE.DirectionalLight(0xfff2e0, 1.6);
key.position.set(2.5, -1.5, 4);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -3; key.shadow.camera.right = 3;
key.shadow.camera.top = 3; key.shadow.camera.bottom = -3;
key.shadow.camera.far = 12;
key.shadow.bias = -0.0003;
scene.add(key);
const rim = new THREE.DirectionalLight(0x8fb0ff, 0.5);
rim.position.set(-3, 2.5, 2);
scene.add(rim);

// ring-side ambience: dark mat + yellow ring (decor only, not physics).
// NOTE: three.js CircleGeometry/RingGeometry live in the XY plane — already
// horizontal in this z-up world. Only GridHelper (XZ) needs a quarter turn.
const mat = new THREE.Mesh(
  new THREE.CircleGeometry(2.6, 64),
  new THREE.MeshPhongMaterial({ color: 0x232733, shininess: 4 }),
);
mat.position.z = 0.001;
mat.receiveShadow = true;
scene.add(mat);

const ring = new THREE.Mesh(
  new THREE.RingGeometry(1.32, 1.38, 96),
  new THREE.MeshBasicMaterial({ color: 0xd8b83a, side: THREE.DoubleSide }),
);
ring.position.z = 0.004;
scene.add(ring);

const grid = new THREE.GridHelper(10, 20, 0x3a4256, 0x272d3d);
grid.rotation.x = Math.PI / 2; // GridHelper is built in the XZ plane; lay it flat for z-up
grid.position.z = 0.002;
scene.add(grid);

// ------------------------------------------------------------------ mjv scene
const mjvScene = new mujoco.MjvScene(model, 20000);
const mjvOption = new mujoco.MjvOption();
const mjvPerturb = new mujoco.MjvPerturb();
const mjvCamera = new mujoco.MjvCamera();
mjvOption.geomgroup[0] = 1; // floor
mjvOption.geomgroup[1] = 1; // ring line
mjvOption.geomgroup[2] = 1; // visual meshes
mjvOption.geomgroup[3] = 0; // collision hulls off
mjvOption.geomgroup[4] = 1;
mjvOption.geomgroup[5] = 0; // sites off

// geometry cache: mjvGeom (type,size,dataid) -> BufferGeometry
const geomCache = new Map();
const meshGeoms = new Map(); // mesh id -> BufferGeometry

function getMeshGeometry(mid) {
  if (meshGeoms.has(mid)) return meshGeoms.get(mid);
  const va = model.mesh_vertadr[mid], vn = model.mesh_vertnum[mid];
  const fa = model.mesh_faceadr[mid], fn = model.mesh_facenum[mid];
  const positions = new Float32Array(vn * 3);
  for (let i = 0; i < vn * 3; i++) positions[i] = model.mesh_vert[3 * va + i];
  const indices = new Uint32Array(fn * 3);
  for (let i = 0; i < fn * 3; i++) indices[i] = model.mesh_face[3 * fa + i];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  meshGeoms.set(mid, g);
  return g;
}

function getGeometry(mjvGeom) {
  const type = Number(mjvGeom.type), dataid = Number(mjvGeom.dataid);
  const size = Array.from(mjvGeom.size).map(Number);
  if (type === mujoco.mjtGeom.mjGEOM_MESH.value) {
    return getMeshGeometry(dataid);
  }
  const key = JSON.stringify([type, size]);
  if (geomCache.has(key)) return geomCache.get(key);
  let g;
  if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
    g = new THREE.PlaneGeometry(size[0] ? 2 * size[0] : 12, size[1] ? 2 * size[1] : 12);
  } else if (type === mujoco.mjtGeom.mjGEOM_SPHERE.value) {
    g = new THREE.SphereGeometry(size[0], 24, 16);
  } else if (type === mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
    g = new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 16);
    g.rotateX(Math.PI / 2);
  } else if (type === mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
    g = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 28);
    g.rotateX(Math.PI / 2);
  } else if (type === mujoco.mjtGeom.mjGEOM_BOX.value) {
    g = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
  } else {
    g = new THREE.SphereGeometry(0.02, 8, 6);
  }
  geomCache.set(key, g);
  return g;
}

const meshes = [];      // one three mesh per mjv geom slot
const group = new THREE.Group();
scene.add(group);

const gloveGeoms = new Map();
const coveredHandMeshes = new Set();
for (const side of ['A', 'B']) {
  for (const hand of ['left', 'right']) {
    const fistId = id(OBJ.mjOBJ_GEOM, `${side}_${hand}_fist`);
    if (fistId < 0) continue;
    // AMO nests the fist inside the rubber-hand body; tracking puts it on
    // the wrist directly. Account for that offset so both cuffs fit the wrist.
    const rubberBody = id(OBJ.mjOBJ_BODY, `${side}_${hand}_rubber_hand`);
    const bodyId = Number(model.geom_bodyid[fistId]);
    const handOffset = bodyId === rubberBody ? Number(model.body_pos[3 * bodyId]) : 0;
    gloveGeoms.set(fistId, {
      hand, wristX: -Number(model.geom_pos[3 * fistId]) - handOffset,
    });
    for (const name of [`${side}_${hand}_rubber_hand`, `${hand}_rubber_hand`]) {
      const meshId = id(OBJ.mjOBJ_MESH, name);
      if (meshId >= 0) coveredHandMeshes.add(meshId);
    }
  }
}

function syncRenderer() {
  mujoco.mjv_updateScene(model, data, mjvOption, mjvPerturb, mjvCamera,
    mujoco.mjtCatBit.mjCAT_ALL.value, mjvScene);
  const geoms = mjvScene.geoms; // embind vector wrapper — delete() before returning
  const n = geoms.size();
  try {
    for (let i = 0; i < n; i++) {
      const g = geoms.get(i);
      const type = Number(g.type), dataid = Number(g.dataid);
      const objtype = Number(g.objtype), objid = Number(g.objid);
      const size = Array.from(g.size).map(Number);
      const rgba = Array.from(g.rgba);
      const mat9 = Array.from(g.mat);
      const pos = Array.from(g.pos);
      g.delete();

      // WORKAROUND: this @mujoco/mujoco WASM build returns a corrupted
      // mjvGeom.dataid for mesh geoms (it is 2x the real mesh id), which made
      // every link render with a mismatched STL and the robot looked exploded.
      // Recover the true mesh id from the geom's objid instead.
      let meshId = dataid;
      if (type === mujoco.mjtGeom.mjGEOM_MESH.value &&
          objtype === OBJ.mjOBJ_GEOM.value && objid >= 0 && objid < model.ngeom) {
        meshId = Number(model.geom_dataid[objid]);
      }

      let mesh = meshes[i];
      const glove = objtype === OBJ.mjOBJ_GEOM.value ? gloveGeoms.get(objid) : null;
      const cacheKey = JSON.stringify([type, size, meshId, glove ? objid : null]);
      if (!mesh || mesh.userData.key !== cacheKey) {
        if (mesh) group.remove(mesh);
        const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
        const geo = glove ? null : getGeometry(mjvGeomShim(type, meshId, size));
        mesh = glove ? createBoxingGlove(color, glove.hand, glove.wristX) : new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          transparent: rgba[3] < 1, opacity: rgba[3],
          shininess: 24, specular: new THREE.Color(0x222222),
        }));
        mesh.castShadow = type !== mujoco.mjtGeom.mjGEOM_PLANE.value;
        mesh.receiveShadow = true;
        mesh.userData.key = cacheKey;
        meshes[i] = mesh;
        group.add(mesh);
      } else if (!glove) {
        mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
        mesh.material.opacity = rgba[3];
        mesh.material.transparent = rgba[3] < 1;
      }
      mesh.matrixAutoUpdate = false;
      mesh.matrix.set(
        mat9[0], mat9[1], mat9[2], pos[0],
        mat9[3], mat9[4], mat9[5], pos[1],
        mat9[6], mat9[7], mat9[8], pos[2],
        0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
      mesh.visible = !(type === mujoco.mjtGeom.mjGEOM_MESH.value && coveredHandMeshes.has(meshId));
    }
  } finally {
    geoms.delete();
  }
  for (let i = n; i < meshes.length; i++) if (meshes[i]) meshes[i].visible = false;
}

// plain-object shim so geometry creation can happen after the mjvGeom handle is freed
function mjvGeomShim(type, dataid, size) {
  return { type, dataid, size };
}

// ------------------------------------------------------------------ UI wiring
const ui = {
  ptsA: $('ptsA'), ptsB: $('ptsB'), hitsA: $('hitsA'), hitsB: $('hitsB'),
  hpA: $('hpA'), hpB: $('hpB'), time: $('time'), round: $('round'),
  banner: $('banner'), flash: $('flash'), stats: $('stats'),
};
let roundNo = 1;
let paused = false;
let simSpeed = 1;

function banner(html, ms = 1400) {
  ui.banner.innerHTML = html;
  ui.banner.classList.add('show');
  clearTimeout(banner._t);
  banner._t = setTimeout(() => ui.banner.classList.remove('show'), ms);
  recSnapshotBanner(ms); // 视频录制快照（低频调用；合成器按快照绘制横幅大字）
}

function worldToScreen(x, y, z) {
  const v = new THREE.Vector3(x, y, z).project(camera);
  return [(v.x * 0.5 + 0.5) * innerWidth, (-v.y * 0.5 + 0.5) * innerHeight, v.z < 1];
}

function floatText(text, color, wx, wy, wz) {
  const [px, py, vis] = worldToScreen(wx, wy, wz);
  if (!vis) return;
  const el = document.createElement('div');
  el.className = 'float';
  el.textContent = text;
  el.style.cssText = `left:${px - 40}px;top:${py - 20}px;color:${color};`;
  $('hud').appendChild(el);
  setTimeout(() => el.remove(), 1000);
  recFloats.push({ text, color, x: px - 40, y: py - 20, t0: performance.now() }); // 视频录制镜像（见下方"视频录制"节）
}

function flash(side) {
  const f = ui.flash;
  f.className = side === 'A' ? 'left' : 'right';
  void f.offsetWidth;
  f.classList.add('pop');
}

function refreshScore() {
  const A = ctl.fighters.A, B = ctl.fighters.B;
  ui.ptsA.textContent = A.scores.points;
  ui.ptsB.textContent = B.scores.points;
  ui.hitsA.textContent = A.scores.hits;
  ui.hitsB.textContent = B.scores.hits;
  ui.hpA.style.width = Math.max(0, 100 - ctl.damage.A * 20) + '%';
  ui.hpB.style.width = Math.max(0, 100 - ctl.damage.B * 20) + '%';
}

function handleEvents() {
  for (const ev of ctl.events.splice(0)) {
    if (ev.type === 'hit') {
      const color = ev.attacker === 'A' ? '#ff6b6b' : '#7b9bff';
      const label = ev.points === 2 ? `+2 击中头部！` : `+1 击中躯干`;
      const b = ev.victim === 'A' ? ctl.ids.headA : ctl.ids.headB;
      floatText(label, color, data.site_xpos[3 * b], data.site_xpos[3 * b + 1], data.site_xpos[3 * b + 2]);
      flash(ev.attacker);
    } else if (ev.type === 'block') {
      floatText('格挡！', '#ffd166',
        (data.site_xpos[3 * ctl.ids.fistLA] + data.site_xpos[3 * ctl.ids.fistRB]) / 2,
        (data.site_xpos[3 * ctl.ids.fistLA + 1] + data.site_xpos[3 * ctl.ids.fistRB + 1]) / 2,
        (data.site_xpos[3 * ctl.ids.fistLA + 2] + data.site_xpos[3 * ctl.ids.fistRB + 2]) / 2);
    } else if (ev.type === 'ko') {
      const w = ev.winner === 'A' ? '红方 G1-A' : '蓝方 G1-B';
      const c = ev.winner === 'A' ? 'var(--red)' : 'var(--blue)';
      banner(`<span style="color:${c}">K O !</span><span class="small">${w} 获胜</span>`, 2600);
    } else if (ev.type === 'round') {
      roundNo++;
      ui.round.textContent = roundNo;
      banner(`ROUND ${roundNo}`, 1200);
    }
  }
  refreshScore();
}

$('pauseBtn').onclick = e => {
  paused = !paused;
  e.target.textContent = paused ? '继续' : '暂停';
  e.target.classList.toggle('on', paused);
};
$('resetBtn').onclick = () => {
  roundNo = 1; ui.round.textContent = 1;
  ctl.fighters.A.scores = { hits: 0, points: 0 };
  ctl.fighters.B.scores = { hits: 0, points: 0 };
  ctl.resetRound();
  banner('ROUND 1', 1000);
};
$('assistBtn').onclick = e => {
  ctl.assist = !ctl.assist;
  e.target.textContent = ctl.assist ? '平衡辅助 开' : '平衡辅助 关';
  e.target.classList.toggle('on', ctl.assist);
};
$('contactBtn').onclick = e => {
  const idx = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT.value;
  mjvOption.flags[idx] = mjvOption.flags[idx] ? 0 : 1;
  e.target.textContent = mjvOption.flags[idx] ? '接触点 开' : '接触点 关';
  e.target.classList.toggle('on', !!mjvOption.flags[idx]);
};
$('orbitBtn').onclick = e => setOrbit(!controls.autoRotate);
function setOrbit(on) {
  controls.autoRotate = on;
  const b = $('orbitBtn');
  b.textContent = on ? '镜头环绕 开' : '镜头环绕 关';
  b.classList.toggle('on', on);
}
// 固定评估视角（方案 §5：人工验收使用同一视角）——所有录像/对比用这个机位
$('evalCamBtn').onclick = () => {
  camera.position.set(3.1, -2.4, 1.7);
  controls.target.set(0, 0, 0.95);
  setOrbit(false);
};
$('aggr').oninput = e => { ctl.aggression = Number(e.target.value); };

// ------------------------------------------------------------------ 拳套轨迹
// §5 持续记录辅助指标的可视化：双机四只拳套的最近轨迹，用于人工核对出拳
// 目标指向与收拳路径。轨迹点按帧采样，长度约 1.5s（90 帧 @60fps）。
const TRAIL_N = 90;
const fistTrailSites = [
  { id: ctl.ids.fistLA, color: 0xff8a8a },
  { id: ctl.ids.fistRA, color: 0xff4d4d },
  { id: ctl.ids.fistLB, color: 0x8ab0ff },
  { id: ctl.ids.fistRB, color: 0x4d7dff },
].map(({ id, color }) => {
  const pos = new Float32Array(3 * TRAIL_N);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setDrawRange(0, 0);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.65 }));
  line.visible = false;
  line.frustumCulled = false;
  scene.add(line);
  return { id, pos, geo, line, n: 0 };
});
let trailsOn = false;
$('trailBtn').onclick = e => {
  trailsOn = !trailsOn;
  e.target.textContent = trailsOn ? '拳套轨迹 开' : '拳套轨迹 关';
  e.target.classList.toggle('on', trailsOn);
  for (const t of fistTrailSites) { t.n = 0; t.geo.setDrawRange(0, 0); t.line.visible = trailsOn; }
};
function updateFistTrails() {
  if (!trailsOn) return;
  for (const t of fistTrailSites) {
    const i = 3 * t.id;
    if (t.n < TRAIL_N) {
      t.pos.set([data.site_xpos[i], data.site_xpos[i + 1], data.site_xpos[i + 2]], 3 * t.n);
      t.n++;
    } else {
      t.pos.copyWithin(0, 3);   // shift left by one point
      t.pos[3 * (TRAIL_N - 1)] = data.site_xpos[i];
      t.pos[3 * (TRAIL_N - 1) + 1] = data.site_xpos[i + 1];
      t.pos[3 * (TRAIL_N - 1) + 2] = data.site_xpos[i + 2];
    }
    t.geo.setDrawRange(0, t.n);
    t.geo.attributes.position.needsUpdate = true;
  }
}

// ------------------------------------------------------------------ tactics policy
// Learned self-play tactics (optional file vendor/tactics/tactics.json): when
// present it replaces the weighted-random combo picker / distance thresholds.
// 404-tolerant — without it the old behaviour stays.
let tacticsPolicy = null;
let tacticsOn = false;
fetch('./vendor/tactics/tactics.json').then(r => {
  if (!r.ok) throw new Error('tactics.json HTTP ' + r.status);
  return r.json();
}).then(json => {
  tacticsPolicy = TacticsPolicy.fromJSON(json);
  tacticsOn = true;
  ctl.setTactics(tacticsPolicy);
  const btn = $('tacticsBtn');
  btn.disabled = false;
  btn.textContent = '战术 AI 开';
  btn.classList.add('on');
  console.log(`tactics policy loaded (${json.trainedEpisodes ?? '?'} training episodes)`);
}).catch(err => {
  const btn = $('tacticsBtn');
  if (btn) btn.remove();
  console.log('tactics weights not available — weighted-random picker stays', err.message);
});
$('tacticsBtn')?.addEventListener('click', e => {
  if (!tacticsPolicy) return;
  tacticsOn = !tacticsOn;
  if (tacticsOn) ctl.setTactics(tacticsPolicy);
  else ctl.clearTactics();
  e.target.textContent = tacticsOn ? '战术 AI 开' : '战术 AI 关';
  e.target.classList.toggle('on', tacticsOn);
});

// ------------------------------------------------------------------ AMO policy mode
// Per-robot toggle. RL is enabled by default at boot (both robots attach the
// policy before the first frame — see the load sequence above); the buttons
// below can switch either robot back to the scripted fallback or re-enable it.
const rlState = {
  A: !!(ctl.fighters.A.amo || ctl.fighters.A.tracking),
  B: !!(ctl.fighters.B.amo || ctl.fighters.B.tracking),
};

function rlLabel(side) {
  const f = ctl.fighters[side];
  const name = f.tracking ? `追踪:${f.clip}` : (f.amo ? 'AMO 策略' : '脚本');
  return `${side}: ${name}`;
}

async function toggleAMO(side) {
  const btn = $(side === 'A' ? 'rlABtn' : 'rlBBtn');
  btn.disabled = true;
  try {
    if (TRACKING) {
      // tracking scene: toggle the whole clip stack per robot
      if (rlState[side]) {
        rlState[side] = false;
        ctl.clearTracking(side);
      } else {
        banner('<span class="small">正在加载拳击片段策略…</span>', 8000);
        // fight 生效（运行时 fightActive，非编译期 FIGHT_MODE）时重开挂单策略
        // fight 槽；回退后走片段栈——与启动加载同一运行时语义（P1-1）
        const nets = fightActive ? { fight: await loadFightNet() } : await loadTrackingNets();
        ctl.setTracking(side, nets);
        rlState[side] = true;
      }
    } else if (rlState[side]) {
      rlState[side] = false;
      ctl.clearAMO(side);
    } else {
      banner('<span class="small">正在加载 AMO 策略权重…</span>', 8000);
      const net = await loadAMONetwork();
      ctl.setAMO(side, net);
      rlState[side] = true;
    }
  } catch (err) {
    console.error('failed to toggle policy', err);
    banner('<span class="small">策略加载失败，请重试</span>', 2500);
    rlState[side] = false;
    ctl.clearAMO(side);
    ctl.clearTracking(side);
  }
  btn.disabled = false;
  btn.textContent = rlLabel(side);
  btn.classList.toggle('on', rlState[side]);
}

$('rlABtn').onclick = () => toggleAMO('A');
$('rlBBtn').onclick = () => toggleAMO('B');
// reflect the default-on state in the buttons
for (const side of ['A', 'B']) {
  const btn = $(side === 'A' ? 'rlABtn' : 'rlBBtn');
  btn.textContent = rlLabel(side);
  btn.classList.toggle('on', rlState[side]);
}
document.querySelectorAll('[data-speed]').forEach(b => {
  b.onclick = () => {
    simSpeed = Number(b.dataset.speed);
    document.querySelectorAll('[data-speed]').forEach(x => x.classList.toggle('on', x === b));
  };
});

// ------------------------------------------------------------------ 模式中心 & 新手引导（纯 UI）
// 模式中心：当前模式卡片高亮，其余卡片点击整页跳转。四种玩法对应不同场景
// 模型与策略栈，切换必然重新加载页面（卡片 title 已注明）。currentMode 与
// 图例 modeLine 同一判定源：fight 编译期常量只反映 meta 探测，运行时以
// fightActive 为准（权重损坏回退后按追踪/守卫高亮，与 modeLine 一致）。
const MODE_INFO = {
  fight: '真实对打 —— 两台 G1 共享对打策略自主攻防，含 KO 与回合',
  combo: '组合拳表演 —— 双机循环演练 35 秒连续组合拳，纯观赏不记 KO',
  guard: '守卫练习 —— 双机各自循环基础拳法片段，适合看清动作细节',
  amo: 'AMO 自由模式 —— 23 DoF 全身 RL 策略自由行走的开放场景',
};
const currentMode = FIGHT_MODE && fightActive ? 'fight' : COMBO_MODE ? 'combo' : TRACKING ? 'guard' : 'amo';
const modeGrid = $('modeGrid');
if (modeGrid) {
  for (const card of modeGrid.querySelectorAll('.mode-card')) {
    if (card.dataset.mode === currentMode) card.classList.add('current');
    else card.addEventListener('click', () => { location.href = card.dataset.url; });
  }
}
// sceneBtn（保留原 id）：旧「追踪模式 开/关」URL 开关与模式中心完全重复，
// 改为「更多模式」——滚动到模式中心并高亮一闪，把视线引向模式卡片。
const sceneBtn = $('sceneBtn');
if (sceneBtn && modeGrid) {
  sceneBtn.onclick = () => {
    modeGrid.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    modeGrid.classList.remove('flash');
    void modeGrid.offsetWidth;
    modeGrid.classList.add('flash');
  };
}
// 首次引导浮层：localStorage 无标记时在加载完成后弹出，「开始观看」关闭并
// 写标记（隐私模式等 localStorage 异常一律静默降级：只是每次首屏都弹）。
// 「？」按钮随时重看——重看不写标记，下次首访仍会弹。
const GUIDE_KEY = 'g1boxing_seen_guide_v1';
const guideOverlay = $('guideOverlay');
if (guideOverlay) {
  const guideMode = $('guideMode');
  if (guideMode) guideMode.textContent = '当前模式：' + MODE_INFO[currentMode];
  let guideFirstTime = false;
  try { guideFirstTime = !localStorage.getItem(GUIDE_KEY); } catch { /* localStorage 不可用，按首访处理 */ }
  if (guideFirstTime) guideOverlay.classList.remove('hidden');
  $('guideCloseBtn').onclick = () => {
    guideOverlay.classList.add('hidden');
    if (guideFirstTime) {
      try { localStorage.setItem(GUIDE_KEY, '1'); } catch { /* 写不进则下次首访仍弹 */ }
      guideFirstTime = false;
    }
  };
  $('helpBtn').onclick = () => guideOverlay.classList.remove('hidden');
}
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ------------------------------------------------------------------ 视频录制
// 一键录制"所看到的"画面：渲染循环每帧把 three.js 画布与轻量 HUD（比分/命中、
// 时间/回合、横幅大字、命中浮字）合成到一张离屏 2D canvas，再
// captureStream(60) → MediaRecorder 编码保存。零外部依赖。
//
// 帧挂钩（侵入最小的方案）：frame() 内 renderer.render() 之后同步调用
// recCompose()——同一任务内 WebGL 绘制缓冲仍有效（preserveDrawingBuffer=false
// 下跨任务 drawImage 会读到已清空的缓冲），watchdog 兜底帧同样被覆盖；未录制
// 时 recCompose() 只是一次布尔短路。模式切换/刷新是整页跳转，录制随页面卸载
// 自然中断，无需挂钩模式卡片。
const REC_MIMES = [
  'video/mp4;codecs="avc1.640028"',   // H.264 High 档，同码率画质最好，探测失败自动回退
  'video/mp4;codecs="avc1.42E01E"',
  'video/mp4',
  'video/webm;codecs=h264',
  'video/webm;codecs=vp9',
  'video/webm',
];
const recMime = typeof MediaRecorder !== 'undefined'
  ? REC_MIMES.find(m => MediaRecorder.isTypeSupported(m)) || null
  : null;
const recCssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const REC_COLOR_BG = recCssVar('--bg') || '#0b0e14';
const REC_COLOR_RED = recCssVar('--red') || '#ff4d4d';
const REC_COLOR_BLUE = recCssVar('--blue') || '#4d7dff';
// 横幅行内色可能是 CSS 变量（canvas 不认识 var(...)），换算成具体色值
const recResolveColor = c => {
  if (!c) return null;
  const m = /^var\(--([a-z]+)\)/.exec(c);
  if (!m) return c;
  return { red: REC_COLOR_RED, blue: REC_COLOR_BLUE }[m[1]] || null;
};

let recState = 'idle';   // idle | recording | stopping
let recorder = null;
let recChunks = [];
let recLastBlob = null;
let recStartT = 0;       // performance.now() 起点（停止后 banner 报告录制时长用）
let recTick = null;      // 录制中按钮刷新 interval
let recFonts = null;     // 按 start 时缩放预生成的 font 串（避免每帧构造/measureText）
const recFloats = [];    // 命中浮字录制镜像 {text,color,x,y,t0}（floatText 推入）
let recBanner = null;    // 横幅快照 {lines:[{text,small,color}], t0, ms}

// 离屏合成画布：尺寸取录制开始时的渲染画布设备像素；录制中途窗口缩放只做
// 缩放绘制（流分辨率稳定，避免编码器中途改分辨率）。alpha:false 直接给编码器
// 不透明帧，也省去透明合成开销。
const compCanvas = document.createElement('canvas');
const compCtx = compCanvas.getContext('2d', { alpha: false });

// banner() 钩子：把当前横幅内容解析成行快照（banner() 低频调用，解析开销可忽略）。
// <br> 与 .small（display:block）都换行；行内 style 的 color 解析成画布可用色。
function recSnapshotBanner(ms) {
  const lines = [];
  let cur = null;
  const flush = () => { if (cur && cur.text.trim()) lines.push(cur); cur = null; };
  const walk = node => {
    for (const child of node.childNodes) {
      if (child.nodeType === 3) {                     // 文本节点
        if (!cur) cur = { text: '', small: false, color: null };
        cur.text += child.textContent;
      } else if (child.nodeType === 1) {              // 元素节点
        if (child.tagName === 'BR') { flush(); continue; }
        const small = child.classList.contains('small');
        if (small && cur) flush();
        if (!cur) cur = { text: '', small, color: null };
        const style = child.getAttribute('style');
        if (style) {
          const cm = /color:\s*([^;]+)/.exec(style);
          if (cm && !cur.color) cur.color = recResolveColor(cm[1].trim());
        }
        walk(child);
      }
    }
  };
  walk(ui.banner);
  flush();
  recBanner = lines.length ? { lines, t0: performance.now(), ms } : null;
}

function buildRecFonts(k) {
  const fam = '"Segoe UI", "Microsoft YaHei", system-ui, sans-serif';
  return {
    name: `600 ${14 * k}px ${fam}`,
    pts: `700 ${30 * k}px ${fam}`,
    sub: `${11 * k}px ${fam}`,
    clock: `700 ${24 * k}px ${fam}`,
    banner: `800 ${54 * k}px ${fam}`,
    bannerSmall: `500 ${20 * k}px ${fam}`,
    float: `800 ${22 * k}px ${fam}`,
  };
}

// 记分板（与 DOM 同数据源：textContent + refreshScore 同式的 ctl.damage）
function drawRecScoreboard(ctx, k) {
  const boxW = 170 * k, boxH = 94 * k, clockW = 96 * k, gap = 2 * k, top = 14 * k;
  const x0 = (compCanvas.width - (boxW * 2 + clockW + gap * 2)) / 2;
  const box = (x, color, name, pts, hits, hpPct) => {
    ctx.fillStyle = 'rgba(13,17,27,.82)';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, top, boxW, boxH, [10 * k, 10 * k, 0, 0]);
    else ctx.rect(x, top, boxW, boxH);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.fillRect(x, top, boxW, 3 * k);
    const tx = x + 18 * k;
    ctx.textAlign = 'left';
    ctx.fillStyle = color;
    ctx.font = recFonts.name; ctx.fillText(name, tx, top + 24 * k);
    ctx.font = recFonts.pts;  ctx.fillText(pts, tx, top + 58 * k);
    ctx.fillStyle = '#8a93a8';
    ctx.font = recFonts.sub;  ctx.fillText('命中 ' + hits, tx, top + 77 * k);
    const barW = boxW - 36 * k;
    ctx.fillStyle = '#1d2333'; ctx.fillRect(tx, top + 82 * k, barW, 5 * k);
    ctx.fillStyle = color;     ctx.fillRect(tx, top + 82 * k, barW * hpPct / 100, 5 * k);
  };
  const clockX = x0 + boxW + gap;
  ctx.fillStyle = 'rgba(13,17,27,.82)';
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(clockX, top, clockW, boxH, 10 * k);
  else ctx.rect(clockX, top, clockW, boxH);
  ctx.fill();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#e8eaf0';
  ctx.font = recFonts.clock; ctx.fillText(ui.time.textContent, clockX + clockW / 2, top + 44 * k);
  ctx.fillStyle = '#8a93a8';
  ctx.font = recFonts.sub;   ctx.fillText('回合 ' + ui.round.textContent, clockX + clockW / 2, top + 70 * k);
  box(x0, REC_COLOR_RED, '红方 · G1-A', ui.ptsA.textContent, ui.hitsA.textContent,
    Math.max(0, 100 - ctl.damage.A * 20));
  box(clockX + clockW + gap, REC_COLOR_BLUE, 'G1-B · 蓝方', ui.ptsB.textContent, ui.hitsB.textContent,
    Math.max(0, 100 - ctl.damage.B * 20));
}

// 横幅大字：透明度直接读 #banner 的计算样式（与 CSS 过渡实时同步），位置按
// DOM 的 top:34% 居中；行快照来自 recSnapshotBanner。
function drawRecBanner(ctx, k, now) {
  if (!recBanner || recBanner.t0 + recBanner.ms + 200 < now) return;
  const alpha = Number(getComputedStyle(ui.banner).opacity);
  if (alpha < 0.02) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.textAlign = 'center';
  ctx.shadowColor = 'rgba(0,0,0,.7)';
  ctx.shadowBlur = 30 * k;
  if ('letterSpacing' in ctx) ctx.letterSpacing = (6 * k) + 'px';
  const cx = compCanvas.width / 2;
  const rows = recBanner.lines.map(l => ({ ...l, h: (l.small ? 26 : 60) * k }));
  const total = rows.reduce((s, r) => s + r.h, 0);
  let y = compCanvas.height * 0.34 - total / 2;
  for (const r of rows) {
    y += r.h * 0.7;
    ctx.font = r.small ? recFonts.bannerSmall : recFonts.banner;
    if ('letterSpacing' in ctx) ctx.letterSpacing = ((r.small ? 3 : 6) * k) + 'px';
    ctx.fillStyle = r.small ? '#cfd6e4' : (r.color || '#e8eaf0');
    ctx.fillText(r.text.trim(), cx, y);
    y += r.h * 0.3;
  }
  ctx.restore();
  if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
}

// 命中浮字：按 floatUp 动画复刻（15% 淡入回位，之后上浮 46px 淡出；DOM 的
// scale 微差省略），坐标为 worldToScreen 的 CSS px × k。
function drawRecFloats(ctx, k, now) {
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,.8)';
  ctx.shadowBlur = 8 * k;
  ctx.textAlign = 'left';
  for (let i = recFloats.length - 1; i >= 0; i--) {
    const f = recFloats[i];
    const age = (now - f.t0) / 1000;
    if (age >= 1) { recFloats.splice(i, 1); continue; }
    const p = age < 0.15 ? age / 0.15 : 1;
    const out = age < 0.15 ? 0 : (age - 0.15) / 0.85;
    ctx.globalAlpha = p * (1 - out);
    ctx.fillStyle = f.color;
    ctx.font = recFonts.float;
    ctx.fillText(f.text, f.x * k, (f.y + 19 + (1 - p) * 6 - out * 46) * k);
  }
  ctx.restore();
}

// 每帧合成（frame() 内 renderer.render() 之后调用；未录制时布尔短路）
function recCompose() {
  if (recState !== 'recording') return;
  try {
    const ctx = compCtx, W = compCanvas.width, H = compCanvas.height;
    ctx.globalAlpha = 1;
    ctx.fillStyle = REC_COLOR_BG;          // WebGL 画布透明区域在 DOM 下是 --bg
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(renderer.domElement, 0, 0, W, H);
    const k = W / innerWidth;
    drawRecScoreboard(ctx, k);
    drawRecBanner(ctx, k, performance.now());
    drawRecFloats(ctx, k, performance.now());
  } catch (err) {
    console.error('recCompose failed', err);   // 合成失败不影响仿真帧循环
  }
}

function recStamp() {
  const d = new Date(), p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function recUpdateBtn() {
  // 录制中按钮只显示红点（rec-on 红色脉动），不显示状态词/时长；
  // 时长计数由 recStartT 保留，停止后 recFinalize 的 banner 用它报告录制时长
  recBtn.textContent = '●';
}

// 只清录制侧状态与 UI；正常收尾走 onstop → recFinalize → recTeardown，
// 出错路径直接调用（先摘 handler 再停轨，不会误触发 recFinalize）。
function recTeardown() {
  if (recTick) { clearInterval(recTick); recTick = null; }
  if (recorder) {
    const stream = recorder.stream;
    recorder.ondataavailable = null; recorder.onerror = null; recorder.onstop = null;
    recorder = null;
    if (stream) for (const t of stream.getTracks()) t.stop();
  }
  recState = 'idle';
  recBtn.disabled = false;
  recBtn.classList.remove('rec-on');
  recBtn.textContent = '● 录制视频';
}

function recStart() {
  if (recState !== 'idle') return false;
  if (typeof MediaRecorder === 'undefined' || !recMime || !compCanvas.captureStream) {
    banner('<span style="color:var(--red)">当前浏览器不支持视频录制</span>', 3000);
    return false;
  }
  compCanvas.width = renderer.domElement.width;
  compCanvas.height = renderer.domElement.height;
  // 码率按画布高度自适应：≥1440 24Mbps / ≥1080 16Mbps / ≥720 10Mbps / 其余 6Mbps
  const bitrate = compCanvas.height >= 1440 ? 24e6 : compCanvas.height >= 1080 ? 16e6 : compCanvas.height >= 720 ? 10e6 : 6e6;
  recFonts = buildRecFonts(compCanvas.width / innerWidth);
  recChunks = [];
  recFloats.length = 0;
  recBanner = null;
  try {
    recorder = new MediaRecorder(compCanvas.captureStream(60), {
      mimeType: recMime, videoBitsPerSecond: bitrate,
    });
  } catch (err) {
    console.error('MediaRecorder 创建失败', err);
    banner('<span style="color:var(--red)">录制启动失败</span><span class="small">' +
      String(err && err.message || err) + '</span>', 4000);
    return false;
  }
  recorder.ondataavailable = e => { if (e.data && e.data.size) recChunks.push(e.data); };
  recorder.onerror = e => {
    console.error('MediaRecorder error', e.error || e);
    banner('<span style="color:var(--red)">录制出错，已终止</span><span class="small">' +
      String(e.error && e.error.message || '编码器异常') + '</span>', 4000);
    recTeardown();
  };
  recorder.onstop = recFinalize;
  recorder.start(1000);   // 每秒一个 chunk，异常中断也能保住已录内容
  recStartT = performance.now();
  recState = 'recording';
  recBtn.classList.add('rec-on');
  recUpdateBtn();
  recTick = setInterval(recUpdateBtn, 500);
  return true;
}

function recStop() {
  if (recState !== 'recording') return false;
  recState = 'stopping';
  recBtn.disabled = true;
  try {
    recorder.stop();          // onstop → recFinalize 收尾
  } catch (err) {
    console.error('recorder.stop failed', err);
    recFinalize();
  }
  return true;
}

function recFinalize() {
  const durMs = Math.max(0, performance.now() - recStartT);
  recTeardown();
  try {
    const blob = new Blob(recChunks, { type: recMime });
    recChunks = [];
    recLastBlob = blob;
    const ext = recMime.startsWith('video/mp4') ? 'mp4' : 'webm';
    const name = `g1-boxing-${recStamp()}.${ext}`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);   // 留缓冲，防下载未起即吊销
    const webmNote = ext === 'webm'
      ? '<span class="small">当前浏览器不支持 MP4 直录，已保存 WebM</span>' : '';
    banner(`已保存 ${name}（${(blob.size / 1048576).toFixed(1)} MB · ${(durMs / 1000).toFixed(1)} 秒）${webmNote}`, 4200);
  } catch (err) {
    console.error('录制文件生成失败', err);
    banner('<span style="color:var(--red)">录制文件生成失败</span>', 3000);
  }
}

const recBtn = $('recBtn');
if (recBtn) {
  recBtn.onclick = () => {
    if (recState === 'recording') recStop();
    else if (recState === 'idle') recStart();
  };  // stopping 期间的点击忽略
}

// 调试句柄（仿 window.__box）：供无 DOM 自动化验收——start/stop/state()/mime/lastBlob
window.__rec = {
  start: () => (recState === 'idle' ? recStart() : false),
  stop: () => (recState === 'recording' ? recStop() : false),
  state: () => recState,
  mime: recMime,
  get lastBlob() { return recLastBlob; },
};

// ------------------------------------------------------------------ main loop
let last = performance.now();
let acc = 0;
let fps = 0, fpsN = 0, fpsT = 0;
let physMs = 0;

function frame(now) {
  requestAnimationFrame(frame);
  lastFrameAt = performance.now();
  try {
    let dtWall = Math.min(0.05, (now - last) / 1000);
    last = now;

    if (!paused) {
      acc += dtWall * simSpeed;
      let steps = 0;
      const t0 = performance.now();
      while (acc >= DT && steps < 60) {
        ctl.update(DT);
        mujoco.mj_step(model, data);
        acc -= DT;
        steps++;
      }
      physMs = performance.now() - t0;
    }

    syncRenderer();
    controls.update();
    updateFistTrails();
    handleEvents();
    renderer.render(scene, camera);
    recCompose();   // 视频录制合成挂钩（帧末；未录制时是一次布尔短路）

    fpsN++; fpsT += dtWall;
    if (fpsT >= 0.5) {
      fps = Math.round(fpsN / fpsT); fpsN = 0; fpsT = 0;
      const minZ = Math.min(data.xpos[3 * ctl.ids.pelvisA + 2], data.xpos[3 * ctl.ids.pelvisB + 2]);
      const clipInfo = TRACKING
        ? ` · A[${ctl.fighters.A.clip}@${ctl.fighters.A.tracking ? ctl.fighters.A.tracking.timeStep : '-'}] B[${ctl.fighters.B.clip}@${ctl.fighters.B.tracking ? ctl.fighters.B.tracking.timeStep : '-'}]<br>`
        : '<br>';
      // §4.6 辅助用量与 §4.5 命中分类计数：验收报告直接引用这里的数字。
      // 追踪模式辅助默认关断（trackingAssist，策略无外力训练）——用量为 0。
      const au = ctl.assistUsage;
      const auSum = u => (u.fx + u.fy + u.fz + u.tx + u.ty + u.tz);
      const assistNote = TRACKING ? (ctl.trackingAssist ? '追踪辅助开' : '追踪辅助关断') : (ctl.assist ? '开' : '关');
      ui.stats.innerHTML =
        `FPS ${fps} · 物理 ${physMs.toFixed(1)}ms/帧 · sim t=${data.time.toFixed(1)}s${clipInfo}` +
        `接触 ${data.ncon} · 最低骨盆 ${minZ.toFixed(2)}m · A伤害 ${ctl.damage.A.toFixed(1)} / B伤害 ${ctl.damage.B.toFixed(1)}<br>` +
        `辅助用量 |F|+|τ|: A ${auSum(au.A).toFixed(0)} B ${auSum(au.B).toFixed(0)} ` +
        `(${assistNote}) · 命中 ${ctl.fighters.A.scores.hits}/${ctl.fighters.B.scores.hits}`;
      // keep the per-side buttons in sync with the running clip
      if (TRACKING) {
        for (const side of ['A', 'B']) {
          const btn = $(side === 'A' ? 'rlABtn' : 'rlBBtn');
          const on = !!(ctl.fighters[side].tracking || ctl.fighters[side].amo);
          btn.textContent = rlLabel(side);
          btn.classList.toggle('on', on);
        }
      }
    }
    const t = data.time;
    ui.time.textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  } catch (e) {
    window.__frameErr = (e && e.stack) || String(e);
    console.error(e);
    document.getElementById('stats').innerHTML =
      '<span style="color:#ff6b6b">帧循环错误: ' + String(e && e.message || e) + '</span>';
  }
}

// 图例首行标明当前模式（对打/追踪/AMO），便于确认加载的是哪套控制
{
  const ml = document.getElementById('modeLine');
  if (ml) ml.textContent = TRACKING
    ? (fightActive
      ? `对打模式 · 168 维观测共享策略（${FIGHT_BIN}，opponent_state 14 维）· KO/60s 回合/记分启用`
      : `追踪模式 · 29 DoF 片段策略（guard/jab/cross/hook${COMBO_MODE ? ' + combo 组合循环' : ''}${FACEOFF_MODE ? ' · faceoff 面对面对抗' : ''}）`)
    : 'AMO 模式 · 23 DoF 全身策略';
}

refreshScore();
banner('ROUND 1<br><span class="small">Fight!</span>', 1600);
let lastFrameAt = 0;
requestAnimationFrame(frame);

// watchdog: if rAF stalls (hidden tab / throttled compositor), keep the loop
// alive with a timer so the simulation keeps running in the background.
setInterval(() => {
  if (performance.now() - lastFrameAt > 300) frame(performance.now());
}, 50);

// debug handle for automated testing
window.__box = {
  mujoco, model, data, ctl, pausedRef: () => paused,
  setPaused: v => { paused = v; },
  camera, controls, meshes, mjvScene, renderer, scene, sync: syncRenderer, THREE,
  geomWorlds: () => {
    const out = [];
    const n = mjvScene.geoms.size();
    for (let i = 0; i < n; i++) {
      const g = mjvScene.geoms.get(i);
      out.push({ type: Number(g.type), dataid: Number(g.dataid), pos: Array.from(g.pos).slice(0, 3) });
      g.delete();
    }
    return out;
  },
};
