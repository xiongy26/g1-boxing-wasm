// G1 robot boxing — browser app: MuJoCo WASM physics + Three.js rendering.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BoxingController } from './boxing_ai.mjs';
import { AMONetwork } from './rl_policy.mjs';
import { TrackingNetwork } from './tracking_policy.mjs';
import { TacticsPolicy } from './tactics.mjs';

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

// stage-4 clip policies (boxing_{guard,jab,cross,hook}); present only after the
// GPU training pipeline has been run — 404-tolerant like tactics.json
const TRACKING_CLIPS = ['guard', 'jab', 'cross', 'hook'];
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
    ]).then(([buf, meta]) => [name, new TrackingNetwork(buf, meta)])))
      .then(entries => Object.fromEntries(entries))
      .catch(err => { trackingNetsPromise = null; throw err; });
  }
  return trackingNetsPromise;
}

setProgress(15, '正在下载场景与机器人网格…');
// scene selection: default is the AMO scene; `?scene=tracking` runs the
// stage-4 29-DoF tracking scene — but only if the trained clip policies are
// actually deployed; otherwise fall back to AMO with a note.
const TRACKING_REQUESTED = new URLSearchParams(location.search).get('scene') === 'tracking';
let trackingAvailable = false;
if (TRACKING_REQUESTED) {
  trackingAvailable = await fetch('./vendor/policy/boxing_guard_meta.json')
    .then(r => r.ok).catch(() => false);
  if (!trackingAvailable) {
    setProgress(16, '追踪片段策略尚未就绪，回退 AMO 模式…');
  }
}
const TRACKING = TRACKING_REQUESTED && trackingAvailable;
const SCENE_FILE = TRACKING ? './models/scene_boxing_tracking.xml' : './models/scene_boxing_amo.xml';
const xml = await fetch(SCENE_FILE).then(r => r.text());
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
if (TRACKING) {
  setProgress(94, '正在加载拳击片段策略…');
  try {
    const nets = await loadTrackingNets();
    ctl.setTracking('A', nets);
    ctl.setTracking('B', nets);
    console.log('tracking clip policies loaded:', Object.keys(nets).join(', '));
  } catch (err) {
    console.error('tracking policies failed to load at boot — scripted fallback', err);
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
      const cacheKey = JSON.stringify([type, size, meshId]);
      if (!mesh || mesh.userData.key !== cacheKey) {
        if (mesh) group.remove(mesh);
        const geo = getGeometry(mjvGeomShim(type, meshId, size));
        mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          transparent: rgba[3] < 1, opacity: rgba[3],
          shininess: 24, specular: new THREE.Color(0x222222),
        }));
        mesh.castShadow = type !== mujoco.mjtGeom.mjGEOM_PLANE.value;
        mesh.receiveShadow = true;
        mesh.userData.key = cacheKey;
        meshes[i] = mesh;
        group.add(mesh);
      } else {
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
      mesh.visible = true;
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
$('aggr').oninput = e => { ctl.aggression = Number(e.target.value); };

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
        const nets = await loadTrackingNets();
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
// stage-4 toggle: reload with the tracking scene
{
  const sceneBtn = $('sceneBtn');
  if (sceneBtn) {
    sceneBtn.textContent = TRACKING ? '追踪模式 开' : '追踪模式 关';
    sceneBtn.classList.toggle('on', TRACKING);
    sceneBtn.onclick = () => {
      const u = new URL(location);
      if (TRACKING) u.searchParams.delete('scene');
      else u.searchParams.set('scene', 'tracking');
      location.href = u;
    };
  }
}
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
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

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
    handleEvents();
    renderer.render(scene, camera);

    fpsN++; fpsT += dtWall;
    if (fpsT >= 0.5) {
      fps = Math.round(fpsN / fpsT); fpsN = 0; fpsT = 0;
      const minZ = Math.min(data.xpos[3 * ctl.ids.pelvisA + 2], data.xpos[3 * ctl.ids.pelvisB + 2]);
      const clipInfo = TRACKING
        ? ` · A[${ctl.fighters.A.clip}@${ctl.fighters.A.tracking ? ctl.fighters.A.tracking.timeStep : '-'}] B[${ctl.fighters.B.clip}@${ctl.fighters.B.tracking ? ctl.fighters.B.tracking.timeStep : '-'}]<br>`
        : '<br>';
      ui.stats.innerHTML =
        `FPS ${fps} · 物理 ${physMs.toFixed(1)}ms/帧 · sim t=${data.time.toFixed(1)}s${clipInfo}` +
        `接触 ${data.ncon} · 最低骨盆 ${minZ.toFixed(2)}m · A伤害 ${ctl.damage.A.toFixed(1)} / B伤害 ${ctl.damage.B.toFixed(1)}`;
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

// 图例首行标明当前模式（追踪/AMO），便于确认加载的是哪套控制
{
  const ml = document.getElementById('modeLine');
  if (ml) ml.textContent = TRACKING
    ? '追踪模式 · 29 DoF 片段策略（guard/jab/cross/hook）'
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
