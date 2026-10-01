// TEMP diagnostic: render an ASCII side-view silhouette from the REAL mjv geoms
// (exactly the data src/main.js feeds to three.js) so we can see if the robot is
// assembled or exploded without a GPU.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const loadMujoco = (await import('../vendor/mujoco/mujoco.js')).default;
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
const assetsDir = path.join(root, 'models/unitree_g1/assets');
for (const f of fs.readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl'))
    vfs.addBuffer('assets/' + f, new Uint8Array(fs.readFileSync(path.join(assetsDir, f))));
}
const xml = fs.readFileSync(path.join(root, 'models/scene_boxing.xml'), 'utf8');
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

const MESH = mujoco.mjtGeom.mjGEOM_MESH.value;
// ---- mesh local bboxes & geom sizes -----------------------------------------
console.log('--- mesh local bbox (metres) ---');
for (let m = 0; m < model.nmesh; m++) {
  const va = model.mesh_vertadr[m], vn = model.mesh_vertnum[m];
  let lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
  for (let i = 0; i < vn; i++)
    for (let k = 0; k < 3; k++) {
      const v = model.mesh_vert[3 * (va + i) + k];
      lo[k] = Math.min(lo[k], v); hi[k] = Math.max(hi[k], v);
    }
  const mp = model.mesh(m); const nm = mp.name; mp.delete?.();
  const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]].map(v => v.toFixed(3)).join(' x ');
  if (m < 6 || m >= model.nmesh - 3)
    console.log(`  mesh ${String(m).padStart(2)} ${nm.padEnd(26)} verts=${String(vn).padStart(6)} extent=${ext}`);
}
console.log('--- mesh geom scale (geom_size) samples ---');
let shown = 0;
for (let g = 0; g < model.ngeom && shown < 8; g++) {
  if (Number(model.geom_type[g]) !== MESH) continue;
  console.log(`  geom ${g} body=${model.geom_bodyid[g]} size=[${model.geom_size[3 * g].toFixed(4)},` +
    `${model.geom_size[3 * g + 1].toFixed(4)},${model.geom_size[3 * g + 2].toFixed(4)}] dataid=${model.geom_dataid[g]}`);
  shown++;
}

// ---- ASCII silhouette --------------------------------------------------------
const scene = new mujoco.MjvScene(model, 20000);
const opt = new mujoco.MjvOption();
const pert = new mujoco.MjvPerturb();
const cam = new mujoco.MjvCamera();
mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene);
const n = scene.geoms.size();
const pts = [];
const W = 76, H = 44;
for (let i = 0; i < n; i++) {
  const g = scene.geoms.get(i);
  const type = Number(g.type), did = Number(g.dataid);
  const p = [Number(g.pos[0]), Number(g.pos[1]), Number(g.pos[2])];
  const mat = Array.from(g.mat).map(Number);
  g.delete();
  if (type !== MESH) continue;
  const va = model.mesh_vertadr[did], vn = model.mesh_vertnum[did];
  const step = Math.max(1, Math.floor(vn / 250));
  for (let v = 0; v < vn; v += step) {
    const lx = model.mesh_vert[3 * (va + v)], ly = model.mesh_vert[3 * (va + v) + 1], lz = model.mesh_vert[3 * (va + v) + 2];
    pts.push([
      mat[0] * lx + mat[1] * ly + mat[2] * lz + p[0],
      mat[3] * lx + mat[4] * ly + mat[5] * lz + p[1],
      mat[6] * lx + mat[7] * ly + mat[8] * lz + p[2],
    ]);
  }
}
let lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
for (const q of pts) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
console.log(`\n--- silhouette (side view: horizontal=y  vertical=z) ---`);
console.log(`verts=${pts.length} bbox x[${lo[0].toFixed(2)},${hi[0].toFixed(2)}] y[${lo[1].toFixed(2)},${hi[1].toFixed(2)}] z[${lo[2].toFixed(2)},${hi[2].toFixed(2)}]`);
const ylo = lo[1] - 0.05, yhi = hi[1] + 0.05, zlo = lo[2] - 0.05, zhi = hi[2] + 0.05;
const grid = Array.from({ length: H }, () => new Array(W).fill(' '));
for (const q of pts) {
  const cx = Math.round((q[1] - ylo) / (yhi - ylo) * (W - 1));
  const cy = Math.round((1 - (q[2] - zlo) / (zhi - zlo)) * (H - 1));
  if (cx >= 0 && cx < W && cy >= 0 && cy < H) grid[cy][cx] = '#';
}
console.log('+' + '-'.repeat(W) + '+');
for (let r = 0; r < H; r++) console.log('|' + grid[r].join('') + '|');
console.log('+' + '-'.repeat(W) + '+');
console.log('z-axis: top=' + zhi.toFixed(2) + 'm  bottom=' + zlo.toFixed(2) + 'm,  y-axis: left=' + ylo.toFixed(2) + ' right=' + yhi.toFixed(2));
