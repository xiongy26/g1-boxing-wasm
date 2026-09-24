// Milestone B acceptance (docs/plan-mjlab-gpu-training.md): the JS
// TrackingNetwork forward pass must match the ONNX reference (~1e-6, the AMO
// alignment standard) on exported test vectors.
//
//   node tools/test_tracking.mjs vendor/policy/spinkick
//
// Exits 0 with PASS when actions and motion-table gathers align.
import fs from 'node:fs';
import { TrackingNetwork } from '../src/tracking_policy.mjs';

const prefix = process.argv[2] ?? 'vendor/policy/spinkick';
const meta = JSON.parse(fs.readFileSync(prefix + '_meta.json', 'utf8'));
const vectors = JSON.parse(fs.readFileSync(prefix + '_test_vectors.json', 'utf8'));
const bin = fs.readFileSync(prefix + '.bin');
const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);

const net = new TrackingNetwork(buffer, meta);
const nq = net.nq, nb = net.nBody;

let maxAct = 0, maxRef = 0, maxActMag = 0;
for (let v = 0; v < vectors.obs.length; v++) {
  const obs = Float32Array.from(vectors.obs[v]);
  const act = net.infer(obs);
  for (let i = 0; i < nq; i++) {
    maxAct = Math.max(maxAct, Math.abs(act[i] - vectors.actions[v][i]));
    maxActMag = Math.max(maxActMag, Math.abs(vectors.actions[v][i]));
  }
  const ref = net.refAt(vectors.time_step[v]);
  const rp = vectors.ref_joint_pos[v].flat(), rv = vectors.ref_joint_vel[v].flat();
  const bp = vectors.ref_body_pos_w[v].flat(), bq = vectors.ref_body_quat_w[v].flat();
  for (let i = 0; i < nq; i++) {
    maxRef = Math.max(maxRef, Math.abs(ref.joint_pos[i] - rp[i]));
    maxRef = Math.max(maxRef, Math.abs(ref.joint_vel[i] - rv[i]));
  }
  for (let i = 0; i < nb * 3; i++) maxRef = Math.max(maxRef, Math.abs(ref.body_pos_w[i] - bp[i]));
  for (let i = 0; i < nb * 4; i++) maxRef = Math.max(maxRef, Math.abs(ref.body_quat_w[i] - bq[i]));
}

// the JS dot product accumulates in float64 while ONNX Gemm stays in float32,
// so the residual is the reference's own rounding: hold it to 1e-6 relative
// (absolute floor 1e-6), same spirit as the AMO ~1e-7 standard
const actBar = Math.max(1e-6, 1e-6 * maxActMag);
const PASS = maxAct < actBar && maxRef < 1e-6;
console.log(`tracking forward alignment (${vectors.obs.length} vectors, ${meta.joint_names.length} joints):`);
console.log(`  max |action| = ${maxActMag.toFixed(3)}, max |action_js - action_onnx| = ${maxAct.toExponential(3)} (bar ${actBar.toExponential(1)})`);
console.log(`  max |motion_js - motion_onnx| = ${maxRef.toExponential(3)}`);
console.log(`RESULT: ${PASS ? 'PASS' : 'FAIL'}`);
process.exit(PASS ? 0 : 1);
