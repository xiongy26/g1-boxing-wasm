// Build models/scene_boxing.xml: two Unitree G1 robots (mujoco_menagerie) merged
// into one arena, facing each other, with boxing gloves added at the wrists.
// Run:  node tools/build_scene.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const g1xml = fs.readFileSync(path.join(root, 'models/unitree_g1/g1.xml'), 'utf8');

// ---------------------------------------------------------------- parameters
const STANDOFF = Number(process.env.STANDOFF || 0.45); // half distance between pelvises (m), along x
const GUARD = {
  // arm joint targets for the boxing guard (same for both arms, mirrored roll)
  shoulder_pitch: -0.72, shoulder_roll: 0.10, shoulder_yaw: 0.00,
  elbow: 1.85, wrist_roll: 0.0, wrist_pitch: -0.35, wrist_yaw: 0.0,
};
const FIST_POS = '0.082 0.003 0'; // glove sphere center, wrist_yaw_link frame
const FIST_SIZE = '0.047';

// material colors per team: metal-colored links get the team tint
const COLORS = {
  A: { metal: '0.78 0.30 0.26 1', black: '0.42 0.15 0.13 1', glove: '0.88 0.16 0.12 1' },
  B: { metal: '0.28 0.44 0.86 1', black: '0.13 0.20 0.48 1', glove: '0.18 0.36 0.95 1' },
};
const MAT_RGBA = { black: '0.13 0.13 0.13 1', metal: '0.72 0.72 0.72 1' };

// ------------------------------------------------------------ transforms
function makeTeamCopy(team) {
  let s = g1xml;

  // drop the sensor and keyframe blocks (we write our own keyframe)
  s = s.replace(/  <sensor>[\s\S]*?<\/sensor>\n/, '');
  s = s.replace(/  <keyframe>[\s\S]*?<\/keyframe>\n/, '');

  // prefix all instance names (mesh/material names stay shared)
  s = s.replace(/<body name="/g, `<body name="${team}_`);
  s = s.replace(/<freejoint name="/g, `<freejoint name="${team}_`);
  s = s.replace(/<joint name="/g, `<joint name="${team}_`);
  s = s.replace(/<site name="/g, `<site name="${team}_`);
  s = s.replace(/<position class="g1" name="/g, `<position class="g1" name="${team}_`);

  // explicit geom colors instead of shared materials (per-team tint)
  s = s.replace(/ material="black"/g, ` rgba="${COLORS[team].black}"`);
  s = s.replace(/ material="metal"/g, ` rgba="${COLORS[team].metal}"`);

  // place & orient: A faces +x at -STANDOFF, B faces -x at +STANDOFF
  if (team === 'A') {
    s = s.replace(`<body name="${team}_pelvis" pos="0 0 0.793"`,
      `<body name="${team}_pelvis" pos="${-STANDOFF} 0 0.793"`);
  } else {
    s = s.replace(`<body name="${team}_pelvis" pos="0 0 0.793"`,
      `<body name="${team}_pelvis" pos="${STANDOFF} 0 0.793" quat="0 0 0 1"`);
  }

  // name the hit-classifiable collision geoms (head / torso)
  s = s.replace('class="collision" rgba="0.13 0.13 0.13 1" mesh="head_link"/>',
    `class="collision" rgba="0.13 0.13 0.13 1" name="${team}_head_hit" mesh="head_link"/>`);
  s = s.replace('class="collision" mesh="torso_link"/>',
    `class="collision" name="${team}_torso_hit" mesh="torso_link"/>`);
  s = s.replace('class="collision" rgba="0.13 0.13 0.13 1" mesh="logo_link"/>',
    `class="collision" rgba="0.13 0.13 0.13 1" name="${team}_torso_hit2" mesh="logo_link"/>`);

  // boxing gloves + fist/head sites
  for (const hand of ['left', 'right']) {
    const hy = hand === 'left' ? '0.003' : '-0.003';
    const bodyTag = `<body name="${team}_${hand}_wrist_yaw_link"`;
    const i = s.indexOf(bodyTag);
    if (i < 0) throw new Error(`wrist body not found for ${team} ${hand}`);
    // insert glove geoms right before the closing of that body's child list:
    // anchor on the rubber-hand visual geom line inside this body
    const anchor = s.indexOf(`mesh="${hand}_rubber_hand"/>`, i);
    if (anchor < 0) throw new Error(`rubber hand geom not found for ${team} ${hand}`);
    const insertAt = anchor + `mesh="${hand}_rubber_hand"/>`.length;
    const glove =
      `\n          <geom name="${team}_${hand}_fist" type="sphere" size="${FIST_SIZE}" pos="${FIST_POS}"` +
      ` rgba="${COLORS[team].glove}" mass="0.12" group="2" contype="1" conaffinity="1"` +
      ` friction="0.8 0.005 0.0001" priority="1"/>` +
      `\n          <site name="${team}_${hand}_fist_site" pos="${FIST_POS}" size="0.015" rgba="1 1 0 0.4"/>`;
    s = s.slice(0, insertAt) + glove + s.slice(insertAt);
  }
  // head site (for hit distance / diagnostics), torso frame — CRLF-safe insert
  s = s.replace(`<body name="${team}_left_shoulder_pitch_link"`,
    `<site name="${team}_head" pos="0.004 0 0.27" size="0.015"/>\n            ` +
    `<body name="${team}_left_shoulder_pitch_link"`);
  return s;
}

function extractRobot(team, s) {
  const start = s.indexOf(`<body name="${team}_pelvis"`);
  const end = s.indexOf('</worldbody>');
  if (start < 0 || end < 0) throw new Error('pelvis/worldbody not found');
  return s.slice(start, end).trimEnd(); // includes pelvis' closing </body>
}

function extractBlock(s, tag) {
  const events = [];
  const openRe = new RegExp(`<${tag}(?=[\\s>])`, 'g');
  const closeRe = new RegExp(`</${tag}>`, 'g');
  let m;
  while ((m = openRe.exec(s))) events.push([m.index, 1]);
  while ((m = closeRe.exec(s))) events.push([m.index, -1]);
  events.sort((a, b) => a[0] - b[0]);
  let depth = 0, start = -1;
  for (const [pos, kind] of events) {
    if (kind === 1) { if (depth === 0) start = pos; depth++; }
    else { depth--; if (depth === 0 && start >= 0) { const end = s.indexOf(`</${tag}>`, pos); return s.slice(start, end + tag.length + 3); } }
  }
  throw new Error('unbalanced block: ' + tag);
}

const copyA = makeTeamCopy('A');
const copyB = makeTeamCopy('B');
const robotA = extractRobot('A', copyA);
const robotB = extractRobot('B', copyB);
const defaults = extractBlock(g1xml, 'default');
const assets = extractBlock(g1xml, 'asset');

// actuators from the original xml with prefixed name= and joint=
function actuators(team) {
  const block = extractBlock(g1xml, 'actuator');
  return `  <!-- ${team} actuators -->\n` + block
    .replace(/<position class="g1" name="/g, `<position class="g1" name="${team}_`)
    .replace(/ joint="/g, ` joint="${team}_`)
    .replace('<actuator>', '').replace('</actuator>', '').trimEnd();
}
const actA = actuators('A');
const actB = actuators('B');

// ------------------------------------------------------------------ keyframe
// joint order == actuator order of g1.xml
const JOINTS = [];
for (const side of ['left', 'right']) {
  for (const j of ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']) JOINTS.push(`${side}_${j}_joint`);
}
for (const j of ['waist_yaw', 'waist_roll', 'waist_pitch']) JOINTS.push(`${j}_joint`);
for (const side of ['left', 'right']) {
  for (const j of ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']) JOINTS.push(`${side}_${j}_joint`);
}
const ARM = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
  'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
  'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'];

function ctrlForTeam(team) {
  const c = new Array(29).fill(0);
  for (let i = 0; i < 29; i++) {
    const jn = JOINTS[i];
    if (jn.endsWith('shoulder_pitch_joint')) c[i] = GUARD.shoulder_pitch;
    else if (jn.endsWith('shoulder_roll_joint')) c[i] = (jn.startsWith('left') ? -1 : 1) * GUARD.shoulder_roll;
    else if (jn.endsWith('shoulder_yaw_joint')) c[i] = (jn.startsWith('left') ? -1 : 1) * GUARD.shoulder_yaw;
    else if (jn.endsWith('elbow_joint')) c[i] = GUARD.elbow;
    else if (jn.endsWith('wrist_pitch_joint')) c[i] = GUARD.wrist_pitch;
    else if (jn.endsWith('wrist_roll_joint') || jn.endsWith('wrist_yaw_joint')) c[i] = 0;
  }
  return c;
}
function qposForTeam(team, x) {
  const yawQ = team === 'B' ? '0 0 0 1' : '1 0 0 0';
  const q = [x, '0', '0.793', ...yawQ.split(' ')];
  // arms already in the guard pose at spawn (legs/waist straight)
  for (const jn of JOINTS) {
    if (jn.endsWith('shoulder_pitch_joint')) q.push(String(GUARD.shoulder_pitch));
    else if (jn.endsWith('shoulder_roll_joint')) q.push(((jn.startsWith('left') ? -1 : 1) * GUARD.shoulder_roll).toFixed(4));
    else if (jn.endsWith('elbow_joint')) q.push(String(GUARD.elbow));
    else if (jn.endsWith('wrist_pitch_joint')) q.push(String(GUARD.wrist_pitch));
    else if (jn.endsWith('knee_joint')) q.push('0.20');
    else if (jn.endsWith('hip_pitch_joint')) q.push('0.10');
    else if (jn.endsWith('ankle_pitch_joint')) q.push('-0.10');
    else q.push('0');
  }
  return q.join(' ');
}
const ctrl = [...ctrlForTeam('A'), ...ctrlForTeam('B')].map(v => v.toFixed(4)).join(' ');
const qpos = `${qposForTeam('A', (-STANDOFF).toFixed(3))}  ${qposForTeam('B', STANDOFF.toFixed(3))}`;

const scene = `<mujoco model="g1_boxing_arena">
  <compiler angle="radian" meshdir="assets"/>
  <option timestep="0.002" integrator="implicitfast"/>

${defaults}

${assets}

  <worldbody>
    <light pos="0 0 3.2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.32 0.33 0.38 1" condim="3" friction="0.9 0.02 0.001"/>
${robotA}

${robotB}
  </worldbody>

  <actuator>
${actA}
${actB}
  </actuator>

  <keyframe>
    <key name="ready" qpos="${qpos}" ctrl="${ctrl}"/>
  </keyframe>
</mujoco>
`;

fs.writeFileSync(path.join(root, 'models/scene_boxing.xml'), scene);
console.log('wrote models/scene_boxing.xml');
console.log('qpos len:', qpos.split(' ').length, 'ctrl len:', ctrl.split(' ').length);
console.log('gloves:', (scene.match(/_fist"/g) || []).length, 'actuators:', (scene.match(/<position/g) || []).length);
