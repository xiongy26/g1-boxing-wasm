import * as THREE from 'three';

// Render-only glove: local +X points from the wrist toward the knuckles.
// Keeping this separate from the MuJoCo fist preserves policy/contact dynamics.
const paddingGeometry = new THREE.SphereGeometry(1, 40, 28);
const cuffGeometry = new THREE.CylinderGeometry(0.038, 0.036, 0.044, 40);
cuffGeometry.rotateZ(Math.PI / 2);
const pipingGeometry = new THREE.TorusGeometry(0.037, 0.002, 8, 40);
pipingGeometry.rotateY(Math.PI / 2);

export function createBoxingGlove(color, hand, wristX) {
  const glove = new THREE.Group();
  const leather = new THREE.MeshStandardMaterial({ color, roughness: 0.48, metalness: 0 });
  const palm = leather.clone();
  palm.color.multiplyScalar(0.76);
  const trim = new THREE.MeshStandardMaterial({ color: 0x18202b, roughness: 0.7 });
  const thread = new THREE.MeshStandardMaterial({ color: 0xe6e0d5, roughness: 0.8 });
  const thumbSide = hand === 'left' ? -1 : 1;

  function add(geometry, material, x, y, z, scale) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(wristX + x, y, z);
    if (scale) mesh.scale.set(...scale);
    mesh.castShadow = mesh.receiveShadow = true;
    glove.add(mesh);
    return mesh;
  }

  // Broad padded knuckles, a tapered palm, and an attached curled thumb.
  add(paddingGeometry, palm, 0.072, 0, -0.008, [0.049, 0.044, 0.038]);
  add(paddingGeometry, leather, 0.105, 0, 0.008, [0.061, 0.049, 0.048]);
  const thumb = add(paddingGeometry, leather, 0.076, thumbSide * 0.043, -0.018,
    [0.034, 0.021, 0.024]);
  thumb.rotation.z = thumbSide * 0.35;
  add(cuffGeometry, leather, 0.035, 0, 0);
  // Dark fastening band and fine stitching make the wrist opening readable.
  const band = add(cuffGeometry, trim, 0.025, 0, 0);
  band.scale.set(0.30, 1.025, 1.025);
  add(pipingGeometry, thread, 0.014, 0, 0);
  add(pipingGeometry, palm, 0.055, 0, 0);
  return glove;
}
