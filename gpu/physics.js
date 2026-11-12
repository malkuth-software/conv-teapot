// The rigid body on the GPU. The teapot is one body: thirteen numbers
// (centre of mass, attitude as a quaternion, velocity, spin) in four texels.
// One step is one fragment shader run over those four texels. It applies
// gravity, the grab's soft pin, and contact with the floor, then moves the
// body. The floor is a plane and the collision shape is convex hulls, so the
// contacts are exactly the hull's points that reach the floor: the shader
// goes over every point of every hull, eight times, taking from each that
// approaches the floor the impulse that stops it there and the friction
// that impulse allows. Lengths are in the model's unit; time in seconds.
//
// This is not the CPU version's solver (zc_phys solves all contacts of a
// frame together); it is a sequential one, which a shader can run as a loop.
const PHYSICS_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uState;    // four texels: position, quaternion, velocity, spin
uniform sampler2D uPoints;   // the points that may meet the floor, at rest; alpha 0: none in this texel
uniform int uCount, uWide;   // how many texels, and how many across
uniform vec3 uCom;           // the centre of mass at rest
uniform float uMass, uStep, uGravity, uGround, uFriction, uRestitution, uPushMost, uBounceFrom, uRollingArm;
uniform mat3 uInverseInertia;   // about the centre of mass, in the body's axes
uniform int uGrab;
uniform vec3 uGrabLocal, uGrabTarget;   // the held point in the body's axes from the centre of mass; where the pointer has it
out vec4 outState;
mat3 turn(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
              2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
              2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y));
}
void main() {
  vec3 x = texelFetch(uState, ivec2(0, 0), 0).xyz;
  vec4 q = texelFetch(uState, ivec2(1, 0), 0);
  vec3 v = texelFetch(uState, ivec2(2, 0), 0).xyz;
  vec3 w = texelFetch(uState, ivec2(3, 0), 0).xyz;
  mat3 R = turn(q);
  mat3 inertia = R * uInverseInertia * transpose(R);   // inverse inertia in the room's axes
  float h = uStep;
  v.y -= uGravity * h;

  if (uGrab != 0) {
    // The grab: the impulse at the held point that a critically damped
    // spring to the pointer would give over the step, found implicitly.
    vec3 r = R * uGrabLocal;
    vec3 error = x + r - uGrabTarget;
    vec3 speed = v + cross(w, r);
    float omega = 6.283185307 * 8.0;
    float stiffness = uMass * omega * omega, damping = 2.0 * uMass * omega;
    float gamma = 1.0 / (h * (damping + h * stiffness)), beta = h * stiffness * gamma;
    mat3 k;
    for (int c = 0; c < 3; ++c) {
      vec3 unit = vec3(c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0);
      k[c] = unit / uMass + cross(inertia * cross(r, unit), r) + unit * gamma;
    }
    vec3 impulse = inverse(k) * (-(speed + error * beta));
    float limit = 6.0 * uMass * uGravity * h;   // a hand pulls with at most a few times the weight
    float size = length(impulse);
    if (size > limit) { impulse *= limit / size; }
    v += impulse / uMass;
    w += inertia * cross(r, impulse);
  }

  // Contact with the floor. Every point that reaches the floor answers for
  // itself from the velocity the pass began with: the push that stops its
  // approach, and the friction that push allows. The body takes the mean of
  // those answers. (Taking them one after another would depend on the order
  // of the points; the mean does not, and a body that starts symmetric about
  // a plane stays so.)
  vec3 v0 = v, w0 = w;
  const vec3 n = vec3(0.0, 1.0, 0.0);
  float pushed = 0.0;   // the floor's whole push this step
  float deepest = 0.0;  // how far the lowest point is in the floor
  for (int pass = 0; pass < 10; ++pass) {
    vec3 dv = vec3(0.0), dw = vec3(0.0);
    float count = 0.0, pass_push = 0.0;
    // The teapot is the same on both sides of the plane z = 0 at rest, and its
    // contact points are taken so too: each point and its mirror image. A
    // teapot dropped in that plane then has nothing to turn it out of it.
    for (int i = 0; i < 2 * uCount; ++i) {
      int j = i < uCount ? i : i - uCount;
      vec4 point = texelFetch(uPoints, ivec2(j % uWide, j / uWide), 0);
      if (point.a < 0.5) { continue; }
      if (i >= uCount) { point.z = -point.z; }
      vec3 r = R * (point.xyz - uCom);
      float height = x.y + r.y - uGround;
      if (height > 0.05) { continue; }
      if (pass == 0) { deepest = max(deepest, -height); }
      vec3 u = v + cross(w, r);
      // The point may approach the floor no faster than brings it there this step.
      float least = height > 0.0 ? -height / h : 0.0;
      float before = v0.y + cross(w0, r).y;
      if (height + h * before < 0.0 && before < -uBounceFrom) { least = max(least, -uRestitution * before); }
      if (u.y >= least) { continue; }
      float kn = 1.0 / uMass + dot(cross(inertia * cross(r, n), r), n);
      float push = (least - u.y) / kn;
      vec3 impulse = n * push;
      // Friction: against the sliding there would be after the push, by at most what the push allows.
      vec3 after = u + impulse / uMass + cross(inertia * cross(r, impulse), r);
      vec3 slide = after - n * after.y;
      float speed = length(slide);
      if (speed > 1e-7) {
        vec3 t = slide / speed;
        float kt = 1.0 / uMass + dot(cross(inertia * cross(r, t), r), t);
        impulse -= t * min(speed / kt, uFriction * push);
      }
      dv += impulse / uMass;
      dw += inertia * cross(r, impulse);
      pass_push += push;
      count += 1.0;
    }
    if (count == 0.0) { break; }
    v += dv / count;
    w += dw / count;
    pushed += pass_push / count;
  }

  // Rolling resistance: the floor's push, at a small lever arm, against the spin.
  float spin = length(w);
  if (spin > 1e-7 && pushed > 0.0) {
    vec3 axis = w / spin;
    float answer = dot(axis, inertia * axis);
    w -= inertia * axis * min(spin / answer, uRollingArm * pushed);
  }
  // Damping, as the CPU's engine has it.
  v *= 1.0 - 0.02 * h;
  w *= 1.0 - 0.05 * h;
  x += v * h;
  // A body in the floor is lifted out of it, at most so fast, without being given the speed.
  x.y += min(uPushMost * h, 0.5 * deepest);
  q += 0.5 * h * vec4(w * q.w + cross(w, q.xyz), -dot(w, q.xyz));
  q = normalize(q);
  int which = int(gl_FragCoord.x);
  outState = which == 0 ? vec4(x, 0.0) : which == 1 ? q : which == 2 ? vec4(v, 0.0) : vec4(w, 0.0);
}
`;

// The points of the surface that can meet a floor, found on the GPU once per
// teapot. A plane touches a body only at corners of its convex hull, and a
// corner of the hull is the farthest point of the surface from the centre
// of mass in its own direction. So every grid point is drawn into a small
// picture, placed by its direction from the centre of mass (an octahedral
// map) with its distance as depth, farthest in front: what remains in each
// texel is the farthest point of the surface in that direction, and the
// hull's corners are among those. This takes the place of the engine's
// convex hulls.
const CANDIDATE_VERTEX_GLSL = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uGridPoints, uGridNormals;
uniform vec4 uRect;
uniform vec3 uCom;
uniform float uReach;
out vec3 vRest;
void main() {
  ivec2 texel = ivec2(int(uRect.x) + gl_VertexID % int(uRect.z), int(uRect.y) + gl_VertexID / int(uRect.z));
  vec3 point = texelFetch(uGridPoints, texel, 0).xyz;
  vec3 arm = point - uCom;
  vec3 d = arm / max(1e-9, abs(arm.x) + abs(arm.y) + abs(arm.z));
  vec2 at = d.y >= 0.0 ? d.xz : (1.0 - abs(d.zx)) * vec2(d.x >= 0.0 ? 1.0 : -1.0, d.z >= 0.0 ? 1.0 : -1.0);
  vRest = point;
  gl_Position = vec4(at * (35.0 / 36.0), clamp(1.0 - 2.0 * length(arm) / uReach, -0.99, 0.99), 1.0);
  if (texelFetch(uGridNormals, texel, 0).a < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }   // under a join: not part of the surface
  gl_PointSize = 1.0;
}
`;
const CANDIDATE_FRAGMENT_GLSL = `#version 300 es
precision highp float;
in vec3 vRest;
out vec4 outPoint;
void main() { outPoint = vec4(vRest, 1.0); }
`;

class TeapotPhysics {
  constructor(gpu) {
    this.gpu = gpu;
    const gl = gpu.gl;
    this.step = gpu.program(ROOM_VERTEX_GLSL, PHYSICS_GLSL);
    this.candidates = gpu.program(CANDIDATE_VERTEX_GLSL, CANDIDATE_FRAGMENT_GLSL);
    this.states = [0, 1].map(() => {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 4, 1);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      return { texture, framebuffer };
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.state = new Float32Array(16);
    this.state[7] = 1;
    this.accumulator = 0;
    this.grab = null;
    this.still = 0;
    this.running = false;
    this.stepMs = 0;
    this.onSurface = true;
  }

  // The body from the engine: mass, centre of mass, inverse inertia, poke radius, hull points.
  setBody(engine) {
    const pointer = engine._tp_body();
    const head = new Float64Array(engine.HEAPF64.buffer, pointer, 15);
    this.mass = head[0];
    this.com = [head[1], head[2], head[3]];
    this.inverseInertia = new Float32Array([head[4], head[7], head[10], head[5], head[8], head[11], head[6], head[9], head[12]]);   // by columns
    this.pokeRadius = head[13];
    this.count = head[14];
    const points = new Float64Array(engine.HEAPF64.buffer, pointer + 15 * 8, this.count * 3);
    const texels = new Float32Array(this.count * 4);
    for (let k = 0; k < this.count; ++k) { texels.set([points[k * 3], points[k * 3 + 1], points[k * 3 + 2], 1], k * 4); }
    this.gpu.floatTexture('hullPoints', texels, 4);
    this.hullCount = this.count;
    // The surface's own contact points: the farthest grid point in each direction from the centre of mass.
    const gl = this.gpu.gl, c = this.candidates.uniforms;
    const farthest = this.gpu.target('farthestPoints', 36, 36, gl.RGBA32F);   // 1296 directions, about as many points as the engine's hulls have
    gl.useProgram(this.candidates.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, farthest.framebuffer);
    gl.viewport(0, 0, 36, 36);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE12); gl.bindTexture(gl.TEXTURE_2D, this.gpu.targets.mapSurface.first); gl.uniform1i(c.uGridPoints, 12);
    gl.activeTexture(gl.TEXTURE13); gl.bindTexture(gl.TEXTURE_2D, this.gpu.targets.mapSurface.second); gl.uniform1i(c.uGridNormals, 13);
    gl.uniform3fv(c.uCom, this.com);
    gl.uniform1f(c.uReach, 12.0);
    gl.bindVertexArray(this.gpu.emptyVao);
    for (const part of this.gpu.parts) {
      gl.uniform4fv(c.uRect, part.rect);
      gl.drawArrays(gl.POINTS, 0, part.rect[2] * part.rect[3]);
    }
    gl.depthFunc(gl.LEQUAL);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.useSurface(this.onSurface);
  }
  // Which points meet the floor: the surface's own, evaluated on the GPU, or the CPU's hull points.
  useSurface(on) {
    this.onSurface = on;
    this.count = on ? 1296 : this.hullCount;
  }

  // Take the state from a pose (rotation by rows a, translation b), at rest.
  fromPose(a, b) {
    const c = this.com;
    const x = [a[0] * c[0] + a[1] * c[1] + a[2] * c[2] + b[0], a[3] * c[0] + a[4] * c[1] + a[5] * c[2] + b[1], a[6] * c[0] + a[7] * c[1] + a[8] * c[2] + b[2]];
    // Quaternion of the rotation.
    const trace = a[0] + a[4] + a[8];
    let q;
    if (trace > 0) {
      const s = 2 * Math.sqrt(trace + 1);
      q = [(a[7] - a[5]) / s, (a[2] - a[6]) / s, (a[3] - a[1]) / s, s / 4];
    } else if (a[0] > a[4] && a[0] > a[8]) {
      const s = 2 * Math.sqrt(1 + a[0] - a[4] - a[8]);
      q = [s / 4, (a[1] + a[3]) / s, (a[2] + a[6]) / s, (a[7] - a[5]) / s];
    } else if (a[4] > a[8]) {
      const s = 2 * Math.sqrt(1 + a[4] - a[0] - a[8]);
      q = [(a[1] + a[3]) / s, s / 4, (a[5] + a[7]) / s, (a[2] - a[6]) / s];
    } else {
      const s = 2 * Math.sqrt(1 + a[8] - a[0] - a[4]);
      q = [(a[2] + a[6]) / s, (a[5] + a[7]) / s, s / 4, (a[3] - a[1]) / s];
    }
    this.state.fill(0);
    this.state.set(x, 0);
    this.state.set(q, 4);
    this.write();
    this.accumulator = 0;
    this.still = 0;
  }

  write() {
    const gl = this.gpu.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.states[0].texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4, 1, gl.RGBA, gl.FLOAT, this.state);
  }

  // The pose the state stands for: rotation by rows, then translation.
  pose() {
    const s = this.state, [x, y, z, w] = [s[4], s[5], s[6], s[7]];
    const a = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
               2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
               2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
    const c = this.com;
    const b = [s[0] - (a[0] * c[0] + a[1] * c[1] + a[2] * c[2]), s[1] - (a[3] * c[0] + a[4] * c[1] + a[5] * c[2]), s[2] - (a[6] * c[0] + a[7] * c[1] + a[8] * c[2])];
    return { a, b };
  }

  // Hold the body at a point of the room that is on it now.
  hold(point) {
    const { a } = this.pose(), s = this.state;
    const d = [point[0] - s[0], point[1] - s[1], point[2] - s[2]];
    this.grab = { local: [a[0] * d[0] + a[3] * d[1] + a[6] * d[2], a[1] * d[0] + a[4] * d[1] + a[7] * d[2], a[2] * d[0] + a[5] * d[1] + a[8] * d[2]], target: point.slice() };
    this.running = true;
    this.still = 0;
  }
  moveHold(point) { if (this.grab) { this.grab.target = point; } }
  // A push: a change of velocity at a point along a direction.
  poke(point, direction, speed) {
    const s = this.state;
    const arm = [point[0] - s[0], point[1] - s[1], point[2] - s[2]], push = direction.map((d) => d * speed / 0.07);   // speed in m/s; a model unit is 0.07 m
    const r2 = this.pokeRadius * this.pokeRadius;
    s[8] += push[0]; s[9] += push[1]; s[10] += push[2];
    s[12] += (arm[1] * push[2] - arm[2] * push[1]) / r2;
    s[13] += (arm[2] * push[0] - arm[0] * push[2]) / r2;
    s[14] += (arm[0] * push[1] - arm[1] * push[0]) / r2;
    this.write();
    this.running = true;
    this.still = 0;
  }

  // Advance by dt seconds of the simulation's time. Returns true if the body moved.
  advance(dt, ground) {
    if (!this.running) { return false; }
    const gl = this.gpu.gl, u = this.step.uniforms, h = 1 / 240;
    const started = performance.now();
    this.accumulator += dt;
    let steps = 0;
    gl.useProgram(this.step.program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, 4, 1);
    const lowest = this.onSurface ? this.gpu.targets.farthestPoints : null;
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, lowest ? lowest.texture : this.gpu.textures.hullPoints); gl.uniform1i(u.uPoints, 1);
    gl.uniform1i(u.uWide, lowest ? 36 : 2048);
    gl.uniform1i(u.uCount, this.count);
    gl.uniform3fv(u.uCom, this.com);
    gl.uniform1f(u.uMass, this.mass);
    gl.uniform1f(u.uStep, h);
    gl.uniform1f(u.uGravity, 9.81 / 0.07);
    gl.uniform1f(u.uGround, ground);
    // The CPU engine's values, in the model's unit (0.07 m): friction of porcelain on
    // the floor, restitution and the speed below which nothing bounces, the fastest an
    // overlap is pushed apart, the floor's rolling resistance.
    gl.uniform1f(u.uFriction, Math.sqrt(0.5 * 0.9));
    gl.uniform1f(u.uRestitution, 0.25);
    gl.uniform1f(u.uBounceFrom, 1.0 / 0.07);
    gl.uniform1f(u.uPushMost, 0.5 / 0.07);
    gl.uniform1f(u.uRollingArm, 0.004 / 0.07);
    gl.uniformMatrix3fv(u.uInverseInertia, false, this.inverseInertia);
    gl.uniform1i(u.uGrab, this.grab ? 1 : 0);
    if (this.grab) {
      gl.uniform3fv(u.uGrabLocal, this.grab.local);
      gl.uniform3fv(u.uGrabTarget, this.grab.target);
    }
    gl.bindVertexArray(this.gpu.emptyVao);
    while (this.accumulator >= h && steps < 16) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.states[1].framebuffer);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.states[0].texture); gl.uniform1i(u.uState, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.states.reverse();
      this.accumulator -= h;
      ++steps;
    }
    if (steps === 16) { this.accumulator = 0; }
    if (steps > 0) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.states[0].framebuffer);
      gl.readPixels(0, 0, 4, 1, gl.RGBA, gl.FLOAT, this.state);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    // Slower than the engine's sleep thresholds for half a second, and not held: stop.
    const s = this.state, speed = Math.hypot(s[8], s[9], s[10]), spin = Math.hypot(s[12], s[13], s[14]);
    this.still = speed < 0.004 / 0.07 && spin < 0.02 && !this.grab ? this.still + steps * h : 0;
    if (this.still > 0.5) {
      this.running = false;
      s.fill(0, 8, 16);
      this.write();
    }
    this.stepMs = performance.now() - started;
    this.steps = steps;
    return steps > 0;
  }
}
