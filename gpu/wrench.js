// PlaySuite's wrench-transport contact solver (zc::phys, third_party/zc_phys)
// for the teapot's one body on a flat floor, as a fragment shader.
//
// This file is a port of PlaySuite's code and is under PlaySuite's licence,
// not this repository's MIT licence; see third_party/zc_phys/PROVENANCE.md.
// Ported with the owner's permission.
//
// What is the same as the CPU's engine: the contact points (every hull vertex
// within the margin of the floor), the impulses each point retains from frame
// to frame, the compliance set against the mass a contact carries, the
// sliding shift and the rebound, the rolling-resistance row, the passes, the
// cost each pass minimises, Newton's method on it with the exact line
// search, the damping, the speed caps and the rule for falling asleep.
//
// What is adapted to the GPU:
//  - One run of the shader is one whole frame of the solver. The frame's
//    rows live in arrays inside the shader (at most 96 contacts).
//  - A fragment can write only its own texel. The body's new state is a few
//    texels and each fragment that writes one runs the frame. The impulses
//    the contact points retain are one pair of texels per hull vertex: each
//    vertex in contact runs the same frame again and keeps its own impulse.
//    The frame is deterministic, so they agree.
//  - Numbers are 32-bit floats. The CPU stops Newton at a relative gradient
//    of 1e-9, which a float cannot reach; here it stops at 1e-5, and the
//    line search at 1e-5 of the starting slope.
//  - With one body and a floor whose normal is fixed, the contact frame is
//    constant and the block factorisation is a 6 x 6 elimination.
const WRENCH_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uState;      // position; quaternion; velocity; spin; rolling impulse and whether it is held; quiet time, asleep
uniform sampler2D uWarm;       // per hull vertex two texels: (normal impulse, tangent impulses, 0); (rebound owed, in contact, 0, 0)
uniform sampler2D uVertices;   // hull vertices in the body's frame
uniform int uCount, uMode;     // mode 0: the body's state; 1: the vertices' retained impulses
uniform float uMass, uRadius, uFriction, uRestitution, uRolling, uH;
uniform mat3 uInertia, uInverseInertia;
uniform int uGrab;
uniform vec3 uGrabLocal, uGrabTarget;
out vec4 outValue;

const float kGravity = 9.81;
const float kLinearDamping = 0.02, kAngularDamping = 0.05;
const float kPushMax = 0.5, kLinearSlop = 0.0005, kSpeculative = 0.002, kBounceFrom = 1.0;
const float kSleepLinear = 0.004, kSleepAngular = 0.02, kSleepTime = 0.5;
const float kRegularization = 0.03, kRelaxation = 0.1, kLoose = 1.0e-3, kTight = 1.0e-5;
const float kPassTolerance = 1.0e-3, kShiftTolerance = 1.0e-3;
const int kMaxNewton = 30, kMaxPasses = 6;
const int kMode0pen = 0, kModeStick = 1, kModeSlide = 2;
// The contact frame on the floor: tangent 1, tangent 2, normal (the engine's tangent_basis of (0, 0, 1)).
const vec3 kT1 = vec3(0.0, -1.0, 0.0), kT2 = vec3(1.0, 0.0, 0.0), kN = vec3(0.0, 0.0, 1.0);

const int kRows = 96;
int rowVertex[kRows];
vec3 rowArm[kRows];
float rowSeparation[kRows], rowR[kRows], rowBase[kRows], rowTarget[kRows], rowShift[kRows], rowBounce[kRows];
vec3 rowLambda[kRows], rowGamma[kRows];
mat3 rowG[kRows];
int rowMode[kRows];
int rowCount;
bool rolling;
float rollR, rollLimit;
vec3 rollLambda, rollGamma;
mat3 rollG;
int rollMode;
// The frame's mass matrix and its inverse (rotational part), the free twist and the twist.
mat3 inertia, inverseInertia;
vec3 freeV, freeW, V, W;

vec4 warmAt(int vertex, int which) { int i = 2 * vertex + which; return texelFetch(uWarm, ivec2(i & 2047, i >> 11), 0); }
vec4 vertexAt(int i) { return texelFetch(uVertices, ivec2(i & 2047, i >> 11), 0); }
mat3 turn(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
              2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
              2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y));
}
// The contact-frame velocity of a point at 'arm' from the centre for a twist.
vec3 contactVelocity(vec3 arm, vec3 v, vec3 w) {
  vec3 u = v + cross(w, arm);
  return vec3(dot(kT1, u), dot(kT2, u), dot(kN, u));
}

// Friction cone |g_t| <= mu g_n: impulse, mode and Hessian of one contact for contact-frame velocity c.
void evaluateContact(int k, vec3 c) {
  float inverseR = 1.0 / rowR[k];
  float y0 = rowLambda[k].x - c.x * inverseR;
  float y1 = rowLambda[k].y - c.y * inverseR;
  float yn = rowLambda[k].z - (c.z - rowTarget[k]) * inverseR;
  float yr = sqrt(y0 * y0 + y1 * y1);
  float mu = uFrictionPair;
  if (yr <= mu * yn) {
    rowMode[k] = kModeStick;
    rowGamma[k] = vec3(y0, y1, yn);
    rowG[k] = mat3(inverseR);
    return;
  }
  if (yn <= -mu * yr) {
    rowMode[k] = kMode0pen;
    rowGamma[k] = vec3(0.0);
    rowG[k] = mat3(0.0);
    return;
  }
  rowMode[k] = kModeSlide;
  float factor = 1.0 / (1.0 + mu * mu);
  float gammaN = (yn + mu * yr) * factor;
  float t0 = y0 / yr, t1 = y1 / yr;
  rowGamma[k] = vec3(mu * gammaN * t0, mu * gammaN * t1, gammaN);
  float a = factor * inverseR;
  float p = mu * gammaN * inverseR / yr;
  float g01 = a * mu * mu * t0 * t1 - p * t0 * t1;
  rowG[k] = mat3(a * mu * mu * t0 * t0 + p * (1.0 - t0 * t0), g01, a * mu * t0,
                 g01, a * mu * mu * t1 * t1 + p * (1.0 - t1 * t1), a * mu * t1,
                 a * mu * t0, a * mu * t1, a);
}
// A 3-vector bounded in length: the rolling resistance, on the spin.
void evaluateRolling(vec3 c) {
  float inverseR = 1.0 / rollR;
  vec3 y = rollLambda - c * inverseR;
  float size = length(y);
  if (size <= rollLimit) {
    rollMode = rollLimit > 0.0 ? kModeStick : kMode0pen;
    rollGamma = y;
    rollG = mat3(inverseR);
    return;
  }
  float shrink = rollLimit / size;
  rollMode = rollLimit > 0.0 ? kModeSlide : kMode0pen;
  rollGamma = y * shrink;
  float p = shrink * inverseR;
  vec3 u = y / size;
  rollG = mat3(p) - p * outerProduct(u, u);
}
void evaluateRows(vec3 v, vec3 w) {
  for (int k = 0; k < rowCount; ++k) { evaluateContact(k, contactVelocity(rowArm[k], v, w)); }
  if (rolling) { evaluateRolling(w); }
}
// The impulses of all rows as a wrench on the body.
void rowsWrench(out vec3 force, out vec3 torque) {
  force = vec3(0.0);
  torque = vec3(0.0);
  for (int k = 0; k < rowCount; ++k) {
    vec3 f = kT1 * rowGamma[k].x + kT2 * rowGamma[k].y + kN * rowGamma[k].z;
    force += f;
    torque += cross(rowArm[k], f);
  }
  if (rolling) { torque += rollGamma; }
}
float energyNorm(vec3 linear, vec3 angular) { return sqrt(dot(linear, linear) / uMass + dot(angular, inverseInertia * angular)); }

// First and second derivative of the cost along a step at parameter alpha.
void lineDerivatives(vec3 stepV, vec3 stepW, float slopeLinear, float curvatureMass, float alpha, out float slope, out float curvature) {
  slope = slopeLinear + alpha * curvatureMass;
  curvature = curvatureMass;
  vec3 v = V + alpha * stepV, w = W + alpha * stepW;
  for (int k = 0; k < rowCount; ++k) {
    vec3 d = contactVelocity(rowArm[k], stepV, stepW);
    evaluateContact(k, contactVelocity(rowArm[k], v, w));
    slope -= dot(d, rowGamma[k]);
    curvature += dot(d, rowG[k] * d);
  }
  if (rolling) {
    evaluateRolling(w);
    slope -= dot(stepW, rollGamma);
    curvature += dot(stepW, rollG * stepW);
  }
}
// The step length that minimises the convex cost along the Newton direction.
float exactLineSearch(vec3 stepV, vec3 stepW, float slopeLinear, float curvatureMass, float slope0) {
  float slope, curvature;
  float tolerance = 1.0e-5 * abs(slope0);
  lineDerivatives(stepV, stepW, slopeLinear, curvatureMass, 1.0, slope, curvature);
  if (abs(slope) <= tolerance) { return 1.0; }
  float low = 0.0, high = 1.0;
  if (slope < 0.0) {
    low = 1.0;
    high = 2.0;
    for (int grow = 0; grow < 20; ++grow) {
      lineDerivatives(stepV, stepW, slopeLinear, curvatureMass, high, slope, curvature);
      if (slope >= 0.0) { break; }
      low = high;
      high *= 2.0;
    }
  }
  float alpha = 0.5 * (low + high);
  for (int refine = 0; refine < 40; ++refine) {
    lineDerivatives(stepV, stepW, slopeLinear, curvatureMass, alpha, slope, curvature);
    if (abs(slope) <= tolerance) { break; }
    if (slope > 0.0) { high = alpha; } else { low = alpha; }
    float next = alpha - slope / curvature;
    if (!(next > low && next < high)) { next = 0.5 * (low + high); }
    if (abs(next - alpha) <= 1.0e-6 * max(1.0, alpha)) { alpha = next; break; }
    alpha = next;
  }
  return alpha;
}
// Solves the 6 x 6 system a x = b by elimination with pivoting.
bool solve6(inout float a[36], inout float b[6]) {
  for (int column = 0; column < 6; ++column) {
    int pivot = column;
    float best = abs(a[column * 6 + column]);
    for (int r = column + 1; r < 6; ++r) {
      if (abs(a[r * 6 + column]) > best) { best = abs(a[r * 6 + column]); pivot = r; }
    }
    if (!(best > 1.0e-30)) { return false; }
    if (pivot != column) {
      for (int c = 0; c < 6; ++c) { float t = a[column * 6 + c]; a[column * 6 + c] = a[pivot * 6 + c]; a[pivot * 6 + c] = t; }
      float t = b[column]; b[column] = b[pivot]; b[pivot] = t;
    }
    for (int r = column + 1; r < 6; ++r) {
      float f = a[r * 6 + column] / a[column * 6 + column];
      for (int c = column; c < 6; ++c) { a[r * 6 + c] -= f * a[column * 6 + c]; }
      b[r] -= f * b[column];
    }
  }
  for (int r = 5; r >= 0; --r) {
    float sum = b[r];
    for (int c = r + 1; c < 6; ++c) { sum -= a[r * 6 + c] * b[c]; }
    b[r] = sum / a[r * 6 + r];
  }
  return true;
}
// Newton's method with an exact line search from the twist (V, W).
void newtonSolve(float loose) {
  float previous = 1.0e30;
  int stalls = 0;
  for (int iteration = 0; iteration <= kMaxNewton; ++iteration) {
    evaluateRows(V, W);
    vec3 force, torque;
    rowsWrench(force, torque);
    vec3 differenceV = V - freeV, differenceW = W - freeW;
    vec3 gradientV = uMass * differenceV - force, gradientW = inertia * differenceW - torque;
    float residual = energyNorm(gradientV, gradientW);
    float scale = max(sqrt(uMass * dot(V, V) + dot(W, inertia * W)), energyNorm(force, torque));
    if (residual <= 1.0e-14 + kTight * scale) { return; }
    if (iteration > 0 && residual <= loose * scale) { return; }
    if (iteration == kMaxNewton) { return; }
    // Round-off floor: accept once the residual stops falling near zero.
    if (residual > 0.5 * previous && residual <= 1.0e-3 * scale) {
      stalls += 1;
      if (stalls >= 3) { return; }
    } else {
      stalls = 0;
    }
    previous = residual;
    // Hessian: the mass, plus one relation per row that presses.
    float h[36];
    for (int i = 0; i < 36; ++i) { h[i] = 0.0; }
    h[0] = uMass; h[7] = uMass; h[14] = uMass;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) { h[(3 + r) * 6 + 3 + c] = inertia[c][r]; }
    }
    for (int k = 0; k < rowCount; ++k) {
      if (rowMode[k] == kMode0pen) { continue; }
      vec3 direction[3] = vec3[3](kT1, kT2, kN);
      float j[18];
      for (int d = 0; d < 3; ++d) {
        vec3 arm = cross(rowArm[k], direction[d]);
        j[d * 6] = direction[d].x; j[d * 6 + 1] = direction[d].y; j[d * 6 + 2] = direction[d].z;
        j[d * 6 + 3] = arm.x; j[d * 6 + 4] = arm.y; j[d * 6 + 5] = arm.z;
      }
      mat3 g = rowG[k];
      for (int i = 0; i < 6; ++i) {
        vec3 column = vec3(j[i], j[6 + i], j[12 + i]);
        vec3 gi = column * g;   // row vector times G
        for (int c = 0; c < 6; ++c) { h[i * 6 + c] += gi.x * j[c] + gi.y * j[6 + c] + gi.z * j[12 + c]; }
      }
    }
    if (rolling && rollMode != kMode0pen) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) { h[(3 + r) * 6 + 3 + c] += rollG[c][r]; }
      }
    }
    float b[6] = float[6](-gradientV.x, -gradientV.y, -gradientV.z, -gradientW.x, -gradientW.y, -gradientW.z);
    if (!solve6(h, b)) { return; }
    vec3 stepV = vec3(b[0], b[1], b[2]), stepW = vec3(b[3], b[4], b[5]);
    vec3 massStepV = uMass * stepV, massStepW = inertia * stepW;
    float slopeLinear = dot(massStepV, differenceV) + dot(massStepW, differenceW);
    float curvatureMass = dot(massStepV, stepV) + dot(massStepW, stepW);
    float slope, curvature;
    lineDerivatives(stepV, stepW, slopeLinear, curvatureMass, 0.0, slope, curvature);
    if (!(slope < 0.0)) { return; }
    float alpha = exactLineSearch(stepV, stepW, slopeLinear, curvatureMass, slope);
    V += alpha * stepV;
    W += alpha * stepW;
  }
}

void main() {
  vec3 position = texelFetch(uState, ivec2(0, 0), 0).xyz;
  vec4 q = texelFetch(uState, ivec2(1, 0), 0);
  V = texelFetch(uState, ivec2(2, 0), 0).xyz;
  W = texelFetch(uState, ivec2(3, 0), 0).xyz;
  vec4 rollMemory = texelFetch(uState, ivec2(4, 0), 0);
  vec4 rest = texelFetch(uState, ivec2(5, 0), 0);   // quiet time, asleep
  int texel = int(gl_FragCoord.y) * 2048 + int(gl_FragCoord.x);
  if (rest.y > 0.5) {
    // Asleep: nothing moves and what the contacts retain is kept.
    outValue = uMode == 0 ? texelFetch(uState, ivec2(texel, 0), 0) : texelFetch(uWarm, ivec2(gl_FragCoord.xy), 0);
    return;
  }
  mat3 R = turn(q);
  inertia = R * uInertia * transpose(R);
  inverseInertia = R * uInverseInertia * transpose(R);
  float h = uH;

  if (uGrab != 0) {
    // The grab (src/sim.hpp, pull): the impulse at the held point that a critically damped spring to the pointer gives over the frame.
    vec3 r = R * uGrabLocal;
    vec3 error = position + r - uGrabTarget;
    vec3 speed = V + cross(W, r);
    float omega = 6.283185307 * 8.0;
    float stiffness = uMass * omega * omega, damping = 2.0 * uMass * omega;
    float gamma = 1.0 / (h * (damping + h * stiffness)), beta = h * stiffness * gamma;
    mat3 k;
    for (int c = 0; c < 3; ++c) {
      vec3 unit = vec3(c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0);
      k[c] = unit / uMass + cross(inverseInertia * cross(r, unit), r) + unit * gamma;
    }
    vec3 impulse = inverse(k) * (-(speed + error * beta));
    float limit = 6.0 * uMass * kGravity * h;
    float size = length(impulse);
    if (size > limit) { impulse *= limit / size; }
    V += impulse / uMass;
    W += inverseInertia * cross(r, impulse);
  }

  // Contacts: every hull vertex within the margin of the floor.
  float margin = min(kSpeculative + (length(V) + length(W) * uRadius) * h, 0.05);
  int own = uMode == 1 ? texel / 2 : -1;
  if (uMode == 1) {
    vec3 mine = position + R * vertexAt(own).xyz;
    if (own >= uCount || !(mine.z < margin)) { outValue = vec4(0.0); return; }
  }
  rowCount = 0;
  float load = 0.0;
  for (int i = 0; i < uCount && rowCount < kRows; ++i) {
    vec3 arm = R * vertexAt(i).xyz;
    float separation = position.z + arm.z;
    if (!(separation < margin)) { continue; }
    int k = rowCount++;
    rowVertex[k] = i;
    rowArm[k] = arm;
    rowSeparation[k] = separation;
    // What the point retained, if it was in contact last frame.
    vec4 warm = warmAt(i, 0), more = warmAt(i, 1);
    bool matched = more.y > 0.5;
    rowLambda[k] = matched ? vec3(warm.y, warm.z, warm.x) : vec3(0.0);
    float pending = matched ? more.x : 0.0;
    float trace = 0.0;
    vec3 direction[3] = vec3[3](kT1, kT2, kN);
    for (int d = 0; d < 3; ++d) {
      vec3 a = cross(arm, direction[d]);
      trace += 1.0 / uMass + dot(a, inverseInertia * a);
    }
    float localInverseMass = trace / 3.0;
    // Compliance against the mass the contact carries: the body rests on the floor with all its mass.
    rowR[k] = kRegularization / max(1.0 / localInverseMass, uMass);
    // Normal target: a gap may close exactly; an overlap decays, never pops.
    rowBase[k] = separation > 0.0 ? -separation / h : min(-separation / (h + kRelaxation), kPushMax);
    rowBounce[k] = pending;   // read below, then overwritten with what is owed next
    load += rowLambda[k].z;
  }
  // Rolling resistance: a torque on the spin bounded by a lever arm times the normal impulse.
  rolling = rowCount > 0;
  if (rolling) {
    rollR = kRegularization * (inverseInertia[0][0] + inverseInertia[1][1] + inverseInertia[2][2]) / 3.0;
    rollLimit = max(uRolling, 0.004) * load;
    rollLambda = rollMemory.w > 0.5 ? rollMemory.xyz : vec3(0.0);
  }
  // A point that carried normal impulse g last frame was holding up a mass of g / (|gravity| h).
  for (int k = 0; k < rowCount; ++k) {
    float held = rowLambda[k].z / (kGravity * h);
    if (held * rowR[k] > kRegularization) { rowR[k] = kRegularization / held; }
  }
  // Sliding shift and restitution from the incoming velocities.
  for (int k = 0; k < rowCount; ++k) {
    vec3 c = contactVelocity(rowArm[k], V, W);
    float pending = rowBounce[k];
    rowBounce[k] = 0.0;
    if (pending > 0.0) {
      rowBase[k] = max(rowBase[k], pending);
    } else if (c.z < -kBounceFrom && rowSeparation[k] + h * c.z < 0.0) {
      if (rowSeparation[k] > kLinearSlop) {
        rowBounce[k] = -uRestitution * c.z;
      } else {
        rowBase[k] = max(rowBase[k], -uRestitution * c.z);
      }
    }
    rowShift[k] = uFrictionPair * length(c.xy);
    rowTarget[k] = rowBase[k] - rowShift[k];
  }
  // The free twist: gravity and damping.
  float linearDecay = 1.0 / (1.0 + h * kLinearDamping), angularDecay = 1.0 / (1.0 + h * kAngularDamping);
  freeV = (V + h * vec3(0.0, 0.0, -kGravity)) * linearDecay;
  freeW = W * angularDecay;

  // The passes: each minimises the cost by Newton; its impulses become the retained field.
  float accuracy = kLoose;
  for (int pass = 0; pass < kMaxPasses; ++pass) {
    newtonSolve(accuracy);
    float change = 0.0, magnitude = 0.0, shiftChange = 0.0;
    float pressed = 0.0;
    for (int k = 0; k < rowCount; ++k) {
      vec3 d = rowGamma[k] - rowLambda[k];
      change += rowR[k] * dot(d, d);
      magnitude += rowR[k] * dot(rowGamma[k], rowGamma[k]);
      rowLambda[k] = rowGamma[k];
      vec3 c = contactVelocity(rowArm[k], V, W);
      float shift = uFrictionPair * length(c.xy);
      shiftChange = max(shiftChange, abs(shift - rowShift[k]));
      rowShift[k] = shift;
      rowTarget[k] = rowBase[k] - shift;
      pressed += rowGamma[k].z;
    }
    if (rolling) {
      vec3 d = rollGamma - rollLambda;
      change += rollR * dot(d, d);
      magnitude += rollR * dot(rollGamma, rollGamma);
      rollLambda = rollGamma;
      rollLimit = max(uRolling, 0.004) * pressed;
    }
    float relative = sqrt(change) / max(sqrt(magnitude), 1.0e-30);
    if (relative <= kPassTolerance && shiftChange <= kShiftTolerance) { break; }
    accuracy = min(kLoose, max(kTight, 0.1 * relative));
  }

  if (uMode == 1) {
    // This vertex's retained impulses and the rebound it is owed.
    int which = texel % 2;
    outValue = vec4(0.0);
    for (int k = 0; k < rowCount; ++k) {
      if (rowVertex[k] != own) { continue; }
      outValue = which == 0 ? vec4(rowGamma[k].z, rowGamma[k].x, rowGamma[k].y, 0.0) : vec4(rowBounce[k], 1.0, 0.0, 0.0);
    }
    return;
  }

  // The body moves with the twist found.
  position += V * h;
  q = normalize(q + 0.5 * h * vec4(W * q.w + cross(W, q.xyz), -dot(W, q.xyz)));
  // Safety nets: speeds are capped.
  float speed = length(V), spin = length(W);
  if (speed > 20.0) { V *= 20.0 / speed; }
  if (spin > 50.0) { W *= 50.0 / spin; }
  // Asleep after half a second below both thresholds, unless held.
  float quiet = speed < kSleepLinear && spin < kSleepAngular ? rest.x + h : 0.0;
  bool asleep = quiet >= kSleepTime && uGrab == 0;
  if (asleep) { V = vec3(0.0); W = vec3(0.0); }
  outValue = texel == 0 ? vec4(position, float(rowCount)) : texel == 1 ? q : texel == 2 ? vec4(V, 0.0) : texel == 3 ? vec4(W, 0.0)
           : texel == 4 ? vec4(rollGamma, rolling ? 1.0 : 0.0) : vec4(quiet, asleep ? 1.0 : 0.0, 0.0, 0.0);
}
`.replace(/uFrictionPair/g, 'sqrt(uFriction * 0.9)');

// The body's mass, centre of mass and inertia, integrated on the GPU over the
// faces of its convex hulls: each triangle with the origin is a tetrahedron,
// and the volume, first moments and second moments of the solid are sums over
// the tetrahedra (the divergence theorem). The shader goes over all the
// triangles and writes five texels: mass and centre of mass; then the
// inertia about the centre of mass, by rows.
const MASS_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uTriangles;   // three texels a triangle
uniform int uCount;
uniform float uDensity;
out vec4 outValue;
void main() {
  float volume = 0.0;
  vec3 first = vec3(0.0);
  mat3 second = mat3(0.0);
  for (int t = 0; t < uCount; ++t) {
    int i = 3 * t;
    vec3 a = texelFetch(uTriangles, ivec2(i & 2047, i >> 11), 0).xyz;
    vec3 b = texelFetch(uTriangles, ivec2((i + 1) & 2047, (i + 1) >> 11), 0).xyz;
    vec3 c = texelFetch(uTriangles, ivec2((i + 2) & 2047, (i + 2) >> 11), 0).xyz;
    float six = dot(a, cross(b, c));   // six times the tetrahedron's signed volume
    vec3 sum = a + b + c;
    volume += six / 6.0;
    first += sum * (six / 24.0);
    second += (outerProduct(a, a) + outerProduct(b, b) + outerProduct(c, c) + outerProduct(sum, sum)) * (six / 120.0);
  }
  vec3 centre = first / volume;
  float mass = uDensity * volume;
  // Inertia about the origin from the second moments, then moved to the centre of mass.
  mat3 about = uDensity * (mat3(second[0][0] + second[1][1] + second[2][2]) - second);
  mat3 inertia = about - mass * (mat3(dot(centre, centre)) - outerProduct(centre, centre));
  int texel = int(gl_FragCoord.x);
  outValue = texel == 0 ? vec4(mass, centre) : texel == 1 ? vec4(volume, 0.0, 0.0, 0.0)
           : vec4(inertia[0][texel - 2], inertia[1][texel - 2], inertia[2][texel - 2], 0.0);
}
`;

// The solver's body on the GPU: its state, the hull vertices, and what the contacts retain.
class WrenchPhysics {
  constructor(gpu) {
    this.gpu = gpu;
    const gl = gpu.gl;
    this.program = gpu.program(ROOM_VERTEX_GLSL, WRENCH_GLSL);
    this.massProgram = gpu.program(ROOM_VERTEX_GLSL, MASS_GLSL);
    const make = (width, height) => {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, width, height);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      return { texture, framebuffer, width, height };
    };
    this.make = make;
    this.states = [make(8, 1), make(8, 1)];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.state = new Float32Array(32);
    this.state[7] = 1;
    this.accumulator = 0;
    this.grab = null;
    this.running = false;
    this.stepMs = 0;
    this.steps = 0;
    this.contacts = 0;
  }
  // Model point or direction to the engine's frame (metres, z up) and back.
  static toEngine(p) { return [p[0] * 0.07, -p[2] * 0.07, p[1] * 0.07]; }
  static toModel(p) { return [p[0] / 0.07, p[2] / 0.07, -p[1] / 0.07]; }

  setBody(engine) {
    const gl = this.gpu.gl;
    const pointer = engine._tp_solver_body();
    // Copies: the engine reuses the memory these are read from.
    const head = new Float64Array(engine.HEAPF64.buffer, pointer, 19).slice();
    const vertices = new Float64Array(engine.HEAPF64.buffer, pointer + 19 * 8, head[18] * 4).slice();
    this.mass = head[0];
    this.radius = head[1];
    this.com = [head[2], head[3], head[4]];
    const a = Array.from(head.subarray(5, 14));
    this.inertia = new Float32Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
    const det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    const inv = [(a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det, (a[1] * a[5] - a[2] * a[4]) / det,
                 (a[5] * a[6] - a[3] * a[8]) / det, (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
                 (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det, (a[0] * a[4] - a[1] * a[3]) / det];
    this.inverseInertia = new Float32Array([inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8]]);
    // The same three found on the GPU, by integrating over the hulls' faces; these are the ones used.
    const engineValues = { mass: this.mass, com: this.com.slice(), inertia: a.slice() };
    const facePointer = engine._tp_hull_triangles();
    const triangles = new Float64Array(engine.HEAPF64.buffer, facePointer, 1)[0];
    const corners = new Float64Array(engine.HEAPF64.buffer, facePointer + 8, triangles * 9);
    const texels = new Float32Array(triangles * 12);
    for (let k = 0; k < triangles * 3; ++k) { texels.set([corners[k * 3], corners[k * 3 + 1], corners[k * 3 + 2], 0], k * 4); }
    this.gpu.floatTexture('hullTriangles', texels, 4);
    const target = this.make(5, 1);
    gl.viewport(0, 0, 5, 1);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(this.massProgram.program);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.gpu.textures.hullTriangles); gl.uniform1i(this.massProgram.uniforms.uTriangles, 0);
    gl.uniform1i(this.massProgram.uniforms.uCount, triangles);
    gl.uniform1f(this.massProgram.uniforms.uDensity, 2400);
    gl.bindVertexArray(this.gpu.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const found = new Float32Array(20);
    gl.readPixels(0, 0, 5, 1, gl.RGBA, gl.FLOAT, found);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    const g = [found[8], found[9], found[10], found[12], found[13], found[14], found[16], found[17], found[18]];   // by rows
    this.massCheck = { mass: Math.abs(found[0] - engineValues.mass) / engineValues.mass,
                       centre: Math.hypot(found[1] - engineValues.com[0], found[2] - engineValues.com[1], found[3] - engineValues.com[2]),
                       inertia: Math.max(...g.map((x, k) => Math.abs(x - engineValues.inertia[k]))) / Math.max(...engineValues.inertia.map(Math.abs)), triangles };
    this.mass = found[0];
    this.com = [found[1], found[2], found[3]];
    for (let k = 0; k < 9; ++k) { a[k] = g[k]; }
    this.inertia = new Float32Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
    const det2 = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    const inv2 = [(a[4] * a[8] - a[5] * a[7]) / det2, (a[2] * a[7] - a[1] * a[8]) / det2, (a[1] * a[5] - a[2] * a[4]) / det2,
                  (a[5] * a[6] - a[3] * a[8]) / det2, (a[0] * a[8] - a[2] * a[6]) / det2, (a[2] * a[3] - a[0] * a[5]) / det2,
                  (a[3] * a[7] - a[4] * a[6]) / det2, (a[1] * a[6] - a[0] * a[7]) / det2, (a[0] * a[4] - a[1] * a[3]) / det2];
    this.inverseInertia = new Float32Array([inv2[0], inv2[3], inv2[6], inv2[1], inv2[4], inv2[7], inv2[2], inv2[5], inv2[8]]);
    this.friction = head[14];
    this.restitution = head[15];
    this.rolling = head[16];
    this.h = head[17];
    this.count = head[18];
    this.gpu.floatTexture('wrenchVertices', Float32Array.from(vertices), 4);
    const rows = Math.max(1, Math.ceil(this.count * 2 / 2048));
    this.warm = [this.make(2048, rows), this.make(2048, rows)];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Take the state from a pose of the model (rotation by rows a, translation b), at rest and awake.
  fromPose(a, b) {
    const gl = this.gpu.gl;
    // The centre of mass in the model: the engine's, turned to the model's axes, carried by the pose.
    const c = WrenchPhysics.toModel(this.com);
    const x = [a[0] * c[0] + a[1] * c[1] + a[2] * c[2] + b[0], a[3] * c[0] + a[4] * c[1] + a[5] * c[2] + b[1], a[6] * c[0] + a[7] * c[1] + a[8] * c[2] + b[2]];
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
    this.state.set(WrenchPhysics.toEngine(x), 0);
    this.state.set([q[0], -q[2], q[1], q[3]], 4);   // the rotation's axis turns with the frame
    this.write();
    // Nothing retained.
    for (const target of this.warm) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.accumulator = 0;
  }
  write() {
    const gl = this.gpu.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.states[0].texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 8, 1, gl.RGBA, gl.FLOAT, this.state);
  }
  // The model's pose the state stands for: rotation by rows, then translation.
  pose() {
    const s = this.state, [x, y, z, w] = [s[4], -s[6], s[5], s[7]];   // back to the model's axes: (x, y, z) -> (x, z, -y)
    const qx = s[4], qy = s[6], qz = -s[5];
    const a = [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w),
               2 * (qx * qy + qz * w), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * w),
               2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx * qx + qy * qy)];
    const c = WrenchPhysics.toModel(this.com), centre = WrenchPhysics.toModel([s[0], s[1], s[2]]);
    const b = [centre[0] - (a[0] * c[0] + a[1] * c[1] + a[2] * c[2]), centre[1] - (a[3] * c[0] + a[4] * c[1] + a[5] * c[2]), centre[2] - (a[6] * c[0] + a[7] * c[1] + a[8] * c[2])];
    return { a, b };
  }
  // Put to rest where it is (a reset or a new pose keeps nothing moving).
  wake() {
    this.state[20] = 0;
    this.state[21] = 0;
    this.running = true;
  }
  // Hold the body at a point of the room (model units) that is on it now.
  hold(point) {
    const s = this.state, p = WrenchPhysics.toEngine(point);
    const d = [p[0] - s[0], p[1] - s[1], p[2] - s[2]];
    // Into the body's frame: the inverse rotation.
    const [x, y, z, w] = [-s[4], -s[5], -s[6], s[7]];
    const cx = y * d[2] - z * d[1], cy = z * d[0] - x * d[2], cz = x * d[1] - y * d[0];
    const tx = cx + w * d[0], ty = cy + w * d[1], tz = cz + w * d[2];
    const local = [d[0] + 2 * (y * tz - z * ty), d[1] + 2 * (z * tx - x * tz), d[2] + 2 * (x * ty - y * tx)];
    this.grab = { local, target: p };
    this.wake();
    this.write();
  }
  moveHold(point) { if (this.grab) { this.grab.target = WrenchPhysics.toEngine(point); } }
  // A push: a change of velocity at a point along a direction (the engine's poke).
  poke(point, direction, speed) {
    const s = this.state, p = WrenchPhysics.toEngine(point), d = WrenchPhysics.toEngine(direction).map((x) => x / 0.07);
    const push = d.map((x) => x * speed), arm = [p[0] - s[0], p[1] - s[1], p[2] - s[2]], r2 = this.radius * this.radius;
    s[8] += push[0]; s[9] += push[1]; s[10] += push[2];
    s[12] += (arm[1] * push[2] - arm[2] * push[1]) / r2;
    s[13] += (arm[2] * push[0] - arm[0] * push[2]) / r2;
    s[14] += (arm[0] * push[1] - arm[1] * push[0]) / r2;
    // The engine's poke puts a new body in the old one's place: nothing its contacts retained is kept.
    s.fill(0, 16, 20);
    const gl = this.gpu.gl;
    for (const target of this.warm) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.wake();
    this.write();
  }
  // Advance by dt seconds in the solver's fixed frames. Returns true if the body moved.
  advance(dt) {
    if (!this.running) { return false; }
    const gl = this.gpu.gl, u = this.program.uniforms;
    const started = performance.now();
    this.accumulator += dt;
    let steps = 0;
    gl.useProgram(this.program.program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.gpu.textures.wrenchVertices); gl.uniform1i(u.uVertices, 2);
    gl.uniform1i(u.uCount, this.count);
    gl.uniform1f(u.uMass, this.mass);
    gl.uniform1f(u.uRadius, this.radius);
    gl.uniform1f(u.uFriction, this.friction);
    gl.uniform1f(u.uRestitution, this.restitution);
    gl.uniform1f(u.uRolling, this.rolling);
    gl.uniform1f(u.uH, this.h);
    gl.uniformMatrix3fv(u.uInertia, false, this.inertia);
    gl.uniformMatrix3fv(u.uInverseInertia, false, this.inverseInertia);
    gl.uniform1i(u.uGrab, this.grab ? 1 : 0);
    if (this.grab) {
      gl.uniform3fv(u.uGrabLocal, this.grab.local);
      gl.uniform3fv(u.uGrabTarget, this.grab.target);
    }
    gl.bindVertexArray(this.gpu.emptyVao);
    while (this.accumulator >= this.h && steps < 8) {
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.states[0].texture); gl.uniform1i(u.uState, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.warm[0].texture); gl.uniform1i(u.uWarm, 1);
      // The body's new state, then what each contact point retains; both from the state before.
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.states[1].framebuffer);
      gl.viewport(0, 0, 8, 1);
      gl.uniform1i(u.uMode, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.warm[1].framebuffer);
      gl.viewport(0, 0, this.warm[1].width, this.warm[1].height);
      gl.uniform1i(u.uMode, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.states.reverse();
      this.warm.reverse();
      this.accumulator -= this.h;
      ++steps;
    }
    if (steps === 8) { this.accumulator = 0; }
    if (steps > 0) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.states[0].framebuffer);
      gl.readPixels(0, 0, 8, 1, gl.RGBA, gl.FLOAT, this.state);
      this.contacts = this.state[3];
      if (this.state[21] > 0.5) { this.running = false; }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    this.stepMs = performance.now() - started;
    this.steps = steps;
    return steps > 0;
  }
}
