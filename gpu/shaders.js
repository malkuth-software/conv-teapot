// Shaders of the GPU version. The CONV evaluator below is the CPU's
// (src/conv_line.hpp, src/surface.hpp) written for a shader:
//
//  - The admitted u-cells of every lattice row are compiled once on the CPU
//    and read here from a texture; a row's value at the query is one cell's
//    quintic and bubble.
//  - The v-cell through those eight row values is admitted here, at every
//    query, because its inputs depend on u. Admission is piecewise linear in
//    its inputs: every step is a sum, a multiple, or a choice of one of two
//    values by comparing them. So the derivative along u is carried through
//    it as the second component of a vec2, taking at each choice the
//    derivative of the value chosen. That is what the CPU's dual numbers do.
//  - The derivative along v is the derivative of the basis.
//
// Numbers are 32-bit floats here and 64-bit on the CPU; the page measures
// the difference.
const CONV_GLSL = `
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uCells;   // admitted u-cells: per (row, cell) six texels, two per coordinate
uniform sampler2D uShapes;  // per v-cell two texels: jets (left, right, prev, next); has_prev, has_next, six, rows read
struct Lat { int cells; int shapes; int cu; int cv; int nv; int wrap; };

vec4 lin(sampler2D t, int i) { return texelFetch(t, ivec2(i & 2047, i >> 11), 0); }

struct Basis { float tau[5]; float psi; float dtau[5]; float dpsi; };
Basis makeBasis(float t) {
  Basis b;
  float s = 1.0 - t;
  float t2 = t * t, t3 = t2 * t, t4 = t2 * t2;
  float s2 = s * s, s3 = s2 * s, s4 = s2 * s2;
  b.tau[4] = t4 * t;
  b.tau[3] = b.tau[4] + 5.0 * t4 * s;
  b.tau[2] = b.tau[3] + 10.0 * t3 * s2;
  b.tau[1] = b.tau[2] + 10.0 * t2 * s3;
  b.tau[0] = b.tau[1] + 5.0 * t * s4;
  b.psi = t3 * s3 * (2.0 * t - 1.0);
  // The derivative of a tail of Bernstein polynomials of degree 5 is 5 times one of degree 4.
  b.dtau[0] = 5.0 * s4;
  b.dtau[1] = 20.0 * t * s3;
  b.dtau[2] = 30.0 * t2 * s2;
  b.dtau[3] = 20.0 * t3 * s;
  b.dtau[4] = 5.0 * t4;
  b.dpsi = 3.0 * t2 * s3 * (2.0 * t - 1.0) - 3.0 * t3 * s2 * (2.0 * t - 1.0) + 2.0 * t3 * s3;
  return b;
}

// A value with its derivative along u: choices are made by the value.
vec2 tmin(vec2 a, vec2 b) { return b.x < a.x ? b : a; }
vec2 tmax(vec2 a, vec2 b) { return b.x > a.x ? b : a; }
vec2 pos(vec2 x) { return x.x > 0.0 ? x : vec2(0.0); }
vec2 neg(vec2 x) { return x.x < 0.0 ? -x : vec2(0.0); }

// Scaled jet (m, q) of the node c[3] from its neighbours c[0..6].
void jet(int kind, vec2 c[7], out vec2 m, out vec2 q) {
  if (kind == 0) { // interior
    m = (c[1] - 8.0 * c[2] + 8.0 * c[4] - c[5]) / 12.0;
    q = (16.0 * c[4] - c[5] - 30.0 * c[3] + 16.0 * c[2] - c[1]) / 12.0;
  } else if (kind == 1) { // paper_first
    m = (4.0 * c[4] - 3.0 * c[3] - c[5]) / 2.0;
    q = 2.0 * c[3] - 5.0 * c[4] + 4.0 * c[5] - c[6];
  } else if (kind == 2 || kind == 9) { // paper_central, central3
    m = (c[4] - c[2]) / 2.0;
    q = c[2] - 2.0 * c[3] + c[4];
  } else if (kind == 3) { // paper_last
    m = (3.0 * c[3] - 4.0 * c[2] + c[1]) / 2.0;
    q = 2.0 * c[3] - 5.0 * c[2] + 4.0 * c[1] - c[0];
  } else if (kind == 4) { // first4
    m = (18.0 * c[4] - 11.0 * c[3] - 9.0 * c[5] + 2.0 * c[6]) / 6.0;
    q = 2.0 * c[3] - 5.0 * c[4] + 4.0 * c[5] - c[6];
  } else if (kind == 5) { // second4
    m = (6.0 * c[4] - 2.0 * c[2] - 3.0 * c[3] - c[5]) / 6.0;
    q = c[2] - 2.0 * c[3] + c[4];
  } else if (kind == 6) { // penult4
    m = (2.0 * c[4] + 3.0 * c[3] - 6.0 * c[2] + c[1]) / 6.0;
    q = c[2] - 2.0 * c[3] + c[4];
  } else if (kind == 7) { // last4
    m = (11.0 * c[3] - 18.0 * c[2] + 9.0 * c[1] - 2.0 * c[0]) / 6.0;
    q = 2.0 * c[3] - 5.0 * c[2] + 4.0 * c[1] - c[0];
  } else if (kind == 8) { // first3
    m = (4.0 * c[4] - 3.0 * c[3] - c[5]) / 2.0;
    q = c[3] - 2.0 * c[4] + c[5];
  } else if (kind == 10) { // last3
    m = (3.0 * c[3] - 4.0 * c[2] + c[1]) / 2.0;
    q = c[3] - 2.0 * c[2] + c[1];
  } else if (kind == 11) { // first2
    m = c[4] - c[3];
    q = vec2(0.0);
  } else { // last2
    m = c[3] - c[2];
    q = vec2(0.0);
  }
}

void order(inout vec2 a, inout vec2 b) {
  vec2 low = tmin(a, b), high = tmax(a, b);
  a = low;
  b = high;
}

// The largest of five prefix forms after a nine-comparator sort.
vec2 capacityThreshold(vec2 w[5], vec2 mass) {
  order(w[0], w[1]);
  order(w[3], w[4]);
  order(w[2], w[4]);
  order(w[2], w[3]);
  order(w[1], w[4]);
  order(w[0], w[3]);
  order(w[0], w[2]);
  order(w[1], w[3]);
  order(w[1], w[2]);
  vec2 result = mass / 5.0;
  vec2 sum = w[0];
  result = tmax(result, (mass - sum) / 4.0);
  sum += w[1];
  result = tmax(result, (mass - sum) / 3.0);
  sum += w[2];
  result = tmax(result, (mass - sum) / 2.0);
  sum += w[3];
  result = tmax(result, mass - sum);
  return result;
}

// Admission: one cell of a CONV line from its window w[0..7] (each value with
// a derivative carried beside it): the five admitted currents and the bubble.
void admit(vec2 w[8], vec4 sa, vec4 sb, out vec2 c[5], out vec2 bubble) {
  int left = int(sa.x), right = int(sa.y), prev = int(sa.z), next = int(sa.w);
  bool hasPrev = sb.x > 0.5, hasNext = sb.y > 0.5, six = sb.z > 0.5;
  vec2 cl[7], cr[7];
  for (int k = 0; k < 7; ++k) { cl[k] = w[k]; cr[k] = w[k + 1]; }
  vec2 ml, ql, mr, qr;
  jet(left, cl, ml, ql);
  jet(right, cr, mr, qr);
  vec2 delta = w[4] - w[3];
  vec2 a[5];
  a[0] = ml / 5.0;
  a[1] = ml / 5.0 + ql / 20.0;
  a[3] = mr / 5.0 - qr / 20.0;
  a[4] = mr / 5.0;
  a[2] = delta - a[0] - a[1] - a[3] - a[4];
  vec2 prevDelta = hasPrev ? w[3] - w[2] : vec2(0.0);
  vec2 nextDelta = hasNext ? w[5] - w[4] : vec2(0.0);
  vec2 prevLast = vec2(0.0), nextFirst = vec2(0.0);
  if (hasPrev) {
    vec2 m = ml, q;
    if (prev != left) { jet(prev, cl, m, q); }
    prevLast = m / 5.0;
  }
  if (hasNext) {
    vec2 m = mr, q;
    if (next != right) { jet(next, cr, m, q); }
    nextFirst = m / 5.0;
  }
  vec2 roomPos[5], roomNeg[5];
  vec2 headPos = tmin(pos(prevDelta), pos(pos(a[0]) - neg(prevLast)));
  vec2 headNeg = tmin(neg(prevDelta), pos(neg(a[0]) - pos(prevLast)));
  for (int j = 0; j < 5; ++j) {
    headPos = tmin(headPos, pos(a[j]));
    headNeg = tmin(headNeg, neg(a[j]));
    roomPos[j] = headPos;
    roomNeg[j] = headNeg;
  }
  vec2 tailPos = tmin(pos(nextDelta), pos(pos(a[4]) - neg(nextFirst)));
  vec2 tailNeg = tmin(neg(nextDelta), pos(neg(a[4]) - pos(nextFirst)));
  for (int j = 4; j >= 0; --j) {
    tailPos = tmin(tailPos, pos(a[j]));
    tailNeg = tmin(tailNeg, neg(a[j]));
    roomPos[j] += tailPos;
    roomNeg[j] += tailNeg;
  }
  vec2 totalPos = vec2(0.0), totalNeg = vec2(0.0);
  for (int j = 0; j < 5; ++j) {
    totalPos += roomPos[j];
    totalNeg += roomNeg[j];
  }
  vec2 plus = pos(delta), minus = neg(delta);
  vec2 upperBase = plus + tmin(totalNeg, 4.0 * plus);
  vec2 lowerBase = minus + tmin(totalPos, 4.0 * minus);
  vec2 q[5], down[5], up[5];
  vec2 sum = vec2(0.0);
  for (int j = 0; j < 5; ++j) {
    vec2 high = roomPos[j] + upperBase;
    vec2 low = -(roomNeg[j] + lowerBase);
    q[j] = tmin(high, tmax(low, a[j]));
    down[j] = q[j] - low;
    up[j] = high - q[j];
    sum += q[j];
  }
  vec2 residual = sum - delta;
  bool lower = residual.x > 0.0;
  vec2 capacity[5];
  for (int j = 0; j < 5; ++j) { capacity[j] = lower ? down[j] : up[j]; }
  vec2 mass = lower ? residual : -residual;
  vec2 threshold = capacityThreshold(capacity, mass);
  for (int j = 0; j < 5; ++j) {
    vec2 shift = tmin(threshold, capacity[j]);
    c[j] = lower ? q[j] - shift : q[j] + shift;
  }
  // The bubble, with the continuous bound.
  vec2 fit = vec2(0.0);
  if (six) {
    vec2 fifth = (w[6] - w[1]) + 5.0 * (w[2] - w[5]) + 10.0 * (w[4] - w[3]);
    fit = (-41.0 / 30.0) * fifth;
  }
  fit += 35.0 * (a[0] - c[0]) + 87.5 * (a[1] - c[1]) + 112.5 * (a[2] - c[2]) + 87.5 * (a[3] - c[3]) + 35.0 * (a[4] - c[4]);
  vec2 d2 = (c[0] + 8.0 * c[1] + 6.0 * c[2]) / 3.0;
  vec2 d3 = c[1] + 3.0 * c[2] + c[3];
  vec2 d4 = (6.0 * c[2] + 8.0 * c[3] + c[4]) / 3.0;
  vec2 bound = tmin(tmin(5.0 * tmax(d2, -d2), 2.5 * tmax(d3, -d3)), 5.0 * tmax(d4, -d4));
  bubble = tmin(bound, tmax(-bound, fit));
}

// One v-cell from its window of row values (each with its derivative along
// u), evaluated with the basis bt: value, derivative along u, derivative along v.
vec3 cellAlong(vec2 w[8], vec4 sa, vec4 sb, Basis bt) {
  vec2 c[5], bubble;
  admit(w, sa, sb, c, bubble);
  vec2 value = w[3] + bubble * bt.psi;
  float along = bubble.x * bt.dpsi;
  for (int j = 0; j < 5; ++j) {
    value += c[j] * bt.tau[j];
    along += c[j].x * bt.dtau[j];
  }
  return vec3(value, along);
}

void locate(float x, int cells, bool periodic, out int cell, out float t) {
  float n = float(cells);
  if (periodic) {
    x = x - floor(x / n) * n;
    if (x >= n) { x = 0.0; }
  } else {
    x = clamp(x, 0.0, n);
  }
  float whole = floor(x);
  if (whole >= n) { whole = n - 1.0; }
  cell = int(whole);
  t = x - whole;
}

// A lattice at (u, v) in node units: point and the two derivatives.
void evalLat(Lat L, float u, float v, out vec3 p, out vec3 du, out vec3 dv) {
  int cellU, cellV;
  float s, t;
  locate(u, L.cu, L.wrap != 0, cellU, s);
  locate(v, L.cv, false, cellV, t);
  Basis bs = makeBasis(s), bt = makeBasis(t);
  vec4 sa = lin(uShapes, L.shapes + 2 * cellV), sb = lin(uShapes, L.shapes + 2 * cellV + 1);
  int reads = int(sb.w);
  vec2 wx[8], wy[8], wz[8];
  for (int k = 0; k < 8; ++k) {
    int row = cellV - 3 + k;
    bool used = row >= 0 && row < L.nv && !(k == 0 && (reads & 1) == 0) && !(k == 7 && (reads & 2) == 0);
    if (!used) {
      wx[k] = vec2(0.0); wy[k] = vec2(0.0); wz[k] = vec2(0.0);
      continue;
    }
    int base = L.cells + (row * L.cu + cellU) * 6;
    for (int channel = 0; channel < 3; ++channel) {
      vec4 a = lin(uCells, base + 2 * channel), b = lin(uCells, base + 2 * channel + 1);
      vec2 value = vec2(a.x + a.y * bs.tau[0] + a.z * bs.tau[1] + a.w * bs.tau[2] + b.x * bs.tau[3] + b.y * bs.tau[4] + b.z * bs.psi,
                        a.y * bs.dtau[0] + a.z * bs.dtau[1] + a.w * bs.dtau[2] + b.x * bs.dtau[3] + b.y * bs.dtau[4] + b.z * bs.dpsi);
      if (channel == 0) { wx[k] = value; } else if (channel == 1) { wy[k] = value; } else { wz[k] = value; }
    }
  }
  vec3 x = cellAlong(wx, sa, sb, bt), y = cellAlong(wy, sa, sb, bt), z = cellAlong(wz, sa, sb, bt);
  p = vec3(x.x, y.x, z.x);
  du = vec3(x.y, y.y, z.y);
  dv = vec3(x.z, y.z, z.z);
}

// The two other surfaces the page offers through the same nodes: the nodes
// joined bilinearly, and the original bicubic Bezier patches.
uniform sampler2D uNodes, uPatches;
struct Flat { int base; int nu; int nv; int wrap; };
struct Bezier { int base; int pu; int pv; int peru; int perv; int wrap; };
vec3 nodeAt(Flat f, int i, int j) { return lin(uNodes, f.base + j * f.nu + i).xyz; }
void evalFlat(Flat f, float u, float v, out vec3 p, out vec3 du, out vec3 dv) {
  int cu, cv;
  float s, t;
  locate(u, f.wrap != 0 ? f.nu : f.nu - 1, f.wrap != 0, cu, s);
  locate(v, f.nv - 1, false, cv, t);
  int u1 = f.wrap != 0 ? (cu + 1) % f.nu : cu + 1;
  vec3 a = nodeAt(f, cu, cv), b = nodeAt(f, u1, cv), c = nodeAt(f, cu, cv + 1), d = nodeAt(f, u1, cv + 1);
  p = mix(mix(a, b, s), mix(c, d, s), t);
  du = mix(b - a, d - c, t);
  dv = mix(c - a, d - b, s);
}
void evalBezier(Bezier z, float u, float v, out vec3 p, out vec3 du, out vec3 dv) {
  int pu, pv;
  float s, t;
  locate(u / float(z.peru), z.pu, z.wrap != 0, pu, s);
  locate(v / float(z.perv), z.pv, false, pv, t);
  float a = 1.0 - s, b = 1.0 - t;
  float bs[4] = float[4](a * a * a, 3.0 * a * a * s, 3.0 * a * s * s, s * s * s);
  float ds[4] = float[4](-3.0 * a * a, 3.0 * a * a - 6.0 * a * s, 6.0 * a * s - 3.0 * s * s, 3.0 * s * s);
  float bt[4] = float[4](b * b * b, 3.0 * b * b * t, 3.0 * b * t * t, t * t * t);
  float dt[4] = float[4](-3.0 * b * b, 3.0 * b * b - 6.0 * b * t, 6.0 * b * t - 3.0 * t * t, 3.0 * t * t);
  int base = z.base + (pv * z.pu + pu) * 16;
  p = vec3(0.0); du = vec3(0.0); dv = vec3(0.0);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      vec3 c = lin(uPatches, base + i * 4 + j).xyz;
      p += c * (bs[i] * bt[j]);
      du += c * (ds[i] * bt[j]);
      dv += c * (bs[i] * dt[j]);
    }
  }
  du /= float(z.peru);
  dv /= float(z.perv);
}

// A part of the teapot. A band between two parts keeps where its two sides
// are on those parts, and is the blend of them with a quintic weight.
uniform Lat uOwn, uToward, uOn, uTube;
uniform int uBand;
uniform float uSpread;
uniform int uKind;        // 0: CONV; 1: the Bezier patches; 2: the nodes joined bilinearly
uniform Flat uFlat;
uniform Bezier uBezier;
void evalSurface(float u, float v, out vec3 p, out vec3 du, out vec3 dv) {
  if (uKind == 1) { evalBezier(uBezier, u, v, p, du, dv); return; }
  if (uKind == 2) { evalFlat(uFlat, u, v, p, du, dv); return; }
  if (uBand == 0) {
    evalLat(uOwn, u, v, p, du, dv);
    return;
  }
  float scale = 1.0 / uSpread;
  vec3 ap, adu, adv, bp, bdu, bdv;
  evalLat(uOwn, u, v * scale, ap, adu, adv);
  evalLat(uToward, u, v * scale, bp, bdu, bdv);
  adv *= scale;
  bdv *= scale;
  vec3 Ap, Adu, Adv, Bp, Bdu, Bdv;
  evalLat(uOn, ap.x, ap.y, Ap, Adu, Adv);
  evalLat(uTube, u, bp.x, Bp, Bdu, Bdv);
  vec3 sideAdu = Adu * adu.x + Adv * adu.y, sideAdv = Adu * adv.x + Adv * adv.y;
  vec3 sideBdu = Bdu + Bdv * bdu.x, sideBdv = Bdv * bdv.x;
  float rows = float(uOwn.cv) * uSpread;
  float tau = clamp(v / rows, 0.0, 1.0);
  float beta = tau * tau * tau * (10.0 + tau * (6.0 * tau - 15.0));
  float slope = 30.0 * tau * tau * (1.0 - tau) * (1.0 - tau) / rows;
  p = Ap + (Bp - Ap) * beta;
  du = sideAdu + (sideBdu - sideAdu) * beta;
  dv = sideAdv + (sideBdv - sideAdv) * beta + (Bp - Ap) * slope;
}
`;

const VERTEX_GLSL = `#version 300 es
${CONV_GLSL}
layout(location = 0) in vec3 aUvs;     // u, v in node units; which side of a crease the vertex belongs to
layout(location = 1) in vec3 aNormal;  // the baked mesh's normal at rest: used at a pole, where the surface has none
uniform mat3 uPose;
uniform vec3 uShift;
uniform mat4 uViewProj;
uniform vec3 uEyeAt;
uniform float uNearer;   // a band is drawn where it and the part it lies on are both there: projected this fraction nearer the eye
uniform int uProbe;      // > 0: the vertices are written to a row of that many pixels, for reading the surface back
out vec3 vWorld;
out vec3 vNormal;
out vec3 vUvs;
void main() {
  vec3 p, du, dv;
  evalSurface(aUvs.x, aUvs.y, p, du, dv);
  if (aUvs.z != 0.0) {
    vec3 q;
    evalSurface(aUvs.x, aUvs.y + aUvs.z * 1e-4, q, du, dv);
  }
  vec3 n = cross(du, dv);
  n = dot(n, n) > 1e-18 ? normalize(n) : aNormal;
  vWorld = uPose * p + uShift;
  vNormal = uPose * n;
  vUvs = aUvs;
  if (uProbe > 0) {
    gl_Position = vec4((float(gl_VertexID) + 0.5) / float(uProbe) * 2.0 - 1.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    return;
  }
  gl_Position = uViewProj * vec4(vWorld + (uEyeAt - vWorld) * uNearer, 1.0);
}
`;

// What every pass that draws the teapot decides first: is this point of the part there at all?
const TRIM_GLSL = `
uniform sampler2D uTrimData;
uniform int uTrimCount;
uniform vec4 uTrimA[6];   // kind (0 loop, 1 cut), offset, count, below
uniform vec4 uTrimB[6];   // loop: centre u, period u
uniform vec4 uTrimC[6];   // loop: bounds of u and of v
uniform int uCut;
uniform vec4 uCutPlane;
float trimValue(int i) { return texelFetch(uTrimData, ivec2(i & 2047, i >> 11), 0).r; }
bool hidden(vec3 world, float u, float v) {
  if (uCut != 0 && dot(uCutPlane.xyz, world) > uCutPlane.w) { return true; }
  for (int k = 0; k < uTrimCount; ++k) {
    int offset = int(uTrimA[k].y), count = int(uTrimA[k].z);
    if (uTrimA[k].x > 0.5) {
      float n = float(count);
      float x = u - floor(u / n) * n;
      int i = int(x) % count;
      float f = x - floor(x);
      float c = mix(trimValue(offset + i), trimValue(offset + (i + 1) % count), f);
      if (uTrimA[k].w > 0.5 ? v < c : v > c) { return true; }
      continue;
    }
    float x = u;
    if (uTrimB[k].y > 0.0) { x -= uTrimB[k].y * floor((x - uTrimB[k].x) / uTrimB[k].y + 0.5); }
    if (x < uTrimC[k].x || x > uTrimC[k].y || v < uTrimC[k].z || v > uTrimC[k].w) { continue; }
    bool inside = false;
    vec2 pj = vec2(trimValue(offset + 2 * (count - 1)), trimValue(offset + 2 * (count - 1) + 1));
    for (int i = 0; i < count; ++i) {
      vec2 pi = vec2(trimValue(offset + 2 * i), trimValue(offset + 2 * i + 1));
      if ((pi.y > v) != (pj.y > v) && x < (pj.x - pi.x) * (v - pi.y) / (pj.y - pi.y) + pi.x) { inside = !inside; }
      pj = pi;
    }
    if (inside) { return true; }
  }
  return false;
}
`;

// The room, as the CPU renderer has it (src/render.hpp): sky and window,
// floor pattern averaged over a pixel, shadow map lookup.
const ROOM_GLSL = `
uniform vec3 uLight;        // direction towards the light
uniform vec3 uEye;
uniform float uGround;
uniform int uShadows;
uniform sampler2D uShadowMap;   // distance along the light, nearest the light
uniform vec3 uShadowX, uShadowY, uShadowCentre;
uniform float uShadowRadius, uShadowSize;

float smooth01(float a, float b, float x) {
  float t = clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
float fifth(float x) { float x2 = x * x; return x2 * x2 * x; }
float ninetieth(float x) {
  float x2 = x * x, x4 = x2 * x2, x8 = x4 * x4, x16 = x8 * x8, x32 = x16 * x16, x64 = x32 * x32;
  return x64 * x16 * x8 * x2;
}
vec3 ambientOf(vec3 n) {
  float up = 0.5 * (n.y + 1.0);
  return mix(vec3(0.30, 0.28, 0.26), vec3(0.46, 0.50, 0.56), up);
}
vec3 backgroundOf(vec3 dir) {
  float t = 0.5 * (normalize(dir).y + 1.0);
  return mix(vec3(0.84, 0.86, 0.89), vec3(0.93, 0.95, 0.98), t);
}
vec3 environmentOf(vec3 d) {
  float t = 0.5 * (d.y + 1.0);
  vec3 sky = mix(vec3(0.20, 0.24, 0.30), vec3(0.78, 0.84, 0.92), t);
  if (d.y < 0.0) { sky = vec3(0.16, 0.15, 0.14) * (1.0 + d.y) + vec3(0.08) * (-d.y); }
  float r = sqrt(d.x * d.x + d.z * d.z);
  float c = r > 1e-12 ? d.x / r : 1.0, sn = r > 1e-12 ? d.z / r : 0.0;
  float cos3 = 4.0 * c * c * c - 3.0 * c, sin3 = 3.0 * sn - 4.0 * sn * sn * sn;
  float lobe = cos3 * 0.8253356149096783 - sin3 * 0.5646424733950354;
  float window = exp(-18.0 * (d.y - 0.55) * (d.y - 0.55)) * (0.5 + 0.5 * lobe);
  return sky + vec3(1.0, 0.98, 0.92) * (0.9 * window * window);
}
// The area emitter and sky seen along a direction.
vec3 skyOf(vec3 origin, vec3 dir) {
  vec3 d = normalize(dir);
  float t = 0.5 * (d.y + 1.0);
  vec3 colour = d.y >= 0.0 ? mix(vec3(0.50, 0.56, 0.66), vec3(0.80, 0.86, 0.95), t)
                           : vec3(0.30, 0.31, 0.33) * (1.0 + d.y) + vec3(0.12, 0.12, 0.13) * (-d.y);
  vec3 centre = vec3(0.0, 1.3, 0.0) + uLight * 14.0;
  float facing = -dot(d, uLight);
  if (facing < -1e-6) {
    float reach = dot(centre - origin, uLight) / dot(d, uLight);
    if (reach > 0.0) {
      vec3 local = origin + d * reach - centre;
      vec3 helper = abs(uLight.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
      vec3 ax = normalize(cross(helper, uLight)), ay = cross(uLight, ax);
      float x = abs(dot(local, ax)), y = abs(dot(local, ay));
      float window = (1.0 - smooth01(3.6, 4.2, x)) * (1.0 - smooth01(3.6, 4.2, y));
      float bars = smooth01(0.10, 0.22, x) * smooth01(0.10, 0.22, y);
      colour += vec3(4.2, 4.0, 3.7) * (window * (0.25 + 0.75 * bars));
    }
  }
  return colour;
}
// The floor's checker averaged over a square of side width about p.
float triangleWave(float t) {
  float m = t - 2.0 * floor(0.5 * t);
  return m < 1.0 ? m : 2.0 - m;
}
float squareWave(float x, float w) {
  if (!(w > 1e-6)) { return (x - 2.0 * floor(0.5 * x)) < 1.0 ? 1.0 : -1.0; }
  return (triangleWave(x + 0.5 * w) - triangleWave(x - 0.5 * w)) / w;
}
vec3 checkerOf(vec3 p, float width) {
  float m = 0.5 + 0.5 * squareWave(p.x, width) * squareWave(p.z, width);
  return mix(vec3(0.50, 0.52, 0.55), vec3(0.66, 0.68, 0.71), m);
}
// The fraction of the light that reaches p: the four texels around p, each
// tested, blended by where p lies between their centres.
float litAt(vec3 p, float bias) {
  if (uShadows == 0) { return 1.0; }
  vec3 d = p - uShadowCentre;
  float reach = uShadowRadius * (1.0 + 4.0 / uShadowSize);
  if (abs(dot(d, uShadowX)) > reach || abs(dot(d, uShadowY)) > reach) { return 1.0; }
  float sx = (dot(d, uShadowX) / uShadowRadius * 0.5 + 0.5) * uShadowSize;
  float sy = (dot(d, uShadowY) / uShadowRadius * 0.5 + 0.5) * uShadowSize;
  float sz = -dot(d, uLight);
  float x = sx - 0.5, y = sy - 0.5;
  float fx = floor(x), fy = floor(y);
  float ax = x - fx, ay = y - fy;
  ivec2 base = ivec2(int(fx), int(fy));
  float open[4];
  for (int k = 0; k < 4; ++k) {
    ivec2 at = base + ivec2(k & 1, k >> 1);
    open[k] = (at.x < 0 || at.y < 0 || at.x >= int(uShadowSize) || at.y >= int(uShadowSize)) ? 1.0
              : (sz - bias <= texelFetch(uShadowMap, at, 0).r ? 1.0 : 0.0);
  }
  return mix(mix(open[0], open[1], ax), mix(open[2], open[3], ax), ay);
}
float shadowBias(float nl) { return 0.02 + 0.02 * min(12.0, sqrt(max(0.0, 1.0 - nl * nl)) / nl); }
vec3 floorColour(vec3 p, float width, float reach, float lit) {
  vec3 base = checkerOf(p, width);
  float light = 0.55 + 0.45 * lit * max(0.0, uLight.y);
  float fade = smooth01(18.0, 60.0, reach);
  return mix(base * light, vec3(0.80, 0.84, 0.90), fade);
}
vec4 encode(vec3 linear) { return vec4(sqrt(clamp(linear, 0.0, 1.0)), 1.0); }   // gamma 2, as the CPU's bytes
`;

// Rays traced in the shader against the teapot. The teapot is a grid of
// surface points the GPU has evaluated, one grid per part, side by side in
// one texture (TeapotGpu.bakeSurface). Over the cells of that grid stands a
// pyramid of boxes, each level a quarter of the one below, made on the GPU
// by taking the smallest and largest corner of four boxes at a time. A ray
// is taken into the teapot's frame and walks down the pyramid, nearer box
// first, to the cells it may meet; a cell is two triangles.
const TRACE_GLSL = `
// The tree over the mesh, in the order a ray walks it (mesh.js): one texel of four whole numbers a node, its box in
// steps of the teapot's own box and the node to go on from on a miss (or, for one pair of triangles, the first of them).
uniform highp usampler2D uTree;
uniform vec3 uTreeLow;
uniform float uTreeStep;
uniform sampler2D uMeshPoints, uMeshNormals, uMeshWhere;   // three texels a triangle: point and shown; normal; u, v, part, cell
uniform int uTreeCount, uStepCap;
uniform mat3 uTracePose;      // the teapot's rotation
uniform vec3 uTraceShift;
uniform int uTraceCut;        // the teapot is cut: what is beyond the plane through its origin is not there
uniform vec3 uTraceCutNormal; // in the teapot's frame
uniform int uTraceOn;
vec4 listAt(sampler2D list, int i) { return texelFetch(list, ivec2(i & 2047, i >> 11), 0); }
int traceSteps = 0, traceLeaves = 0;   // counted for the page's measurements
bool meetTriangle(vec3 o, vec3 d, vec3 p0, vec3 p1, vec3 p2, float tMin, float tMax, out float t, out vec2 weights) {
  vec3 e1 = p1 - p0, e2 = p2 - p0;
  vec3 pv = cross(d, e2);
  float det = dot(e1, pv);
  if (abs(det) < 1e-14) { return false; }
  vec3 tv = o - p0;
  float wu = dot(tv, pv) / det;
  vec3 qv = cross(tv, e1);
  float wv = dot(d, qv) / det;
  if (wu < 0.0 || wv < 0.0 || wu + wv > 1.0) { return false; }
  t = dot(e2, qv) / det;
  weights = vec2(wu, wv);
  return t > tMin && t < tMax;
}
// The nearest triangle along a ray of the room between tMin and tMax (or any, for a shadow).
// The walk keeps no stack: a node the ray meets is followed by its first child, which is the
// next node; one it misses names the node to go on from.
// 'from', 'onFrom' and 'bands': the nodes of the part the ray leaves (first, and the one after its last) and the bands of
// that part's rows the ray may meet, as bits; the others it is known to be clear of. from -1: nothing is known.
// 'ahead': the tangent plane the ray leaves (unit normal on the ray's side, and normal . point), or zero. The ray stays in
// front of it, so a box wholly behind it cannot be met.
bool traceRay(vec3 origin, vec3 direction, float tMin, float tMax, bool any, int from, int onFrom, int bands, vec4 ahead, out float tHit, out int triangle, out vec2 weights) {
  vec3 o = (origin - uTraceShift) * uTracePose, d = direction * uTracePose;   // into the teapot's frame
  vec3 facing = ahead.xyz * uTracePose;
  float level = ahead.w - dot(ahead.xyz, uTraceShift);
  vec3 inv = 1.0 / mix(d, vec3(1e-20), lessThan(abs(d), vec3(1e-20)));
  tHit = tMax;
  triangle = -1;
  int node = 0;
  for (int step = 0; step < uStepCap && node < uTreeCount; ++step) {
    if (node == from && bands == 0) { node = onFrom; continue; }
    uvec4 packed = texelFetch(uTree, ivec2(node & 2047, node >> 11), 0);
    ++traceSteps;
    int kind = int(packed.a >> 24), link = int(packed.a & 0xFFFFFFu);
    bool pair = kind >= 128;
    int onMiss = pair ? node + 1 : link;
    // A block of the part the ray leaves that it is known to be clear of.
    if (kind > 0 && !pair && node > from && node < onFrom && ((bands >> (kind - 1)) & 1) == 0) { node = onMiss; continue; }
    vec3 low = uTreeLow + uTreeStep * vec3(packed.r & 0xFFFFu, packed.r >> 16, packed.g & 0xFFFFu);
    vec3 high = uTreeLow + uTreeStep * vec3(packed.g >> 16, packed.b & 0xFFFFu, packed.b >> 16);
    vec3 t0 = (low - o) * inv, t1 = (high - o) * inv;
    vec3 a = min(t0, t1), b = max(t0, t1);
    float near = max(max(a.x, a.y), max(a.z, tMin)), far = min(min(b.x, b.y), min(b.z, tHit));
    // Missed, empty, or wholly behind the plane the ray leaves.
    vec3 reach = max(facing * low, facing * high);
    if (low.x > high.x || near > far || reach.x + reach.y + reach.z < level - 1e-4) { node = onMiss; continue; }
    if (pair) {
      // One pair of triangles.
      ++traceLeaves;
      for (int k = 0; k < 2; ++k) {
        int at = 3 * (link + k);
        vec4 p0 = listAt(uMeshPoints, at), p1 = listAt(uMeshPoints, at + 1), p2 = listAt(uMeshPoints, at + 2);
        float t;
        vec2 w;
        if (p0.w + p1.w + p2.w < 0.5 || !meetTriangle(o, d, p0.xyz, p1.xyz, p2.xyz, tMin, tHit, t, w)) { continue; }
        if (uTraceCut != 0 && dot(uTraceCutNormal, o + d * t) > 0.0) { continue; }
        tHit = t;
        triangle = link + k;
        weights = w;
        if (any) { return true; }
      }
    }
    node += 1;
  }
  return triangle >= 0;
}
// The lit fraction at p as the CPU finds it: where the shadow map's texels
// around p agree, their answer; where they differ, light rays traced from a
// rotated grid inside the pixel (dx, dy: the pixel's extent at p).
float litTraced(vec3 p, float bias, vec3 dx, vec3 dy) {
  if (uShadows == 0) { return 1.0; }
  float coarse = litAt(p, bias);
  if (uTraceOn == 0) { return coarse; }
  vec3 d = p - uShadowCentre;
  float reach = uShadowRadius * (1.0 + 4.0 / uShadowSize);
  if (abs(dot(d, uShadowX)) > reach || abs(dot(d, uShadowY)) > reach) { return 1.0; }
  ivec2 at = ivec2(floor((vec2(dot(d, uShadowX), dot(d, uShadowY)) / uShadowRadius * 0.5 + 0.5) * uShadowSize));
  float sz = -dot(d, uLight);
  int open = 0, shut = 0;
  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      ivec2 texel = at + ivec2(i, j);
      bool outside = texel.x < 0 || texel.y < 0 || texel.x >= int(uShadowSize) || texel.y >= int(uShadowSize);
      if (outside || sz - bias <= texelFetch(uShadowMap, texel, 0).r) { ++open; } else { ++shut; }
    }
  }
  if (shut == 0) { return 1.0; }
  if (open == 0) { return 0.0; }
  float t;
  int triangle;
  vec2 weights;
  float first = traceRay(p + dx * -0.375 + dy * -0.125, uLight, 1e-3, 1e30, true, -1, 0, 0, vec4(0.0), t, triangle, weights) ? 0.0 : 1.0;
  float second = traceRay(p + dx * 0.375 + dy * 0.125, uLight, 1e-3, 1e30, true, -1, 0, 0, vec4(0.0), t, triangle, weights) ? 0.0 : 1.0;
  if (first == second) { return first; }
  float third = traceRay(p + dx * 0.125 + dy * -0.375, uLight, 1e-3, 1e30, true, -1, 0, 0, vec4(0.0), t, triangle, weights) ? 0.0 : 1.0;
  float fourth = traceRay(p + dx * -0.125 + dy * 0.375, uLight, 1e-3, 1e30, true, -1, 0, 0, vec4(0.0), t, triangle, weights) ? 0.0 : 1.0;
  return 0.25 * (first + second + third + fourth);
}
`;

const FRAGMENT_GLSL = `#version 300 es
${CONV_GLSL}
${TRIM_GLSL}
${ROOM_GLSL}
${TRACE_GLSL}
in vec3 vWorld;
in vec3 vNormal;
in vec3 vUvs;
in float vClear;
uniform int uOwnRoot, uOwnNext;   // the part's nodes in the tree
uniform int uTricks;              // for measurements: 1 the blocks a cell is clear of are skipped, 2 boxes behind the tangent plane are skipped
uniform mat3 uPose;
uniform int uMode;          // 0 picture, 1 distance along the light, 2 which part and where, 3 point, 4 normal, 5 presence
uniform float uPartIndex;
uniform float uOrientation, uInteriorFrom;
uniform int uTwoSided, uPerPixel, uOnBand, uCountSteps;
uniform int uMaterial, uGloss, uLighting, uWire, uTextureOn, uMapping, uGroundOn;
uniform Lat uTexture;
uniform vec2 uPer;          // node intervals per source patch
uniform vec2 uCellCount;    // the part's cells as presented
uniform vec3 uAxis;
uniform int uOpenTop;
uniform int uRoomLight;     // 0: by the normal alone; 1: through what each point sees of the room
uniform sampler2D uOpenFront, uOpenBack;   // per texel: mean open direction (unnormalised) and open weight
uniform vec4 uRect;
uniform float uOpenScale;   // 1 / (the weight a fully open texel has)
uniform sampler2D uThrownFront, uThrownBack;   // the light the body throws on each texel (gathered)
uniform sampler2D uGridNormals;                // per texel of the maps: the surface's normal
uniform vec2 uGridSpan;                        // the part's grid: steps around and along
uniform int uImage;                            // drawing the image in a polished floor: a second bounce shows only the room
out vec4 outColour;

// Where a point of a part is in the maps laid over the parts: the grid has a texel at every step.
vec2 mapPlace(vec2 uv, vec2 cells, vec2 span, vec4 rect) {
  vec2 grid = vec2(uv.x - floor(uv.x / cells.x) * cells.x, uv.y) / cells * span + 0.5;
  return (clamp(grid, vec2(0.5), rect.zw - 0.5) + rect.xy) / vec2(textureSize(uOpenFront, 0));
}
vec2 mapAt() { return mapPlace(vUvs.xy, uCellCount, uGridSpan, uRect); }

// The texture's colour at (x, y): its lattice of colours through CONV, or
// joined bilinearly, or the nearest node (the page's three filters).
uniform int uFilter;      // 0 nearest, 1 bilinear, 2 CONV
uniform Flat uTextureFlat;
vec3 textureColour(float tx, float ty) {
  float u = (tx - floor(tx)) * float(uTexture.cu), v = clamp(ty, 0.0, 1.0) * float(uTexture.cv);
  vec3 c, cu, cv;
  if (uFilter == 2) {
    evalLat(uTexture, u, v, c, cu, cv);
  } else if (uFilter == 1) {
    evalFlat(uTextureFlat, u, v, c, cu, cv);
  } else {
    c = nodeAt(uTextureFlat, int(floor(u + 0.5)) % uTextureFlat.nu, min(uTextureFlat.nv - 1, int(floor(v + 0.5))));
  }
  return clamp(c, 0.0, 1.0);
}

bool facesCavity(float v) { return uInteriorFrom >= 0.0 ? v >= uInteriorFrom : v <= -uInteriorFrom - 1.0; }
bool leavesCavity(vec3 d) { return (uCut != 0 && dot(d, uCutPlane.xyz) > 0.15) || (uOpenTop != 0 && dot(d, uAxis) > 0.6); }
float mirrorDim(float lit, float nl) { return 1.0 - 0.5 * (1.0 - lit) * smooth01(0.0, 0.3, nl); }
float lineInk(float coordinate, float period, float gradient, float halfWidth) {
  float scaled = coordinate / period;
  float distance = abs(scaled - floor(scaled + 0.5)) * period / max(gradient, 1e-12);
  return 1.0 - smooth01(halfWidth - 0.5, halfWidth + 0.5, distance);
}

// Per part, four texels: orientation, interior from, two-sided, 0; cells u, cells v, per u, per v; its rectangle in the maps; its grid's steps around and along.
uniform sampler2D uPartTable;
// A point of the teapot reached by a mirror ray, shaded as the CPU shades a
// surface seen at second hand: its own colour, the sun and the room, and for
// its own gloss only the room.
vec3 shadeHit(vec3 point, vec3 direction, int triangle, vec2 weights) {
  vec4 where0 = listAt(uMeshWhere, 3 * triangle), where1 = listAt(uMeshWhere, 3 * triangle + 1), where2 = listAt(uMeshWhere, 3 * triangle + 2);
  int part = int(where0.z);
  vec4 info = texelFetch(uPartTable, ivec2(part * 4, 0), 0), cells = texelFetch(uPartTable, ivec2(part * 4 + 1, 0), 0), rect = texelFetch(uPartTable, ivec2(part * 4 + 2, 0), 0);
  vec2 span = texelFetch(uPartTable, ivec2(part * 4 + 3, 0), 0).xy;
  // Where the hit is on its part, and the normal there, between the triangle's corners.
  float wa = 1.0 - weights.x - weights.y;
  vec2 uv = where0.xy * wa + where1.xy * weights.x + where2.xy * weights.y;
  vec3 raw = normalize(uPose * (listAt(uMeshNormals, 3 * triangle).xyz * wa + listAt(uMeshNormals, 3 * triangle + 1).xyz * weights.x + listAt(uMeshNormals, 3 * triangle + 2).xyz * weights.y));
  vec3 normal = raw * info.x;
  vec3 incoming = normalize(direction), toEye = -incoming;
  bool back = dot(normal, toEye) < 0.0;
  if (back) { normal = -normal; }
  bool inside = back && info.z < 0.5;
  vec3 base_colour = vec3(0.93, 0.92, 0.89);
  if (uTextureOn != 0) {
    float tx, ty;
    if (uMapping == 1) {
      tx = -uv.x / max(1.0, cells.z);
      ty = uv.y / max(1.0, cells.w);
      ty = ty - floor(ty);
    } else {
      tx = 0.70 - uv.x / cells.x * (cells.x >= 16.0 ? 3.0 : 1.0);
      ty = uv.y / cells.y;
    }
    base_colour = textureColour(tx, ty);
  }
  if (inside && uRoomLight == 0) { base_colour *= 0.55; }
  if (uLighting == 0) { return base_colour; }
  float nl = dot(normal, uLight);
  float lit = nl > 0.0 ? litAt(point + normal * 0.02, shadowBias(nl)) : 1.0;
  vec3 ambient = ambientOf(normal);
  if (uRoomLight != 0) {
    vec2 at = mapPlace(uv, cells.xy, span, rect);
    bool backSide = dot(raw, toEye) < 0.0;
    vec4 seen = backSide ? texture(uOpenBack, at) : texture(uOpenFront, at);
    vec3 bent = dot(seen.xyz, seen.xyz) > 1e-8 ? uPose * normalize(seen.xyz) : normal;
    ambient = ambientOf(bent) * clamp(seen.a * uOpenScale, 0.0, 1.0);
    if (uRoomLight > 1) { ambient += backSide ? texture(uThrownBack, at).rgb : texture(uThrownFront, at).rgb; }
  }
  vec3 colour = base_colour * (ambient + vec3(0.80, 0.78, 0.74) * max(0.0, nl) * lit);
  bool chrome = uMaterial == 1 && uTextureOn == 0;
  float f0 = chrome ? 0.86 : 0.05;
  float fresnel = f0 + (1.0 - f0) * fifth(1.0 - max(0.0, dot(normal, toEye)));
  vec3 reflected = reflect(incoming, normal);
  if (uGloss != 0) {
    vec3 tint = chrome ? vec3(0.95, 0.96, 0.98) : vec3(1.0);
    vec3 under = chrome ? vec3(0.0) : colour;
    return under * (1.0 - fresnel) + skyOf(point + normal * 2e-3, reflected) * tint * (fresnel * mirrorDim(lit, nl));
  }
  vec3 halfway = normalize(uLight + toEye);
  return colour * (1.0 - 0.6 * fresnel) + environmentOf(reflected) * (0.6 * fresnel) + vec3(0.55 * ninetieth(max(0.0, dot(normal, halfway))) * lit);
}

void main() {
  if (uMode == 3) { outColour = vec4(vWorld, 1.0); return; }
  if (uMode == 4) { outColour = vec4(vNormal, 1.0); return; }
  if (hidden(vWorld, vUvs.x, vUvs.y)) { discard; }
  if (uMode == 5) { outColour = vec4(1.0); return; }   // counted, not drawn
  if (uMode == 1) {
    outColour = vec4(-dot(vWorld - uShadowCentre, uLight), 0.0, 0.0, 1.0);
    return;
  }
  if (uMode == 2) {
    outColour = vec4(uPartIndex + 1.0, vUvs.x, vUvs.y, 1.0);
    return;
  }
  // The normal: the surface's own at this pixel, or the one carried from the vertices.
  vec3 normal = normalize(vNormal);
  if (uPerPixel == 1) {
    // From the map of the surface's normals laid over the part, between its four nearest texels.
    vec2 grid = vec2(vUvs.x - floor(vUvs.x / uCellCount.x) * uCellCount.x, vUvs.y) / uCellCount * uGridSpan;
    ivec2 corner = ivec2(floor(grid)), size = ivec2(uRect.zw);
    vec2 f = grid - vec2(corner);
    bool closes = size.x == int(uGridSpan.x);
    int x0 = clamp(corner.x, 0, size.x - 1), x1 = closes ? (corner.x + 1) % size.x : min(corner.x + 1, size.x - 1);
    int y0 = clamp(corner.y, 0, size.y - 1), y1 = min(corner.y + 1, size.y - 1);
    ivec2 origin = ivec2(uRect.xy);
    vec3 n = mix(mix(texelFetch(uGridNormals, origin + ivec2(x0, y0), 0).xyz, texelFetch(uGridNormals, origin + ivec2(x1, y0), 0).xyz, f.x),
                 mix(texelFetch(uGridNormals, origin + ivec2(x0, y1), 0).xyz, texelFetch(uGridNormals, origin + ivec2(x1, y1), 0).xyz, f.x), f.y);
    if (dot(n, n) > 1e-12) { normal = uPose * normalize(n); }
  } else if (uPerPixel == 2) {
    // The surface's own normal, evaluated here.
    vec3 p, du, dv;
    float side = vUvs.z > 1e-6 ? 1.0 : vUvs.z < -1e-6 ? -1.0 : 0.0;
    evalSurface(vUvs.x, vUvs.y + side * 1e-4, p, du, dv);
    vec3 n = cross(du, dv);
    if (dot(n, n) > 1e-18) { normal = uPose * normalize(n); }
  }
  normal *= uOrientation;
  vec3 incoming = normalize(vWorld - uEye), toEye = -incoming;
  bool back = dot(normal, toEye) < 0.0;
  if (back) { normal = -normal; }
  bool inside = back && uTwoSided == 0;

  vec3 base = vec3(0.93, 0.92, 0.89);
  if (uTextureOn != 0) {
    float tx, ty;
    if (uMapping == 1) {
      tx = -vUvs.x / max(1.0, uPer.x);
      ty = vUvs.y / max(1.0, uPer.y);
      ty = ty - floor(ty);
    } else {
      tx = 0.70 - vUvs.x / uCellCount.x * (uCellCount.x >= 16.0 ? 3.0 : 1.0);
      ty = vUvs.y / uCellCount.y;
    }
    base = textureColour(tx, ty);
  }
  bool chrome = uMaterial == 1 && uTextureOn == 0;
  if (uMaterial == 2 && uTextureOn == 0 && uWire != 2) {
    // Reflection lines: stripes of the mirror direction.
    vec3 reflected = reflect(incoming, normal);
    float stripe = sin(14.0 * atan(reflected.y, length(reflected.xz)) + 9.0 * atan(reflected.z, reflected.x));
    float ink = smooth01(-0.12, 0.12, stripe);
    float shade = 0.55 + 0.45 * max(0.0, dot(normal, toEye));
    outColour = encode(mix(vec3(0.06, 0.07, 0.09), vec3(0.96, 0.96, 0.94), ink) * shade);
    return;
  }
  if (inside && uRoomLight == 0) { base *= 0.55; }
  if (uWire == 2) { base = vec3(0.97, 0.97, 0.96); }

  vec3 colour = base;
  vec3 debugHit = vec3(0.0);
  int shadowSteps = 0;
  if (uLighting != 0) {
    float nl = dot(normal, uLight);
    float lit = 1.0;
    if (nl > 0.0) {
      // The shadow map's own answer toward the terminator, where a traced ray
      // meets the surface's own neighbourhood at a glancing angle; the traced
      // one where the surface faces the light squarely or where something
      // well clear of it is in the way.
      float bias = shadowBias(nl);
      lit = litAt(vWorld + normal * 0.02, bias);
      float trust = smooth01(0.2, 0.5, nl);
      if (uTraceOn != 0 && trust < 1.0 && litAt(vWorld + normal * 0.02, 0.3) < 1.0) { trust = 1.0; }
      if (uTraceOn != 0 && trust > 0.0) { lit += (litTraced(vWorld + normal * 0.02, bias, dFdx(vWorld), dFdy(vWorld)) - lit) * trust; }
    }
    shadowSteps = traceSteps;
    float diffuse = max(0.0, nl) * lit;
    vec3 ambient = ambientOf(normal);
    if (uRoomLight != 0) {
      // The room's light through what this point sees of it, from the mean open direction.
      vec2 at = mapAt();
      bool backSide = dot(normalize(vNormal), toEye) < 0.0;
      vec4 seen = backSide ? texture(uOpenBack, at) : texture(uOpenFront, at);
      float open = clamp(seen.a * uOpenScale, 0.0, 1.0);
      vec3 bent = dot(seen.xyz, seen.xyz) > 1e-8 ? uPose * normalize(seen.xyz) : normal;
      ambient = ambientOf(bent) * open;
      // And what the body itself throws on the point.
      if (uRoomLight > 1) { ambient += backSide ? texture(uThrownBack, at).rgb : texture(uThrownFront, at).rgb; }
    }
    colour = base * (ambient + vec3(0.80, 0.78, 0.74) * diffuse);
    if (uWire != 2) {
      float nv = max(0.0, dot(normal, toEye));
      float f0 = chrome ? 0.86 : 0.05;
      float fresnel = f0 + (1.0 - f0) * fifth(1.0 - nv);
      vec3 reflected = reflect(incoming, normal);
      if (uGloss != 0) {
        // What the mirror direction shows, answered by emitters: the sky and
        // its window above, the floor below with its pattern and the shadow on it.
        // A join's band begins on the part it leaves, a hair under it where the tube is inside the wall:
        // a mirror ray from the band starts clear of that part's sheet.
        vec3 origin = vWorld + normal * (uOnBand != 0 ? 0.03 : 2e-3);
        vec3 mirror;
        float tHit;
        int triangle;
        vec2 weights;
        float floorAt = uGroundOn != 0 && reflected.y < -1e-9 && origin.y > uGround ? (uGround - origin.y) / reflected.y : 1e30;
        if (uImage == 0 && uTraceOn != 0 && traceRay(origin, reflected, 1e-3, floorAt, false, back || (uTricks & 1) == 0 ? -1 : uOwnRoot, uOwnNext, int(vClear + 0.5), (uTricks & 2) == 0 ? vec4(0.0) : vec4(normal, dot(normal, vWorld)), tHit, triangle, weights)) {
          // The teapot shows itself: the point the mirror ray reaches, lit as a surface seen at second hand.
          mirror = shadeHit(origin + reflected * tHit, reflected, triangle, weights);
          debugHit = vec3(listAt(uMeshWhere, 3 * triangle).z + 1.0, tHit, float(triangle));
        } else if (uTraceOn == 0 && facesCavity(vUvs.y) && !leavesCavity(reflected)) {
          mirror = vec3(0.27, 0.26, 0.25);
        } else if (uImage == 0 && uGroundOn != 0 && reflected.y < -1e-9 && origin.y > uGround) {
          float reach = (uGround - origin.y) / reflected.y;
          vec3 q = origin + reflected * reach;
          float width = max(length(dFdx(q.xz)), length(dFdy(q.xz)));
          mirror = floorColour(q, width, reach, litAt(q + vec3(0.0, 0.004, 0.0), 0.02));
        } else {
          mirror = skyOf(origin, reflected);
        }
        vec3 tint = chrome ? vec3(0.95, 0.96, 0.98) : vec3(1.0);
        vec3 under = chrome ? vec3(0.0) : colour;
        colour = under * (1.0 - fresnel) + mirror * tint * (fresnel * mirrorDim(lit, nl));
      } else {
        vec3 halfway = normalize(uLight + toEye);
        float specular = ninetieth(max(0.0, dot(normal, halfway))) * lit;
        colour = colour * (1.0 - 0.6 * fresnel) + environmentOf(reflected) * (0.6 * fresnel) + vec3(0.55 * specular);
      }
    }
  }
  if (uWire != 0 && uImage == 0) {
    // Node lines and patch outlines from the point's own parameters.
    float gu = length(vec2(dFdx(vUvs.x), dFdy(vUvs.x))), gv = length(vec2(dFdx(vUvs.y), dFdy(vUvs.y)));
    float node = max(lineInk(vUvs.x, 1.0, gu, 0.55), lineInk(vUvs.y, 1.0, gv, 0.55));
    float outline = uPer.x == 0.0 ? 0.0 : max(lineInk(vUvs.x, uPer.x, gu, 1.1), lineInk(vUvs.y, uPer.y, gv, 1.1));
    vec3 nodeInk = inside ? vec3(0.45, 0.50, 0.58) : vec3(0.16, 0.30, 0.52);
    colour = mix(colour, nodeInk, 0.75 * node);
    colour = mix(colour, vec3(0.08, 0.09, 0.12), outline);
  }
  outColour = encode(colour);
  if (uCountSteps == 1) { outColour = vec4(float(traceSteps % 256) / 255.0, float(traceSteps / 256) / 255.0, float(min(traceLeaves, 255)) / 255.0, 1.0); }
  if (uCountSteps == 3) { outColour = vec4(debugHit.x / 16.0, clamp(debugHit.y, 0.0, 1.0), uPartIndex / 16.0, 1.0); }
  if (uCountSteps == 4) { int tri = int(debugHit.z) + (debugHit.x > 0.0 ? 1 : 0); outColour = vec4(float(tri % 256) / 255.0, float((tri / 256) % 256) / 255.0, float(tri / 65536) / 255.0, 1.0); }
  if (uCountSteps == 2) { outColour = vec4(back ? 1.0 : 0.0, vClear / 65535.0 > 0.0 ? 0.5 + 0.5 * fract(vClear / 256.0) : 0.0, float(min(shadowSteps, 255)) / 255.0, 1.0); }
}
`;

// The floor and what is beyond it: one triangle over the whole picture. Each
// pixel's ray either meets the floor or shows the room.
const ROOM_VERTEX_GLSL = `#version 300 es
out vec2 vScreen;
void main() {
  vec2 corner = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  vScreen = corner;
  gl_Position = vec4(corner, 0.0, 1.0);
}
`;
const ROOM_FRAGMENT_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
${ROOM_GLSL}
${TRACE_GLSL}
in vec2 vScreen;
uniform vec3 uForward, uRight, uUp;   // right and up scaled so that forward + x right + y up is the ray at (x, y) in [-1, 1]
uniform mat4 uViewProj;
uniform int uGroundOn, uGloss, uLighting;
uniform float uSink;   // the floor hides the teapot only where it is this far below the floor
uniform int uPolished;        // a polished floor shows the teapot's image
uniform sampler2D uImageMap;  // that image, drawn from the eye mirrored in the floor
out vec4 outColour;
void main() {
  vec3 dir = uForward + uRight * vScreen.x + uUp * vScreen.y;
  float t = uGroundOn != 0 && abs(dir.y) > 1e-9 ? (uGround - uEye.y) / dir.y : -1.0;
  if (!(t > 0.0)) {
    outColour = encode(uGloss != 0 ? skyOf(uEye, dir) : backgroundOf(dir));
    gl_FragDepth = 1.0;
    return;
  }
  vec3 p = uEye + dir * t;
  float far = length(p - uEye);
  float width = far / (1.0 / length(dFdy(vScreen.y) * uUp)) / max(0.08, abs(uEye.y - p.y) / max(far, 1e-9));
  vec3 colour = floorColour(p, width, t, uLighting != 0 ? litTraced(p + vec3(0.0, 0.004, 0.0), 0.02, dFdx(p), dFdy(p)) : 1.0);
  if (uPolished != 0 && uLighting != 0 && uEye.y > uGround) {
    vec3 unit = normalize(dir);
    vec4 image = texelFetch(uImageMap, ivec2(gl_FragCoord.xy), 0);
    vec3 seen = image.a > 0.5 ? image.rgb * image.rgb : skyOf(p, vec3(unit.x, -unit.y, unit.z));
    float fresnel = 0.07 + 0.93 * fifth(1.0 - abs(unit.y));
    float keep = fresnel * (1.0 - smooth01(18.0, 60.0, t));
    colour = colour * (1.0 - keep) + seen * keep;
  }
  outColour = encode(colour);
  // Depth of the floor a little below itself, so a body resting slightly in it is not cut by it.
  vec4 clip = uViewProj * vec4(p - vec3(0.0, uSink, 0.0), 1.0);
  gl_FragDepth = clamp(0.5 * clip.z / clip.w + 0.5, 0.0, 1.0);
}
`;

// The cut face of a cutaway: where the cutting plane passes through material.
const CAP_FRAGMENT_GLSL = `#version 300 es
precision highp float;
${ROOM_GLSL}
in vec3 vWorld;
uniform vec4 uCutPlane;
out vec4 outColour;
void main() {
  vec3 clay = vec3(0.36, 0.26, 0.22);
  outColour = encode(clay * (0.55 + 0.6 * max(0.0, dot(uCutPlane.xyz, uLight))));
}
`;
const CAP_VERTEX_GLSL = `#version 300 es
layout(location = 0) in vec3 aPoint;
uniform mat4 uViewProj;
out vec3 vWorld;
void main() {
  vWorld = aPoint;
  gl_Position = uViewProj * vec4(aPoint, 1.0);
}
`;

// The teapot drawn from the grid of surface points the GPU has evaluated
// (TeapotGpu.bakeSurface). There is no vertex buffer: a vertex's number says
// which cell of the part's grid it belongs to and which corner, and its
// point and normal are read from the grid.
const MAP_VERTEX_GLSL = `#version 300 es
uniform vec4 uRect;      // x, y, width, height in texels
uniform vec2 uMapSize;
void main() {
  vec2 corner = vec2(gl_VertexID == 1 || gl_VertexID == 2 || gl_VertexID == 4 ? 1.0 : 0.0, gl_VertexID == 2 || gl_VertexID == 4 || gl_VertexID == 5 ? 1.0 : 0.0);
  gl_Position = vec4((uRect.xy + corner * uRect.zw) / uMapSize * 2.0 - 1.0, 0.0, 1.0);
}
`;
// The surface at every texel of a part's grid, at rest: its point, its unit
// normal, and whether the part is there (not under a join).
const MAP_SURFACE_GLSL = `#version 300 es
${CONV_GLSL}
${TRIM_GLSL}
uniform vec4 uRect;
uniform vec2 uCellCount, uGridSpan;
layout(location = 0) out vec4 outPoint;
layout(location = 1) out vec4 outNormal;
void main() {
  vec2 at = gl_FragCoord.xy - 0.5 - uRect.xy;
  float u = at.x / uGridSpan.x * uCellCount.x, v = at.y / uGridSpan.y * uCellCount.y;
  vec3 p, du, dv;
  evalSurface(u, v, p, du, dv);
  vec3 n = cross(du, dv);
  if (dot(n, n) < 1e-16) {
    // At a pole: the normal a little way along the part.
    vec3 q;
    evalSurface(u, v < 0.5 * uCellCount.y ? v + 0.05 : v - 0.05, q, du, dv);
    n = cross(du, dv);
  }
  outPoint = vec4(p, 1.0);
  outNormal = vec4(normalize(n), hidden(p, u, v) ? 0.0 : 1.0);
}
`;
// Peeling along a direction: the next layer of the body behind the one taken
// last, with the light of the side that faces away from the direction.
const PEEL_FRAGMENT_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
${TRIM_GLSL}
in vec3 vWorld;
in vec3 vNormal;
in vec3 vUvs;
uniform vec3 uLight, uShadowCentre;
uniform vec4 uRect;
uniform vec2 uCellCount, uGridSpan, uMapSize;
uniform sampler2D uThrownFront, uThrownBack, uDirectFront, uDirectBack, uPrevLayer;
uniform int uPeel;
out vec4 outColour;
void main() {
  if (hidden(vWorld, vUvs.x, vUvs.y)) { discard; }
  float sz = -dot(vWorld - uShadowCentre, uLight);
  if (uPeel > 0 && sz <= texelFetch(uPrevLayer, ivec2(gl_FragCoord.xy), 0).r + 2e-3) { discard; }
  vec2 grid = vec2(vUvs.x - floor(vUvs.x / uCellCount.x) * uCellCount.x, vUvs.y) / uCellCount * uGridSpan + 0.5;
  vec2 at = (clamp(grid, vec2(0.5), uRect.zw - 0.5) + uRect.xy) / uMapSize;
  vec3 light = dot(vNormal, uLight) < 0.0 ? texture(uDirectFront, at).rgb + texture(uThrownFront, at).rgb : texture(uDirectBack, at).rgb + texture(uThrownBack, at).rgb;
  outColour = vec4(sz, light);
}
`;
// One direction of the room. Every texel, on each of its two sides, adds what
// that direction contributes if nothing of the body is in the way: the
// direction weighted by its cosine, and the cosine. Summed over directions
// spread evenly over the sphere this is the open fraction of the texel's
// cosine-weighted hemisphere and its mean open direction. Where the CPU
// casts rays from each texel, here the body is drawn once from the
// direction and every texel looks itself up in that picture.
const MAP_OPEN_GLSL = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPoints, uNormals, uDepth;
uniform vec3 uDirection, uDepthX, uDepthY, uDepthCentre;
uniform float uDepthRadius, uDepthSize;
layout(location = 0) out vec4 outFront;
layout(location = 1) out vec4 outBack;
float seen(vec3 p) {
  vec3 d = p - uDepthCentre;
  vec2 at = (vec2(dot(d, uDepthX), dot(d, uDepthY)) / uDepthRadius * 0.5 + 0.5) * uDepthSize - 0.5;
  float sz = -dot(d, uDirection);
  // The four texels around the point, each tested, blended by where the point lies between them.
  ivec2 base = ivec2(floor(at));
  vec2 f = at - floor(at);
  float open[4];
  for (int k = 0; k < 4; ++k) { open[k] = sz - 0.03 <= texelFetch(uDepth, base + ivec2(k & 1, k >> 1), 0).r ? 1.0 : 0.0; }
  return mix(mix(open[0], open[1], f.x), mix(open[2], open[3], f.x), f.y);
}
void main() {
  ivec2 texel = ivec2(gl_FragCoord.xy);
  vec4 point = texelFetch(uPoints, texel, 0);
  if (point.a < 0.5) { discard; }
  vec3 n = texelFetch(uNormals, texel, 0).xyz;
  float cosine = dot(n, uDirection);
  outFront = vec4(0.0);
  outBack = vec4(0.0);
  if (cosine > 0.0) {
    float w = cosine * seen(point.xyz + n * 0.02);
    outFront = vec4(uDirection * w, w);
  } else if (cosine < 0.0) {
    float w = -cosine * seen(point.xyz - n * 0.02);
    outBack = vec4(uDirection * w, w);
  }
}
`;

// The light every texel has from the room and the sun, before the body's own:
// the room through the open fraction, from the mean open direction, and the
// sun where the shadow map (drawn in the teapot's frame) lets it through.
const MAP_DIRECT_GLSL = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPoints, uNormals, uOpenFront, uOpenBack, uDepth;
uniform vec3 uDirection, uDepthX, uDepthY, uDepthCentre;   // the sun, in the teapot's frame
uniform float uDepthRadius, uDepthSize, uOpenScale;
uniform mat3 uPose;
uniform int uSun;
layout(location = 0) out vec4 outFront;
layout(location = 1) out vec4 outBack;
vec3 ambientOf(vec3 n) {
  float up = 0.5 * (n.y + 1.0);
  return mix(vec3(0.30, 0.28, 0.26), vec3(0.46, 0.50, 0.56), up);
}
float seen(vec3 p, float cosine) {
  vec3 d = p - uDepthCentre;
  vec2 at = (vec2(dot(d, uDepthX), dot(d, uDepthY)) / uDepthRadius * 0.5 + 0.5) * uDepthSize;
  float bias = 0.02 + 0.02 * min(12.0, sqrt(max(0.0, 1.0 - cosine * cosine)) / cosine);
  return -dot(d, uDirection) - bias <= texelFetch(uDepth, ivec2(at), 0).r ? 1.0 : 0.0;
}
vec4 side(vec3 p, vec3 n, vec4 open) {
  vec3 bent = dot(open.xyz, open.xyz) > 1e-8 ? normalize(open.xyz) : n;
  vec3 light = ambientOf(uPose * bent) * clamp(open.a * uOpenScale, 0.0, 1.0);
  float cosine = dot(n, uDirection);
  if (cosine > 0.0) { light += vec3(0.80, 0.78, 0.74) * cosine * (uSun != 0 ? seen(p + n * 0.02, cosine) : 1.0); }
  return vec4(light, 1.0);
}
void main() {
  ivec2 texel = ivec2(gl_FragCoord.xy);
  vec4 point = texelFetch(uPoints, texel, 0);
  if (point.a < 0.5) { discard; }
  vec3 n = texelFetch(uNormals, texel, 0).xyz;
  outFront = side(point.xyz, n, texelFetch(uOpenFront, texel, 0));
  outBack = side(point.xyz, -n, texelFetch(uOpenBack, texel, 0));
}
`;
// One direction of one gather. A texel side that faces the direction and
// has the body in the way receives the light of what it faces: the layer of
// the body just in front of it along the direction. Summed over directions,
// times the albedo, this is the CPU's gather through its transfer; the
// transfer is not stored here but found again by peeling each time.
const MAP_GATHER_GLSL = `#version 300 es
precision highp float;
precision highp sampler2D;
precision highp sampler2DArray;
uniform sampler2D uPoints, uNormals;
uniform sampler2DArray uLayers;
uniform int uLayerCount;
uniform vec3 uDirection, uDepthX, uDepthY, uDepthCentre;
uniform float uDepthRadius, uDepthSize, uWeight;   // albedo / (the weight of a whole hemisphere)
layout(location = 0) out vec4 outFront;
layout(location = 1) out vec4 outBack;
vec3 facedAt(ivec2 at, float sz) {
  float nearest = -1e30;
  vec3 light = vec3(0.0);
  for (int k = 0; k < uLayerCount; ++k) {
    vec4 layer = texelFetch(uLayers, ivec3(at, k), 0);
    if (layer.r < sz - 0.03 && layer.r > nearest) {
      nearest = layer.r;
      light = layer.gba;
    }
  }
  return light;
}
vec3 faced(vec3 p) {
  vec3 d = p - uDepthCentre;
  vec2 at = (vec2(dot(d, uDepthX), dot(d, uDepthY)) / uDepthRadius * 0.5 + 0.5) * uDepthSize - 0.5;
  float sz = -dot(d, uDirection);
  ivec2 base = ivec2(floor(at));
  vec2 f = at - floor(at);
  return mix(mix(facedAt(base, sz), facedAt(base + ivec2(1, 0), sz), f.x), mix(facedAt(base + ivec2(0, 1), sz), facedAt(base + ivec2(1, 1), sz), f.x), f.y);
}
void main() {
  ivec2 texel = ivec2(gl_FragCoord.xy);
  vec4 point = texelFetch(uPoints, texel, 0);
  if (point.a < 0.5) { discard; }
  vec3 n = texelFetch(uNormals, texel, 0).xyz;
  float cosine = dot(n, uDirection);
  outFront = vec4(0.0);
  outBack = vec4(0.0);
  if (cosine > 0.0) {
    outFront = vec4(faced(point.xyz + n * 0.02) * (cosine * uWeight), 0.0);
  } else if (cosine < 0.0) {
    outBack = vec4(faced(point.xyz - n * 0.02) * (-cosine * uWeight), 0.0);
  }
}
`;

// A 1-2-1 average each way over a pair of maps, among texels that belong to a part.
const MAP_SMOOTH_GLSL = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPoints, uFront, uBack;
layout(location = 0) out vec4 outFront;
layout(location = 1) out vec4 outBack;
void main() {
  ivec2 texel = ivec2(gl_FragCoord.xy);
  if (texelFetch(uPoints, texel, 0).a < 0.5) { discard; }
  vec4 front = vec4(0.0), back = vec4(0.0);
  float total = 0.0;
  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      ivec2 at = texel + ivec2(i, j);
      if (texelFetch(uPoints, at, 0).a < 0.5) { continue; }
      float w = float((2 - abs(i)) * (2 - abs(j)));
      front += texelFetch(uFront, at, 0) * w;
      back += texelFetch(uBack, at, 0) * w;
      total += w;
    }
  }
  outFront = front / total;
  outBack = back / total;
}
`;

// For reading the surface back (the check against the CPU) and for evaluating the vertices: nothing is shaded.
const PROBE_FRAGMENT_GLSL = `#version 300 es
precision highp float;
in vec3 vWorld;
in vec3 vNormal;
in vec3 vUvs;
uniform int uMode;
out vec4 outColour;
void main() { outColour = vec4(uMode == 4 ? vNormal : vWorld, 1.0); }
`;

// The CPU's compile(): the admitted u-cells of every row of a lattice, from
// its nodes. Each texel of the cell texture is one of the two halves of one coordinate of
// one cell; the fragment that owns it admits that cell from the eight nodes
// around it in its row.
const COMPILE_FRAGMENT_GLSL = `#version 300 es
${CONV_GLSL}
uniform int uCellsBase, uNodesBase, uShapesBase, uCu, uNu, uNv, uWrap;
out vec4 outCell;
void main() {
  int i = int(gl_FragCoord.y) * 2048 + int(gl_FragCoord.x) - uCellsBase;
  if (i < 0 || i >= uNv * uCu * 6) { discard; }
  int row = i / (uCu * 6), cell = (i / 6) % uCu, which = i % 6, channel = which / 2;
  vec2 w[8];
  for (int k = 0; k < 8; ++k) {
    int index = cell - 3 + k;
    if (uWrap != 0) {
      index = ((index % uNu) + uNu) % uNu;
    } else if (index < 0 || index >= uNu) {
      w[k] = vec2(0.0);
      continue;
    }
    w[k] = vec2(lin(uNodes, uNodesBase + row * uNu + index)[channel], 0.0);
  }
  vec2 c[5], bubble;
  admit(w, lin(uShapes, uShapesBase + 2 * cell), lin(uShapes, uShapesBase + 2 * cell + 1), c, bubble);
  outCell = which % 2 == 0 ? vec4(w[3].x, c[0].x, c[1].x, c[2].x) : vec4(c[3].x, c[4].x, bubble.x, 0.0);
}
`;
