// Where the nodes are, on the GPU: the rows of the unified teapot's body and
// lid, placed by CONV's own error (src/unified.hpp: plan_rows_by_error,
// plan_rows, ProjectiveGrading, ProfileCurve), and the nodes on those rows.
//
// The engine hands over the source: the meridian the sampled patches give.
// The steps of the CPU's planner become passes over small textures:
//   scale  one texel: the bisection that finds the one scale of the wanted
//          spacing for which the gradings hold the asked number of rows;
//   rows   one texel a row: the row's arc length from the projective grading;
//   error  one texel a cell: how far the CONV lines through the rows stray
//          from the meridian there, over the fourth power of the cell's length;
//   best   one texel a row: the plan with the smallest error so far;
//   nodes  the lattice's nodes on the best plan's rows, written into the node texture.
const ROWS_GLSL = `#version 300 es
${CONV_GLSL}
uniform sampler2D uProfile;     // the meridian: fine table, coarse table, runs' samples
uniform int uFine, uFineCount, uCoarse, uCoarseCount, uRunBase, uRunPoints, uCreaseCount;
uniform float uCreases[12];
uniform float uLength, uFinest, uCoarsest, uGrowth;
uniform int uRows, uMode, uShape, uHavePrevious, uHardnessCount, uOnAxis;
uniform sampler2D uPlan, uHardness, uPrevious, uBest, uScale, uStations;
uniform int uNodesBase, uAround, uPlanCount;
out vec4 outValue;

vec4 profileAt(int i) { return texelFetch(uProfile, ivec2(i & 2047, i >> 11), 0); }
float fineS(int k) { return profileAt(uFine + 2 * k).x; }
// The first entry of the fine table beyond s.
int fineUpper(float s) {
  int low = 0, high = uFineCount;
  for (int step = 0; step < 20 && low < high; ++step) {
    int mid = (low + high) / 2;
    if (fineS(mid) <= s) { low = mid + 1; } else { high = mid; }
  }
  return low;
}
// The point of the meridian's polyline at arc length s.
vec2 fineAt(float s) {
  int k = max(0, fineUpper(s) - 1);
  if (k + 1 >= uFineCount) { k = uFineCount - 2; }
  vec4 a = profileAt(uFine + 2 * k), b = profileAt(uFine + 2 * k + 2);
  float span = b.x - a.x, t = span > 0.0 ? (s - a.x) / span : 0.0;
  return mix(a.yz, b.yz, t);
}
vec2 coarseAt(float s) {
  int low = 0, high = uCoarseCount;
  for (int step = 0; step < 20 && low < high; ++step) {
    int mid = (low + high) / 2;
    if (profileAt(uCoarse + mid).x <= s) { low = mid + 1; } else { high = mid; }
  }
  int k = max(0, low - 1);
  if (k + 1 >= uCoarseCount) { k = uCoarseCount - 2; }
  vec4 a = profileAt(uCoarse + k), b = profileAt(uCoarse + k + 1);
  float span = b.x - a.x, t = span > 0.0 ? (s - a.x) / span : 0.0;
  return mix(a.yz, b.yz, t);
}
// Radius of curvature near arc length s, from three points 'reach' apart.
float radiusAt(float s, float reach) {
  float a = max(0.0, s - reach), c = min(uLength, s + reach), b = 0.5 * (a + c);
  vec2 pa = fineAt(a), pb = fineAt(b), pc = fineAt(c);
  float area = abs((pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x));
  return area > 1e-14 ? length(pb - pa) * length(pc - pb) * length(pa - pc) / (2.0 * area) : 1e9;
}

// Which difference stencil a node takes, by its place in its segment.
int pickJet(int dl, int dr) {
  if (dl >= 2 && dr >= 2) { return 0; }
  int span = dl + dr;
  if (span >= 3) { return dl == 0 ? 4 : dr == 0 ? 7 : dl == 1 ? 5 : 6; }
  if (span == 2) { return dl == 0 ? 8 : dr == 0 ? 10 : 9; }
  return dl == 0 ? 11 : 12;
}
// One cell of an open CONV line with cubic ends from its window, at t.
float lineValue(vec2 w[8], vec4 sa, vec4 sb, float t) {
  vec2 c[5], bubble;
  admit(w, sa, sb, c, bubble);
  Basis b = makeBasis(t);
  float value = w[3].x + bubble.x * b.psi;
  for (int j = 0; j < 5; ++j) { value += c[j].x * b.tau[j]; }
  return value;
}
// A run of the meridian at parameter t: an open CONV line through its samples.
vec2 runPoint(int run, float t) {
  int n = uRunPoints;
  int cell = min(int(t), n - 2);
  vec2 wx[8], wy[8];
  for (int k = 0; k < 8; ++k) {
    int index = cell - 3 + k;
    vec2 p = index >= 0 && index < n ? profileAt(uRunBase + run * n + index).xy : vec2(0.0);
    wx[k] = vec2(p.x, 0.0);
    wy[k] = vec2(p.y, 0.0);
  }
  int left = pickJet(cell, n - 1 - cell), right = pickJet(cell + 1, n - 2 - cell);
  vec4 sa = vec4(float(left), float(right), float(left), float(right));
  vec4 sb = vec4(cell > 0 ? 1.0 : 0.0, cell + 2 < n ? 1.0 : 0.0, cell >= 2 && cell + 3 < n ? 1.0 : 0.0, 0.0);
  return vec2(lineValue(wx, sa, sb, t - float(cell)), lineValue(wy, sa, sb, t - float(cell)));
}
// The meridian itself at arc length s: the run that holds it and the parameter within the run.
vec2 truthAt(float arc) {
  float s = clamp(arc, 0.0, uLength);
  int k = max(0, fineUpper(s) - 1);
  if (k + 1 >= uFineCount) { k = uFineCount - 2; }
  vec4 a = profileAt(uFine + 2 * k), b = profileAt(uFine + 2 * k + 2);
  float runA = profileAt(uFine + 2 * k + 1).x, runB = profileAt(uFine + 2 * k + 3).x;
  float span = b.x - a.x, f = span > 0.0 ? clamp((s - a.x) / span, 0.0, 1.0) : 0.0;
  if (runA != runB) { return runPoint(int(runB), b.w * f); }   // the step that crosses into the next run starts at its parameter 0
  return runPoint(int(runA), a.w + (b.w - a.w) * f);
}

// The plan in hand: rows' arc lengths and creases.
float planArc(int k) { return texelFetch(uPlan, ivec2(k, 0), 0).x; }
bool planSeam(int k, int n) { return k <= 0 || k >= n - 1 || texelFetch(uPlan, ivec2(k, 0), 0).y > 0.5; }
int jetInCell(int cell, int node, int n) {
  int start = cell;
  for (int step = 0; step < 300 && !planSeam(start, n); ++step) { --start; }
  int end = cell + 1;
  for (int step = 0; step < 300 && !planSeam(end, n); ++step) { ++end; }
  return pickJet(node - start, end - node);
}
// The lattice row k of the plan: the meridian's point there (a pole at each end).
vec2 rowPoint(int k, int n) {
  vec2 p = truthAt(planArc(k));
  if (uOnAxis != 0 && (k == 0 || k == n - 1)) { p.x = 0.0; }
  return p;
}
// How far a point is from the meridian's polyline within 'reach' of arc length 'arc'.
float distanceTo(vec2 p, float arc, float reach) {
  // The first entry not before arc - reach, then one back.
  int low = 0, high = uFineCount;
  for (int step = 0; step < 20 && low < high; ++step) {
    int mid = (low + high) / 2;
    if (fineS(mid) < arc - reach) { low = mid + 1; } else { high = mid; }
  }
  int k = max(0, low - 1);
  float best = 1e30;
  for (int step = 0; step < 6000 && k + 1 < uFineCount; ++step, ++k) {
    vec4 a = profileAt(uFine + 2 * k);
    if (!(a.x <= arc + reach)) { break; }
    vec2 d = profileAt(uFine + 2 * k + 2).yz - a.yz;
    float len2 = dot(d, d);
    float t = len2 > 0.0 ? clamp(dot(p - a.yz, d) / len2, 0.0, 1.0) : 0.0;
    best = min(best, length(a.yz + t * d - p));
  }
  return best;
}

// The hardness in hand: per cell its middle arc length and e / h^4.
int hardnessCell(float s) {
  int low = 0, high = uHardnessCount;
  for (int step = 0; step < 12 && low < high; ++step) {
    int mid = (low + high) / 2;
    if (texelFetch(uHardness, ivec2(mid, 0), 0).x <= s) { low = mid + 1; } else { high = mid; }
  }
  return low == 0 ? 0 : min(uHardnessCount - 1, low - 1);
}
// The wanted spacing at an arc length, up to one common scale.
float wantedSpacing(float q) {
  if (uShape == 0) { return radiusAt(q, 0.001 * uLength); }
  int j = hardnessCell(q), j1 = min(uHardnessCount - 1, j + 1);
  return pow(max(texelFetch(uHardness, ivec2(j, 0), 0).y, texelFetch(uHardness, ivec2(j1, 0), 0).y), -0.25);
}
// The projective grading of one section between creases, for a scale.
float station[98], root[98];
int stations;
void gradingFor(int section, float a, float b, float scale) {
  int n = max(2, int(ceil(96.0 * (b - a) / uLength)));
  stations = n + 1;
  for (int k = 0; k <= n; ++k) {
    station[k] = a + (b - a) * float(k) / float(n);
    root[k] = sqrt(min(uCoarsest, max(uFinest, scale * texelFetch(uStations, ivec2(k, section), 0).x)));
  }
  // Limit how fast the spacing may change between stations, both ways.
  for (int pass = 0; pass < 2; ++pass) {
    for (int k = 1; k <= n; ++k) { root[k] = min(root[k], root[k - 1] + uGrowth * (station[k] - station[k - 1]) / root[k - 1]); }
    for (int k = n - 1; k >= 0; --k) { root[k] = min(root[k], root[k + 1] + uGrowth * (station[k + 1] - station[k]) / root[k + 1]); }
  }
}
float rowsInSpan(int k) { return (station[k + 1] - station[k]) / (root[k] * root[k + 1]); }
float rowsInAll() {
  float total = 0.0;
  for (int k = 0; k + 1 < stations; ++k) { total += rowsInSpan(k); }
  return total;
}
// Arc length holding 'count' rows from the first station.
float arcHolding(float count) {
  int k = 0;
  for (int step = 0; step < 98 && k + 2 < stations && count > rowsInSpan(k); ++step) {
    count -= rowsInSpan(k);
    ++k;
  }
  float d = station[k + 1] - station[k], p0 = root[k], p1 = root[k + 1];
  float x = count * p0 * p0 / (1.0 - count * p0 * (p1 - p0) / d);
  return station[k] + min(d, max(0.0, x));
}
float sectionStart(int k) { return k == 0 ? 0.0 : uCreases[k - 1]; }
float sectionEnd(int k) { return k == uCreaseCount ? uLength : uCreases[k]; }

void main() {
  int at = int(gl_FragCoord.x);
  if (uMode == 5) {
    // The wanted spacing at one station of one section, before the scale:
    // the tightest within reach of the station.
    int section = int(gl_FragCoord.y);
    float a = sectionStart(section), b = sectionEnd(section);
    int n = max(2, int(ceil(96.0 * (b - a) / uLength)));
    if (section > uCreaseCount || at > n) { outValue = vec4(0.0); return; }
    float s = a + (b - a) * float(at) / float(n);
    float radius = 1e9;
    for (int j = -2; j <= 2; ++j) {
      float q = min(b, max(a, s + float(j) * 0.25 * (b - a) / float(n)));
      radius = min(radius, wantedSpacing(q));
    }
    outValue = vec4(radius, 0.0, 0.0, 0.0);
    return;
  }
  if (uMode == 2) {
    // One scale for the whole meridian, by bisection on the number of rows.
    float low = 1e-9, high = 1e9;
    for (int iteration = 0; iteration < 120; ++iteration) {
      float mid = sqrt(low * high);
      float count = 0.0;
      for (int k = 0; k <= uCreaseCount; ++k) {
        gradingFor(k, sectionStart(k), sectionEnd(k), mid);
        count += rowsInAll();
      }
      if (count > float(uRows)) { low = mid; } else { high = mid; }
    }
    outValue = vec4(sqrt(low * high), 0.0, 0.0, 0.0);
    return;
  }
  if (uMode == 3) {
    // This row's arc length: its section's rows, spread by the grading.
    float scale = texelFetch(uScale, ivec2(0, 0), 0).x;
    int before = 0;
    float arc = uLength, seam = 0.0;
    bool placed = false;
    for (int k = 0; k <= uCreaseCount; ++k) {
      gradingFor(k, sectionStart(k), sectionEnd(k), scale);
      float exact = rowsInAll();
      int n = max(4, int(floor(exact + 0.5)));
      if (!placed && at < before + n) {
        int row = at - before;
        arc = arcHolding(exact * float(row) / float(n));
        seam = row == 0 && k > 0 ? 1.0 : 0.0;
        placed = true;
      }
      before += n;
    }
    // The last row is the meridian's end.
    outValue = at <= before ? vec4(arc, seam, float(before + 1), 0.0) : vec4(0.0);
    return;
  }
  if (uMode == 0) {
    // This cell's error: the CONV lines through the rows against the meridian.
    int n = uPlanCount;
    if (at + 1 >= n) { outValue = vec4(0.0); return; }
    float a0 = planArc(at), h = planArc(at + 1) - a0;
    vec2 wx[8], wy[8];
    for (int k = 0; k < 8; ++k) {
      int index = at - 3 + k;
      vec2 p = index >= 0 && index < n ? rowPoint(index, n) : vec2(0.0);
      wx[k] = vec2(p.x, 0.0);
      wy[k] = vec2(p.y, 0.0);
    }
    bool hasPrev = at > 0, hasNext = at + 2 < n;
    int left = jetInCell(at, at, n), right = jetInCell(at, at + 1, n);
    vec4 sa = vec4(float(left), float(right), float(hasPrev ? jetInCell(at - 1, at, n) : left), float(hasNext ? jetInCell(at + 1, at + 1, n) : right));
    bool six = !planSeam(at, n) && !planSeam(at - 1, n) && !planSeam(at + 1, n) && !planSeam(at + 2, n) && !(at < 2 || at + 3 >= n);
    vec4 sb = vec4(hasPrev ? 1.0 : 0.0, hasNext ? 1.0 : 0.0, six ? 1.0 : 0.0, 0.0);
    float e = 0.0;
    float samples[7] = float[7](0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9);
    for (int k = 0; k < 7; ++k) {
      float t = samples[k];
      e = max(e, distanceTo(vec2(lineValue(wx, sa, sb, t), lineValue(wy, sa, sb, t)), a0 + t * h, 2.0 * h));
    }
    float middle = a0 + 0.5 * h;
    float g = max(1e-7, e) / (h * h * h * h);
    // Blend with the previous estimate so the iteration settles.
    if (uHavePrevious != 0) {
      int low = 0, high = uHardnessCount;
      for (int step = 0; step < 12 && low < high; ++step) {
        int mid = (low + high) / 2;
        if (texelFetch(uPrevious, ivec2(mid, 0), 0).x <= middle) { low = mid + 1; } else { high = mid; }
      }
      int j = low == 0 ? 0 : min(uHardnessCount - 1, low - 1);
      g = sqrt(g * texelFetch(uPrevious, ivec2(j, 0), 0).y);
    }
    outValue = vec4(middle, g, e, 1.0);
    return;
  }
  if (uMode == 1) {
    // Keep the plan with the smallest error.
    float worst = 0.0;
    for (int k = 0; k + 1 < uPlanCount; ++k) { worst = max(worst, texelFetch(uHardness, ivec2(k, 0), 0).z); }
    vec4 kept = texelFetch(uBest, ivec2(at, 0), 0);
    vec4 plan = texelFetch(uPlan, ivec2(at, 0), 0);
    outValue = worst < texelFetch(uBest, ivec2(0, 0), 0).w ? vec4(plan.x, plan.y, float(uPlanCount), worst) : kept;
    return;
  }
  // The nodes: every row a true circle of the meridian's radius there.
  int i = int(gl_FragCoord.y) * 2048 + at - uNodesBase;
  if (i < 0 || i >= uAround * uPlanCount) { discard; }
  int v = i / uAround, u = i % uAround;
  vec2 p = coarseAt(texelFetch(uBest, ivec2(v, 0), 0).x);
  if (v == 0 || v == uPlanCount - 1) { p.x = 0.0; }
  float angle = 6.283185307179586 * float(u) / float(uAround);
  outValue = vec4(p.x * cos(angle), p.y, -p.x * sin(angle), 0.0);
}
`;

class RowPlanner {
  constructor(gpu) {
    this.gpu = gpu;
    const gl = gpu.gl;
    this.program = gpu.program(ROOM_VERTEX_GLSL, ROWS_GLSL);
    const make = (width, height = 1) => {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, width, height);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      return { texture, framebuffer };
    };
    this.plans = [make(256), make(256)];
    this.hardness = [make(256), make(256)];
    this.best = [make(256), make(256)];
    this.scale = make(1);
    this.stations = make(128, 16);
    this.rows = [make(256), make(256)];   // the rows in use, body and lid
    this.nodeFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.report = [];
  }

  // Place the rows of sheet `which` (0 body, 1 lid) from its meridian.
  // With `compare`, the engine also makes its own plan, to set beside this one.
  plan(engine, which, compare = 0) {
    const made = this.planFrom(engine, engine._tp_sheet(which, compare), 'sheetProfile' + which + '_' + engine._tp_rows_scale(), 1, compare);
    if (made) { made.which = which; }
    return made;
  }

  // The rows for the sheet laid out at `pointer` (gpu/main.cpp, fill_sheet). `onAxis`: its two ends are poles.
  planFrom(engine, pointer, key, onAxis, compare) {
    const gl = this.gpu.gl, u = this.program.uniforms;
    if (new Float64Array(engine.HEAPF64.buffer, pointer, 1)[0] < 0.5) { return null; }
    const head = new Float64Array(engine.HEAPF64.buffer, pointer, 10).slice();
    const [, rows, around, length, runCount, runPoints, creaseCount, fineCount, coarseCount] = head;
    let offset = 10;
    const take = (count) => { const out = new Float64Array(engine.HEAPF64.buffer, pointer + offset * 8, count).slice(); offset += count; return out; };
    if (creaseCount > 12) { return null; }
    const creases = take(creaseCount), runs = take(runCount * runPoints * 2), fine = take(fineCount * 5), coarse = take(coarseCount * 3);
    const own = compare ? { arcs: take(take(1)[0]), error: take(1)[0], first: take(take(1)[0]), given: take(1)[0] } : null;
    // One texture: the fine table (two texels an entry), the coarse table, the runs' samples.
    const data = new Float32Array((fineCount * 2 + coarseCount + runCount * runPoints) * 4);
    for (let k = 0; k < fineCount; ++k) { data.set([fine[k * 5], fine[k * 5 + 1], fine[k * 5 + 2], fine[k * 5 + 3], fine[k * 5 + 4], 0, 0, 0], k * 8); }
    const coarseBase = fineCount * 2;
    for (let k = 0; k < coarseCount; ++k) { data.set([coarse[k * 3], coarse[k * 3 + 1], coarse[k * 3 + 2], 0], (coarseBase + k) * 4); }
    const runBase = coarseBase + coarseCount;
    for (let k = 0; k < runCount * runPoints; ++k) { data.set([runs[k * 2], runs[k * 2 + 1], 0, 0], (runBase + k) * 4); }
    this.gpu.floatTexture(key, data, 4);

    const started = performance.now();
    gl.useProgram(this.program.program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.gpu.emptyVao);
    const bind = (unit, name, texture) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texture); gl.uniform1i(u[name], unit); };
    bind(0, 'uProfile', this.gpu.textures[key]);
    gl.uniform1i(u.uFine, 0); gl.uniform1i(u.uFineCount, fineCount);
    gl.uniform1i(u.uCoarse, coarseBase); gl.uniform1i(u.uCoarseCount, coarseCount);
    gl.uniform1i(u.uRunBase, runBase); gl.uniform1i(u.uRunPoints, runPoints);
    gl.uniform1i(u.uCreaseCount, creaseCount);
    gl.uniform1i(u.uOnAxis, onAxis);
    const padded = new Float32Array(12);
    padded.set(creases);
    gl.uniform1fv(u.uCreases, padded);
    gl.uniform1f(u.uLength, length);
    gl.uniform1i(u.uRows, rows);
    gl.uniform1f(u.uGrowth, 0.2);
    const run = (mode, target, width, height = 1) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, width, height);
      gl.uniform1i(u.uMode, mode);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    const read = (target) => {
      const texels = new Float32Array(256 * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.readPixels(0, 0, 256, 1, gl.RGBA, gl.FLOAT, texels);
      return texels;
    };
    // A plan from a wanted spacing: the stations' spacings, the scale, then the rows.
    const plan = (shape, finest, coarsest, hardnessCount) => {
      gl.uniform1i(u.uShape, shape);
      gl.uniform1f(u.uFinest, finest);
      gl.uniform1f(u.uCoarsest, coarsest);
      gl.uniform1i(u.uHardnessCount, hardnessCount);
      bind(2, 'uHardness', this.hardness[0].texture);
      bind(1, 'uPlan', null);
      bind(7, 'uStations', null);
      bind(4, 'uScale', null);
      run(5, this.stations, 128, 16);
      bind(7, 'uStations', this.stations.texture);
      run(2, this.scale, 1);
      bind(4, 'uScale', this.scale.texture);
      run(3, this.plans[1], 256);
      this.plans.reverse();
    };
    bind(5, 'uBest', null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.best[0].framebuffer);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 1e30]);
    plan(0, length / (40 * rows), 4 * length / rows, 0);
    const first = read(this.plans[0]);
    let planCount = Math.round(first[2]), hardnessCount = 0;
    const iterations = 8;
    for (let iteration = 0; iteration <= iterations; ++iteration) {
      // The cells' errors for the plan in hand, blended with the previous hardness.
      bind(1, 'uPlan', this.plans[0].texture);
      bind(3, 'uPrevious', this.hardness[0].texture);
      bind(2, 'uHardness', null);
      gl.uniform1i(u.uPlanCount, planCount);
      gl.uniform1i(u.uHavePrevious, iteration > 0 ? 1 : 0);
      gl.uniform1i(u.uHardnessCount, hardnessCount);
      run(0, this.hardness[1], 256);
      this.hardness.reverse();
      hardnessCount = planCount - 1;
      // Keep the plan if its error is the smallest so far.
      bind(2, 'uHardness', this.hardness[0].texture);
      bind(3, 'uPrevious', null);
      bind(5, 'uBest', this.best[0].texture);
      run(1, this.best[1], 256);
      this.best.reverse();
      bind(5, 'uBest', null);
      if (iteration === iterations) { break; }
      plan(1, length / (400 * rows), 6 * length / rows, hardnessCount);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.plans[0].framebuffer);
      const texel = new Float32Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, texel);
      planCount = Math.round(texel[2]);
    }
    const kept = read(this.best[0]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    const found = Math.round(kept[2]);
    // The rows as the engine takes them: ends and creases at their exact arc lengths.
    const arcs = new Float64Array(found), seams = new Float64Array(found);
    for (let k = 0; k < found; ++k) {
      arcs[k] = kept[k * 4];
      seams[k] = kept[k * 4 + 1] > 0.5 ? 1 : 0;
      if (seams[k]) { arcs[k] = creases.reduce((best, c) => Math.abs(c - arcs[k]) < Math.abs(best - arcs[k]) ? c : best, creases[0]); }
    }
    arcs[0] = 0;
    arcs[found - 1] = length;
    const result = { rows: found, error: kept[3], ms: performance.now() - started, arcs, seams, profile: key, coarseBase, coarseCount, around };
    if (own) {
      // Beside the engine's own plan: the errors of the two, and how far apart the rows are.
      const apart = (a, b, n) => { let worst = a.length === n ? 0 : -1; for (let k = 0; worst >= 0 && k < n; ++k) { worst = Math.max(worst, Math.abs(a[k] - b[k * 4])); } return worst; };
      result.own = { rows: own.arcs.length, error: own.error, given: own.given, apart: apart(own.arcs, kept, found), firstApart: apart(own.first, first, Math.round(first[2])) };
    }
    return result;
  }

  // Place the rows of the body and the lid and hand them to the engine, which rebuilds on them.
  placeAll(engine) {
    const plans = [this.plan(engine, 0), this.plan(engine, 1)];
    if (!plans[0] || !plans[1]) { return null; }
    const buffer = new Float64Array(engine.HEAPF64.buffer, engine._tp_rows_buffer(), 1024);
    let at = 0;
    for (const p of plans) {
      buffer.set(p.arcs, at);
      buffer.set(p.seams, at + p.rows);
      at += 2 * p.rows;
    }
    engine._tp_rows(plans[0].rows, plans[1].rows);
    return plans;
  }

  // The nodes of sheet `which` on its rows, written into the node texture over the engine's.
  // Returns the largest distance between the two.
  writeNodes(which, lattice, plan, cpuNodes) {
    const gl = this.gpu.gl, u = this.program.uniforms, sheet = plan;
    // The rows in use, in a texture for the pass that makes the nodes.
    const texels = new Float32Array(256 * 4);
    for (let k = 0; k < plan.rows; ++k) { texels[k * 4] = plan.arcs[k]; }
    gl.activeTexture(gl.TEXTURE0 + 5);
    gl.bindTexture(gl.TEXTURE_2D, this.rows[which].texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.FLOAT, texels);
    gl.useProgram(this.program.program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.gpu.emptyVao);
    const bind = (unit, name, texture) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texture); gl.uniform1i(u[name], unit); };
    bind(0, 'uProfile', this.gpu.textures[plan.profile]);
    for (const [unit, name] of [[1, 'uPlan'], [2, 'uHardness'], [3, 'uPrevious'], [4, 'uScale'], [7, 'uStations']]) { bind(unit, name, null); }
    bind(5, 'uBest', this.rows[which].texture);
    gl.uniform1i(u.uCoarse, sheet.coarseBase); gl.uniform1i(u.uCoarseCount, sheet.coarseCount);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.nodeFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.gpu.textures.nodes, 0);
    const base = lattice.nodes, total = sheet.around * plan.rows;
    const first = Math.floor(base / 2048), last = Math.floor((base + total - 1) / 2048);
    gl.viewport(0, 0, 2048, this.gpu.nodeRows);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(0, first, 2048, last - first + 1);
    gl.uniform1i(u.uNodesBase, base);
    gl.uniform1i(u.uAround, sheet.around);
    gl.uniform1i(u.uPlanCount, plan.rows);
    gl.uniform1i(u.uMode, 4);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.SCISSOR_TEST);
    const made = new Float32Array((last - first + 1) * 2048 * 4);
    gl.readPixels(0, first, 2048, last - first + 1, gl.RGBA, gl.FLOAT, made);
    let apart = 0;
    for (let k = 0; k < total; ++k) {
      const at = (base - first * 2048 + k) * 4, from = (base + k) * 4;
      apart = Math.max(apart, Math.hypot(made[at] - cpuNodes[from], made[at + 1] - cpuNodes[from + 1], made[at + 2] - cpuNodes[from + 2]));
    }
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    return apart;
  }
}
