// Building the teapot on the GPU. Where the engine would sample nodes, place a
// tube's rows, make a join or take points of a surface for a collision hull,
// it calls out (BuildHooks, gpu/main.cpp); the page runs one of the passes
// below and hands the result back. The engine keeps the bookkeeping: which
// lattice a part has, the shapes of its cells, the hull builder.
//   1  nodes sampled from the Bezier patches (1975, 1976, 1987);
//   2  the rows of a tube's seam, by CONV's own error (rows.js);
//   3  a tube's nodes: on each row, points around the source patches' surfaces;
//   4  a join: where each column of the tube crosses the body (bisection on
//      the two surfaces), then the two sides of the band walked from there on
//      the body and along the tube, and the trims;
//   5  points of a surface, for the hull builder.
const BUILD_GLSL = `#version 300 es
${CONV_GLSL}
uniform sampler2D uData;
uniform int uMode;
uniform int uRevolved, uCellsU;                       // 1
uniform int uTableCount, uAround, uPoints;            // 3
uniform int uSourceCells[8], uSourceShapes[8];
uniform float uVLow, uVHigh, uWidth, uBodyNu, uRows;  // 4
uniform int uRising, uAtFirst, uAcross;
uniform float uCentreU, uCentreV;
out vec4 outValue;

vec3 bodyAt(float u, float v) { vec3 p, du, dv; evalLat(uOwn, u, v, p, du, dv); return p; }
vec3 tubeAt(float u, float v) { vec3 p, du, dv; evalLat(uTube, u, v, p, du, dv); return p; }
// The body as a surface of revolution: its row at a height, on one branch.
float rowAt(float y) {
  float a = uVLow, b = uVHigh;
  for (int k = 0; k < 32; ++k) {
    float m = 0.5 * (a + b);
    bool above = bodyAt(0.0, m).y > y;
    if (above == (uRising != 0)) { b = m; } else { a = m; }
  }
  return 0.5 * (a + b);
}
float radiusAt(float y) { return length(bodyAt(0.0, rowAt(y)).xz); }
// Is this point of the tube nearer the axis than the wall it meets?
bool buried(float u, float w) {
  vec3 p = tubeAt(u, w);
  return length(p.xz) < radiusAt(p.y);
}

void main() {
  ivec2 at = ivec2(gl_FragCoord.xy);
  vec3 p, du, dv;
  if (uMode == 1) {
    // A node of a lattice over the Bezier patches.
    if (uRevolved != 0) {
      evalBezier(uBezier, 0.0, float(at.y), p, du, dv);
      float angle = 6.283185307179586 * float(at.x) / float(uCellsU);
      outValue = vec4(p.x * cos(angle), p.y, p.x * sin(angle), 0.0);
    } else {
      evalBezier(uBezier, float(at.x), float(at.y), p, du, dv);
      outValue = vec4(p, 0.0);
    }
    return;
  }
  if (uMode == 3) {
    // A node of a tube: the row's place on the seam says which source patch and where along it.
    float arc = lin(uData, uTableCount + at.y).x;
    int low = 0, high = uTableCount;
    for (int step = 0; step < 16 && low < high; ++step) {
      int mid = (low + high) / 2;
      if (lin(uData, mid).x <= arc) { low = mid + 1; } else { high = mid; }
    }
    int k = max(0, low - 1);
    if (k + 1 >= uTableCount) { k = uTableCount - 2; }
    vec4 a = lin(uData, k), b = lin(uData, k + 1);
    float span = b.x - a.x, f = span > 0.0 ? clamp((arc - a.x) / span, 0.0, 1.0) : 0.0;
    int run = int(a.z);
    float t = a.y + (b.y - a.y) * f;
    if (a.z != b.z) { run = int(b.z); t = b.y * f; }
    int side = uAround / 2;
    bool mirrored = at.x > side;
    int step = mirrored ? uAround - at.x : at.x;
    Lat source = uOwn;
    source.cells = uSourceCells[run];
    source.shapes = uSourceShapes[run];
    evalLat(source, float(uPoints - 1) * float(step) / float(side), t, p, du, dv);
    if (mirrored) { p.z = -p.z; }
    outValue = vec4(p, 0.0);
    return;
  }
  if (uMode == 4) {
    // Where this column of the tube leaves the body: the crossing along the tube and its place on the body.
    float u = float(at.x), end = uAtFirst != 0 ? 0.0 : uRows, direction = uAtFirst != 0 ? 1.0 : -1.0;
    p = tubeAt(u, end);
    bool near = abs(length(p.xz) - radiusAt(p.y)) <= 0.35;
    float a = end, b = end;
    bool found = !buried(u, end);
    int last = int(floor(0.6 * uRows * 4.0));
    for (int i = 1; i <= last && !found; ++i) {
      b = end + direction * 0.25 * float(i);
      if (!buried(u, b)) { found = true; break; }
      a = b;
    }
    for (int it = 0; it < 30; ++it) {
      float m = 0.5 * (a + b);
      if (buried(u, m)) { a = m; } else { b = m; }
    }
    float crossing = 0.5 * (a + b);
    p = tubeAt(u, crossing);
    outValue = vec4(crossing, atan(-p.z, p.x) / 6.283185307179586 * uBodyNu, rowAt(p.y), near && found ? 1.0 : 0.0);
    return;
  }
  if (uMode == 5) {
    // The band from the crossing: a walk of a given length on the body, away from the loop's centre, and one along the tube.
    float u = float(at.x), direction = uAtFirst != 0 ? 1.0 : -1.0;
    vec4 column = lin(uData, at.x);   // crossing, body u, body v
    evalLat(uOwn, column.y, column.z, p, du, dv);
    vec2 away = normalize(vec2((column.y - uCentreU) * length(du), (column.z - uCentreV) * length(dv)));
    float distances[4];
    int j = at.y;
    float tau = float(j) / float(uAcross - 1);
    float keep = 0.93 * uWidth;
    // What this texel holds: a node of the two sides, or the column's trims.
    float onBody = j < uAcross ? uWidth * (1.0 - tau) : keep;
    float onBodyToo = uWidth;
    float onTube = j < uAcross ? uWidth * tau : keep;
    float onTubeToo = uWidth;
    vec2 q[2];
    float w[2];
    for (int which = 0; which < 2; ++which) {
      float distance = which == 0 ? onBody : onBodyToo;
      vec2 r = column.yz;
      for (int step = 0; step < 8; ++step) {
        evalLat(uOwn, r.x, r.y, p, du, dv);
        r += distance / 8.0 * away / max(vec2(1e-9), vec2(length(du), length(dv)));
      }
      q[which] = r;
      distance = which == 0 ? onTube : onTubeToo;
      float s = column.x;
      for (int step = 0; step < 8; ++step) {
        evalLat(uTube, u, s, p, du, dv);
        s += direction * distance / 8.0 / max(1e-9, length(dv));
      }
      w[which] = s;
    }
    outValue = j < uAcross ? vec4(q[0], w[0], 0.0) : j == uAcross ? vec4(w[0], w[1], 0.0, 0.0) : vec4(q[0], q[1]);
    return;
  }
  if (uMode == 7) {
    // A node of a sheet of revolution: the meridian at the row's arc length, turned about the axis.
    float arc = lin(uData, uTableCount + at.y).x;
    int low = 0, high = uTableCount;
    for (int step = 0; step < 16 && low < high; ++step) {
      int mid = (low + high) / 2;
      if (lin(uData, mid).x <= arc) { low = mid + 1; } else { high = mid; }
    }
    int k = max(0, low - 1);
    if (k + 1 >= uTableCount) { k = uTableCount - 2; }
    vec4 a = lin(uData, k), b = lin(uData, k + 1);
    float span = b.x - a.x, f = span > 0.0 ? (arc - a.x) / span : 0.0;
    vec2 m = mix(a.yz, b.yz, f);
    if (at.y == 0 || at.y == uPoints - 1) { m.x = 0.0; }   // the two poles
    float angle = 6.283185307179586 * float(at.x) / float(uAround);
    outValue = vec4(m.x * cos(angle), m.y, -m.x * sin(angle), 0.0);
    return;
  }
  // 6: a point of the surface.
  vec4 where = lin(uData, at.x);
  evalLat(uOwn, where.x, where.y, p, du, dv);
  outValue = vec4(p, 0.0);
}
`;

class TeapotBuilder {
  constructor(gpu) {
    this.gpu = gpu;
    const gl = gpu.gl;
    this.program = gpu.program(ROOM_VERTEX_GLSL, BUILD_GLSL);
    this.width = 512;
    this.height = 256;
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, this.width, this.height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.counts = [0, 0, 0, 0, 0, 0, 0];
    this.rowPlans = [];   // how often each kind of work was done here
    this.ms = 0;
  }

  floats(kind) { const e = this.gpu.engine; return new Float32Array(e.HEAPF32.buffer, e._tp_hook_buffer(kind), e._tp_hook_length(kind)).slice(); }
  doubles(kind) { const e = this.gpu.engine; return new Float64Array(e.HEAPF64.buffer, e._tp_hook_buffer(kind), e._tp_hook_length(kind)); }

  // The lattices the engine laid out, with their u-cells admitted on the GPU.
  // They take the place of the teapot's in the renderer's textures until its next upload.
  lattices() {
    const g = this.gpu, e = g.engine, meta = this.floats(2);
    g.floatTexture('nodes', this.floats(0), 4);
    g.floatTexture('shapes', this.floats(1), 4);
    g.floatTexture('cells', new Float32Array(Math.max(4, e._tp_hook_length(7))), 4);
    g.lattice = (index) => ({ cells: meta[index * 12], shapes: meta[index * 12 + 1], cu: meta[index * 12 + 2], cv: meta[index * 12 + 3], nv: meta[index * 12 + 4], wrap: meta[index * 12 + 5],
                              nu: meta[index * 12 + 6], nodes: meta[index * 12 + 7], shapesAround: meta[index * 12 + 8] });
    g.compileCells(meta);
    this.made = meta.length / 12;
  }

  // One pass into a width x height corner of the target, read back.
  pass(mode, width, height, set) {
    const g = this.gpu, gl = g.gl, u = this.program.uniforms;
    if (width > this.width || height > this.height) { throw new Error('too large for a build pass'); }
    gl.useProgram(this.program.program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.SCISSOR_TEST);
    gl.bindVertexArray(g.emptyVao);
    const bind = (unit, name, texture) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texture || null); gl.uniform1i(u[name], unit); };
    bind(0, 'uCells', g.textures.cells);
    bind(1, 'uShapes', g.textures.shapes);
    bind(2, 'uNodes', g.textures.nodes);
    bind(3, 'uPatches', g.textures.patches);
    bind(4, 'uData', g.textures.buildData);
    gl.uniform1i(u.uMode, mode);
    set(gl, u);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, width, height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const out = new Float32Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    return out;
  }

  // The first three of every four floats, as the result.
  give(texels, count) {
    const out = this.doubles(5);
    for (let k = 0; k < count; ++k) { out[k * 3] = texels[k * 4]; out[k * 3 + 1] = texels[k * 4 + 1]; out[k * 3 + 2] = texels[k * 4 + 2]; }
  }

  hook(kind) {
    if (!this.gpu.options.buildOnGpu || this.gpu.gl.isContextLost()) { return 0; }
    if (this.skip && this.skip.includes(kind)) { return 0; }   // for checks: leave this kind of work to the engine
    if (kind === 0) { this.counts.fill(0); this.ms = 0; this.rowPlans = []; return 1; }
    const started = performance.now();
    let done = 0;
    try { done = this.run(kind); } catch (error) { console.error('build pass', kind, error); done = 0; }
    this.ms += performance.now() - started;
    if (done) { ++this.counts[kind]; }
    return done;
  }

  run(kind) {
    const g = this.gpu, e = g.engine, args = Array.from(this.doubles(4));
    if (kind === 1) {
      const [pu, pv, peru, perv, wrap, nu, nv, revolved, cellsU] = args;
      g.floatTexture('patches', this.floats(3), 4);
      const out = this.pass(1, nu, nv, (gl, u) => {
        [0, pu, pv, peru, perv, wrap].forEach((value, k) => gl.uniform1i(u['uBezier.' + ['base', 'pu', 'pv', 'peru', 'perv', 'wrap'][k]], value));
        gl.uniform1i(u.uRevolved, revolved);
        gl.uniform1i(u.uCellsU, cellsU);
      });
      this.give(out, nu * nv);
      return 1;
    }
    if (kind === 2) {
      g.rowPlanner = g.rowPlanner || new RowPlanner(g);
      const plan = g.rowPlanner.planFrom(e, e._tp_hook_buffer(6), 'seamProfile', args[0] > 0.5 ? 1 : 0, 0);
      if (!plan) { return 0; }
      const out = this.doubles(5);
      out[0] = plan.rows;
      out.set(plan.arcs, 1);
      out.set(plan.seams, 1 + plan.rows);
      this.rowPlans.push({ which: this.rowPlans.length, rows: plan.rows, error: plan.error, ms: plan.ms });
      return 1;
    }
    if (kind === 3) {
      const [tableCount, rows, around, points, sources] = args;
      this.lattices();
      g.floatTexture('buildData', this.floats(3), 4);
      const cells = new Int32Array(8), shapes = new Int32Array(8);
      for (let k = 0; k < sources; ++k) { cells[k] = g.lattice(k).cells; shapes[k] = g.lattice(k).shapes; }
      const out = this.pass(3, around, rows, (gl, u) => {
        g.setLattice(u, 'uOwn', g.lattice(0));
        gl.uniform1iv(u.uSourceCells, cells);
        gl.uniform1iv(u.uSourceShapes, shapes);
        gl.uniform1i(u.uTableCount, tableCount);
        gl.uniform1i(u.uAround, around);
        gl.uniform1i(u.uPoints, points);
      });
      this.give(out, around * rows);
      return 1;
    }
    if (kind === 4) {
      const [vLow, vHigh, rising, atFirst, width, bodyNu, across, columns, tubeRows] = args;
      this.lattices();
      const common = (gl, u) => {
        g.setLattice(u, 'uOwn', g.lattice(0));
        g.setLattice(u, 'uTube', g.lattice(1));
        gl.uniform1f(u.uVLow, vLow); gl.uniform1f(u.uVHigh, vHigh); gl.uniform1i(u.uRising, rising); gl.uniform1i(u.uAtFirst, atFirst);
        gl.uniform1f(u.uWidth, width); gl.uniform1f(u.uBodyNu, bodyNu); gl.uniform1f(u.uRows, tubeRows); gl.uniform1i(u.uAcross, across);
      };
      const found = this.pass(4, columns, 1, common);
      // The loop on the body in parameters: unwound around the body from column to column, and its centre.
      const data = new Float32Array(columns * 4);
      let centreU = 0, centreV = 0;
      for (let k = 0; k < columns; ++k) {
        if (found[k * 4 + 3] < 0.5) { return 0; }   // not joined here, or not found: the engine decides
        let bu = found[k * 4 + 1];
        if (k > 0) { bu -= bodyNu * Math.round((bu - data[(k - 1) * 4 + 1]) / bodyNu); }
        data.set([found[k * 4], bu, found[k * 4 + 2], 0], k * 4);
        centreU += bu / columns;
        centreV += found[k * 4 + 2] / columns;
      }
      g.floatTexture('buildData', data, 4);
      const band = this.pass(5, columns, across + 2, (gl, u) => { common(gl, u); gl.uniform1f(u.uCentreU, centreU); gl.uniform1f(u.uCentreV, centreV); });
      const out = this.doubles(5);
      out[0] = 1;
      out[1] = centreU;
      for (let k = 0; k < columns * across; ++k) { out[2 + k * 3] = band[k * 4]; out[3 + k * 3] = band[k * 4 + 1]; out[4 + k * 3] = band[k * 4 + 2]; }
      const trims = 2 + columns * across * 3;
      for (let k = 0; k < columns; ++k) {
        const cut = (across * columns + k) * 4, loop = ((across + 1) * columns + k) * 4;
        out.set([band[cut], band[cut + 1], band[loop], band[loop + 1], band[loop + 2], band[loop + 3]], trims + k * 6);
      }
      return 1;
    }
    if (kind === 6) {
      const [tableCount, rows, around] = args;
      g.floatTexture('buildData', this.floats(3), 4);
      const out = this.pass(7, around, rows, (gl, u) => { gl.uniform1i(u.uTableCount, tableCount); gl.uniform1i(u.uAround, around); gl.uniform1i(u.uPoints, rows); });
      this.give(out, around * rows);
      return 1;
    }
    if (kind === 5) {
      const [count, fresh] = args;
      if (fresh > 0.5) { this.lattices(); }
      g.floatTexture('buildData', this.floats(3), 4);
      const out = this.pass(6, count, 1, (gl, u) => { g.setLattice(u, 'uOwn', g.lattice(0)); });
      this.give(out, count);
      return 1;
    }
    return 0;
  }
}
