// The GPU version's renderer: WebGL 2. The engine (WebAssembly) builds the
// teapot, runs the physics and answers the pointer; this draws.
class TeapotGpu {
  constructor(canvas, engine) {
    this.canvas = canvas;
    this.engine = engine;
    const gl = canvas.getContext('webgl2', { antialias: true, stencil: true, depth: true, alpha: false, preserveDrawingBuffer: true });
    if (!gl) { throw new Error('WebGL 2 is not available'); }
    if (!gl.getExtension('EXT_color_buffer_float')) { throw new Error('rendering to float textures (EXT_color_buffer_float) is not available'); }
    this.gl = gl;
    this.options = { material: 0, gloss: 0, lighting: 1, shadows: 1, ground: 1, wire: 0, texture: 0, mapping: 0, perPixel: 1, shadowSize: 2048, polished: 0, trace: 1, filter: 2, placeRows: 1, gatherPerFrame: 3, roomLight: 2, directions: 256, gatherDirections: 48, gathers: 5, layers: 6 };
    // The surface from the lattice at every vertex (used once, to evaluate them, and to check against the CPU);
    // the teapot from those evaluated vertices; the maps of what each point sees of the room.
    this.conv = this.program(VERTEX_GLSL, PROBE_FRAGMENT_GLSL);
    this.main = this.program(GRID_VERTEX_GLSL, FRAGMENT_GLSL);
    this.peel = this.program(GRID_VERTEX_GLSL, PEEL_FRAGMENT_GLSL);
    this.mapInfo = this.program(MAP_VERTEX_GLSL, MAP_INFO_GLSL);
    this.boxCell = this.program(MAP_VERTEX_GLSL, BOX_CELL_GLSL);
    this.boxParent = this.program(MAP_VERTEX_GLSL, BOX_PARENT_GLSL);
    this.mapSurface = this.program(MAP_VERTEX_GLSL, MAP_SURFACE_GLSL);
    this.mapOpen = this.program(MAP_VERTEX_GLSL, MAP_OPEN_GLSL);
    this.mapDirect = this.program(MAP_VERTEX_GLSL, MAP_DIRECT_GLSL);
    this.mapGather = this.program(MAP_VERTEX_GLSL, MAP_GATHER_GLSL);
    this.mapSmooth = this.program(MAP_VERTEX_GLSL, MAP_SMOOTH_GLSL);
    this.compiler = this.program(ROOM_VERTEX_GLSL, COMPILE_FRAGMENT_GLSL);
    this.thrownKey = '';
    this.mapSize = 1024;
    this.roomKey = '';
    this.room = this.program(ROOM_VERTEX_GLSL, ROOM_FRAGMENT_GLSL);
    this.cap = this.program(CAP_VERTEX_GLSL, CAP_FRAGMENT_GLSL);
    this.textures = {};
    this.capVao = gl.createVertexArray();
    this.capBuffer = gl.createBuffer();
    this.probeVao = gl.createVertexArray();
    this.probeBuffer = gl.createBuffer();
    this.emptyVao = gl.createVertexArray();
    this.targets = {};
    this.parts = [];
    this.check = null;
  }

  program(vertexSource, fragmentSource, feedback) {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) { throw new Error('shader: ' + gl.getShaderInfoLog(shader)); }
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    if (feedback) { gl.transformFeedbackVaryings(program, feedback, gl.INTERLEAVED_ATTRIBS); }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { throw new Error('program: ' + gl.getProgramInfoLog(program)); }
    const uniforms = {};
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let k = 0; k < count; ++k) {
      const name = gl.getActiveUniform(program, k).name.replace(/\[0\]$/, '');
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return { program, uniforms };
  }

  // A texture of floats addressed by one index: 2048 across.
  floatTexture(key, data, channels) {
    const gl = this.gl;
    const perRow = 2048 * channels;
    const rows = Math.max(1, Math.ceil(data.length / perRow));
    const padded = new Float32Array(rows * perRow);
    padded.set(data);
    if (!this.textures[key]) { this.textures[key] = gl.createTexture(); }
    if (key === 'cells') { this.cellRows = rows; }
    if (key === 'nodes') { this.nodeRows = rows; }
    gl.bindTexture(gl.TEXTURE_2D, this.textures[key]);
    gl.texImage2D(gl.TEXTURE_2D, 0, channels === 4 ? gl.RGBA32F : gl.R32F, 2048, rows, 0, channels === 4 ? gl.RGBA : gl.RED, gl.FLOAT, padded);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // Take from the engine what is shown now, if it has changed.
  upload() {
    const e = this.engine, gl = this.gl;
    // Where the nodes are: the rows of the unified teapot's body and lid are placed here,
    // by CONV's own error (rows.js), and the engine builds on them.
    if (!this.rowPlanner) { this.rowPlanner = new RowPlanner(this); }
    if (this.rowsAsked !== this.options.placeRows) { this.rowsAsked = this.options.placeRows; e._tp_set(40, this.options.placeRows); }
    if (e._tp_rows_wanted()) {
      const scale = e._tp_rows_scale();
      this.rowPlans = this.rowPlans || {};
      this.rowPlans[scale] = this.rowPlanner.placeAll(e);
    }
    if (e._tp_build() === 0 && this.parts.length > 0) { return false; }
    const floats = (kind) => new Float32Array(e.HEAPF32.buffer, e._tp_buffer(kind), e._tp_length(kind)).slice();
    this.floatTexture('cells', floats(0), 4);
    this.floatTexture('shapes', floats(1), 4);
    this.floatTexture('trims', floats(7), 1);
    const cpuNodes = floats(12);
    this.floatTexture('nodes', cpuNodes, 4);
    this.floatTexture('patches', floats(13), 4);
    const lattices = floats(2), meta = floats(3), table = floats(6), exact = floats(9), other = floats(14);
    this.lattice = (index) => index < 0 ? { cells: 0, shapes: 0, cu: 1, cv: 1, nv: 2, wrap: 0, nu: 2, nodes: 0, shapesAround: 0 }
      : { cells: lattices[index * 12], shapes: lattices[index * 12 + 1], cu: lattices[index * 12 + 2], cv: lattices[index * 12 + 3], nv: lattices[index * 12 + 4], wrap: lattices[index * 12 + 5],
          nu: lattices[index * 12 + 6], nodes: lattices[index * 12 + 7], shapesAround: lattices[index * 12 + 8] };
    // The nodes on those rows are made here too, and written over the engine's.
    this.rowsPlaced = null;
    if (e._tp_rows_given()) {
      this.rowsPlaced = this.rowPlans[e._tp_rows_scale()];
      for (const plan of this.rowsPlaced) { plan.nodes = this.rowPlanner.writeNodes(plan.which, this.lattice(plan.which), plan, cpuNodes); }
    }
    // The admitted u-cells of every lattice, made here from its nodes.
    this.compileCells(lattices);
    const count = e._tp_length(8);
    this.parts = [];
    const trims = (source, first, n) => {
      const a = new Float32Array(24), b = new Float32Array(24), c = new Float32Array(24);
      for (let k = 0; k < Math.min(6, n); ++k) {
        const row = source.subarray((first + k) * 12, (first + k) * 12 + 12);
        a.set(row.subarray(0, 4), k * 4);
        b.set(row.subarray(4, 8), k * 4);
        c.set(row.subarray(8, 12), k * 4);
      }
      return { a, b, c, count: Math.min(6, n) };
    };
    for (let k = 0; k < count; ++k) {
      const m = meta.subarray(k * 24, k * 24 + 24);
      this.parts.push({
        own: this.lattice(m[0]), toward: this.lattice(m[1]), on: this.lattice(m[2]), tube: this.lattice(m[3]), band: m[17] > 0.5 ? 1 : 0,
        spread: m[4], orientation: m[5], interiorFrom: m[6], first: m[7], count: m[8], trims: trims(table, m[9], m[10]), exact: trims(exact, m[9], m[10]),
        group: m[11], per: [m[12], m[13]], twoSided: m[14] > 0.5 ? 1 : 0, cells: [m[15], m[16]], firstVertex: m[18], vertexCount: m[19],
        kind: other[m[21] * 8], other: Array.from(other.subarray(m[21] * 8 + 1, m[21] * 8 + 8)),
      });
      // Does the part close on itself around?
      const made = this.parts[this.parts.length - 1];
      made.wrap = made.kind === 1 ? made.other[5] > 0.5 : made.kind === 2 ? made.other[3] > 0.5 : made.band || made.own.wrap > 0.5;
    }
    this.textureLattices = [0, 1, 2].map((k) => this.lattice(meta[count * 24 + k]));
    this.check = null;
    // The grid of surface points of each part: a texel at every step around and
    // along, the steps dividing the node intervals evenly so that every node row
    // and column is a grid line. The grids lie side by side in one texture,
    // apart so that smoothing does not mix parts; coarser if they do not fit.
    for (let scale = 1; scale > 0.05; scale *= 0.7) {
      let x = 0, y = 0, shelf = 0, fits = true;
      for (const part of this.parts) {
        // Around: 96 steps on the round parts, 48 on the tubes and bands. Along: four steps to a
        // node interval, but at least 96 and at most 288 in all.
        // A part that does not close on itself is one patch of many: three steps to a node interval.
        const around = part.cells[0] * (part.wrap ? Math.max(1, Math.min(16, Math.round((part.cells[0] >= 16 ? 48 : 96) * scale / part.cells[0]))) : 3);
        const along = part.cells[1] * (part.wrap ? Math.max(1, Math.min(16, Math.round(Math.max(part.band ? 32 : 96, Math.min(288, 4 * part.cells[1])) * scale / part.cells[1]))) : 3);
        const w = part.wrap ? around : around + 1, h = along + 1;
        if (x + w + 4 > this.mapSize) { x = 0; y += shelf; shelf = 0; }
        part.rect = [x + 2, y + 2, w, h];
        part.span = [around, along];
        x += w + 4;
        shelf = Math.max(shelf, h + 4);
        fits = fits && y + h + 4 <= this.mapSize;
      }
      if (fits) { break; }
    }
    this.vertexCount = this.parts.reduce((sum, part) => sum + part.rect[2] * part.rect[3], 0);
    this.triangleCount = this.parts.reduce((sum, part) => sum + 2 * part.span[0] * part.span[1], 0);
    // What a traced ray needs of the part it reaches.
    const partTable = new Float32Array(Math.max(1, this.parts.length) * 16);
    this.parts.forEach((part, k) => {
      partTable.set([part.orientation, part.interiorFrom, part.twoSided, 0, part.cells[0], part.cells[1], part.per[0], part.per[1], ...part.rect, part.span[0], part.span[1], part.wrap ? 1 : 0, 0], k * 16);
    });
    if (!this.textures.partTable) { this.textures.partTable = gl.createTexture(); }
    gl.bindTexture(gl.TEXTURE_2D, this.textures.partTable);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, partTable.length / 4, 1, 0, gl.RGBA, gl.FLOAT, partTable);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.bakeSurface();
    this.roomKey = '';
    return true;
  }

  // The CPU's compile() on the GPU: every lattice's admitted u-cells from its nodes, drawn into the cell texture.
  compileCells(lattices) {
    const gl = this.gl, u = this.compiler.uniforms;
    if (!this.cellFramebuffer) { this.cellFramebuffer = gl.createFramebuffer(); }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.cellFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.cells, 0);
    gl.viewport(0, 0, 2048, this.cellRows);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(this.compiler.program);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.textures.shapes); gl.uniform1i(u.uShapes, 1);
    gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, this.textures.nodes); gl.uniform1i(u.uNodes, 15);
    gl.bindVertexArray(this.emptyVao);
    gl.enable(gl.SCISSOR_TEST);
    for (let k = 0; k * 12 < lattices.length; ++k) {
      const l = this.lattice(k), count = l.nv * l.cu * 6;
      gl.uniform1i(u.uCellsBase, l.cells);
      gl.uniform1i(u.uNodesBase, l.nodes);
      gl.uniform1i(u.uShapesBase, l.shapesAround);
      gl.uniform1i(u.uCu, l.cu);
      gl.uniform1i(u.uNu, l.nu);
      gl.uniform1i(u.uNv, l.nv);
      gl.uniform1i(u.uWrap, l.wrap);
      const first = Math.floor(l.cells / 2048), last = Math.floor((l.cells + count - 1) / 2048);
      gl.scissor(0, first, 2048, last - first + 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
  }

  // The surface on the GPU, once per teapot: its point and normal at every
  // texel of every part's grid; what a ray needs to know of each texel; and
  // the pyramid of boxes over the grid's cells. Everything drawn, lit,
  // traced or collided afterwards starts from these.
  bakeSurface() {
    const gl = this.gl, size = this.mapSize;
    const surface = this.pair('mapSurface', size, gl.RGBA32F);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, surface.framebuffer);
    gl.viewport(0, 0, size, size);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    gl.useProgram(this.mapSurface.program);
    let u = this.mapSurface.uniforms;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.textures.cells); gl.uniform1i(u.uCells, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.textures.shapes); gl.uniform1i(u.uShapes, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.textures.trims); gl.uniform1i(u.uTrimData, 2);
    gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, this.textures.nodes); gl.uniform1i(u.uNodes, 15);
    gl.activeTexture(gl.TEXTURE11); gl.bindTexture(gl.TEXTURE_2D, this.textures.patches); gl.uniform1i(u.uPatches, 11);
    gl.uniform2f(u.uMapSize, size, size);
    gl.uniform1i(u.uCut, 0);
    gl.bindVertexArray(this.emptyVao);
    this.parts.forEach((part, index) => {
      this.setPart(part, index, false, u);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    });
    // Per texel: its part's place in the texture, whether a cell starts there, the part.
    const info = this.target('gridInfo', size, size, gl.RGBA32F);
    gl.bindFramebuffer(gl.FRAMEBUFFER, info.framebuffer);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.useProgram(this.mapInfo.program);
    u = this.mapInfo.uniforms;
    gl.uniform2f(u.uMapSize, size, size);
    this.parts.forEach((part, index) => {
      gl.uniform4fv(u.uRect, part.rect);
      gl.uniform2f(u.uGridSpan, part.span[0], part.span[1]);
      gl.uniform1f(u.uPartIndex, index);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    });
    // The pyramid of boxes: one box per cell, then the box of four boxes, up to one box for all.
    const levels = Math.round(Math.log2(size)) + 1;
    if (!this.boxes) {
      const make = () => {
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texStorage2D(gl.TEXTURE_2D, levels, gl.RGBA32F, size, size);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        return texture;
      };
      this.boxes = { low: make(), high: make(), framebuffer: gl.createFramebuffer() };
    }
    const boxes = this.boxes;
    this.topLevel = levels - 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, boxes.framebuffer);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    for (let level = 0; level < levels; ++level) {
      const extent = size >> level;
      for (const texture of [boxes.low, boxes.high]) {
        // While a level is drawn, only the level below it may be read.
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, Math.max(0, level - 1));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, Math.max(0, level - 1));
      }
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, boxes.low, level);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, boxes.high, level);
      gl.viewport(0, 0, extent, extent);
      const program = level === 0 ? this.boxCell : this.boxParent;
      gl.useProgram(program.program);
      u = program.uniforms;
      gl.uniform4f(u.uRect, 0, 0, extent, extent);
      gl.uniform2f(u.uMapSize, extent, extent);
      if (level === 0) {
        gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, null);
        gl.activeTexture(gl.TEXTURE9); gl.bindTexture(gl.TEXTURE_2D, null);
        gl.activeTexture(gl.TEXTURE12); gl.bindTexture(gl.TEXTURE_2D, surface.first); gl.uniform1i(u.uGridPoints, 12);
        gl.activeTexture(gl.TEXTURE13); gl.bindTexture(gl.TEXTURE_2D, surface.second); gl.uniform1i(u.uGridNormals, 13);
        gl.activeTexture(gl.TEXTURE10); gl.bindTexture(gl.TEXTURE_2D, info.texture); gl.uniform1i(u.uGridInfo, 10);
      } else {
        gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, boxes.low); gl.uniform1i(u.uBoxMin, 8);
        gl.activeTexture(gl.TEXTURE9); gl.bindTexture(gl.TEXTURE_2D, boxes.high); gl.uniform1i(u.uBoxMax, 9);
      }
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    for (const texture of [boxes.low, boxes.high]) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, levels - 1);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
  }

  // A framebuffer of two float textures (no depth), for the room-light maps.
  pair(key, size, format) {
    const gl = this.gl;
    if (this.targets[key]) { return this.targets[key]; }
    const make = () => {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, format, size, size);
      const filter = format === gl.RGBA16F ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return texture;
    };
    const first = make(), second = make();
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, first, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, second, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return (this.targets[key] = { framebuffer, first, second, width: size, height: size });
  }

  // Smooth a pair of maps (twice a 1-2-1 average each way), leaving the result in `maps`.
  smooth(maps, scratch) {
    const gl = this.gl, size = this.mapSize, u = this.mapSmooth.uniforms;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(this.mapSmooth.program);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.targets.mapSurface.first); gl.uniform1i(u.uPoints, 0);
    gl.uniform4f(u.uRect, 0, 0, size, size);
    gl.uniform2f(u.uMapSize, size, size);
    gl.bindVertexArray(this.emptyVao);
    gl.viewport(0, 0, size, size);
    for (const [from, to] of [[maps, scratch], [scratch, maps]]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, to.framebuffer);
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
      gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, from.first); gl.uniform1i(u.uFront, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, from.second); gl.uniform1i(u.uBack, 2);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // A view of the teapot at rest along a direction, as the shadow map has one along the light.
  along(state, d) {
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const s = state.slice();
    const rest = [s[33], s[34], s[35]], radius = s[36];
    const n = [s[0] * s[23] + s[3] * s[24] + s[6] * s[25], s[1] * s[23] + s[4] * s[24] + s[7] * s[25], s[2] * s[23] + s[5] * s[24] + s[8] * s[25]];
    s[23] = n[0]; s[24] = n[1]; s[25] = n[2]; s[26] = 0;
    s[19] = d[0]; s[20] = d[1]; s[21] = d[2];
    const dx = unit(cross(Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], d)), dy = cross(d, dx), scale = 1 / (2 * radius);
    return { state: s, eye: [0, 0, 0], viewProj: null, pose: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), shift: [0, 0, 0], shadowX: dx, shadowY: dy, shadowCentre: rest, shadowRadius: radius,
      lightMatrix: new Float32Array([dx[0] / radius, dy[0] / radius, -d[0] * scale, 0, dx[1] / radius, dy[1] / radius, -d[1] * scale, 0, dx[2] / radius, dy[2] / radius, -d[2] * scale, 0,
                                     -dot(dx, rest) / radius, -dot(dy, rest) / radius, dot(d, rest) * scale, 1]) };
  }
  mapUniforms(u, view, d, size) {
    const gl = this.gl;
    gl.uniform3fv(u.uDirection, d);
    gl.uniform3fv(u.uDepthX, view.shadowX);
    gl.uniform3fv(u.uDepthY, view.shadowY);
    gl.uniform3fv(u.uDepthCentre, view.shadowCentre);
    gl.uniform1f(u.uDepthRadius, view.shadowRadius);
    gl.uniform1f(u.uDepthSize, size);
    gl.uniform4f(u.uRect, 0, 0, this.mapSize, this.mapSize);
    gl.uniform2f(u.uMapSize, this.mapSize, this.mapSize);
  }

  // The light the body throws on itself, for the pose and sun as they are:
  // the direct light of every texel, then `gathers` gathers of it. In each,
  // for each direction, the body is peeled into layers along the direction,
  // each layer carrying the light of the side that faces away, and every
  // texel that has the body in front of it takes the layer it faces.
  // It is a sequence of steps, one direction of one gather each, so that it can be spread over frames
  // while the picture keeps the light found before.
  *gatherSteps(state) {
    const gl = this.gl, size = this.mapSize, o = this.options;
    const surface = this.targets.mapSurface, open = this.targets.mapOpen;
    const direct = this.pair('mapDirect', size, gl.RGBA16F);
    // Two maps to gather between, neither the one the picture is using.
    const [a, b] = ['mapThrownA', 'mapThrownB', 'mapThrownC'].map((key) => this.pair(key, size, gl.RGBA16F)).filter((maps) => maps !== this.thrown);
    const layerSize = 256;
    if (!this.layerTexture) {
      this.layerTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.layerTexture);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.RGBA32F, layerSize, layerSize, 8);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    }
    const peel = [this.target('peel0', layerSize, layerSize, gl.RGBA32F), this.target('peel1', layerSize, layerSize, gl.RGBA32F)];
    const quad = (program, target) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, size, size);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(program.program);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, surface.first); gl.uniform1i(program.uniforms.uPoints, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, surface.second); gl.uniform1i(program.uniforms.uNormals, 1);
      gl.bindVertexArray(this.emptyVao);
    };
    // The sun in the teapot's frame, and the teapot's shadow map there.
    const s = state;
    const sun = [s[0] * s[19] + s[3] * s[20] + s[6] * s[21], s[1] * s[19] + s[4] * s[20] + s[7] * s[21], s[2] * s[19] + s[5] * s[20] + s[8] * s[21]];
    const sunView = this.along(state, sun);
    const depth = this.target('mapSun', 1024, 1024, gl.R32F);
    gl.bindFramebuffer(gl.FRAMEBUFFER, depth.framebuffer);
    gl.viewport(0, 0, 1024, 1024);
    gl.clearBufferfv(gl.COLOR, 0, [1e30, 0, 0, 0]);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    this.beginTeapot(sunView, 1);
    this.drawTeapot(false);
    quad(this.mapDirect, direct);
    let u = this.mapDirect.uniforms;
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, depth.texture); gl.uniform1i(u.uDepth, 2);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, open.first); gl.uniform1i(u.uOpenFront, 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, open.second); gl.uniform1i(u.uOpenBack, 5);
    this.mapUniforms(u, sunView, sun, 1024);
    gl.uniform1f(u.uOpenScale, 4 / o.directions);
    gl.uniformMatrix3fv(u.uPose, false, [s[0], s[3], s[6], s[1], s[4], s[7], s[2], s[5], s[8]]);
    gl.uniform1i(u.uSun, o.shadows ? 1 : 0);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    let from = a, to = b;
    gl.bindFramebuffer(gl.FRAMEBUFFER, from.framebuffer);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    const count = o.gatherDirections;
    for (let pass = 0; pass < o.gathers; ++pass) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, to.framebuffer);
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
      gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
      for (let k = 0; k < count; ++k) {
        const z = 1 - (2 * k + 1) / count, r = Math.sqrt(1 - z * z), angle = 2.399963229728653 * k + 0.7;
        const d = [r * Math.cos(angle), z, r * Math.sin(angle)];
        const view = this.along(state, d);
        // Peel the body along the direction.
        gl.enable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        for (let layer = 0; layer < o.layers; ++layer) {
          // Each layer is drawn behind the one before it, then kept in the stack.
          gl.bindFramebuffer(gl.FRAMEBUFFER, peel[layer % 2].framebuffer);
          gl.viewport(0, 0, layerSize, layerSize);
          gl.clearBufferfv(gl.COLOR, 0, [1e30, 0, 0, 0]);
          gl.clear(gl.DEPTH_BUFFER_BIT);
          const pu = this.peel.uniforms, vs = view.state;
          this.current = this.peel;
          gl.useProgram(this.peel.program);
          gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.textures.trims); gl.uniform1i(pu.uTrimData, 2);
          gl.activeTexture(gl.TEXTURE12); gl.bindTexture(gl.TEXTURE_2D, surface.first); gl.uniform1i(pu.uGridPoints, 12);
          gl.activeTexture(gl.TEXTURE13); gl.bindTexture(gl.TEXTURE_2D, surface.second); gl.uniform1i(pu.uGridNormals, 13);
          gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, from.first); gl.uniform1i(pu.uThrownFront, 6);
          gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, from.second); gl.uniform1i(pu.uThrownBack, 7);
          gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, direct.first); gl.uniform1i(pu.uDirectFront, 4);
          gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, direct.second); gl.uniform1i(pu.uDirectBack, 5);
          gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, layer > 0 ? peel[(layer + 1) % 2].texture : null); gl.uniform1i(pu.uPrevLayer, 3);
          gl.uniformMatrix3fv(pu.uPose, false, view.pose);
          gl.uniform3fv(pu.uShift, view.shift);
          gl.uniformMatrix4fv(pu.uViewProj, false, view.lightMatrix);
          gl.uniform3f(pu.uEyeAt, 0, 0, 0);
          gl.uniform2f(pu.uImageOf, 0, 0);
          gl.uniform1i(pu.uCut, vs[22] > 0.5 ? 1 : 0);
          gl.uniform4f(pu.uCutPlane, vs[23], vs[24], vs[25], vs[26]);
          gl.uniform3f(pu.uLight, d[0], d[1], d[2]);
          gl.uniform3fv(pu.uShadowCentre, view.shadowCentre);
          gl.uniform2f(pu.uMapSize, size, size);
          gl.uniform1i(pu.uPeel, layer);
          gl.bindVertexArray(this.emptyVao);
          this.drawTeapot(true);
          gl.activeTexture(gl.TEXTURE11);
          gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.layerTexture);
          gl.copyTexSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, layer, 0, 0, layerSize, layerSize);
          gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
        }
        // Every texel takes the layer it faces.
        quad(this.mapGather, to);
        u = this.mapGather.uniforms;
        gl.activeTexture(gl.TEXTURE10); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.layerTexture); gl.uniform1i(u.uLayers, 10);
        gl.uniform1i(u.uLayerCount, o.layers);
        this.mapUniforms(u, view, d, layerSize);
        gl.uniform1f(u.uWeight, 0.9 * 4 / count);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.disable(gl.BLEND);
        yield;
      }
      [from, to] = [to, from];
    }
    this.smooth(from, to);
    this.thrown = from;
  }

  // Goes on with the gather for the pose in `state`, starting it if `key` is new: `steps` steps of it now.
  // Returns whether it has finished.
  gatherThrown(state, key, steps) {
    const gl = this.gl;
    if (this.gatherFor !== key) {
      this.gatherJob = this.gatherSteps(state.slice());
      this.gatherFor = key;
      this.gatherStarted = performance.now();
      this.gatherFrames = 0;
    }
    let done = false;
    for (let k = 0; k < steps && !done; ++k) { done = this.gatherJob.next().done; }
    ++this.gatherFrames;
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (done) {
      if (this.gatherFrames === 1) {
        const sync = new Uint8Array(4);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, sync);
      }
      this.thrownMs = performance.now() - this.gatherStarted;
      this.thrownFrames = this.gatherFrames;
      this.gatherJob = null;
      this.gatherFor = '';
    }
    return done;
  }

  // What each point of the teapot sees of the room, in the teapot's own
  // frame: for every texel of every part's map and each of its two sides, the
  // open fraction of its hemisphere and the mean open direction. The body is
  // drawn from each of `directions` directions spread over the sphere, and
  // every texel looks itself up in each picture.
  bakeRoom(state) {
    const gl = this.gl, size = this.mapSize, o = this.options;
    const started = performance.now();
    const surface = this.targets.mapSurface;
    const open = this.pair('mapOpen', size, gl.RGBA16F);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    let u;
    // The teapot at rest, with its cutting plane if it has one.
    const s = state.slice();
    const rest = [s[33], s[34], s[35]], radius = s[36];
    const n = [s[0] * s[23] + s[3] * s[24] + s[6] * s[25], s[1] * s[23] + s[4] * s[24] + s[7] * s[25], s[2] * s[23] + s[5] * s[24] + s[8] * s[25]];
    s[23] = n[0]; s[24] = n[1]; s[25] = n[2]; s[26] = 0;
    const depth = this.target('mapDepth', 512, 512, gl.R32F);
    gl.bindFramebuffer(gl.FRAMEBUFFER, open.framebuffer);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for (let k = 0; k < o.directions; ++k) {
      const z = 1 - (2 * k + 1) / o.directions, r = Math.sqrt(1 - z * z), angle = 2.399963229728653 * k;
      const d = [r * Math.cos(angle), z, r * Math.sin(angle)];
      const dx = unit(cross(Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], d)), dy = cross(d, dx), scale = 1 / (2 * radius);
      s[19] = d[0]; s[20] = d[1]; s[21] = d[2];
      const view = { state: s, eye: [0, 0, 0], viewProj: null, pose: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), shift: [0, 0, 0], shadowX: dx, shadowY: dy, shadowCentre: rest, shadowRadius: radius,
        lightMatrix: new Float32Array([dx[0] / radius, dy[0] / radius, -d[0] * scale, 0, dx[1] / radius, dy[1] / radius, -d[1] * scale, 0, dx[2] / radius, dy[2] / radius, -d[2] * scale, 0,
                                       -dot(dx, rest) / radius, -dot(dy, rest) / radius, dot(d, rest) * scale, 1]) };
      // The body from this direction: the distance along it of what is nearest.
      gl.bindFramebuffer(gl.FRAMEBUFFER, depth.framebuffer);
      gl.viewport(0, 0, 512, 512);
      gl.clearBufferfv(gl.COLOR, 0, [1e30, 0, 0, 0]);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      this.beginTeapot(view, 1);
      this.drawTeapot(true);
      // Every texel adds what this direction gives it.
      gl.bindFramebuffer(gl.FRAMEBUFFER, open.framebuffer);
      gl.viewport(0, 0, size, size);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.mapOpen.program);
      u = this.mapOpen.uniforms;
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, surface.first); gl.uniform1i(u.uPoints, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, surface.second); gl.uniform1i(u.uNormals, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, depth.texture); gl.uniform1i(u.uDepth, 2);
      gl.uniform3fv(u.uDirection, d);
      gl.uniform3fv(u.uDepthX, dx);
      gl.uniform3fv(u.uDepthY, dy);
      gl.uniform3fv(u.uDepthCentre, rest);
      gl.uniform1f(u.uDepthRadius, radius);
      gl.uniform1f(u.uDepthSize, 512);
      gl.uniform4f(u.uRect, 0, 0, size, size);
      gl.uniform2f(u.uMapSize, size, size);
      gl.bindVertexArray(this.emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.disable(gl.BLEND);
    this.smooth(open, this.pair('mapScratch', size, gl.RGBA16F));
    gl.enable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const sync = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, sync);
    this.roomMs = performance.now() - started;
  }

  target(key, width, height, format) {
    const gl = this.gl;
    const existing = this.targets[key];
    if (existing && existing.width === width && existing.height === height) { return existing; }
    if (existing) {
      gl.deleteFramebuffer(existing.framebuffer);
      gl.deleteTexture(existing.texture);
      gl.deleteRenderbuffer(existing.depth);
    }
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, format, width, height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const depth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return (this.targets[key] = { framebuffer, texture, depth, width, height });
  }

  setLattice(u, name, lattice) {
    const gl = this.gl;
    for (const field of ['cells', 'shapes', 'cu', 'cv', 'nv', 'wrap']) {
      const location = u[name + '.' + field];
      if (location) { gl.uniform1i(location, lattice[field]); }
    }
  }

  // Uniforms that hold for a whole pass over the teapot.
  beginTeapot(view, mode, program = this.main) {
    const gl = this.gl, u = program.uniforms, o = this.options, s = view.state;
    this.current = program;
    gl.useProgram(program.program);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.textures.cells); gl.uniform1i(u.uCells, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.textures.shapes); gl.uniform1i(u.uShapes, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.textures.trims); gl.uniform1i(u.uTrimData, 2);
    gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, this.textures.nodes); gl.uniform1i(u.uNodes, 15);
    gl.activeTexture(gl.TEXTURE11); gl.bindTexture(gl.TEXTURE_2D, this.textures.patches); gl.uniform1i(u.uPatches, 11);
    gl.uniform1i(u.uFilter, o.filter);
    const flat = this.textureLattices[Math.max(0, o.texture - 1)];
    if (u['uTextureFlat.base']) {
      gl.uniform1i(u['uTextureFlat.base'], flat.nodes);
      gl.uniform1i(u['uTextureFlat.nu'], flat.nu);
      gl.uniform1i(u['uTextureFlat.nv'], flat.nv);
      gl.uniform1i(u['uTextureFlat.wrap'], flat.wrap);
    }
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, mode === 1 ? null : this.targets.shadow.texture); gl.uniform1i(u.uShadowMap, 3);
    gl.uniformMatrix3fv(u.uPose, false, view.pose);
    gl.uniform3fv(u.uShift, view.shift);
    const fromLight = mode === 1;   // drawn along a direction, not from the eye
    gl.uniformMatrix4fv(u.uViewProj, false, fromLight ? view.lightMatrix : view.viewProj);
    gl.uniform3fv(u.uEyeAt, fromLight ? [0, 0, 0] : view.eye);
    gl.uniform1i(u.uProbe, 0);
    const room = o.roomLight && mode === 0 && this.targets.mapOpen ? (o.roomLight > 1 && this.thrown ? 2 : 1) : 0;
    gl.uniform1i(u.uRoomLight, room);
    const thrown = this.thrown;
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, thrown ? thrown.first : null); gl.uniform1i(u.uThrownFront, 6);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, thrown ? thrown.second : null); gl.uniform1i(u.uThrownBack, 7);
    // The grid of surface points the teapot is drawn from.
    gl.activeTexture(gl.TEXTURE12); gl.bindTexture(gl.TEXTURE_2D, this.targets.mapSurface.first); gl.uniform1i(u.uGridPoints, 12);
    gl.activeTexture(gl.TEXTURE13); gl.bindTexture(gl.TEXTURE_2D, this.targets.mapSurface.second); gl.uniform1i(u.uGridNormals, 13);
    gl.uniform1i(u.uImage, view.image ? 1 : 0);
    gl.uniform2f(u.uImageOf, view.image ? 1 : 0, s[31]);
    const maps = room && this.targets.mapOpen;
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, maps ? this.targets.mapOpen.first : null); gl.uniform1i(u.uOpenFront, 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, maps ? this.targets.mapOpen.second : null); gl.uniform1i(u.uOpenBack, 5);
    gl.uniform1f(u.uOpenScale, 4 / o.directions);
    gl.uniform1i(u.uMode, mode);
    gl.uniform1i(u.uCut, s[22] > 0.5 ? 1 : 0);
    gl.uniform4f(u.uCutPlane, s[23], s[24], s[25], s[26]);
    this.roomUniforms(u, view, mode);
    gl.uniform1i(u.uPerPixel, o.perPixel);
    gl.uniform1i(u.uMaterial, o.material);
    gl.uniform1i(u.uGloss, o.gloss);
    gl.uniform1i(u.uLighting, o.lighting);
    gl.uniform1i(u.uWire, o.wire);
    gl.uniform1i(u.uTextureOn, o.texture > 0 ? 1 : 0);
    gl.uniform1i(u.uMapping, o.mapping);
    gl.uniform1i(u.uGroundOn, o.ground);
    this.setLattice(u, 'uTexture', this.textureLattices[Math.max(0, o.texture - 1)]);
    gl.uniform3f(u.uAxis, s[37], s[38], s[39]);
    gl.uniform1i(u.uOpenTop, s[40] > 0.5 ? 1 : 0);
    gl.bindVertexArray(this.emptyVao);
  }

  roomUniforms(u, view, mode) {
    const gl = this.gl, s = view.state, o = this.options;
    // Rays traced against the teapot: its tree and where it is.
    gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, this.boxes.low); gl.uniform1i(u.uBoxMin, 8);
    gl.activeTexture(gl.TEXTURE9); gl.bindTexture(gl.TEXTURE_2D, this.boxes.high); gl.uniform1i(u.uBoxMax, 9);
    gl.activeTexture(gl.TEXTURE10); gl.bindTexture(gl.TEXTURE_2D, this.targets.gridInfo.texture); gl.uniform1i(u.uGridInfo, 10);
    gl.activeTexture(gl.TEXTURE12); gl.bindTexture(gl.TEXTURE_2D, this.targets.mapSurface.first); gl.uniform1i(u.uGridPoints, 12);
    gl.activeTexture(gl.TEXTURE13); gl.bindTexture(gl.TEXTURE_2D, this.targets.mapSurface.second); gl.uniform1i(u.uGridNormals, 13);
    gl.uniform1i(u.uTopLevel, this.topLevel);
    gl.activeTexture(gl.TEXTURE14); gl.bindTexture(gl.TEXTURE_2D, this.textures.partTable); gl.uniform1i(u.uPartTable, 14);
    gl.uniformMatrix3fv(u.uTracePose, false, view.pose);
    gl.uniform3fv(u.uTraceShift, view.shift);
    gl.uniform1i(u.uTraceOn, o.trace && mode === 0 ? 1 : 0);
    gl.uniform1i(u.uTraceCut, s[22] > 0.5 ? 1 : 0);
    gl.uniform3f(u.uTraceCutNormal, s[0] * s[23] + s[3] * s[24] + s[6] * s[25], s[1] * s[23] + s[4] * s[24] + s[7] * s[25], s[2] * s[23] + s[5] * s[24] + s[8] * s[25]);
    gl.uniform3f(u.uLight, s[19], s[20], s[21]);
    gl.uniform3fv(u.uEye, view.eye);
    gl.uniform1f(u.uGround, s[31]);
    gl.uniform1i(u.uShadows, o.shadows && o.lighting && mode !== 1 && mode !== 6 ? 1 : 0);
    gl.uniform3fv(u.uShadowX, view.shadowX);
    gl.uniform3fv(u.uShadowY, view.shadowY);
    gl.uniform3fv(u.uShadowCentre, view.shadowCentre);
    gl.uniform1f(u.uShadowRadius, view.shadowRadius);
    gl.uniform1f(u.uShadowSize, o.shadowSize);
  }

  setPart(part, index, exactTrims, u = this.current.uniforms) {
    const gl = this.gl;
    gl.uniform4fv(u.uRect, part.rect);
    gl.uniform2f(u.uGridSpan, part.span[0], part.span[1]);
    gl.uniform1i(u.uPeriodic, part.wrap ? 1 : 0);
    this.setLattice(u, 'uOwn', part.own);
    this.setLattice(u, 'uToward', part.toward);
    this.setLattice(u, 'uOn', part.on);
    this.setLattice(u, 'uTube', part.tube);
    gl.uniform1i(u.uBand, part.band);
    gl.uniform1f(u.uSpread, part.spread);
    gl.uniform1i(u.uKind, part.kind);
    if (part.kind === 1) {
      ['base', 'pu', 'pv', 'peru', 'perv', 'wrap'].forEach((field, k) => { if (u['uBezier.' + field]) { gl.uniform1i(u['uBezier.' + field], part.other[k]); } });
    } else if (part.kind === 2) {
      ['base', 'nu', 'nv', 'wrap'].forEach((field, k) => { if (u['uFlat.' + field]) { gl.uniform1i(u['uFlat.' + field], part.other[k]); } });
    }
    gl.uniform1f(u.uNearer, part.band && this.current !== this.conv ? 0.0015 : 0.0);
    gl.uniform1f(u.uPartIndex, index);
    gl.uniform1f(u.uOrientation, part.orientation);
    gl.uniform1f(u.uInteriorFrom, part.interiorFrom);
    gl.uniform1i(u.uTwoSided, part.twoSided);
    gl.uniform2f(u.uPer, part.per[0], part.per[1]);
    gl.uniform2f(u.uCellCount, part.cells[0], part.cells[1]);
    const trims = exactTrims ? part.exact : part.trims;
    gl.uniform1i(u.uTrimCount, trims.count);
    gl.uniform4fv(u.uTrimA, trims.a);
    gl.uniform4fv(u.uTrimB, trims.b);
    gl.uniform4fv(u.uTrimC, trims.c);
  }

  drawTeapot(exactTrims) {
    const gl = this.gl;
    this.parts.forEach((part, index) => {
      this.setPart(part, index, exactTrims);
      gl.drawArrays(gl.TRIANGLES, 0, 6 * part.span[0] * part.span[1]);
    });
  }

  // Camera, pose and light for this frame, from the engine's state.
  view(state, width, height) {
    const s = state;
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
    const eye = [s[12], s[13], s[14]], target = [s[15], s[16], s[17]];
    const forward = unit(sub(target, eye)), right = unit(cross(forward, [0, 1, 0])), up = cross(right, forward);
    const tan = Math.tan(0.5 * s[18]), aspect = width / height, near = 0.05, far = 400;
    // Column-major: projection times the camera's frame.
    const fx = 1 / (tan * aspect), fy = 1 / tan, a = -(far + near) / (far - near), b = -2 * far * near / (far - near);
    const rows = [right, up, [-forward[0], -forward[1], -forward[2]]];
    const t = rows.map((r) => -dot(r, eye));
    const viewProj = new Float32Array([
      fx * rows[0][0], fy * rows[1][0], a * rows[2][0], -rows[2][0],
      fx * rows[0][1], fy * rows[1][1], a * rows[2][1], -rows[2][1],
      fx * rows[0][2], fy * rows[1][2], a * rows[2][2], -rows[2][2],
      fx * t[0], fy * t[1], a * t[2] + b, -t[2]]);
    // The pose: rotation by rows in the state, by columns for the shader.
    const pose = new Float32Array([s[0], s[3], s[6], s[1], s[4], s[7], s[2], s[5], s[8]]);
    const shift = [s[9], s[10], s[11]];
    const light = [s[19], s[20], s[21]];
    // The shadow map looks along the light at a sphere that holds the teapot.
    const rest = [s[33], s[34], s[35]];
    const centre = [s[0] * rest[0] + s[1] * rest[1] + s[2] * rest[2] + shift[0], s[3] * rest[0] + s[4] * rest[1] + s[5] * rest[2] + shift[1],
                    s[6] * rest[0] + s[7] * rest[1] + s[8] * rest[2] + shift[2]];
    const radius = s[36];
    const helper = Math.abs(light[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const shadowX = unit(cross(helper, light)), shadowY = cross(light, shadowX);
    const depth = 1 / (2 * radius);
    const lightMatrix = new Float32Array([
      shadowX[0] / radius, shadowY[0] / radius, -light[0] * depth, 0,
      shadowX[1] / radius, shadowY[1] / radius, -light[1] * depth, 0,
      shadowX[2] / radius, shadowY[2] / radius, -light[2] * depth, 0,
      -dot(shadowX, centre) / radius, -dot(shadowY, centre) / radius, dot(light, centre) * depth, 1]);
    return { state: s, eye, forward, right: right.map((x) => x * tan * aspect), up: up.map((x) => x * tan), viewProj, pose, shift, light, lightMatrix,
             shadowX, shadowY, shadowCentre: centre, shadowRadius: radius, width, height };
  }

  draw(state) {
    const gl = this.gl, o = this.options;
    const width = this.canvas.width, height = this.canvas.height;
    const view = this.view(state, width, height);
    const shadow = this.target('shadow', o.shadowSize, o.shadowSize, gl.R32F);
    // What each point sees of the room changes only with the geometry and the cut.
    const roomKey = [this.parts.length, this.vertexCount, state[22], state[22] > 0.5 ? Math.sign(state[23] * state[2] + state[24] * state[5] + state[25] * state[8]) : 0, o.directions].join(',');
    if (o.roomLight && o.lighting && roomKey !== this.roomKey) {
      this.bakeRoom(state);
      this.roomKey = roomKey;
      this.thrownKey = '';
      this.thrown = null;
      this.gatherJob = null;
      this.gatherFor = '';
    }
    // What the body throws on itself changes when the room or the sun turns
    // relative to it: found again when the teapot is at rest in a new attitude.
    const thrownKey = roomKey + '|' + Array.from(state.slice(0, 9), (x) => x.toFixed(5)).join(',') + '|' + o.shadows;
    if (o.roomLight > 1 && o.lighting && thrownKey !== this.thrownKey && state[27] < 0.5) {
      // All at once when there is nothing to show meanwhile; otherwise a few directions a frame.
      const steps = this.thrown && o.gatherPerFrame > 0 ? o.gatherPerFrame : 1e9;
      if (this.gatherThrown(state, thrownKey, steps)) { this.thrownKey = thrownKey; }
    } else if (this.gatherJob) {
      // The body moved again: what was being gathered is dropped.
      this.gatherJob = null;
      this.gatherFor = '';
    }
    this.gathering = !!this.gatherJob;
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.STENCIL_TEST);

    // 1. The distance along the light of what is nearest the light.
    if (o.shadows && o.lighting) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, shadow.framebuffer);
      gl.viewport(0, 0, shadow.width, shadow.height);
      gl.clearBufferfv(gl.COLOR, 0, [1e30, 0, 0, 0]);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      this.beginTeapot(view, 1);
      this.drawTeapot(false);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.87, 0.88, 0.9, 1);
    gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);

    // 2. A cutaway: count, at every pixel, how often its ray crosses the
    // surface in the half that is kept. An odd count means the cutting
    // plane is inside material there.
    const cut = state[22] > 0.5;
    if (cut) {
      gl.enable(gl.STENCIL_TEST);
      gl.stencilFunc(gl.ALWAYS, 0, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.INVERT);
      gl.colorMask(false, false, false, false);
      gl.depthMask(false);
      gl.disable(gl.DEPTH_TEST);
      this.beginTeapot(view, 5);
      this.drawTeapot(true);
      gl.colorMask(true, true, true, true);
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.STENCIL_TEST);
    }

    // 3. The teapot: first only where it is (depth), then its colour at the
    // nearest surface of each pixel, so that the surface is evaluated and lit
    // once per pixel and not once per layer of the teapot behind it.
    gl.colorMask(false, false, false, false);
    this.beginTeapot(view, 5);
    this.drawTeapot(false);
    gl.colorMask(true, true, true, true);
    gl.depthMask(false);
    this.beginTeapot(view, 0);
    this.drawTeapot(false);
    gl.depthMask(true);

    // A polished floor: the teapot's image in it, as the eye mirrored in the floor would see the teapot.
    const polished = o.polished && o.ground && o.lighting && view.eye[1] > state[31];
    if (polished) {
      const image = this.target('image', width, height, gl.RGBA8);
      gl.bindFramebuffer(gl.FRAMEBUFFER, image.framebuffer);
      gl.viewport(0, 0, width, height);
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      this.beginTeapot({ ...view, eye: [view.eye[0], 2 * state[31] - view.eye[1], view.eye[2]], image: true }, 0);
      gl.uniform3fv(this.main.uniforms.uEyeAt, view.eye);
      this.drawTeapot(false);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
    }

    // 4. The cut face, where the count was odd and nothing kept is in front.
    if (cut) {
      const n = [state[23], state[24], state[25]], offset = state[26];
      const eyeSide = n[0] * view.eye[0] + n[1] * view.eye[1] + n[2] * view.eye[2] > offset;
      if (eyeSide) {
        const helper = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
        const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let ax = cross(helper, n);
        const l = Math.hypot(ax[0], ax[1], ax[2]);
        ax = ax.map((x) => x / l);
        const ay = cross(n, ax);
        const c = view.shadowCentre, d = offset - (n[0] * c[0] + n[1] * c[1] + n[2] * c[2]);
        const origin = [c[0] + n[0] * d, c[1] + n[1] * d, c[2] + n[2] * d];
        const r = view.shadowRadius * 1.5;
        const corner = (i, j) => [0, 1, 2].map((k) => origin[k] + ax[k] * r * i + ay[k] * r * j);
        const quad = new Float32Array([...corner(-1, -1), ...corner(1, -1), ...corner(1, 1), ...corner(-1, -1), ...corner(1, 1), ...corner(-1, 1)]);
        gl.useProgram(this.cap.program);
        gl.bindVertexArray(this.capVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.capBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        const u = this.cap.uniforms;
        gl.uniformMatrix4fv(u.uViewProj, false, view.viewProj);
        gl.uniform4f(u.uCutPlane, n[0], n[1], n[2], offset);
        gl.uniform3f(u.uLight, o.lighting ? state[19] : 0, o.lighting ? state[20] : 0, o.lighting ? state[21] : 0);
        gl.uniform3fv(u.uEye, view.eye);
        gl.enable(gl.STENCIL_TEST);
        gl.stencilFunc(gl.EQUAL, 1, 1);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.disable(gl.STENCIL_TEST);
      }
    }

    // 5. The floor and the room, wherever nothing nearer was drawn.
    gl.useProgram(this.room.program);
    const u = this.room.uniforms;
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, shadow.texture); gl.uniform1i(u.uShadowMap, 3);
    this.roomUniforms(u, view, 0);
    gl.uniform3fv(u.uForward, view.forward);
    gl.uniform3fv(u.uRight, view.right);
    gl.uniform3fv(u.uUp, view.up);
    gl.uniformMatrix4fv(u.uViewProj, false, view.viewProj);
    gl.uniform1i(u.uGroundOn, o.ground);
    gl.uniform1i(u.uGloss, o.gloss || o.polished ? 1 : 0);
    gl.uniform1i(u.uLighting, o.lighting);
    gl.uniform1f(u.uSink, 0.012);
    gl.uniform1i(u.uPolished, polished ? 1 : 0);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, polished ? this.targets.image.texture : null); gl.uniform1i(u.uImageMap, 4);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    this.lastView = view;
  }

  // Which part is at a pixel of the canvas, and where on it: [part, u, v], part < 0 for none.
  pick(x, y) {
    if (!this.lastView) { return [-1, 0, 0]; }
    const gl = this.gl, view = this.lastView;
    const target = this.target('pick', view.width, view.height, gl.RGBA32F);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.viewport(0, 0, view.width, view.height);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    this.beginTeapot(view, 2);
    this.drawTeapot(false);
    const pixel = new Float32Array(4);
    gl.readPixels(Math.floor(x), view.height - 1 - Math.floor(y), 1, 1, gl.RGBA, gl.FLOAT, pixel);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return [Math.round(pixel[0]) - 1, pixel[1], pixel[2]];
  }

  // The shaders' surface against the CPU's, at points spread over every part:
  // the largest distance between the two, and the largest angle between their normals.
  // The rows placed here beside the engine's own plan: rows, errors (the engine's
  // measure of both plans) and the largest distance between a row and its counterpart.
  rowsCheck() {
    if (!this.engine._tp_rows_given()) { return null; }
    return [0, 1].map((which) => {
      const made = this.rowPlanner.plan(this.engine, which, 1);
      return { which, rows: made.rows, error: made.error, measured: made.own.given, ownRows: made.own.rows, ownError: made.own.error, apart: made.own.apart, first: made.own.firstApart };
    });
  }

  compare(perPart = 1500) {
    const gl = this.gl, e = this.engine;
    if (!this.lastView) { return null; }
    let worst = 0, worstAngle = 0, points = 0;
    const identity = { state: this.lastView.state, eye: [0, 0, 0], viewProj: this.lastView.viewProj, lightMatrix: this.lastView.lightMatrix, pose: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
                       shift: [0, 0, 0], shadowX: [1, 0, 0], shadowY: [0, 1, 0], shadowCentre: [0, 0, 0], shadowRadius: 1 };
    const target = this.target('probe', perPart, 1, gl.RGBA32F);
    let seed = 12345;
    const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    this.parts.forEach((part, index) => {
      const uvs = new Float32Array(perPart * 6);
      for (let k = 0; k < perPart; ++k) {
        uvs[k * 6] = random() * part.cells[0];
        uvs[k * 6 + 1] = 0.02 + random() * (part.cells[1] - 0.04);
        uvs[k * 6 + 4] = 1;
      }
      gl.bindVertexArray(this.probeVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.probeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      const read = (mode) => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
        gl.viewport(0, 0, perPart, 1);
        gl.disable(gl.DEPTH_TEST);
        this.beginTeapot(identity, mode, this.conv);
        gl.bindVertexArray(this.probeVao);
        this.setPart(part, index, false);
        gl.uniform1i(this.conv.uniforms.uProbe, perPart);
        gl.drawArrays(gl.POINTS, 0, perPart);
        const out = new Float32Array(perPart * 4);
        gl.readPixels(0, 0, perPart, 1, gl.RGBA, gl.FLOAT, out);
        return out;
      };
      const position = read(3), normal = read(4);
      for (let k = 0; k < perPart; ++k) {
        const c = new Float64Array(e.HEAPF64.buffer, e._tp_evaluate(index, uvs[k * 6], uvs[k * 6 + 1]), 9);
        worst = Math.max(worst, Math.hypot(position[k * 4] - c[0], position[k * 4 + 1] - c[1], position[k * 4 + 2] - c[2]));
        const n = [c[4] * c[8] - c[5] * c[7], c[5] * c[6] - c[3] * c[8], c[3] * c[7] - c[4] * c[6]];
        const length = Math.hypot(n[0], n[1], n[2]);
        if (length > 1e-9) {
          const cosine = (n[0] * normal[k * 4] + n[1] * normal[k * 4 + 1] + n[2] * normal[k * 4 + 2]) / length;
          worstAngle = Math.max(worstAngle, Math.acos(Math.min(1, Math.abs(cosine))) * 180 / Math.PI);
        }
        ++points;
      }
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    this.check = { worst, worstAngle, points };
    return this.check;
  }
}
