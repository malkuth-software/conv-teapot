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
    this.options = { material: 0, gloss: 0, lighting: 1, shadows: 1, ground: 1, wire: 0, texture: 0, mapping: 0, perPixel: 1, shadowSize: 2048 };
    this.main = this.program(VERTEX_GLSL, FRAGMENT_GLSL);
    this.room = this.program(ROOM_VERTEX_GLSL, ROOM_FRAGMENT_GLSL);
    this.cap = this.program(CAP_VERTEX_GLSL, CAP_FRAGMENT_GLSL);
    this.textures = {};
    this.vao = gl.createVertexArray();
    this.vertexBuffer = gl.createBuffer();
    this.indexBuffer = gl.createBuffer();
    this.capVao = gl.createVertexArray();
    this.capBuffer = gl.createBuffer();
    this.probeVao = gl.createVertexArray();
    this.probeBuffer = gl.createBuffer();
    this.emptyVao = gl.createVertexArray();
    this.targets = {};
    this.parts = [];
    this.check = null;
  }

  program(vertexSource, fragmentSource) {
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
    if (e._tp_build() === 0 && this.parts.length > 0) { return false; }
    const floats = (kind) => new Float32Array(e.HEAPF32.buffer, e._tp_buffer(kind), e._tp_length(kind)).slice();
    this.floatTexture('cells', floats(0), 4);
    this.floatTexture('shapes', floats(1), 4);
    this.floatTexture('trims', floats(7), 1);
    const lattices = floats(2), meta = floats(3), table = floats(6), exact = floats(9);
    this.lattice = (index) => index < 0 ? { cells: 0, shapes: 0, cu: 1, cv: 1, nv: 2, wrap: 0 }
      : { cells: lattices[index * 8], shapes: lattices[index * 8 + 1], cu: lattices[index * 8 + 2], cv: lattices[index * 8 + 3], nv: lattices[index * 8 + 4], wrap: lattices[index * 8 + 5] };
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
      });
    }
    this.textureLattices = [0, 1, 2].map((k) => this.lattice(meta[count * 24 + k]));
    const vertices = floats(4);
    const indices = new Uint32Array(e.HEAPU32.buffer, e._tp_buffer(5), e._tp_length(5)).slice();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.vertexCount = vertices.length / 6;
    this.triangleCount = indices.length / 3;
    this.check = null;
    return true;
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
  beginTeapot(view, mode) {
    const gl = this.gl, u = this.main.uniforms, o = this.options, s = view.state;
    gl.useProgram(this.main.program);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.textures.cells); gl.uniform1i(u.uCells, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.textures.shapes); gl.uniform1i(u.uShapes, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.textures.trims); gl.uniform1i(u.uTrimData, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, mode === 1 ? null : this.targets.shadow.texture); gl.uniform1i(u.uShadowMap, 3);
    gl.uniformMatrix3fv(u.uPose, false, view.pose);
    gl.uniform3fv(u.uShift, view.shift);
    gl.uniformMatrix4fv(u.uViewProj, false, mode === 1 ? view.lightMatrix : view.viewProj);
    gl.uniform3fv(u.uEyeAt, view.eye);
    gl.uniform1i(u.uProbe, 0);
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
    gl.bindVertexArray(this.vao);
  }

  roomUniforms(u, view, mode) {
    const gl = this.gl, s = view.state, o = this.options;
    gl.uniform3f(u.uLight, s[19], s[20], s[21]);
    gl.uniform3fv(u.uEye, view.eye);
    gl.uniform1f(u.uGround, s[31]);
    gl.uniform1i(u.uShadows, o.shadows && o.lighting && mode !== 1 ? 1 : 0);
    gl.uniform3fv(u.uShadowX, view.shadowX);
    gl.uniform3fv(u.uShadowY, view.shadowY);
    gl.uniform3fv(u.uShadowCentre, view.shadowCentre);
    gl.uniform1f(u.uShadowRadius, view.shadowRadius);
    gl.uniform1f(u.uShadowSize, o.shadowSize);
  }

  setPart(part, index, exactTrims) {
    const gl = this.gl, u = this.main.uniforms;
    this.setLattice(u, 'uOwn', part.own);
    this.setLattice(u, 'uToward', part.toward);
    this.setLattice(u, 'uOn', part.on);
    this.setLattice(u, 'uTube', part.tube);
    gl.uniform1i(u.uBand, part.band);
    gl.uniform1f(u.uSpread, part.spread);
    gl.uniform1f(u.uNearer, part.band ? 0.0015 : 0.0);
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
      if (part.count === 0) { return; }
      this.setPart(part, index, exactTrims);
      gl.drawElements(gl.TRIANGLES, part.count, gl.UNSIGNED_INT, part.first * 4);
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

    // 3. The teapot.
    this.beginTeapot(view, 0);
    this.drawTeapot(false);

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
    gl.uniform1i(u.uGloss, o.gloss);
    gl.uniform1i(u.uLighting, o.lighting);
    gl.uniform1f(u.uSink, 0.012);
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
  compare(perPart = 1500) {
    const gl = this.gl, e = this.engine;
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
        this.beginTeapot(identity, mode);
        gl.bindVertexArray(this.probeVao);
        this.setPart(part, index, false);
        gl.uniform1i(this.main.uniforms.uProbe, perPart);
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
