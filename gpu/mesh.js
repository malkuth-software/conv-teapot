// The mesh the teapot is drawn and traced from: the bounded form of the
// paper's layouts from the metric (tools/mesh_study.cpp, "nodes bounded,
// checked"). The seeds are the lattice's own nodes; each lattice cell is
// meshed alone, with more vertices only inside the cell that needs them; the
// test is the checked one: a chord may stray from the surface by no more
// than a distance, and the normal may turn along it by no more than an angle.
//
// Adapted to how a GPU works, each step is a pass in which every cell, edge
// or vertex is found independently:
//   factors   one fragment a lattice cell: along the cell's two edges and its
//             middle line each way, the fewest equal steps (1, 2, 4, 8 or 16)
//             that pass the test. The cell's interior takes the largest each
//             way; each edge keeps its own count. An edge's count depends on
//             the edge alone, so the two cells that share it agree without
//             being told;
//   layout    the cells' triangle counts are summed in order (on the CPU: a
//             few thousand numbers) so that every triangle has a place;
//   vertices  one fragment a triangle corner: which cell, which corner of
//             which step, snapped along an edge to that edge's own count so
//             that neighbours meet; the surface there, its normal, whether it
//             is shown. Three texels a triangle, in order: drawing and tracing
//             read them straight through;
//   boxes     one fragment a pair of triangles (one step of a cell), its box;
//             one fragment a cell, the box of its pairs; then one fragment a
//             node of a tree, the box of the cells or pairs under it. The
//             tree halves a part's cells by where they are in space, down to
//             one cell, and goes on halving the cell's steps down to one
//             pair. Which cells or pairs a node stands for, and where it lies
//             in the list, is planned on the CPU from the cells' boxes, which
//             the GPU finds (a few thousand boxes); the nodes are written
//             in the order a ray walks them, each with the node to go on from
//             if the ray misses it. A ray needs no stack and makes no
//             decision but "hit or miss".
const MESH_COMMON_GLSL = `
uniform vec2 uCellCount;
uniform int uPeriodic;
// The surface and its unit normal; at a pole, the normal a little way along the part.
void surfaceAt(float u, float v, out vec3 p, out vec3 n) {
  vec3 du, dv;
  evalSurface(u, v, p, du, dv);
  n = cross(du, dv);
  if (dot(n, n) < 1e-16) {
    vec3 q;
    evalSurface(u, v < 0.5 * uCellCount.y ? v + 0.05 : v - 0.05, q, du, dv);
    n = cross(du, dv);
  }
  n = normalize(n);
}
// A column around a part that closes on itself is evaluated where its twin is: the last is the first.
float around(float u) { return uPeriodic != 0 && u >= uCellCount.x ? u - uCellCount.x : u; }
`;

// The surface at eight places to a cell each way, every part's samples side by side: point and
// whether it is shown; unit normal. Each is found once and read by the cells on both sides of it.
const MESH_SAMPLE_GLSL = `#version 300 es
${CONV_GLSL}
${TRIM_GLSL}
${MESH_COMMON_GLSL}
uniform vec4 uRect;
layout(location = 0) out vec4 outPoint;
layout(location = 1) out vec4 outNormal;
void main() {
  vec2 at = floor(gl_FragCoord.xy - uRect.xy) / 8.0;
  vec3 p, n;
  surfaceAt(at.x, at.y, p, n);
  outPoint = vec4(p, hidden(p, at.x, at.y) ? 0.0 : 1.0);
  outNormal = vec4(n, 0.0);
}
`;

const MESH_FACTOR_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uSamplePoints, uSampleNormals;
uniform vec4 uRect;          // the part's cells in the target
uniform vec4 uSampleRect;    // the part's samples: eight to a cell
uniform int uPeriodic;
uniform float uTolerance, uAngle;
out vec4 outFactors;
ivec2 sampleAt(ivec2 at) {
  int wide = int(uSampleRect.z);
  return ivec2(uSampleRect.xy) + ivec2(uPeriodic != 0 ? at.x % wide : at.x, at.y);
}
// The fewest equal steps along one side or middle line of the cell that pass, from its eight
// pieces: a chord of 1/k of the line strays by |P''| / (8 k^2) and the normal turns by
// |N'| / k along it, so k is the larger of sqrt(|P''| / (8 tolerance)) and |N'| / angle where
// they are largest.
float stepsAlong(ivec2 from, ivec2 direction) {
  float bend = 0.0, turn = 0.0, size = 0.0;
  for (int k = 0; k < 8; ++k) {
    ivec2 a = sampleAt(from + direction * k), b = sampleAt(from + direction * (k + 1));
    vec3 pa = texelFetch(uSamplePoints, a, 0).xyz, pb = texelFetch(uSamplePoints, b, 0).xyz;
    size += distance(pa, pb);
    turn = max(turn, acos(clamp(dot(texelFetch(uSampleNormals, a, 0).xyz, texelFetch(uSampleNormals, b, 0).xyz), -1.0, 1.0)));
    if (k > 0) { bend = max(bend, length(texelFetch(uSamplePoints, sampleAt(from + direction * (k - 1)), 0).xyz - 2.0 * pa + pb)); }
  }
  if (size < 1e-7) { return 1.0; }   // a pole: no length
  // bend is |P''| / 64 and turn is |N'| / 8.
  return clamp(ceil(max(sqrt(8.0 * bend / uTolerance), 8.0 * turn / uAngle) - 1e-3), 1.0, 16.0);
}
void main() {
  ivec2 cell = ivec2(floor(gl_FragCoord.xy - uRect.xy)) * 8;
  float low = stepsAlong(cell, ivec2(1, 0)), middle = stepsAlong(cell + ivec2(0, 4), ivec2(1, 0)), high = stepsAlong(cell + ivec2(0, 8), ivec2(1, 0));
  float left = stepsAlong(cell, ivec2(0, 1)), centre = stepsAlong(cell + ivec2(4, 0), ivec2(0, 1)), right = stepsAlong(cell + ivec2(8, 0), ivec2(0, 1));
  float a = max(low, max(middle, high)), b = max(left, max(centre, right));
  // A cell a join's edge crosses is meshed finer, so that its triangles follow the edge.
  float shown = 0.0;
  for (int k = 0; k < 9; ++k) { shown += texelFetch(uSamplePoints, sampleAt(cell + 4 * ivec2(k % 3, k / 3)), 0).w; }
  if (shown > 0.5 && shown < 8.5) { a = max(a, 8.0); b = max(b, 8.0); }
  outFactors = vec4(a, b, low * 64.0 + high, left * 64.0 + right);
}
`;

// The box of each cell's samples that are shown: what the tree is planned from.
const MESH_SAMPLE_BOX_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uSamplePoints;
uniform vec4 uRect, uSampleRect;
uniform int uPeriodic;
layout(location = 0) out vec4 outLow;
layout(location = 1) out vec4 outHigh;
void main() {
  ivec2 cell = ivec2(floor(gl_FragCoord.xy - uRect.xy)) * 8, origin = ivec2(uSampleRect.xy), size = ivec2(uSampleRect.zw);
  vec3 low = vec3(1e30), high = vec3(-1e30);
  for (int y = 0; y <= 8; ++y) {
    for (int x = 0; x <= 8; ++x) {
      ivec2 at = cell + ivec2(x, y);
      if (uPeriodic != 0) { at.x = at.x % size.x; }
      vec4 p = texelFetch(uSamplePoints, origin + at, 0);
      if (p.w < 0.5) { continue; }
      low = min(low, p.xyz);
      high = max(high, p.xyz);
    }
  }
  outLow = vec4(low, 0.0);
  outHigh = vec4(high, 0.0);
}
`;

// Which blocks of its own part a cell is not clear of. A part's tree has up to sixteen blocks
// (its nodes four levels down). A block is clear of a cell if all of it lies on or behind the
// tangent plane at nine places of the cell: a ray that leaves the cell outward stays in front
// of its own tangent plane and so cannot meet that block. The answer is a number whose bit b
// is set if block b is not clear; 0 means the ray has only the other parts to look for.
const MESH_CLEAR_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uSamplePoints, uSampleNormals, uCellBlock;
uniform vec4 uRect, uSampleRect;
uniform int uPeriodic, uCellBase, uCu, uCv;
uniform float uOrientation, uMargin;
out vec4 outClear;
void main() {
  ivec2 cell = ivec2(floor(gl_FragCoord.xy - uRect.xy)) * 8, origin = ivec2(uSampleRect.xy), size = ivec2(uSampleRect.zw);
  vec3 point[9], normal[9];
  for (int k = 0; k < 9; ++k) {
    ivec2 at = cell + 4 * ivec2(k % 3, k / 3);
    if (uPeriodic != 0) { at.x = at.x % size.x; }
    point[k] = texelFetch(uSamplePoints, origin + at, 0).xyz;
    normal[k] = texelFetch(uSampleNormals, origin + at, 0).xyz * uOrientation;
  }
  int mask = 0;
  for (int j = 0; j < uCv; ++j) {
    for (int i = 0; i < uCu; ++i) {
      int index = uCellBase + j * uCu + i;
      int block = int(texelFetch(uCellBlock, ivec2(index & 2047, index >> 11), 0).x);
      if (block < 0 || ((mask >> block) & 1) != 0) { continue; }
      // Every other sample of the other cell, its edges included.
      float above = -1e30;
      for (int y = 0; y <= 8; y += 2) {
        for (int x = 0; x <= 8; x += 2) {
          ivec2 at = ivec2(8 * i + x, 8 * j + y);
          if (uPeriodic != 0) { at.x = at.x % size.x; }
          vec4 p = texelFetch(uSamplePoints, origin + at, 0);
          if (p.w < 0.5) { continue; }   // under a join
          for (int k = 0; k < 9; ++k) { above = max(above, dot(normal[k], p.xyz - point[k])); }
        }
      }
      if (above > uMargin) { mask |= 1 << block; }
    }
  }
  outClear = vec4(float(mask), 0.0, 0.0, 0.0);
}
`;

const MESH_VERTEX_BUILD_GLSL = `#version 300 es
${CONV_GLSL}
${TRIM_GLSL}
${MESH_COMMON_GLSL}
uniform sampler2D uCellTable;   // per cell two texels: first triangle, steps around, steps along, part; column, row, edge counts
uniform sampler2D uClear;       // per cell: which blocks of its own part it is not clear of
uniform sampler2D uCellBlock;   // per cell: its block, or -1 if none of the cell is shown
uniform vec4 uClearRect;
uniform int uCellBase, uCellEnd, uFirstVertex, uVertexEnd;
uniform float uPartIndex;
layout(location = 0) out vec4 outPoint;    // the point; 1 if any of its cell is shown (then a ray meets all of the cell)
layout(location = 1) out vec4 outNormal;   // the unit normal; the blocks of its own part its cell is not clear of
layout(location = 2) out vec4 outWhere;    // u, v, part, cell
vec4 cellAt(int i) { return texelFetch(uCellTable, ivec2(i & 2047, i >> 11), 0); }
void main() {
  int slot = int(gl_FragCoord.y) * 2048 + int(gl_FragCoord.x);
  if (slot < uFirstVertex || slot >= uVertexEnd) { discard; }
  int triangle = slot / 3, corner = slot % 3;
  // The cell this triangle is in: the last whose first triangle is not after it.
  int low = uCellBase, high = uCellEnd;
  for (int step = 0; step < 16 && low + 1 < high; ++step) {
    int mid = (low + high) / 2;
    if (int(cellAt(2 * mid).x) <= triangle) { low = mid; } else { high = mid; }
  }
  vec4 cell = cellAt(2 * low), more = cellAt(2 * low + 1);
  int a = int(cell.y), b = int(cell.z), local = triangle - int(cell.x);
  int quad = local / 2, second = local % 2;
  // The corner of the step: (0,0),(1,0),(1,1) for the first triangle, (0,0),(1,1),(0,1) for the second.
  int dx = corner == 0 ? 0 : (second == 0 || corner == 1) ? 1 : 0;
  int dy = corner == 0 ? 0 : (second == 1 || corner == 2) ? 1 : 0;
  int ci = quad % a + dx, cj = quad / a + dy;
  float fu = float(ci) / float(a), fv = float(cj) / float(b);
  // On an edge, the place of the edge's own vertex at or before this one: the neighbour does the same.
  int low2 = int(more.z) / 64, high2 = int(more.z) % 64, left = int(more.w) / 64, right = int(more.w) % 64;
  if (cj == 0) { fu = float((ci * low2) / a) / float(low2); }
  if (cj == b) { fu = float((ci * high2) / a) / float(high2); }
  if (ci == 0) { fv = float((cj * left) / b) / float(left); }
  if (ci == a) { fv = float((cj * right) / b) / float(right); }
  float u = more.x + fu, v = more.y + fv;
  vec3 p, n;
  surfaceAt(around(u), v, p, n);
  // A join's band and the flat pieces of the part under it do not meet exactly where the part is cut away,
  // so for rays a cell with any of it shown is whole: nothing slips in under the band.
  outPoint = vec4(p, texelFetch(uCellBlock, ivec2(low & 2047, low >> 11), 0).x < -0.5 ? 0.0 : 1.0);
  outNormal = vec4(n, texelFetch(uClear, ivec2(uClearRect.xy) + ivec2(more.xy), 0).x);
  outWhere = vec4(u, v, uPartIndex, float(low));
}
`;

// The box of each pair of triangles (one step of a cell) that is shown, in the triangles' order.
const MESH_QUAD_BOX_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uMeshPoints;
uniform int uQuadCount;
layout(location = 0) out vec4 outLow;
layout(location = 1) out vec4 outHigh;
void main() {
  int index = int(gl_FragCoord.y) * 2048 + int(gl_FragCoord.x);
  outLow = vec4(1e30);
  outHigh = vec4(-1e30);
  if (index >= uQuadCount) { return; }
  vec3 low = vec3(1e30), high = vec3(-1e30);
  float shown = 0.0;
  for (int k = 0; k < 6; ++k) {
    vec4 p = texelFetch(uMeshPoints, ivec2((6 * index + k) & 2047, (6 * index + k) >> 11), 0);
    low = min(low, p.xyz);
    high = max(high, p.xyz);
    shown += p.w;
  }
  if (shown < 0.5) { return; }   // wholly under a join
  outLow = vec4(low, 0.0);
  outHigh = vec4(high, 0.0);
}
`;
// The box of each cell: of its pairs' boxes.
const MESH_CELL_BOX_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uCellTable, uQuadLow, uQuadHigh;
uniform int uCellCount;
layout(location = 0) out vec4 outLow;
layout(location = 1) out vec4 outHigh;
vec4 lin(sampler2D t, int i) { return texelFetch(t, ivec2(i & 2047, i >> 11), 0); }
void main() {
  int index = int(gl_FragCoord.y) * 2048 + int(gl_FragCoord.x);
  outLow = vec4(1e30);
  outHigh = vec4(-1e30);
  if (index >= uCellCount) { return; }
  vec4 cell = lin(uCellTable, 2 * index);
  int first = int(cell.x) / 2, count = int(cell.y) * int(cell.z);
  vec3 low = vec3(1e30), high = vec3(-1e30);
  for (int q = 0; q < count; ++q) {
    low = min(low, lin(uQuadLow, first + q).xyz);
    high = max(high, lin(uQuadHigh, first + q).xyz);
  }
  outLow = vec4(low, 0.0);
  outHigh = vec4(high, 0.0);
}
`;
// A node of the tree, from the plan: its box, the node to go on from on a miss, and its two
// triangles if it is one pair. One texel of four whole numbers a node, so that a step of a ray's
// walk is one read: the box's corners in steps of 1/65535 of the teapot's own box, rounded
// outward (low x | low y << 16, low z | high x << 16, high y | high z << 16); then the node to go
// on from in 24 bits, or for a pair its first triangle (the node to go on from is then the next),
// with the top bits saying which: 128 a pair, otherwise the block of its part it is.
// The plan, three texels a node: (0, first of its cells in the list, how many), (0, go on from)
// for a stretch of cells; (1, cell, first step around, first step along), (end around, end along, go on from, 0) inside a cell;
// then (the block of its part it is, from 1, or 0).
const MESH_TREE_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uPlan, uCellTable, uCellList, uCellLow, uCellHigh, uQuadLow, uQuadHigh;
uniform int uNodeCount;
uniform vec3 uTreeLow;
uniform float uTreeStep;
out uvec4 outNode;
vec4 lin(sampler2D t, int i) { return texelFetch(t, ivec2(i & 2047, i >> 11), 0); }
void main() {
  int node = int(gl_FragCoord.y) * 2048 + int(gl_FragCoord.x);
  if (node >= uNodeCount) { outNode = uvec4(0u); return; }
  vec4 plan = lin(uPlan, 3 * node), more = lin(uPlan, 3 * node + 1);
  float band = lin(uPlan, 3 * node + 2).x;
  vec3 low = vec3(1e30), high = vec3(-1e30);
  float next, pair = 0.0, first = 0.0;
  if (plan.x < 0.5) {
    int listed = int(plan.y), count = int(plan.z);
    next = more.y;
    for (int k = 0; k < count; ++k) {
      int cell = int(lin(uCellList, listed + k).x);
      low = min(low, lin(uCellLow, cell).xyz);
      high = max(high, lin(uCellHigh, cell).xyz);
    }
  } else {
    vec4 cell = lin(uCellTable, 2 * int(plan.y));
    int around = int(cell.y), quads = int(cell.x) / 2;
    int x0 = int(plan.z), y0 = int(plan.w), x1 = int(more.x), y1 = int(more.y);
    next = more.z;
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        low = min(low, lin(uQuadLow, quads + y * around + x).xyz);
        high = max(high, lin(uQuadHigh, quads + y * around + x).xyz);
      }
    }
    if (x1 - x0 == 1 && y1 - y0 == 1) {
      pair = 1.0;
      first = cell.x + float(2 * (y0 * around + x0));
    }
  }
  uvec3 a = uvec3(65535u), b = uvec3(0u);   // nothing in it
  if (low.x <= high.x) {
    a = uvec3(clamp(floor((low - uTreeLow) / uTreeStep), 0.0, 65535.0));
    b = uvec3(clamp(ceil((high - uTreeLow) / uTreeStep), 0.0, 65535.0));
  }
  uint last = pair > 0.5 ? uint(first) | (128u << 24) : uint(next) | (uint(band) << 24);
  outNode = uvec4(a.x | (a.y << 16), a.z | (b.x << 16), b.y | (b.z << 16), last);
}
`;

// A vertex of the mesh: read from the three textures by its number.
const MESH_VERTEX_GLSL = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uMeshPoints, uMeshNormals, uMeshWhere;
uniform int uFirstVertex;
uniform mat3 uPose;
uniform vec3 uShift;
uniform mat4 uViewProj;
uniform vec3 uEyeAt;
uniform float uNearer;
uniform vec2 uImageOf;   // x > 0: draw the teapot's image in a floor at height y (as the eye sees it reflected there)
out vec3 vWorld;
out vec3 vNormal;
out vec3 vUvs;
out float vClear;   // the blocks of its own part the cell is not clear of, as bits
void main() {
  int slot = uFirstVertex + gl_VertexID;
  ivec2 texel = ivec2(slot & 2047, slot >> 11);
  vWorld = uPose * texelFetch(uMeshPoints, texel, 0).xyz + uShift;
  vec4 normal = texelFetch(uMeshNormals, texel, 0);
  vNormal = uPose * normal.xyz;
  vClear = normal.w;
  vUvs = vec3(texelFetch(uMeshWhere, texel, 0).xy, 0.0);
  vec3 placed = vWorld + (uEyeAt - vWorld) * uNearer;
  if (uImageOf.x > 0.0) { placed.y = 2.0 * uImageOf.y - placed.y; }
  gl_Position = uViewProj * vec4(placed, 1.0);
}
`;

class BoundedMesh {
  constructor(gpu) {
    this.gpu = gpu;
    const gl = gpu.gl;
    this.sampleProgram = gpu.program(MAP_VERTEX_GLSL, MESH_SAMPLE_GLSL);
    this.factorProgram = gpu.program(MAP_VERTEX_GLSL, MESH_FACTOR_GLSL);
    this.clearProgram = gpu.program(MAP_VERTEX_GLSL, MESH_CLEAR_GLSL);
    this.sampleBoxProgram = gpu.program(MAP_VERTEX_GLSL, MESH_SAMPLE_BOX_GLSL);
    this.sampleBoxes = this.target(512, 512, 2);
    this.clear = this.target(512, 512, 1);
    this.vertexProgram = gpu.program(ROOM_VERTEX_GLSL, MESH_VERTEX_BUILD_GLSL);
    this.quadBoxProgram = gpu.program(ROOM_VERTEX_GLSL, MESH_QUAD_BOX_GLSL);
    this.cellBoxProgram = gpu.program(ROOM_VERTEX_GLSL, MESH_CELL_BOX_GLSL);
    this.treeProgram = gpu.program(ROOM_VERTEX_GLSL, MESH_TREE_GLSL);
    this.factorSize = 512;
    this.factors = this.target(this.factorSize, this.factorSize, 1);
    this.textures = {};
  }

  // A float target of `count` textures (2048 wide for the lists).
  target(width, height, count) {
    const gl = this.gpu.gl;
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const textures = [];
    for (let k = 0; k < count; ++k) {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, width, height);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, gl.TEXTURE_2D, texture, 0);
      textures.push(texture);
    }
    gl.drawBuffers(textures.map((_, k) => gl.COLOR_ATTACHMENT0 + k));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { framebuffer, textures, width, height };
  }
  // A list of whole numbers, four to a texel.
  wholeList(key, texels) {
    const gl = this.gpu.gl, rows = Math.max(1, Math.ceil(texels / 2048));
    const old = this.textures[key];
    if (old && old.height >= rows) { return old; }
    if (old) { gl.deleteTexture(old.textures[0]); gl.deleteFramebuffer(old.framebuffer); }
    const framebuffer = gl.createFramebuffer(), texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, 2048, rows + 8);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return (this.textures[key] = { framebuffer, textures: [texture], width: 2048, height: rows + 8 });
  }
  // A list target with room for `texels`, kept if it is already large enough.
  list(key, texels, count) {
    const gl = this.gpu.gl, rows = Math.max(1, Math.ceil(texels / 2048));
    const old = this.textures[key];
    if (old && old.height >= rows) { return old; }
    if (old) { old.textures.forEach((texture) => gl.deleteTexture(texture)); gl.deleteFramebuffer(old.framebuffer); }
    return (this.textures[key] = this.target(2048, rows + 8, count));
  }

  build(tolerance, degrees) {
    const g = this.gpu, gl = g.gl, parts = g.parts;
    const started = performance.now();
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.SCISSOR_TEST);
    gl.bindVertexArray(g.emptyVao);
    const surfaceInputs = (u) => {
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, g.textures.cells); gl.uniform1i(u.uCells, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, g.textures.shapes); gl.uniform1i(u.uShapes, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, g.textures.trims); gl.uniform1i(u.uTrimData, 2);
      gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, g.textures.nodes); gl.uniform1i(u.uNodes, 15);
      gl.activeTexture(gl.TEXTURE11); gl.bindTexture(gl.TEXTURE_2D, g.textures.patches); gl.uniform1i(u.uPatches, 11);
      gl.uniform1i(u.uCut, 0);
    };

    // 1. Factors: a rectangle of cells for each part, side by side.
    let x = 0, y = 0, shelf = 0;
    for (const part of parts) {
      const [cu, cv] = part.cells;
      if (x + cu > this.factorSize) { x = 0; y += shelf; shelf = 0; }
      part.factorRect = [x, y, cu, cv];
      x += cu;
      shelf = Math.max(shelf, cv);
      if (y + cv > this.factorSize) { throw new Error('too many cells for the mesh'); }
    }
    // The samples the factors are read from: eight to a cell, one more at the far side of a part that does not close.
    let sampleSize = 1024;
    for (;; sampleSize *= 2) {
      let sx = 0, sy = 0, tall = 0, fits = true;
      for (const part of parts) {
        const w = 8 * part.cells[0] + (part.wrap ? 0 : 1), h = 8 * part.cells[1] + 1;
        if (sx + w > sampleSize) { sx = 0; sy += tall; tall = 0; }
        part.sampleRect = [sx, sy, w, h];
        sx += w;
        tall = Math.max(tall, h);
        fits = fits && w <= sampleSize && sy + h <= sampleSize;
      }
      if (fits || sampleSize >= 4096) { break; }
    }
    if (!this.samples || this.samples.width !== sampleSize) {
      if (this.samples) { this.samples.textures.forEach((texture) => gl.deleteTexture(texture)); gl.deleteFramebuffer(this.samples.framebuffer); }
      this.samples = this.target(sampleSize, sampleSize, 2);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.samples.framebuffer);
    gl.viewport(0, 0, sampleSize, sampleSize);
    gl.useProgram(this.sampleProgram.program);
    let u = this.sampleProgram.uniforms;
    surfaceInputs(u);
    gl.uniform2f(u.uMapSize, sampleSize, sampleSize);
    parts.forEach((part, index) => {
      g.setPart(part, index, false, u);
      gl.uniform4fv(u.uRect, part.sampleRect);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.factors.framebuffer);
    gl.viewport(0, 0, this.factorSize, this.factorSize);
    gl.useProgram(this.factorProgram.program);
    u = this.factorProgram.uniforms;
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.samples.textures[0]); gl.uniform1i(u.uSamplePoints, 3);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.samples.textures[1]); gl.uniform1i(u.uSampleNormals, 4);
    gl.uniform2f(u.uMapSize, this.factorSize, this.factorSize);
    gl.uniform1f(u.uTolerance, tolerance);
    gl.uniform1f(u.uAngle, degrees * Math.PI / 180);
    parts.forEach((part, index) => {
      gl.uniform4fv(u.uRect, part.factorRect);
      gl.uniform4fv(u.uSampleRect, part.sampleRect);
      gl.uniform1i(u.uPeriodic, part.wrap ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    });

    // The tree over each part's cells is planned from where the cells are: the box of each cell's
    // samples is found here and read back, and the cells are halved by the middle of their boxes
    // along the longest way, down to one cell (a few thousand boxes; whole numbers and comparisons).
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sampleBoxes.framebuffer);
    gl.useProgram(this.sampleBoxProgram.program);
    u = this.sampleBoxProgram.uniforms;
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.samples.textures[0]); gl.uniform1i(u.uSamplePoints, 3);
    gl.uniform2f(u.uMapSize, this.factorSize, this.factorSize);
    for (const part of parts) {
      gl.uniform4fv(u.uRect, part.factorRect);
      gl.uniform4fv(u.uSampleRect, part.sampleRect);
      gl.uniform1i(u.uPeriodic, part.wrap ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    let cellTotal = 0;
    for (const part of parts) { part.cellBase = cellTotal; cellTotal += part.cells[0] * part.cells[1]; }
    const cellBlock = new Float32Array(cellTotal * 4).fill(-1);
    const least = [1e30, 1e30, 1e30], most = [-1e30, -1e30, -1e30];
    for (const part of parts) {
      const [fx, fy, cu, cv] = part.factorRect, low = new Float32Array(cu * cv * 4), high = new Float32Array(cu * cv * 4);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(fx, fy, cu, cv, gl.RGBA, gl.FLOAT, low);
      gl.readBuffer(gl.COLOR_ATTACHMENT1);
      gl.readPixels(fx, fy, cu, cv, gl.RGBA, gl.FLOAT, high);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      const middle = (c, axis) => 0.5 * (low[c * 4 + axis] + high[c * 4 + axis]);
      const shown = [];
      for (let c = 0; c < cu * cv; ++c) {
        if (!(low[c * 4] <= high[c * 4])) { continue; }
        shown.push(c);
        for (let a = 0; a < 3; ++a) { least[a] = Math.min(least[a], low[c * 4 + a]); most[a] = Math.max(most[a], high[c * 4 + a]); }
      }
      // A node: its cells (a stretch of `order`), its two halves, and which block it is (from 1) four levels down.
      part.order = shown;
      let blocks = 0;
      const split = (first, count, depth) => {
        const node = { first, count, block: 0, halves: null };
        if (depth === 4 || (count === 1 && depth < 4)) {
          node.block = ++blocks;
          for (let k = first; k < first + count; ++k) { cellBlock[(part.cellBase + shown[k]) * 4] = node.block - 1; }
        }
        if (count > 1) {
          let axis = 0, widest = -1;
          for (let a = 0; a < 3; ++a) {
            let least = 1e30, most = -1e30;
            for (let k = first; k < first + count; ++k) { const m = middle(shown[k], a); least = Math.min(least, m); most = Math.max(most, m); }
            if (most - least > widest) { widest = most - least; axis = a; }
          }
          const part2 = shown.slice(first, first + count).sort((a, b) => middle(a, axis) - middle(b, axis));
          for (let k = 0; k < count; ++k) { shown[first + k] = part2[k]; }
          const half = count >> 1;
          node.halves = [split(first, half, depth + 1), split(first + half, count - half, depth + 1)];
        }
        return node;
      };
      part.root = shown.length > 0 ? split(0, shown.length, 0) : null;
      part.blocks = blocks;
    }
    g.floatTexture('meshCellBlock', cellBlock, 4);
    // The teapot's own box, a little larger, in 65535 steps: the tree's boxes are written in these.
    const widest = Math.max(1e-6, most[0] - least[0], most[1] - least[1], most[2] - least[2]) * 1.02;
    this.treeLow = least.map((x, a) => x - 0.5 * (widest - (most[a] - least[a])) - 0.005 * widest);
    this.treeStep = widest * 1.01 / 65535;

    // Which blocks of its own part each cell is not clear of.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.clear.framebuffer);
    gl.useProgram(this.clearProgram.program);
    u = this.clearProgram.uniforms;
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.samples.textures[0]); gl.uniform1i(u.uSamplePoints, 3);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.samples.textures[1]); gl.uniform1i(u.uSampleNormals, 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, g.textures.meshCellBlock); gl.uniform1i(u.uCellBlock, 5);
    gl.uniform2f(u.uMapSize, this.factorSize, this.factorSize);
    gl.uniform1f(u.uMargin, 1e-3);
    for (const part of parts) {
      gl.uniform4fv(u.uRect, part.factorRect);
      gl.uniform4fv(u.uSampleRect, part.sampleRect);
      gl.uniform1i(u.uPeriodic, part.wrap ? 1 : 0);
      gl.uniform1f(u.uOrientation, part.orientation);
      gl.uniform1i(u.uCellBase, part.cellBase);
      gl.uniform1i(u.uCu, part.cells[0]);
      gl.uniform1i(u.uCv, part.cells[1]);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.factors.framebuffer);
    const stage = [];
    const mark = () => { const texel = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, texel); stage.push(Math.round(performance.now() - started)); };
    mark();
    // 2. Layout: every cell's first triangle, in order.
    const cells = cellTotal;
    let triangles = 0;
    const table = new Float32Array(cells * 8);
    parts.forEach((part, index) => {
      const [fx, fy, cu, cv] = part.factorRect, found = new Float32Array(cu * cv * 4);
      gl.readPixels(fx, fy, cu, cv, gl.RGBA, gl.FLOAT, found);
      part.firstTriangle = triangles;
      for (let j = 0; j < cv; ++j) {
        for (let i = 0; i < cu; ++i) {
          const f = found.subarray((j * cu + i) * 4, (j * cu + i) * 4 + 4), cell = part.cellBase + j * cu + i;
          table.set([triangles, f[0], f[1], index, i, j, f[2], f[3]], cell * 8);
          triangles += 2 * f[0] * f[1];
        }
      }
      part.triangles = triangles - part.firstTriangle;
    });
    g.floatTexture('meshCells', table, 4);
    this.cells = cells;
    this.triangles = triangles;

    // 3. Vertices: three texels a triangle.
    const mesh = this.list('mesh', triangles * 3, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, mesh.framebuffer);
    gl.viewport(0, 0, 2048, mesh.height);
    gl.useProgram(this.vertexProgram.program);
    u = this.vertexProgram.uniforms;
    surfaceInputs(u);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, g.textures.meshCells); gl.uniform1i(u.uCellTable, 3);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.clear.textures[0]); gl.uniform1i(u.uClear, 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, g.textures.meshCellBlock); gl.uniform1i(u.uCellBlock, 5);
    gl.enable(gl.SCISSOR_TEST);
    parts.forEach((part, index) => {
      if (part.triangles === 0) { return; }
      g.setPart(part, index, false, u);
      gl.uniform4fv(u.uClearRect, part.factorRect);
      const first = part.firstTriangle * 3, end = first + part.triangles * 3;
      gl.uniform1i(u.uCellBase, part.cellBase);
      gl.uniform1i(u.uCellEnd, part.cellBase + part.cells[0] * part.cells[1]);
      gl.uniform1i(u.uFirstVertex, first);
      gl.uniform1i(u.uVertexEnd, end);
      gl.uniform1f(u.uPartIndex, index);
      const top = Math.floor(first / 2048), bottom = Math.floor((end - 1) / 2048);
      gl.scissor(0, top, 2048, bottom - top + 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    });
    gl.disable(gl.SCISSOR_TEST);
    mark();

    // 4. The box of every pair of triangles, then of every cell.
    const quads = this.list('quadBoxes', triangles / 2, 2);
    gl.bindFramebuffer(gl.FRAMEBUFFER, quads.framebuffer);
    gl.viewport(0, 0, 2048, quads.height);
    gl.useProgram(this.quadBoxProgram.program);
    u = this.quadBoxProgram.uniforms;
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, mesh.textures[0]); gl.uniform1i(u.uMeshPoints, 4);
    gl.uniform1i(u.uQuadCount, triangles / 2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const boxes = this.list('cellBoxes', cells, 2);
    gl.bindFramebuffer(gl.FRAMEBUFFER, boxes.framebuffer);
    gl.viewport(0, 0, 2048, boxes.height);
    gl.useProgram(this.cellBoxProgram.program);
    u = this.cellBoxProgram.uniforms;
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, g.textures.meshCells); gl.uniform1i(u.uCellTable, 3);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, quads.textures[0]); gl.uniform1i(u.uQuadLow, 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, quads.textures[1]); gl.uniform1i(u.uQuadHigh, 5);
    gl.uniform1i(u.uCellCount, cells);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    mark();

    // 5. The tree's plan: which cells or pairs each node stands for, in the order a ray walks them.
    // A block is halved each way while it is more than one across; a single cell goes on into its steps.
    const plan = [];
    const inCell = (cell, x0, y0, x1, y1, band = 0) => {
      const at = plan.length;
      plan.push(1, cell, x0, y0, x1, y1, 0, 0, band, 0, 0, 0);
      const w = x1 - x0, h = y1 - y0;
      if (w > 1 || h > 1) {
        const xm = w > 1 ? x0 + Math.ceil(w / 2) : x1, ym = h > 1 ? y0 + Math.ceil(h / 2) : y1;
        for (const [ya, yb] of h > 1 ? [[y0, ym], [ym, y1]] : [[y0, y1]]) {
          for (const [xa, xb] of w > 1 ? [[x0, xm], [xm, x1]] : [[x0, x1]]) { inCell(cell, xa, ya, xb, yb); }
        }
      }
      plan[at + 6] = plan.length / 12;
    };
    // A part: its nodes over cells as planned above, each stretch of cells listed in `cellList`; a single cell goes on into its steps.
    const cellList = [];
    const over = (part, node) => {
      if (node.count === 1) {
        const cell = part.cellBase + part.order[node.first];
        inCell(cell, 0, 0, table[cell * 8 + 1], table[cell * 8 + 2], node.block);
        return;
      }
      const at = plan.length;
      plan.push(0, part.listBase + node.first, node.count, 0, 0, 0, 0, 0, node.block, 0, 0, 0);
      over(part, node.halves[0]);
      over(part, node.halves[1]);
      plan[at + 5] = plan.length / 12;
    };
    for (const part of parts) {
      part.listBase = cellList.length / 4;
      for (const c of part.order) { cellList.push(part.cellBase + c, 0, 0, 0); }
      part.treeRoot = plan.length / 12;
      if (part.root) { over(part, part.root); }
      part.treeNext = plan.length / 12;
    }
    g.floatTexture('meshCellList', Float32Array.from(cellList.length ? cellList : [0, 0, 0, 0]), 4);
    const nodes = plan.length / 12;
    g.floatTexture('meshPlan', Float32Array.from(plan), 4);
    const tree = this.wholeList('tree', nodes);
    gl.bindFramebuffer(gl.FRAMEBUFFER, tree.framebuffer);
    gl.viewport(0, 0, 2048, tree.height);
    gl.useProgram(this.treeProgram.program);
    u = this.treeProgram.uniforms;
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, g.textures.meshPlan); gl.uniform1i(u.uPlan, 2);
    gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, g.textures.meshCellList); gl.uniform1i(u.uCellList, 8);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, g.textures.meshCells); gl.uniform1i(u.uCellTable, 3);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, quads.textures[0]); gl.uniform1i(u.uQuadLow, 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, quads.textures[1]); gl.uniform1i(u.uQuadHigh, 5);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, boxes.textures[0]); gl.uniform1i(u.uCellLow, 6);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, boxes.textures[1]); gl.uniform1i(u.uCellHigh, 7);
    gl.uniform1i(u.uNodeCount, nodes);
    gl.uniform3fv(u.uTreeLow, this.treeLow);
    gl.uniform1f(u.uTreeStep, this.treeStep);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.finish();
    stage.push(Math.round(performance.now() - started));
    this.stages = stage;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    this.nodes = nodes;
    this.points = mesh.textures[0];
    this.normals = mesh.textures[1];
    this.where = mesh.textures[2];
    this.tree = tree.textures[0];
    this.ms = performance.now() - started;
  }
}
