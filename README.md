# The Utah Teapot in CONV geometry

The Utah Teapot rebuilt from points that lie on its own surface, traced
without a mesh, and dropped on a floor. A paper and a browser demo are
included.

- **Live demo:** https://malkuth-software.github.io/conv-teapot/ (WebAssembly,
  one CPU thread, no GPU). The page is `docs/index.html`, one self-contained file.
- `paper/conv_teapot.pdf` — the paper (source `paper/conv_teapot.tex`).
- `results/` — the measurements quoted in the paper.

## What is here

**Geometry.** A lattice of nodes reconstructed by Compact Ordered Nodal
Variation (CONV) with ledger-free admission. `src/conv_line.hpp` is one line
cell for any scalar type; `src/surface.hpp` is the surface through a lattice,
with exact derivatives by dual numbers. Two additions make it usable as a
geometry:

- *Seams* with cubic-exact end stencils. With seams on the patch boundaries
  and four node intervals per patch, 560 on-surface nodes reproduce the 28
  original Bézier patches to 2e-15.
- A *continuous bubble limiter*. The standard limiter is discontinuous in the
  data and can crack a surface; the symmetric one is continuous and keeps the
  variation bound. On the teapot itself the bubble moves the surface
  by less than half a pixel, so the demo has no switch for it; it matters on
  rough or deformed lattices.

**Versions** (`src/teapot.hpp`), following the official Utah Teapot page:

- 1975 (Newell's proportions), 1976 (Blinn) and 1987 (Crow, round bottom)
  from the original Bézier data;
- 2026 (Yuksel), rebuilt from 9 × 9 points sampled on each of its 66
  untrimmed surfaces, interior included. No control points, degrees or knots
  of that model are used. Its trimming and chamfers are not included, and the
  reconstruction does not keep its curvature continuity across patch joints
  (normals agree to 0.11° on average).

- 2026 as unified sheets (`src/unified.hpp`, `src/blend.hpp`): the body with
  its interior is one lattice, the lid another, the handle and spout are
  tubes, with rows placed by a piecewise projective warp. The handle and
  spout meet the body in blended joins, which are this project's own and not
  the 2026 chamfers. Along the profile the mean relative curvature jump is
  0.007, against 0.16 for the per-patch version.

Handle, spout, lid, body and interior can be switched off, as on that page.

**Styles** for the 1975–1987 data: exact (the 1976 teapot to rounding), round
(every ring of the body and lid on a true circle: 233 times rounder than the
original with the same nodes), seamless, and an experimental smooth style that
is kept as a measured negative result.

**Rendering** (`src/render.hpp`): correspondence transport. A lattice of
known parameters is sent forward to the screen and each pixel corrects a
prediction with a Newton step. A per-ray BVH tracer over the same lattice is
kept as the baseline; transport is 1.6 to 2.0 times faster and both agree
with a fine reference on 99.9% of surface pixels. Wireframe and patch
outlines are drawn from the hit's own parameters. Options: textures
(magnified by CONV), traced gloss, a polished floor (transport from the
mirrored eye), chrome, reflection lines, edge samples.

**Motion** (`src/sim.hpp`): one rigid body. Its convex hulls are taken from
the lattice nodes and simulated by `third_party/zc_phys`, the rigid-body
engine of PlaySuite's rock-stacking game (wrench-transport solver).

## Node budget, baked mesh, and the rest (second round)

- **How many nodes.** `tools/budget.cpp` gives five methods the same
  on-surface nodes on targets native to none of them
  (`results/node_budget.txt`). CONV needs as few nodes as the globally solved
  C2 interpolating spline, and a half to a quarter of what Catmull-Rom needs,
  while reading only the nodes around a cell. Against the Bezier form the
  teapot was born in it is no better than equal (about 300 control points
  against 324 to 560 nodes). That is the honest size of the claim.
- **Rows by CONV's own error.** The unified sheets place their rows by
  equidistributing the CONV line's own error (`plan_rows_by_error` in
  `src/unified.hpp`). The 2026 teapot went from 5392 nodes within 1.1e-2 of
  its source to 3104 nodes within 7.5e-4.
- **Baked mesh and CPU rasterizer** (`src/raster.hpp`). *Bake vertices*
  samples the surfaces once to triangles; three layouts are offered and a
  fourth (k-means clustering, with e-means updates) was tried and lost
  (`tools/mesh_study.cpp`, `results/mesh_study.txt`). The default is an
  adaptive tree with a distance tolerance and a limit on how far the normal
  may turn along an edge.
- **Baked light.** *Bake light* has the ray tracer shade every texel of a
  light map. With *moving reflections* the map keeps only view-independent
  light, and reflections are added per frame from emitters along each pixel's
  mirror direction: the sky and window above, and below a stand-in for the
  floor that is never drawn (its pattern in closed form, its shadow baked).
  The teapot's reflection of itself is not included there; that needs a ray.
- **Seeded ray tracing.** The rasterized mesh tells each ray where it lands,
  and one Newton solve puts it on the surface itself.
- **Shadow edges** are found on the surface: where the shadow map's texels
  disagree, the light ray is traced from what those texels saw.
- **Grab.** Dragging holds the teapot at the point you picked (a soft pin
  there), so it swings from the hand. This needed one addition to the vendored
  engine, `World::add_velocity`; see `third_party/zc_phys/PROVENANCE.md`.

- **What sets the vertex count** (`tools/curvature_budget.cpp`,
  `out/mesh_study metric`, `results/metric_layout.txt`). The curvature
  integrals that bound a mesh can be read from the lattice nodes alone, within
  1% of the evaluated surface. Layouts built from that metric (seeds at
  random, nucleated, or at the lattice nodes; cells by shortest path; e-means
  relaxation; then a check against the surface) need about 40% fewer vertices
  than the tree when coarse and none fewer by 14 000. Seeding at the lattice
  nodes is the most even and gives the best normals at the fine setting. They
  take a second to build. The engine keeps the tree.

One 960 x 600 frame of the unified teapot, all threads / one thread: per-ray
150 / 410 ms, transport 91 / 261, seeded 37 / 125, mesh shaded per pixel
15 / 45, mesh with baked light and emitters 11 / 36.

## Build

```sh
sh build.sh                 # native tools into out/ (any C++20 compiler; CXX=...)
sh tools/build_web.sh       # docs/index.html (needs Emscripten's em++)
cd paper && tectonic conv_teapot.tex
```

```sh
out/capture out=teapot.png wire=1                      # a still
out/capture out=chrome.png material=1 gloss=1 floor=1
out/capture out=drop.png drop=0.42 lift=6 tilt=1.2     # 0.42 s into a drop
out/capture out=v2026.png version=2026 per=8 hide=lid wire=1   # the 2026 version, lid off
out/capture out=lines.png version=2026 per=8 material=2        # reflection lines
out/capture verify=1                                   # check a frame against the surface and a fine reference
out/study        # deviation from the Bézier teapot; surface continuity
out/qualities    # roundness, joint continuity, creases
out/unified_study  # the unified sheets and their joins
out/line_check   # the line kernel against the stx_conv reference model
```

All of `out/capture`'s options are listed at the top of `tools/capture.cpp`.

## What it does not do

It does not reproduce curvature continuity across seams, trim the handle and
spout to the body, or add chamfers; the 1992 and 2006 versions are not
included. The paper's Section VII says where each later version of the
teapot did that work.
Neither renderer proves that it has found the nearest intersection on a
silhouette. Timings were taken on one shared machine.

## Attribution

The Utah Teapot was created by Martin Newell at the University of Utah and
scaled by Jim Blinn; the 1987 bottom is Frank Crow's and the 2026 version is
Cem Yuksel's. `data/teapot_bezier.txt` is the original Bézier data from the
Utah Model Repository. `data/utah_teapot_samples.json` and `data/utah_teapot_dense.json` hold points sampled
with the mesh generator of the official page
(https://graphics.cs.utah.edu/teapot/): the 2026 surfaces and the 1987
bottom, at eight intervals per patch, symmetric copies stored once. The
models built here are Modified Utah Teapots.

## Notes

`third_party/zc_phys` is used with one addition (`World::add_velocity`, for the grab). Its hull builder reads out of bounds
(hull.cpp, the boundary-loop walk near line 258) when given a point set that
does not close into a hull, before it reports "hull is not closed".
`src/sim.hpp` avoids such sets (it drops coincident nodes and uses one hull
per group of lattices for the sampled version); the engine itself is worth
a guard.

CONV: `malkuth-software/papers` (paper) and `malkuth-software/stx` (stx_conv).
`tests/line_reference.txt` was produced by stx_conv's stream reference model.

## Licence

MIT for this project's own code and paper (see `LICENSE`). `third_party/zc_phys`
remains under the PLAYSUITE Source-Available License, and the teapot data is
subject to the University of Utah's terms of use and attribution.
