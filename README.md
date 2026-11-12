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
  variation bound.

**Versions** (`src/teapot.hpp`), following the official Utah Teapot page:

- 1975 (Newell's proportions), 1976 (Blinn) and 1987 (Crow, round bottom)
  from the original Bézier data;
- 2026 (Yuksel), rebuilt from 9 × 9 points sampled on each of its 66
  untrimmed surfaces, interior included. No control points, degrees or knots
  of that model are used. Its trimming and chamfers are not included, and the
  reconstruction does not keep its curvature continuity across patch joints
  (normals agree to 0.11° on average).

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
Utah Model Repository. `data/utah_teapot_samples.json` holds points sampled
with the mesh generator of the official page
(https://graphics.cs.utah.edu/teapot/): the 2026 surfaces and the 1987
bottom, at eight intervals per patch, symmetric copies stored once. The
models built here are Modified Utah Teapots.

## Notes

`third_party/zc_phys` is used unchanged. Its hull builder reads out of bounds
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
