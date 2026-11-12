# The GPU version

Live: https://malkuth-software.github.io/conv-teapot/conv-teapot-gpu/ (also at `/gpu/`).

The same teapot as the CPU version, drawn and simulated by WebGL 2.

| What | Where it runs | How |
|---|---|---|
| CONV compile | GPU, once per teapot | The engine hands over the lattice nodes and the shape of every cell. A shader admits the u-cells of every row from the nodes into a texture (the CPU's `compile()`). |
| CONV surface | GPU | `shaders.js`: the admitted u-cells of every lattice row are read from that texture; the v-cell through them is admitted in the shader. The derivative along u is carried through admission as the second component of a `vec2`, as the CPU's dual numbers do. |
| Where the surface is evaluated, and the mesh | GPU, once per teapot | Each part has a grid of points, its steps dividing the node intervals evenly; a shader evaluates the surface at every grid point into a texture. There is no vertex buffer and no CPU layout: a vertex's number says which grid cell and corner it is, and its point is read from the texture and carried by the pose. |
| Normals | GPU, every pixel | The fragment shader evaluates the surface again at the pixel's (u, v). |
| Other surfaces and the texture's interpolator | GPU | The Bezier patches and the control mesh are evaluated in the shader from their control points and nodes. A texture is a lattice of colours: interpolated by CONV, bilinearly, or by the nearest node (the page's three filters). |
| Joins | GPU | A band stores where its two sides are on the body and the tube; the shader evaluates both and blends them. Trims are tested per pixel. |
| Shadows | GPU | A map of distance along the light, looked up as the CPU's is; where its texels disagree, light rays traced in the shader. Rays walk a pyramid of boxes over the grid's cells, which the GPU builds by taking the smallest and largest corner of four boxes at a time. |
| Reflections | GPU | The mirror ray is traced one bounce through the same pyramid, so the teapot shows itself; where it misses, emitters: sky and window above, the floor's pattern and shadow below. A polished floor shows the teapot's image, drawn from the mirrored eye. |
| Cutaway | GPU | Crossings of the kept half are counted in the stencil buffer; an odd count shows the cut face. |
| Room light | GPU | The body is drawn from 256 directions; every texel of a map over the parts looks itself up in each picture: open fraction and mean open direction. |
| Light the body throws on itself | GPU | For each direction the body is peeled into depth layers, each carrying its light; every texel takes the layer it faces. Repeated five times. |
| Rigid body | GPU | `wrench.js`: PlaySuite's wrench-transport solver, ported to a fragment shader. One run of the shader is one frame of the solver: contact at every hull vertex within the margin of the floor, the impulses each contact retains, the passes of Newton with exact line search on the 6 x 6 system, rolling resistance, restitution, sleep. The body's state is a few texels and each contact vertex keeps its impulses in its own pair of texels. The state is read back each frame. (`physics.js` keeps an earlier, simpler step, selectable in the page.) |
| Picking | GPU | Part and (u, v) are drawn to a float target and one pixel is read. |
| Mass, centre of mass, inertia | GPU, once per teapot | `wrench.js`: a shader sums signed tetrahedra over the faces of the collision hulls; the solver takes its mass, centre and inertia from that. |
| Where the nodes are (unified teapot's body and lid) | GPU, once per height scale | `rows.js`: the rows of the two sheets are placed in shaders by CONV's own error, as the CPU's `plan_rows_by_error` does. The engine hands over the meridian of the sampled patches. Passes over small textures find the wanted spacing at each station, the one scale that gives the asked number of rows (bisection in one texel), each row's arc length from the projective grading, each cell's distance from the meridian, and keep the plan with the smallest error over nine rounds. The engine builds its lattice on the rows the GPU hands back, and the nodes on those rows are also written by a shader. |
| The rest of the teapot's construction | CPU (WebAssembly), once per teapot | Rows of the handle and spout tubes, the parameter lattices of the joins, the nodes of the other versions (samples of the Bezier patches), the cells' shapes, and the collision hulls. `main.cpp` hands these over as textures. Nothing on the CPU is evaluated per frame, per vertex or per pixel. |

## Measured against the CPU version

- Surface: at 12 000 points, largest distance 2.3e-6, largest angle between normals 0.027 degrees (32-bit floats against 64-bit), with the cells compiled on the GPU. The Bezier patches agree to 1.1e-6 and the control mesh to 4e-7.
- Picture at 960 x 600 against the CPU ray tracer, share of pixels differing by more than 8 of 255: porcelain 0.25%, polished floor 0.25%, texture 0.39%, chrome 0.55%, cutaway 0.96%.
- A frame: 12 ms at 960 x 600, 21 ms at 1920 x 1200 (chrome with its traced bounce: 28 and 54 ms), on the laptop used for the paper, from 66 528 grid points and 132 096 triangles.
- Room light and gathered light against the CPU's baked renderer: mean difference 1 to 2 of 255. The two estimate the same integrals differently (32 rays a texel there, 256 directions for all texels here).
- Rigid body: the shader's solver against the CPU's, frame by frame, largest difference of the pose's position over the whole run (model units; the teapot is about 3 high): a drop of 600 frames 2.6e-3 (2026) and 3e-6 (1976); a grab at the handle dragged and released over 360 frames 8e-5; a poke 2e-5 to 4e-5. Both come to rest in the same pose to 3e-4. The shader works in 32-bit floats and stops Newton at a relative gradient of 1e-5 where the CPU stops at 1e-9. A frame of it takes 1.4 to 2.4 ms with the read-back.
- While the teapot is held and dragged (240 frames, up to 80 contact points): the solver 3.3 ms and the picture 15 ms a frame, the same picture time as at rest. The light the body throws on itself is found again once, when the body comes to rest. (Before 2026-11-12 the page found it again on every frame of motion, which took over a second a frame in the same test.)
- Mass, centre of mass and inertia from the shader against the engine's: mass 8e-7 (relative), centre 2e-7 m, inertia 4e-6 (relative).
- Rows placed on the GPU against the CPU's own plan, for the error both aim at (largest distance of the CONV meridian from the source meridian, measured on the CPU in 64-bit): body 6.95e-4 against 7.03e-4, lid 2.53e-4 against 2.46e-4; at the taller height scale body 6.62e-4 against 5.79e-4, lid 2.25e-4 against 2.33e-4. Same number of rows (73 and 55). The first plan, by curvature, agrees to 2e-4 to 7e-4 in arc length; the kept plans' rows are up to 0.16 apart (0.31 at the taller scale) in arc length on meridians 8.5 long, because the search keeps one of several plans whose errors are within a few percent of each other and 32-bit arithmetic picks a different one. Placing both sheets takes about 0.2 s, once. The nodes the shader writes on those rows are within 6e-7 of the engine's on the same rows.

## Not the same

- The unified teapot's body and lid have their rows where the GPU placed them, not where the CPU version places them; the two surfaces are each within the errors above of the same source.

- Traced rays meet the grid's triangles, not the surface itself.

## Licence

`wrench.js` is a port of PlaySuite's engine (`third_party/zc_phys`) and is under PlaySuite's licence, not MIT; see `third_party/zc_phys/PROVENANCE.md`. The rest of this folder is MIT like the repository.

## Build

`sh gpu/build.sh` (needs Emscripten) compiles the engine and assembles the page; `sh gpu/assemble.sh` reassembles it after a change to the shaders, renderer, physics or page only.
