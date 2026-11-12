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
| Where the nodes are; mass and inertia | CPU (WebAssembly), once per teapot | The teapot itself: which points of the source surface are its nodes (rows placed by bisection on CONV's error), and the body's mass, centre of mass and inertia, which the engine takes from its convex hulls. `main.cpp` hands the nodes over as textures. Nothing on the CPU is evaluated per frame, per vertex or per pixel. |

## Measured against the CPU version

- Surface: at 12 000 points, largest distance 2.3e-6, largest angle between normals 0.027 degrees (32-bit floats against 64-bit), with the cells compiled on the GPU. The Bezier patches agree to 1.1e-6 and the control mesh to 4e-7.
- Picture at 960 x 600 against the CPU ray tracer, share of pixels differing by more than 8 of 255: porcelain 0.25%, polished floor 0.25%, texture 0.39%, chrome 0.55%, cutaway 0.96%.
- A frame: 12 ms at 960 x 600, 21 ms at 1920 x 1200 (chrome with its traced bounce: 28 and 54 ms), on the laptop used for the paper, from 66 528 grid points and 132 096 triangles.
- Room light and gathered light against the CPU's baked renderer: mean difference 1 to 2 of 255. The two estimate the same integrals differently (32 rays a texel there, 256 directions for all texels here).
- Rigid body: the shader's solver against the CPU's, frame by frame, largest difference of the pose's position over the whole run (model units; the teapot is about 3 high): a drop of 600 frames 2.6e-3 (2026) and 3e-6 (1976); a grab at the handle dragged and released over 360 frames 8e-5; a poke 2e-5 to 4e-5. Both come to rest in the same pose to 3e-4. The shader works in 32-bit floats and stops Newton at a relative gradient of 1e-5 where the CPU stops at 1e-9. A frame of it takes 1.4 to 2.4 ms with the read-back.

## Not the same

- Traced rays meet the grid's triangles, not the surface itself.

## Licence

`wrench.js` is a port of PlaySuite's engine (`third_party/zc_phys`) and is under PlaySuite's licence, not MIT; see `third_party/zc_phys/PROVENANCE.md`. The rest of this folder is MIT like the repository.

## Build

`sh gpu/build.sh` (needs Emscripten) compiles the engine and assembles the page; `sh gpu/assemble.sh` reassembles it after a change to the shaders, renderer, physics or page only.
