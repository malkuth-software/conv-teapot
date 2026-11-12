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
| Rigid body | GPU | `physics.js`: one fragment shader run over four texels is one step: gravity, the grab's soft pin, contact with the floor, friction, rolling resistance. The state is read back each frame (13 numbers). The points that can meet the floor are found on the GPU once per teapot: a plane touches a body only at corners of its convex hull, and those are the farthest points of the surface from the centre of mass in their own directions, so every grid point is drawn into a 36 x 36 picture by direction, keeping the farthest. Each point is used with its mirror image in the teapot's plane of symmetry. The engine's hull points can be chosen instead. |
| Picking | GPU | Part and (u, v) are drawn to a float target and one pixel is read. |
| Where the nodes are; mass and inertia | CPU (WebAssembly), once per teapot | The teapot itself: which points of the source surface are its nodes (rows placed by bisection on CONV's error), and the body's mass, centre of mass and inertia, which the engine takes from its convex hulls. `main.cpp` hands the nodes over as textures. Nothing on the CPU is evaluated per frame, per vertex or per pixel. |

## Measured against the CPU version

- Surface: at 12 000 points, largest distance 2.3e-6, largest angle between normals 0.027 degrees (32-bit floats against 64-bit), with the cells compiled on the GPU. The Bezier patches agree to 1.1e-6 and the control mesh to 4e-7.
- Picture at 960 x 600 against the CPU ray tracer, share of pixels differing by more than 8 of 255: porcelain 0.25%, polished floor 0.25%, texture 0.39%, chrome 0.55%, cutaway 0.96%.
- A frame: 12 ms at 960 x 600, 21 ms at 1920 x 1200 (chrome with its traced bounce: 28 and 54 ms), on the laptop used for the paper, from 66 528 grid points and 132 096 triangles.
- Room light and gathered light against the CPU's baked renderer: mean difference 1 to 2 of 255. The two estimate the same integrals differently (32 rays a texel there, 256 directions for all texels here).
- Rigid body: the standard drop comes to rest upright and facing the same way in both (the CPU's is turned 0.3 degrees, the GPU's not at all), 0.16 units apart along its path; the GPU's rests with its surface on the floor, the CPU's 0.007 in. The CPU's solver (zc_phys) solves a frame's contacts together; the shader takes, in ten passes, the mean of what each contact point asks for, so the two are not the same step for step.

## Not the same

- Traced rays meet the grid's triangles, not the surface itself.
- For one body the GPU is the slower place for the physics: a frame's steps with the read-back take 6 ms against 0.13 ms on the CPU. The CPU's engine can be chosen in the page.

## Build

`sh gpu/build.sh` (needs Emscripten) compiles the engine and assembles the page; `sh gpu/assemble.sh` reassembles it after a change to the shaders, renderer, physics or page only.
