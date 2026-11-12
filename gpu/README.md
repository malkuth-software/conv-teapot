# The GPU version

Live: https://malkuth-software.github.io/conv-teapot/conv-teapot-gpu/ (also at `/gpu/`).

The same teapot as the CPU version, drawn and simulated by WebGL 2.

| What | Where it runs | How |
|---|---|---|
| CONV compile | GPU, once per teapot | The engine hands over the lattice nodes and the shape of every cell. A shader admits the u-cells of every row from the nodes into a texture (the CPU's `compile()`). |
| CONV surface | GPU | `shaders.js`: the admitted u-cells of every lattice row are read from that texture; the v-cell through them is admitted in the shader. The derivative along u is carried through admission as the second component of a `vec2`, as the CPU's dual numbers do. |
| Vertices | GPU, once | Transform feedback evaluates the surface at every vertex of the (u, v) layout and keeps the points on the GPU; a frame carries them by the pose. |
| Normals | GPU, every pixel | The fragment shader evaluates the surface again at the pixel's (u, v). |
| Other surfaces and the texture's interpolator | GPU | The Bezier patches and the control mesh are evaluated in the shader from their control points and nodes. A texture is a lattice of colours: interpolated by CONV, bilinearly, or by the nearest node (the page's three filters). |
| Joins | GPU | A band stores where its two sides are on the body and the tube; the shader evaluates both and blends them. Trims are tested per pixel. |
| Shadows | GPU | A map of distance along the light, looked up as the CPU's is; where its texels disagree, light rays traced in the shader against a tree of boxes over the mesh. |
| Reflections | GPU | The mirror ray is traced one bounce against the same tree, so the teapot shows itself; where it misses, emitters: sky and window above, the floor's pattern and shadow below. A polished floor shows the teapot's image, drawn from the mirrored eye. |
| Cutaway | GPU | Crossings of the kept half are counted in the stencil buffer; an odd count shows the cut face. |
| Room light | GPU | The body is drawn from 256 directions; every texel of a map over the parts looks itself up in each picture: open fraction and mean open direction. |
| Light the body throws on itself | GPU | For each direction the body is peeled into depth layers, each carrying its light; every texel takes the layer it faces. Repeated five times. |
| Rigid body | GPU | `physics.js`: one fragment shader run over four texels is one step: gravity, the grab's soft pin, contact with the floor, friction, rolling resistance. The state is read back each frame (13 numbers). What meets the floor is either the engine's hull points (the default, 1280 of them) or the surface itself: each step the GPU draws every evaluated vertex into a 64 x 64 picture by its direction from the centre of mass, keeping the lowest, and the contact pass goes over those. With the surface the teapot rests exactly on the floor (the hulls leave it 0.007 in); a frame's steps then take about 15 ms against 4 ms. |
| Picking | GPU | Part and (u, v) are drawn to a float target and one pixel is read. |
| Where the nodes are, the (u, v) layout of the mesh, the tree for traced rays, the collision hulls | CPU (WebAssembly), once per teapot | Each is a construction that decides as it goes (row placement by bisection, a layout refined where the error is large, a tree split by sorting, convex hulls); `main.cpp` hands the results over as textures and buffers. Nothing on the CPU is evaluated per frame, per vertex or per pixel. |

## Measured against the CPU version

- Surface: at 12 000 points, largest distance 2.3e-6, largest angle between normals 0.027 degrees (32-bit floats against 64-bit), with the cells compiled on the GPU. The Bezier patches agree to 1.1e-6 and the control mesh to 4e-7.
- Picture at 960 x 600 against the CPU ray tracer, share of pixels differing by more than 8 of 255: porcelain 0.25%, polished floor 0.25%, texture 0.46%, chrome 0.56%, cutaway 0.96%.
- A frame: 11 ms at 960 x 600, 18 ms at 1920 x 1200 (chrome with its traced bounce: 22 and 43 ms), on the laptop used for the paper.
- Room light and gathered light against the CPU's baked renderer: mean difference 1 to 2 of 255. The two estimate the same integrals differently (32 rays a texel there, 256 directions for all texels here).
- Rigid body: the standard drop comes to rest upright in both, at the same height, 0.16 units apart and turned 4.7 degrees about the vertical. The CPU's solver (zc_phys) solves a frame's contacts together; the shader takes, in ten passes, the mean of what each contact point asks for.

## Not the same

- Traced rays meet the baked mesh (within 2.5e-3 of the surface), not the surface itself.
- For one body the GPU is the slower place for the physics: a frame's steps with the read-back take 6 ms against 0.13 ms on the CPU. The CPU's engine can be chosen in the page.

## Build

`sh gpu/build.sh` (needs Emscripten) compiles the engine and assembles the page; `sh gpu/assemble.sh` reassembles it after a change to the shaders, renderer, physics or page only.
