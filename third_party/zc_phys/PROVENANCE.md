# zc::phys

The rigid-body engine of PlaySuite's rock-stacking game (Zen Construction),
with its wrench-transport contact solver. Copied from
`playsuite-games/games/zenconstruction/phys` at revision
`78ea30ef6b00fa409e9d0797283eeac287f11693` (2026-10-06).
The teapot uses it for the drop, tumble and settle on the floor: the teapot is
one free body whose convex hulls are taken from the CONV lattice nodes.
See NOTES.md for the engine's own notes.

One addition was made here: `World::add_velocity` (physics.hpp, world.cpp),
which applies a change of velocity from outside. The teapot's grab uses it
to pull on the point of the body that was picked. Nothing else differs from
the PlaySuite revision.

## The shader port

`gpu/wrench.js` is a port of this engine's frame (contacts against the floor,
retained impulses, the wrench-transport solve, rolling resistance, sleep) to a
WebGL 2 fragment shader, for the teapot's one body. It was made with the
owner's permission on 2026-11-12, is derived from the files here, and is
under PlaySuite's licence like them, not under this repository's MIT licence.
