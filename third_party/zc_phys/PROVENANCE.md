# zc::phys

The rigid-body engine of PlaySuite's rock-stacking game (Zen Construction),
with its wrench-transport contact solver. Copied unchanged from
`playsuite-games/games/zenconstruction/phys` at revision
`78ea30ef6b00fa409e9d0797283eeac287f11693` (2026-10-06).
The teapot uses it for the drop, tumble and settle on the floor: the teapot is
one free body whose convex hulls are taken from the CONV lattice nodes.
See NOTES.md for the engine's own notes.
