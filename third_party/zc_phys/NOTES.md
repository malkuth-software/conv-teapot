# zc::phys notes

## Worksite ground and sling contact

PlaySuite supplies an optional immutable `GroundSurface` to the world. The
original infinite plane remains the default for physics consumers. The game
indexes the rendered bank, ledge top and skirt, riverbed and far-bank triangles
into XY cells once. Contact sampling visits a local cell without allocation or
trigonometry; it supplies the actual surface normal to the existing friction
and contact solver. Picking intersects the same triangles. This avoids adding
thousands of static bodies to the all-pairs broad phase. Decorative pebbles and
foliage remain scenery; the bowl and crane boulder retain their static hulls.

An optional positive `HoldParams::sling_length` models the vertical take-up as
the hook moves sideways from a grounded load. The effective vertical error adds
`L - sqrt(max(0, L*L - horizontal_distance_squared))`. This unloads some ground
friction as the sling tilts, so a lowered attached rock can drag rather than
remaining pinned by its full weight. It retains unilateral vertical force,
the gentle 0.35-weight lateral cap and the existing orientation spring. This
is an extension of the game's spring hold, not a replacement cable simulator.
The default zero length preserves the original hold behavior and its tests.

`rockstack_worksite_tests` covers rendered ground correspondence, slope normals,
set-down, sustained grounded dragging, penetration, force limits and subsequent
lifting. It also checks the 105-degree chassis slew limits, full bowl access and
automatic fetch/return travel. The renderer applies the same mechanical stop
in chassis coordinates; travel waypoints go through the allowed sector.

State on 2026-10-03: the engine core is ported from the JavaScript prototype
(kept in `~/bfft/experiments/wrench_transport/`; this is the game's copy). The
specification's sixteen acceptance tests and the prototype's feature checks are
ported to `tests/phys_tests.cpp` (run one by name, e.g. `phys_tests stack`).
All correctness tests pass; see "Acceptance tests" below.

## Build and run (on the Mini)

```sh
/opt/homebrew/bin/cmake -S experiments/wrench_transport/phys -B /tmp/zc_phys_build
/opt/homebrew/bin/cmake --build /tmp/zc_phys_build -j4
/tmp/zc_phys_build/phys_tests
node experiments/wrench_transport/container.mjs --seed 1 --export-text /tmp/zc_scene.txt
/tmp/zc_phys_build/phys_bench /tmp/zc_scene.txt /tmp/zc_cpp60.json 60 8
```

`phys_bench` writes the same result record as the other engines' runners, so
`container_report.mjs` scores it with them.

## Where this departs from the specification

- **Solver.** Section 7's soft-step sequential impulses are replaced by the
  wrench-transport solve (`solver.cpp`; method and measurements in
  `../FINDINGS.md`). `substeps`, `contact_hertz` and `contact_damping` in
  `WorldParams` are kept for the signature and unused.
- **Manifolds** are not reduced to four points.
- **Collision** adds two checks to section 6.2: a face manifold must hold the
  depth its axis measured, and an edge axis must match the hulls' true
  separation along it.
- **Rolling resistance** is a row of the solve, not a correction after it.
  The ground has its own lever arm (`SolverParams::ground_rolling_resistance`).
- **Restitution** is asked for one frame after a gap closes, at the surface.
- **Additions to the API:** `add_static_body`, `SolverParams` and its
  accessors, `rest_report`, `stable_set`, `solve_stats`, `guard_count`,
  `HoldState::lateral_force`, `Shape::hulls`.
- `cook` throws on degenerate input; the specification asks for an assert in
  debug and a fallback tetrahedron in release. Hull simplification above 48
  vertices is not implemented (and no vertex cap is enforced).
- `step()` still allocates when its vectors grow and in `stable_set`,
  `contacts` and `bodies`; steady-state frames of a fixed scene reuse storage,
  but that has not been audited.
- **Islands** join every reported contact between free bodies, loaded or not
  (the specification joins only loaded ones). Waking fires on any contact inside
  the margin, so two quiet groups touching without load would otherwise wake each
  other alternately forever; found by the concave acceptance test (seed 10).
- `Contact::a < b` is not guaranteed: `a` is always a free body and `b` may
  be a static body with a lower id.

## First measurements

Container benchmark, seed 1, 64 bodies, sleeping off, M4 Mini, Release:

| | 60 Hz | 120 Hz |
| --- | ---: | ---: |
| Mean step | 1.46 ms | 1.14 ms |
| Worst step | 9.13 ms | 6.27 ms |
| Mean step while anything moves faster than 4 mm/s | 3.75 ms | 2.67 ms |
| Mean step once quiet (still awake, colliding every pair) | 1.07 ms | 0.91 ms |
| Newton solves / factorisations | 3,237 / 3,232 | 5,717 / 5,712 |
| Wall-clock for 8 s | 0.70 s | 1.09 s |

The JavaScript prototype took 2.00 s and Rapier 0.22 s for the same 8 s. Final
state: no overlapping pair, deepest overlap 0.003 mm, quiet after 1.20 s.

Nothing here is optimised. The quiet-frame cost is almost all narrow phase:
every pair runs the full separating-axis test with a quadratic edge search
each frame, with no cached axis and no broad phase beyond bounding spheres.
The moving-frame cost is about 45 Newton solves per frame at about 60
microseconds each. Both are the next things to work on, against the
specification's 1.5 ms mean for 75 awake bodies.

## Acceptance tests (game copy, 2026-10-03, this Mac)

`phys_tests` (all, ~15 s): every test passes except 16, performance, whose
timings were taken on a machine at load average 15 to 30 and are not
trustworthy (see below). Highlights:

- 1 rest, 150 rocks for 600 s: zero drift; asleep by 1.3 s.
- 2 tall stack, twenty 15-rock stacks for 600 s: zero drift, zero energy asleep.
- 4 overhang: 2% inside stays, 2% outside tips 62 degrees (1%: stays).
- 8 set-down: tension falls monotonically to slack; the stack moves 0.004 mm.
- 9 shove cap: exactly 0.3500 m g.
- 14 determinism: identical checksums in one process and across processes.
- 15 robustness: no guard, no speed beyond free fall, all asleep by 10 s.

Fixed while porting: islands now join every reported contact (see above).
Test-side corrections: the incline test uses a static slab (a free one slides
with the box riding on it, correctly); the consequence test measures the stack
apart from the newcomer's own 1 mm settling drop; the static-table check runs
with sleeping off, as in the prototype.

## Pair cache (2026-10-03)

The narrow phase keeps, per hull pair, the bodies' poses when it last ran and
either the manifold (in each body's frame) or the separating distance it found
(`collide_hulls` now reports it). Next frame:

- an apart pair is skipped while the bodies have moved less than that distance
  minus the margin (exact and conservative: any separating axis bounds the gap
  from below, and motion is bounded by displacement plus pi |vector part| times
  radius);
- a touching pair reuses its manifold while the bodies have moved less than
  0.1 micrometre and the margin hasn't grown by more than 0.1 micrometre.

On the quiet heap, 93% of pair tests are answered from the cache. Clean timing
of the performance scene before the machine got busy: 1.71 ms mean awake
(budget 1.5), 0.007 ms asleep (budget 0.1); the first 60 frames of 75 rocks
landing at once average about 7 ms (Newton solves). A game-like frame (the
crane lowering a rock onto a sleeping 15-rock stack, then letting go) averages
0.26 ms. The quadratic edge query is untouched: it is where the prototype found
two collision faults, and `test_collide.mjs` (the brute-force oracle) isn't
ported to C++ yet. `SolveStats::pair_tests` and `pair_cache_hits` count the
cache's work.
