#!/bin/sh
# Builds the native tools into out/. Needs a C++20 compiler (CXX, default c++).
set -e
cd "$(dirname "$0")"
mkdir -p out
CXX="${CXX:-c++}"
FLAGS="-std=c++20 -O3 -ffp-contract=off -Wall -Wextra"
if [ ! -f out/zc_phys.a ] || [ -n "$(find third_party/zc_phys -newer out/zc_phys.a -name "*.?pp")" ]; then
  for f in hull collide block_ldl solver world; do $CXX $FLAGS -c third_party/zc_phys/$f.cpp -o out/zc_$f.o; done
  ar rcs out/zc_phys.a out/zc_hull.o out/zc_collide.o out/zc_block_ldl.o out/zc_solver.o out/zc_world.o
fi
$CXX $FLAGS tools/capture.cpp out/zc_phys.a -o out/capture
$CXX $FLAGS tools/study.cpp -o out/study
$CXX $FLAGS tools/qualities.cpp -o out/qualities
$CXX $FLAGS tools/unified_study.cpp -o out/unified_study
$CXX $FLAGS tools/budget.cpp -o out/budget
$CXX $FLAGS tools/mesh_study.cpp -o out/mesh_study
$CXX $FLAGS tools/curvature_budget.cpp -o out/curvature_budget
$CXX $FLAGS tests/line_check.cpp -o out/line_check
$CXX $FLAGS tests/grab_check.cpp out/zc_phys.a -o out/grab_check
echo built
