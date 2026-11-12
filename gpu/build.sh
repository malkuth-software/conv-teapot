#!/bin/sh
# Builds the GPU version, one self-contained page, published at two addresses of the
# site: docs/conv-teapot-gpu/index.html and docs/gpu/index.html. The
# engine (teapot, physics, pointer) is compiled to WebAssembly; drawing is
# WebGL 2 (gpu/renderer.js, gpu/shaders.js). Needs Emscripten (em++).
set -e
cd "$(dirname "$0")/.."
mkdir -p out docs/gpu docs/conv-teapot-gpu
em++ -std=c++20 -O3 -ffp-contract=off -fwasm-exceptions -DTCONV_SINGLE_THREAD gpu/main.cpp third_party/zc_phys/*.cpp -o out/teapot_gpu_engine.js \
  -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=8MB -sMODULARIZE=1 -sEXPORT_NAME=TeapotEngine -sENVIRONMENT=web \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAPF64,HEAPU32 \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_tp_init,_tp_set,_tp_build,_tp_buffer,_tp_length,_tp_frame,_tp_evaluate,_tp_pointer,_tp_zoom,_tp_body,_tp_solver_body,_tp_hull_triangles,_tp_sheet,_tp_hook_buffer,_tp_hook_length,_tp_rows_wanted,_tp_rows_given,_tp_rows_scale,_tp_rows_buffer,_tp_rows,_tp_pose,_tp_point,_tp_camera,_tp_fly,_tp_action
python3 - <<'PY'
template = open("gpu/index.template.html").read()
engine = open("out/teapot_gpu_engine.js").read().replace("</script>", "<\\/script>")
page = template.replace("/*ENGINE*/", engine).replace("/*SHADERS*/", open("gpu/shaders.js").read()).replace("/*RENDERER*/", open("gpu/renderer.js").read() + open("gpu/physics.js").read() + open("gpu/wrench.js").read() + open("gpu/rows.js").read() + open("gpu/build.js").read())
import os
for folder in ("docs/conv-teapot-gpu", "docs/gpu"):
    os.makedirs(folder, exist_ok=True)
    open(folder + "/index.html", "w").write(page)
print("docs/conv-teapot-gpu/index.html and docs/gpu/index.html", len(page), "bytes")
PY
