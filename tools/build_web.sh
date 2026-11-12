#!/bin/sh
# Builds docs/index.html (the page GitHub Pages serves): the engine compiled to
# WebAssembly and inlined into one self-contained page, with the paper beside
# it. Needs Emscripten (em++) on the PATH.
set -e
cd "$(dirname "$0")/.."
mkdir -p out
em++ -std=c++20 -O3 -ffp-contract=off -fwasm-exceptions -DTCONV_SINGLE_THREAD web/main.cpp third_party/zc_phys/*.cpp -o out/teapot_engine.js \
  -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=8MB -sMODULARIZE=1 -sEXPORT_NAME=TeapotEngine -sENVIRONMENT=web \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8 \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_tp_init,_tp_draw,_tp_set,_tp_pointer,_tp_zoom,_tp_action,_tp_stat,_tp_export,_tp_export_data
python3 - <<'PY'
template = open("web/index.template.html").read()
engine = open("out/teapot_engine.js").read().replace("</script>", "<\\/script>")
import os, shutil
os.makedirs("docs", exist_ok=True)
open("docs/index.html", "w").write(template.replace("/*ENGINE*/", engine))
open("docs/.nojekyll", "w").close()
if os.path.exists("paper/conv_teapot.pdf"):
    shutil.copyfile("paper/conv_teapot.pdf", "docs/conv_teapot.pdf")
print("docs/index.html", len(template) + len(engine), "bytes")
PY
