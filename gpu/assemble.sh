#!/bin/sh
# Reassembles docs/gpu/index.html from the page, shaders and renderer without recompiling the engine.
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
template = open("gpu/index.template.html").read()
engine = open("out/teapot_gpu_engine.js").read().replace("</script>", "<\\/script>")
page = template.replace("/*ENGINE*/", engine).replace("/*SHADERS*/", open("gpu/shaders.js").read()).replace("/*RENDERER*/", open("gpu/renderer.js").read())
open("docs/gpu/index.html", "w").write(page)
PY
mkdir -p web/gpu && cp docs/gpu/index.html web/gpu/index.html
