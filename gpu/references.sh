#!/bin/sh
# CPU reference pictures for gpu/harness.js, written beside a locally served copy of the page.
# Usage: sh gpu/references.sh DIRECTORY   (needs ./build.sh to have been run)
set -e
cd "$(dirname "$0")/.."
D="${1:-web/gpu}"
C="version=2026u size=960x600 azimuth=68.7549355 elevation=20.6264806 lookx=0.1 lookat=1.3 distance=11 fov=34"
R="$C method=independent ambient=0"
out/capture $R out=$D/ref_porcelain.png >/dev/null
out/capture $R floor=1 out=$D/ref_polished.png >/dev/null
out/capture $R material=1 gloss=1 out=$D/ref_chrome.png >/dev/null
out/capture $R texture=1 out=$D/ref_texture.png >/dev/null
out/capture $R side=left out=$D/ref_cut.png >/dev/null
B="$C method=raster light=1 tolerance=0.0025 angle=8"
out/capture $B ambient=1 out=$D/ref_room.png >/dev/null
out/capture $B ambient=3 out=$D/ref_gather.png >/dev/null
out/capture $B ambient=3 hide=lid elevation=50 out=$D/ref_gather_open.png >/dev/null
echo "references in $D"
