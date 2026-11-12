#!/bin/sh
# Builds and runs the checks: the line kernel against the stx_conv reference,
# exact reproduction of the Bezier teapot, and a frame against the surface.
set -e
cd "$(dirname "$0")/.."
sh build.sh >/dev/null
out/line_check
out/study | awk '/seams at patch boundaries/{seen=1} seen && $1=="4" && $3=="continuous"{ if ($4+0 > 1e-12) { print "FAIL exact reproduction", $4; exit 1 } else { print "exact reproduction at per 4: max deviation", $4 } ; exit 0 }'
out/capture size=320x200 verify=1 | tail -2
out/grab_check
out/engine_check
out/capture size=320x200 version=2026u method=raster light=1 | head -2
echo "checks done"
