// The engine as the web page drives it: every renderer draws a frame, the
// cutaway and the mesh export work, and a drop moves the teapot.
#include "../src/engine.hpp"
#include <cstdio>
using namespace tconv;

int main() {
    Engine engine;
    engine.version = Version::v2026_unified;
    engine.rebuild();
    engine.resize(160, 100);
    bool ok = true;
    for (const Method method : {Method::transport, Method::independent, Method::raster, Method::seeded}) {
        engine.render.method = method;
        for (int side = 0; side < 3; ++side) {
            engine.side = side;
            engine.draw(0.0);
            ok = ok && engine.renderer.stats.hit_pixels > 500;
        }
    }
    engine.side = 0;
    engine.bake_light();
    engine.draw(0.0);
    ok = ok && engine.mesh.lit;
    const std::string obj = engine.export_obj(true);
    ok = ok && obj.size() > 100000;
    engine.drop();
    for (int k = 0; k < 60; ++k) { engine.draw(1.0 / 30.0); }
    ok = ok && engine.sim.moved();
    std::printf("engine: %zu vertices, %zu triangles, obj %zu bytes, light map %zu texels\n", engine.mesh.vertices.size(), engine.mesh.triangle_count(), obj.size(),
                engine.mesh.texels);
    std::puts(ok ? "engine check passed" : "FAIL engine check");
    return ok ? 0 : 1;
}
