// Web front-end glue: a handful of C functions over tconv::Engine.
#include "../src/engine.hpp"
#include <emscripten/emscripten.h>

static tconv::Engine* engine = nullptr;

extern "C" {

EMSCRIPTEN_KEEPALIVE void tp_init(int width, int height) {
    if (engine == nullptr) { engine = new tconv::Engine(); }
    engine->resize(width, height);
}

EMSCRIPTEN_KEEPALIVE const std::uint8_t* tp_draw(double dt) { return engine->draw(dt); }

// Options by name index; see OPTIONS in index.html.
EMSCRIPTEN_KEEPALIVE void tp_set(int key, double value) {
    using namespace tconv;
    Engine& e = *engine;
    switch (key) {
    case 0: e.geometry = Geometry(int(value)); e.rebuild(); break;
    case 1: e.per = std::size_t(value); e.rebuild(); break;
    case 2: e.bubble = Bubble(int(value)); e.rebuild(); break;
    case 3: e.render.method = Method(int(value)); break;
    case 4: e.render.wire = Wire(int(value)); break;
    case 5: e.render.texture = int(value); break;
    case 6: e.render.filter = TextureFilter(int(value)); break;
    case 7: e.render.mapping = TextureMap(int(value)); break;
    case 8: e.render.lighting = value != 0; break;
    case 9: e.render.shadows = value != 0; break;
    case 10: e.render.ground = value != 0; break;
    case 11: e.render.show_work = value != 0; break;
    case 12: e.running = value != 0; break;
    case 14: e.bottom_cap = value != 0; e.rebuild(); break;
    case 15: e.render.subdivision = int(value); break;
    case 16: e.render.material = int(value); break;
    case 17: e.render.gloss = value != 0; break;
    case 18: e.render.floor_reflection = value != 0; break;
    case 19: e.render.edge_samples = value != 0; break;
    case 20: e.version = Version(int(value)); e.blinn_scale = e.version != Version::v1975; if (e.version == Version::v2026 && e.per < 8) { e.per = 8; } e.rebuild(); break;
    case 21: e.blinn_scale = value != 0; e.rebuild(); break;
    case 22: e.show_handle = value != 0; break;
    case 23: e.show_spout = value != 0; break;
    case 24: e.show_lid = value != 0; break;
    case 25: e.show_body = value != 0; break;
    case 26: e.show_interior = value != 0; break;
    default: break;
    }
}

EMSCRIPTEN_KEEPALIVE void tp_pointer(int kind, double x, double y, int orbit) {
    if (kind == 0) { engine->pointer_down(x, y, orbit != 0); }
    if (kind == 1) { engine->pointer_move(x, y); }
    if (kind == 2) { engine->pointer_up(); }
    if (kind == 3) { engine->poke(x, y); }
}
EMSCRIPTEN_KEEPALIVE void tp_zoom(double factor) { engine->zoom(factor); }
EMSCRIPTEN_KEEPALIVE void tp_action(int action) {
    if (action == 0) { engine->drop(); }
    if (action == 1) { engine->reset(); }
}

EMSCRIPTEN_KEEPALIVE double tp_stat(int index) {
    const tconv::RenderStats& s = engine->renderer.stats;
    switch (index) {
    case 0: return s.ms_total;
    case 1: return s.ms_lattice;
    case 2: return s.ms_predict;
    case 3: return s.ms_solve;
    case 4: return double(s.hit_pixels);
    case 5: return double(s.solves);
    case 6: return double(s.iterations);
    case 7: return double(s.lattice_points);
    case 8: return double(engine->teapot().node_count());
    case 9: return engine->ms_sim;
    case 10: return engine->running ? 1.0 : 0.0;
    case 12: return double(engine->sim.contacts());
    case 11: return double(s.node_visits);
    default: return 0.0;
    }
}

} // extern "C"
