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
    // A change to how surfaces are shaded makes a baked light map stale.
    if ((key >= 4 && key <= 10) || (key >= 16 && key <= 18)) { e.mesh.lit = false; }
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
    case 27: e.mesh_options.layout = Layout(int(value)); break;
    case 28: e.mesh_options.tolerance = value; break;
    case 29: e.mesh_options.angle = value; break;
    case 30: e.render.baked_light = value != 0; break;
    case 31: e.render.mesh_lines = value != 0; break;
    case 32: e.render.supersample = int(value); break;
    case 33: e.light_texel = value; break;
    case 34: e.render.live_reflection = value != 0; e.mesh.lit = false; break;
    case 35: e.render.exact_shadows = value != 0; e.mesh.lit = false; break;
    case 36: e.side = int(value); e.mesh.lit = false; break;
    case 37: e.joins = value != 0; e.rebuild(); break;
    case 38: e.mesh_options.match_edge_normals = value != 0; break;
    case 39: e.mesh_options.diagonal = int(value); break;
    case 40: e.quick_while_moving = value != 0; break;
    case 41: e.render.ambient = int(value); e.mesh.lit = false; break;
    case 42: e.render.plain_inside = value != 0; e.mesh.lit = false; break;
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
EMSCRIPTEN_KEEPALIVE void tp_fly(double forward, double right, double up) { engine->fly(forward, right, up); }
EMSCRIPTEN_KEEPALIVE void tp_action(int action) {
    if (action == 0) { engine->drop(); }
    if (action == 1) { engine->reset(); }
    if (action == 2) { engine->bake_vertices(); }
    if (action == 3) { engine->bake_light(); }
    if (action == 4) { engine->home(); }
}

// The baked mesh as OBJ text: returns its length; tp_export_data points at it.
static std::string exported;
EMSCRIPTEN_KEEPALIVE int tp_export(int z_up) {
    exported = engine->export_obj(z_up != 0);
    return int(exported.size());
}
EMSCRIPTEN_KEEPALIVE const char* tp_export_data() { return exported.data(); }

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
    case 13: return double(engine->mesh.vertices.size());
    case 14: return double(engine->mesh.triangle_count());
    case 15: return engine->mesh.ms_build;
    case 16: return engine->mesh.lit ? double(engine->mesh.texels) : 0.0;
    case 17: return engine->mesh.ms_light;
    case 18: return double(engine->mesh.evaluations);
    default: return 0.0;
    }
}

} // extern "C"
