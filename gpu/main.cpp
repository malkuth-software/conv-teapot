// The GPU version's glue: the engine builds the teapot, runs the physics and
// answers the pointer; everything that is drawn is handed to WebGL as data.
// What is handed over is what the CPU evaluator works from: the admitted
// u-cells of every lattice row and the shape of every v-cell, so that the
// shaders evaluate the same CONV surface, and the baked mesh's (u, v) layout,
// which the shaders turn into points by evaluating the surface there.
#include "../src/engine.hpp"
#include <emscripten/emscripten.h>
#include <map>

static tconv::Engine* engine = nullptr;

namespace {

constexpr int atlas_width = 2048;
std::vector<float> cells, shapes, lattice_meta, part_meta, vertices, trim_table, trim_data, trim_table_exact;
double rest_bounds[4] = {0, 0, 0, 1};
std::vector<std::uint32_t> indices;
std::vector<double> state(64, 0.0), evaluated(9, 0.0), body;
std::uint64_t built_for = 0;

struct LatticeEntry {
    int cells_base, shapes_base;
};

void pad(std::vector<float>& texels) {
    const std::size_t row = std::size_t(atlas_width) * 4;
    texels.resize((texels.size() + row - 1) / row * row, 0.0f);
}

void build() {
    using namespace tconv;
    Engine& e = *engine;
    e.bake_vertices();
    const Scene scene = e.scene_now();
    const Teapot& teapot = e.teapot();
    cells.clear(); shapes.clear(); lattice_meta.clear(); part_meta.clear(); vertices.clear(); indices.clear(); trim_table.clear(); trim_data.clear(); trim_table_exact.clear();

    // Every lattice once: the parts', the second side of each band, the textures'.
    std::map<const ConvSurface*, int> index_of;
    const auto add = [&](const ConvSurface& surface) {
        const auto found = index_of.find(&surface);
        if (found != index_of.end()) { return found->second; }
        const int id = int(index_of.size());
        index_of[&surface] = id;
        const Lattice& lattice = surface.lattice;
        const std::size_t cu = lattice.cells_u(), cv = lattice.cells_v();
        lattice_meta.insert(lattice_meta.end(), {float(cells.size() / 4), float(shapes.size() / 4), float(cu), float(cv), float(lattice.nv), lattice.wrap_u ? 1.0f : 0.0f,
                                                 float(lattice.nu), 0.0f});
        const std::vector<Cell<double>>& rows = surface.compiled_rows();
        for (std::size_t k = 0; k < lattice.nv * cu; ++k) {
            for (int channel = 0; channel < 3; ++channel) {
                const Cell<double>& cell = rows[k * 3 + std::size_t(channel)];
                cells.insert(cells.end(), {float(cell.anchor), float(cell.c[0]), float(cell.c[1]), float(cell.c[2]), float(cell.c[3]), float(cell.c[4]), float(cell.bubble), 0.0f});
            }
        }
        const std::vector<CellShape>& along = surface.shapes_along();
        for (std::size_t k = 0; k < cv; ++k) {
            const CellShape& shape = along[k];
            shapes.insert(shapes.end(), {float(int(shape.left)), float(int(shape.right)), float(int(shape.prev)), float(int(shape.next)), shape.has_prev ? 1.0f : 0.0f,
                                         shape.has_next ? 1.0f : 0.0f, shape.six ? 1.0f : 0.0f, float((surface.reads_low(k) ? 1 : 0) + (surface.reads_high(k) ? 2 : 0))});
        }
        return id;
    };
    for (const auto& part : teapot.parts) {
        add(part->conv);
        if (part->conv.toward) { add(*part->conv.toward); }
    }
    for (const auto& texture : e.textures()) { add(texture->conv); }
    pad(cells);
    pad(shapes);

    const Rigid& pose = e.pose();
    const auto to_body = [&](Vec3 d) { return Vec3{pose.a[0] * d.x + pose.a[3] * d.y + pose.a[6] * d.z, pose.a[1] * d.x + pose.a[4] * d.y + pose.a[7] * d.z, pose.a[2] * d.x + pose.a[5] * d.y + pose.a[8] * d.z}; };
    for (const MeshVertex& vertex : e.mesh.vertices) {
        const Vec3 n = to_body(vertex.n);
        vertices.insert(vertices.end(), {vertex.u, vertex.v, vertex.side, float(n.x), float(n.y), float(n.z)});
    }
    indices = e.mesh.triangles;
    // A sphere about the teapot at rest that holds it however it is turned.
    Vec3 low{1e30, 1e30, 1e30}, high{-1e30, -1e30, -1e30};
    for (const MeshVertex& vertex : e.mesh.vertices) {
        const Vec3 p = to_body(vertex.p - pose.b);
        low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
        high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
    }
    const Vec3 centre = (low + high) * 0.5;
    rest_bounds[0] = centre.x; rest_bounds[1] = centre.y; rest_bounds[2] = centre.z;
    rest_bounds[3] = 0.5 * length(high - low) * 1.02 + 1e-6;

    const std::vector<std::size_t>& shown = e.scene_parts();
    for (std::size_t k = 0; k < shown.size(); ++k) {
        const TeapotPart& part = *teapot.parts[shown[k]];
        const RenderPart& render = scene.parts[k];
        const MeshPart& baked = e.mesh.parts[k];
        const auto lattice_index = [&](const Surface* surface) {
            for (const auto& entry : index_of) {
                if (static_cast<const Surface*>(entry.first) == surface) { return float(entry.second); }
            }
            return -1.0f;
        };
        const float group = part.group == "lid" ? 1.0f : part.group == "handle" ? 2.0f : part.group == "spout" ? 3.0f : 0.0f;
        const float trim_first = float(trim_table.size() / 12);
        // Each trim twice: as drawn (stopping inside the band, so no gap shows) and
        // exact (at the band's own edge, so every point belongs to one sheet: for counting crossings).
        for (int exact = 0; exact < 2; ++exact) {
            std::vector<float>& table = exact == 0 ? trim_table : trim_table_exact;
            for (const Trim* trim : render.trims) {
                const std::vector<double>& cut = exact != 0 && !trim->cut_exact.empty() ? trim->cut_exact : trim->cut;
                const auto& loop = exact != 0 && !trim->loop_exact.empty() ? trim->loop_exact : trim->loop;
                const float offset = float(trim_data.size());
                if (!cut.empty()) {
                    for (double c : cut) { trim_data.push_back(float(c)); }
                    table.insert(table.end(), {1.0f, offset, float(cut.size()), trim->below ? 1.0f : 0.0f, 0, 0, 0, 0, 0, 0, 0, 0});
                } else if (!loop.empty()) {
                    float low_u = 1e30f, high_u = -1e30f, low_v = 1e30f, high_v = -1e30f;
                    for (const auto& q : loop) {
                        trim_data.push_back(float(q[0]));
                        trim_data.push_back(float(q[1]));
                        low_u = std::min(low_u, float(q[0])); high_u = std::max(high_u, float(q[0]));
                        low_v = std::min(low_v, float(q[1])); high_v = std::max(high_v, float(q[1]));
                    }
                    table.insert(table.end(), {0.0f, offset, float(loop.size()), 0.0f, float(trim->centre_u), float(trim->period_u), 0, 0, low_u, high_u, low_v, high_v});
                }
            }
        }
        part_meta.insert(part_meta.end(), {
            lattice_index(&part.conv), part.conv.toward ? lattice_index(part.conv.toward.get()) : -1.0f,
            part.conv.on != nullptr ? lattice_index(part.conv.on) : -1.0f, part.conv.toward_on != nullptr ? lattice_index(part.conv.toward_on) : -1.0f,
            float(part.conv.spread), float(render.orientation), float(std::min(1e9, render.interior_from)), float(baked.first_triangle * 3), float(baked.triangle_count * 3),
            trim_first, float(trim_table.size() / 12) - trim_first, group, float(render.per_u), float(render.per_v), render.two_sided ? 1.0f : 0.0f,
            float(part.conv.cells_u()), float(part.conv.cells_v()), part.conv.toward ? 1.0f : 0.0f, float(baked.first_vertex), float(baked.vertex_count),
            render.solid ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f});
    }
    // The textures' lattices, in the order of the texture control.
    for (const auto& texture : e.textures()) { part_meta.push_back(float(index_of[&texture->conv])); }
    trim_data.resize((trim_data.size() + 3) / 4 * 4 + 4, 0.0f);
    built_for = e.shown_signature();
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void tp_init(int width, int height) {
    if (engine == nullptr) {
        engine = new tconv::Engine();
        engine->mesh_options.tolerance = 2.5e-3;
        engine->mesh_options.angle = 8.0;
    }
    engine->resize(width, height);
}

EMSCRIPTEN_KEEPALIVE void tp_set(int key, double value) {
    using namespace tconv;
    Engine& e = *engine;
    switch (key) {
    case 0: e.geometry = Geometry(int(value)); e.rebuild(); break;
    case 1: e.per = std::size_t(value); e.rebuild(); break;
    case 12: e.running = value != 0; break;
    case 14: e.bottom_cap = value != 0; e.rebuild(); break;
    case 20: e.version = Version(int(value)); e.blinn_scale = e.version != Version::v1975; if (e.version == Version::v2026 && e.per < 8) { e.per = 8; } e.rebuild(); break;
    case 21: e.blinn_scale = value != 0; e.rebuild(); break;
    case 22: e.show_handle = value != 0; break;
    case 23: e.show_spout = value != 0; break;
    case 24: e.show_lid = value != 0; break;
    case 25: e.show_body = value != 0; break;
    case 26: e.show_interior = value != 0; break;
    case 28: e.mesh_options.tolerance = value; break;
    case 29: e.mesh_options.angle = value; break;
    case 36: e.side = int(value); break;
    case 37: e.joins = value != 0; e.rebuild(); break;
    default: break;
    }
}

// The buffers for what is shown now; rebuilt when that has changed. Returns 1 if they were rebuilt.
EMSCRIPTEN_KEEPALIVE int tp_build() {
    if (!cells.empty() && built_for == engine->shown_signature()) { return 0; }
    build();
    return 1;
}
EMSCRIPTEN_KEEPALIVE const void* tp_buffer(int kind) {
    switch (kind) {
    case 0: return cells.data();
    case 1: return shapes.data();
    case 2: return lattice_meta.data();
    case 3: return part_meta.data();
    case 4: return vertices.data();
    case 5: return indices.data();
    case 6: return trim_table.data();
    case 7: return trim_data.data();
    case 9: return trim_table_exact.data();
    default: return nullptr;
    }
}
EMSCRIPTEN_KEEPALIVE int tp_length(int kind) {
    switch (kind) {
    case 0: return int(cells.size());
    case 1: return int(shapes.size());
    case 2: return int(lattice_meta.size());
    case 3: return int(part_meta.size());
    case 4: return int(vertices.size());
    case 5: return int(indices.size());
    case 6: return int(trim_table.size());
    case 7: return int(trim_data.size());
    case 9: return int(trim_table_exact.size());
    case 8: return int(engine->scene_parts().size());
    default: return 0;
    }
}

// Advance the simulation; then the state that changes from frame to frame:
// pose (rotation by rows, then translation), eye, target, vertical field of
// view, light direction, cut (flag, normal, offset), running, contacts,
// physics time, node count.
EMSCRIPTEN_KEEPALIVE const double* tp_frame(double dt) {
    using namespace tconv;
    Engine& e = *engine;
    e.advance(dt);
    const Scene scene = e.scene_now();
    const Camera camera = e.camera();
    const Rigid& pose = e.pose();
    for (int k = 0; k < 9; ++k) { state[std::size_t(k)] = pose.a[k]; }
    state[9] = pose.b.x; state[10] = pose.b.y; state[11] = pose.b.z;
    state[12] = camera.eye.x; state[13] = camera.eye.y; state[14] = camera.eye.z;
    state[15] = camera.target.x; state[16] = camera.target.y; state[17] = camera.target.z;
    state[18] = camera.fov_y;
    state[19] = scene.light.x; state[20] = scene.light.y; state[21] = scene.light.z;
    state[22] = scene.cut ? 1.0 : 0.0;
    state[23] = scene.cut_normal.x; state[24] = scene.cut_normal.y; state[25] = scene.cut_normal.z;
    state[26] = scene.cut_offset;
    state[27] = e.running ? 1.0 : 0.0;
    state[28] = double(e.sim.contacts());
    state[29] = e.ms_sim;
    state[30] = double(e.teapot().node_count());
    state[31] = scene.ground_y;
    state[32] = e.sim.grabbing() ? 1.0 : 0.0;
    for (int k = 0; k < 4; ++k) { state[std::size_t(33 + k)] = rest_bounds[k]; }
    state[37] = scene.axis.x; state[38] = scene.axis.y; state[39] = scene.axis.z;
    state[40] = scene.open_top ? 1.0 : 0.0;
    return state.data();
}

// The CPU's own evaluation of a shown part at rest, for checking the GPU's: point, du, dv.
EMSCRIPTEN_KEEPALIVE const double* tp_evaluate(int part, double u, double v) {
    const tconv::SurfacePoint point = engine->teapot().parts[engine->scene_parts()[std::size_t(part)]]->conv.eval(u, v);
    evaluated = {point.p.x, point.p.y, point.p.z, point.du.x, point.du.y, point.du.z, point.dv.x, point.dv.y, point.dv.z};
    return evaluated.data();
}

EMSCRIPTEN_KEEPALIVE void tp_pointer(int kind, double x, double y, int orbit, int part, double u, double v) {
    if (kind == 0) { engine->pointer_down_at(x, y, part, u, v, orbit != 0); }
    if (kind == 1) { engine->pointer_move(x, y); }
    if (kind == 2) { engine->pointer_up(); }
    if (kind == 3) { engine->poke_at(x, y, part, u, v); }
}
// The rigid body for a simulation on the GPU: mass, centre of mass, inverse
// inertia (9), poke radius, number of collision points, then the points.
EMSCRIPTEN_KEEPALIVE const double* tp_body() {
    const tconv::TeapotSim& sim = engine->sim;
    const tconv::Vec3 com = sim.centre_of_mass();
    const std::array<double, 9> inverse = sim.inverse_inertia_model();
    body = {sim.mass(), com.x, com.y, com.z};
    body.insert(body.end(), inverse.begin(), inverse.end());
    body.push_back(sim.poke_radius());
    body.push_back(double(sim.collision_points().size()));
    for (const tconv::Vec3& p : sim.collision_points()) { body.insert(body.end(), {p.x, p.y, p.z}); }
    return body.data();
}
// The pose from that simulation: rotation by rows, then translation.
EMSCRIPTEN_KEEPALIVE void tp_pose(double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double x, double y, double z) {
    tconv::Rigid pose;
    const double a[9] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
    for (int k = 0; k < 9; ++k) { pose.a[k] = a[k]; }
    pose.b = tconv::Vec3{x, y, z};
    engine->set_pose(pose);
}
// A point of a shown part as the pose has it now.
EMSCRIPTEN_KEEPALIVE const double* tp_point(int part, double u, double v) {
    const tconv::Vec3 p = engine->pose().point(engine->teapot().parts[engine->scene_parts()[std::size_t(part)]]->conv.position(u, v));
    evaluated[0] = p.x; evaluated[1] = p.y; evaluated[2] = p.z;
    return evaluated.data();
}
EMSCRIPTEN_KEEPALIVE void tp_zoom(double factor) { engine->zoom(factor); }
// Place the camera: angles in radians about the point looked at.
EMSCRIPTEN_KEEPALIVE void tp_camera(double azimuth, double elevation, double distance, double x, double y, double z) {
    engine->azimuth = azimuth;
    engine->elevation = elevation;
    engine->distance = distance;
    engine->target = tconv::Vec3{x, y, z};
}
EMSCRIPTEN_KEEPALIVE void tp_fly(double forward, double right, double up) { engine->fly(forward, right, up); }
EMSCRIPTEN_KEEPALIVE void tp_action(int action) {
    if (action == 0) { engine->drop(); }
    if (action == 1) { engine->reset(); }
    if (action == 4) { engine->home(); }
}

} // extern "C"
