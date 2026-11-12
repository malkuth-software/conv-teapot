// The GPU version's glue: the engine builds the teapot, runs the physics and
// answers the pointer; everything that is drawn is handed to WebGL as data.
// What is handed over is what the CPU evaluator works from: the admitted
// u-cells of every lattice row and the shape of every v-cell, so that the
// shaders evaluate the same CONV surface. Where the surface is evaluated (a
// grid over each part), the tree for traced rays and what is drawn are the
// GPU's own.
#include "../src/engine.hpp"
#include <emscripten/emscripten.h>
#include <map>

static tconv::Engine* engine = nullptr;

namespace {

constexpr int atlas_width = 2048;
std::vector<float> nodes, patches, flat_meta, cells, shapes, lattice_meta, part_meta, trim_table, trim_data, trim_table_exact;
double rest_bounds[4] = {0, 0, 0, 1};
std::vector<double> state(64, 0.0), evaluated(9, 0.0), body;
std::uint64_t built_for = 0;

struct LatticeEntry {
    int cells_base, shapes_base;
};

void pad(std::vector<float>& texels) {
    const std::size_t row = std::size_t(atlas_width) * 4;
    texels.resize((texels.size() + row - 1) / row * row, 0.0f);
}

// ---- Work done by the page in shaders while a teapot is built (BuildHooks).
// The engine calls out at the point where it would do the work; what the
// shader needs is laid out here for the page to read, and the page writes
// its result into hook_out.
EM_JS(int, gpu_hook, (int kind), { return globalThis.teapotHook ? globalThis.teapotHook(kind) : 0; });
std::vector<float> hook_nodes, hook_shapes, hook_meta, hook_data;
std::vector<double> hook_args(32, 0.0), hook_out, hook_sheet;
std::size_t hook_cells = 0;
const tconv::ConvSurface* hook_surface = nullptr;   // the surface the page last compiled for positions

// A lattice as build() hands lattices over: nodes, cell shapes, where its cells go.
void hook_add(const tconv::ConvSurface& surface) {
    using namespace tconv;
    const Lattice& lattice = surface.lattice;
    const std::size_t cu = lattice.cells_u(), cv = lattice.cells_v();
    hook_meta.insert(hook_meta.end(), {float(hook_cells / 4), float(hook_shapes.size() / 4), float(cu), float(cv), float(lattice.nv), lattice.wrap_u ? 1.0f : 0.0f,
                                       float(lattice.nu), float(hook_nodes.size() / 4), 0.0f, 0.0f, 0.0f, 0.0f});
    for (const Vec3& node : lattice.nodes) { hook_nodes.insert(hook_nodes.end(), {float(node.x), float(node.y), float(node.z), 0.0f}); }
    hook_cells += lattice.nv * cu * 24;
    const std::vector<CellShape>& along = surface.shapes_along();
    for (std::size_t k = 0; k < cv; ++k) {
        const CellShape& shape = along[k];
        hook_shapes.insert(hook_shapes.end(), {float(int(shape.left)), float(int(shape.right)), float(int(shape.prev)), float(int(shape.next)), shape.has_prev ? 1.0f : 0.0f,
                                               shape.has_next ? 1.0f : 0.0f, shape.six ? 1.0f : 0.0f, float((surface.reads_low(k) ? 1 : 0) + (surface.reads_high(k) ? 2 : 0))});
    }
    hook_meta[hook_meta.size() - 4] = float(hook_shapes.size() / 4);
    for (const CellShape& shape : surface.shapes_around()) {
        hook_shapes.insert(hook_shapes.end(), {float(int(shape.left)), float(int(shape.right)), float(int(shape.prev)), float(int(shape.next)), shape.has_prev ? 1.0f : 0.0f,
                                               shape.has_next ? 1.0f : 0.0f, shape.six ? 1.0f : 0.0f, 0.0f});
    }
}
void hook_clear() {
    hook_nodes.clear(); hook_shapes.clear(); hook_meta.clear(); hook_data.clear();
    hook_cells = 0;
    hook_surface = nullptr;
}
// The meridian or seam a planner works from (see tp_sheet).
void fill_sheet(const tconv::ProfileCurve& coarse, std::size_t rows, std::size_t around, std::vector<double>& out) {
    using namespace tconv;
    ProfileCurve fine = coarse;
    fine.build_table(32);
    std::vector<double> creases;
    for (std::size_t run = 1; run < fine.runs.size(); ++run) {
        if (fine.turn_at_joint(run) > 4.0) { creases.push_back(fine.run_start[run]); }
    }
    out = {1.0, double(rows), double(around), fine.length, double(fine.runs.size()), double(fine.runs.empty() ? 0 : fine.runs[0].x.size()),
           double(creases.size()), double(fine.table_s.size()), double(coarse.table_s.size()), 0.0};
    out.insert(out.end(), creases.begin(), creases.end());
    for (const ProfileCurve::Run& run : fine.runs) {
        for (std::size_t k = 0; k < run.x.size(); ++k) { out.insert(out.end(), {run.x[k], run.y[k]}); }
    }
    for (std::size_t k = 0; k < fine.table_s.size(); ++k) {
        out.insert(out.end(), {fine.table_s[k], fine.table_x[k], fine.table_y[k], fine.table_t[k], double(fine.table_run[k])});
    }
    for (std::size_t k = 0; k < coarse.table_s.size(); ++k) { out.insert(out.end(), {coarse.table_s[k], coarse.table_x[k], coarse.table_y[k]}); }
}

void install_hooks(tconv::Engine& e) {
    using namespace tconv;
    e.use_hooks = true;
    e.lazy_sim = true;
    e.hooks.begin = [] { hook_surface = nullptr; gpu_hook(0); };
    // 1: nodes sampled from Bezier patches.
    e.hooks.bezier_nodes = [](const BezierSurface& bezier, bool revolved, Lattice& lattice) {
        hook_clear();
        for (const auto& patch : bezier.patches) {
            for (const Vec3& c : patch) { hook_data.insert(hook_data.end(), {float(c.x), float(c.y), float(c.z), 0.0f}); }
        }
        hook_args = {double(bezier.patches_u), double(bezier.patches_v), double(bezier.per_u), double(bezier.per_v), bezier.wrap_u ? 1.0 : 0.0,
                     double(lattice.nu), double(lattice.nv), revolved ? 1.0 : 0.0, double(bezier.cells_u())};
        hook_out.assign(lattice.nu * lattice.nv * 3, 0.0);
        if (gpu_hook(1) == 0) { return false; }
        for (std::size_t k = 0; k < lattice.nodes.size(); ++k) { lattice.nodes[k] = {hook_out[k * 3], hook_out[k * 3 + 1], hook_out[k * 3 + 2]}; }
        return true;
    };
    // 2: the rows of a seam, by CONV's own error.
    e.hooks.rows = [](const ProfileCurve& seam, std::size_t rows, bool on_axis, SheetPlan& plan) {
        hook_clear();
        fill_sheet(seam, rows, 0, hook_sheet);
        hook_args = {on_axis ? 1.0 : 0.0};
        hook_out.assign(1024, 0.0);
        if (gpu_hook(2) == 0) { return false; }
        const std::size_t n = std::size_t(hook_out[0]);
        if (n < 2 || n > 500) { return false; }
        plan.row_arc.assign(hook_out.begin() + 1, hook_out.begin() + 1 + long(n));
        plan.row_seam.clear();
        for (std::size_t k = 0; k < n; ++k) { plan.row_seam.push_back(hook_out[1 + n + k] > 0.5 ? 1 : 0); }
        return true;
    };
    // 6: the nodes of a sheet of revolution: every row a circle of the meridian's radius there.
    e.hooks.sheet_nodes = [](const ProfileCurve& profile, const SheetPlan& plan, Lattice& lattice) {
        hook_clear();
        for (std::size_t k = 0; k < profile.table_s.size(); ++k) { hook_data.insert(hook_data.end(), {float(profile.table_s[k]), float(profile.table_x[k]), float(profile.table_y[k]), 0.0f}); }
        for (double arc : plan.row_arc) { hook_data.insert(hook_data.end(), {float(arc), 0.0f, 0.0f, 0.0f}); }
        hook_args = {double(profile.table_s.size()), double(lattice.nv), double(lattice.nu)};
        hook_out.assign(lattice.nu * lattice.nv * 3, 0.0);
        if (gpu_hook(6) == 0) { return false; }
        for (std::size_t k = 0; k < lattice.nodes.size(); ++k) { lattice.nodes[k] = {hook_out[k * 3], hook_out[k * 3 + 1], hook_out[k * 3 + 2]}; }
        return true;
    };
    // 3: a tube's nodes: on each row, points of the source patches' surfaces around it.
    e.hooks.tube_nodes = [](const std::vector<ConvSurface>& sources, const ProfileCurve& seam, const SheetPlan& plan, std::size_t around, Lattice& lattice) {
        if (sources.empty() || sources.size() > 8) { return false; }
        hook_clear();
        for (const ConvSurface& source : sources) { hook_add(source); }
        for (std::size_t k = 0; k < seam.table_s.size(); ++k) { hook_data.insert(hook_data.end(), {float(seam.table_s[k]), float(seam.table_t[k]), float(seam.table_run[k]), 0.0f}); }
        for (double arc : plan.row_arc) { hook_data.insert(hook_data.end(), {float(arc), 0.0f, 0.0f, 0.0f}); }
        hook_args = {double(seam.table_s.size()), double(plan.row_arc.size()), double(around), double(sources[0].lattice.nu), double(sources.size())};
        hook_out.assign(lattice.nu * lattice.nv * 3, 0.0);
        if (gpu_hook(3) == 0) { return false; }
        for (std::size_t k = 0; k < lattice.nodes.size(); ++k) { lattice.nodes[k] = {hook_out[k * 3], hook_out[k * 3 + 1], hook_out[k * 3 + 2]}; }
        return true;
    };
    // 4: a join: where each column of the tube crosses the body, the two sides of the band from there, the trims.
    e.hooks.join = [](const ConvSurface& body, const BodyBranch& branch, const ConvSurface& tube, bool at_first, double width, std::size_t body_nu, int across,
                      JoinResult& result) {
        hook_clear();
        hook_add(body);
        hook_add(tube);
        const std::size_t columns = tube.cells_u();
        hook_args = {branch.v_low, branch.v_high, branch.rising ? 1.0 : 0.0, at_first ? 1.0 : 0.0, width, double(body_nu), double(across), double(columns),
                     double(tube.cells_v())};
        const std::size_t rows = std::size_t(across);
        hook_out.assign(2 + columns * rows * 3 + columns * 6, 0.0);
        if (gpu_hook(4) == 0 || hook_out[0] < 0.5) { return false; }
        result = JoinResult{};
        result.blend.wrap_u = true;
        result.blend.closure = Closure::cubic;
        result.blend.nu = columns;
        result.blend.nv = rows;
        result.blend.nodes.resize(columns * rows);
        result.toward = result.blend;
        result.tube_trim.below = at_first;
        result.body_trim.centre_u = hook_out[1];
        result.body_trim.period_u = double(body_nu);
        const double* nodes = hook_out.data() + 2;
        const double* trims = nodes + columns * rows * 3;
        for (std::size_t k = 0; k < columns; ++k) {
            for (std::size_t j = 0; j < rows; ++j) {
                const double* q = nodes + (j * columns + k) * 3;
                result.blend.at(k, j) = Vec3{q[0], q[1], 0.0};
                result.toward.at(k, j) = Vec3{q[2], 0.0, 0.0};
            }
            const double* t = trims + k * 6;
            result.tube_trim.cut.push_back(t[0]);
            result.tube_trim.cut_exact.push_back(t[1]);
            result.body_trim.loop.push_back({t[2], t[3]});
            result.body_trim.loop_exact.push_back({t[4], t[5]});
        }
        round_trim_outlines(result);
        result.ok = true;
        return true;
    };
    // 5: points of a surface.
    e.hooks.positions = [](const ConvSurface& surface, const std::vector<std::array<double, 2>>& at, std::vector<Vec3>& points) {
        if (at.empty() || at.size() > 512 || surface.toward) { return false; }
        const bool fresh = hook_surface != &surface;
        if (fresh) {
            hook_clear();
            hook_add(surface);
            hook_surface = &surface;
        }
        hook_data.clear();
        for (const auto& q : at) { hook_data.insert(hook_data.end(), {float(q[0]), float(q[1]), 0.0f, 0.0f}); }
        hook_args = {double(at.size()), fresh ? 1.0 : 0.0};
        hook_out.assign(at.size() * 3, 0.0);
        if (gpu_hook(5) == 0) { hook_surface = nullptr; return false; }
        points.clear();
        for (std::size_t k = 0; k < at.size(); ++k) { points.push_back({hook_out[k * 3], hook_out[k * 3 + 1], hook_out[k * 3 + 2]}); }
        return true;
    };
}

void build() {
    using namespace tconv;
    Engine& e = *engine;
    const Scene scene = e.scene_now();
    const Teapot& teapot = e.teapot();
    nodes.clear(); patches.clear(); flat_meta.clear(); cells.clear(); shapes.clear(); lattice_meta.clear(); part_meta.clear(); trim_table.clear(); trim_data.clear(); trim_table_exact.clear();

    // Every lattice once: the parts', the second side of each band, the textures'.
    std::map<const ConvSurface*, int> index_of;
    const auto add = [&](const ConvSurface& surface) {
        const auto found = index_of.find(&surface);
        if (found != index_of.end()) { return found->second; }
        const int id = int(index_of.size());
        index_of[&surface] = id;
        const Lattice& lattice = surface.lattice;
        const std::size_t cu = lattice.cells_u(), cv = lattice.cells_v();
        // The nodes themselves and the shape of every u-cell: the GPU admits the
        // u-cells of every row from these (the CPU's compile()), into the place
        // kept for them in the cell texture.
        lattice_meta.insert(lattice_meta.end(), {float(cells.size() / 4), float(shapes.size() / 4), float(cu), float(cv), float(lattice.nv), lattice.wrap_u ? 1.0f : 0.0f,
                                                 float(lattice.nu), float(nodes.size() / 4), 0.0f, 0.0f, 0.0f, 0.0f});
        for (const Vec3& node : lattice.nodes) { nodes.insert(nodes.end(), {float(node.x), float(node.y), float(node.z), 0.0f}); }
        cells.resize(cells.size() + lattice.nv * cu * 24, 0.0f);
        const std::vector<CellShape>& along = surface.shapes_along();
        for (std::size_t k = 0; k < cv; ++k) {
            const CellShape& shape = along[k];
            shapes.insert(shapes.end(), {float(int(shape.left)), float(int(shape.right)), float(int(shape.prev)), float(int(shape.next)), shape.has_prev ? 1.0f : 0.0f,
                                         shape.has_next ? 1.0f : 0.0f, shape.six ? 1.0f : 0.0f, float((surface.reads_low(k) ? 1 : 0) + (surface.reads_high(k) ? 2 : 0))});
        }
        lattice_meta[lattice_meta.size() - 4] = float(shapes.size() / 4);
        for (const CellShape& shape : surface.shapes_around()) {
            shapes.insert(shapes.end(), {float(int(shape.left)), float(int(shape.right)), float(int(shape.prev)), float(int(shape.next)), shape.has_prev ? 1.0f : 0.0f,
                                         shape.has_next ? 1.0f : 0.0f, shape.six ? 1.0f : 0.0f, 0.0f});
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
    // The other two surfaces the page offers: the original Bezier patches, and
    // the nodes joined bilinearly. Per part: where its patches or its nodes are.
    std::vector<std::array<float, 8>> other(teapot.parts.size());
    for (std::size_t k = 0; k < teapot.parts.size(); ++k) {
        const TeapotPart& part = *teapot.parts[k];
        if (e.geometry == Geometry::bezier && !part.sampled) {
            other[k] = {1.0f, float(patches.size() / 4), float(part.bezier.patches_u), float(part.bezier.patches_v), float(part.bezier.per_u), float(part.bezier.per_v),
                        part.bezier.wrap_u ? 1.0f : 0.0f, 0.0f};
            for (const auto& patch : part.bezier.patches) {
                for (const Vec3& c : patch) { patches.insert(patches.end(), {float(c.x), float(c.y), float(c.z), 0.0f}); }
            }
        } else if (e.geometry == Geometry::control_mesh) {
            const Lattice& lattice = *part.linear.lattice;
            other[k] = {2.0f, float(nodes.size() / 4), float(lattice.nu), float(lattice.nv), lattice.wrap_u ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f};
            for (const Vec3& node : lattice.nodes) { nodes.insert(nodes.end(), {float(node.x), float(node.y), float(node.z), 0.0f}); }
        } else {
            other[k] = {0.0f, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    pad(nodes);
    patches.resize(patches.size() + 4, 0.0f);
    pad(patches);

    // A sphere about the teapot at rest that holds it however it is turned: about its nodes, which lie on it.
    {
        Vec3 low{1e30, 1e30, 1e30}, high{-1e30, -1e30, -1e30};
        for (const auto& part : teapot.parts) {
            if (part->conv.on != nullptr) { continue; }   // a join's nodes are parameters
            for (const Vec3& p : part->conv.lattice.nodes) {
                low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
                high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
            }
        }
        const Vec3 centre = (low + high) * 0.5;
        rest_bounds[0] = centre.x; rest_bounds[1] = centre.y; rest_bounds[2] = centre.z;
        rest_bounds[3] = 0.5 * length(high - low) * 1.05 + 1e-6;
    }
    const std::vector<std::size_t>& shown = e.scene_parts();
    for (std::size_t k = 0; k < shown.size(); ++k) {
        const TeapotPart& part = *teapot.parts[shown[k]];
        const RenderPart& render = scene.parts[k];
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
            float(part.conv.spread), float(render.orientation), float(std::min(1e9, render.interior_from)), 0.0f, 0.0f,
            trim_first, float(trim_table.size() / 12) - trim_first, group, float(render.per_u), float(render.per_v), render.two_sided ? 1.0f : 0.0f,
            float(part.conv.cells_u()), float(part.conv.cells_v()), part.conv.toward ? 1.0f : 0.0f, 0.0f, 0.0f,
            render.solid ? 1.0f : 0.0f, float(flat_meta.size() / 8), 0.0f, 0.0f});
        flat_meta.insert(flat_meta.end(), other[shown[k]].begin(), other[shown[k]].end());
    }
    // The textures' lattices, in the order of the texture control.
    for (const auto& texture : e.textures()) { part_meta.push_back(float(index_of[&texture->conv])); }
    flat_meta.resize(flat_meta.size() + 8, 0.0f);
    trim_data.resize((trim_data.size() + 3) / 4 * 4 + 4, 0.0f);
    built_for = e.shown_signature();
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void tp_init(int width, int height) {
    if (engine == nullptr) {
        engine = new tconv::Engine();
        install_hooks(*engine);
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
    case 36: e.side = int(value); break;
    case 37: e.joins = value != 0; e.rebuild(); break;
    case 41: e.use_hooks = value != 0; e.rebuild(); break;
    case 40: e.rows_outside = value != 0; if (e.version == Version::v2026_unified) { e.rebuild(); } break;
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
    case 6: return trim_table.data();
    case 7: return trim_data.data();
    case 9: return trim_table_exact.data();
    case 12: return nodes.data();
    case 13: return patches.data();
    case 14: return flat_meta.data();
    default: return nullptr;
    }
}
EMSCRIPTEN_KEEPALIVE int tp_length(int kind) {
    switch (kind) {
    case 0: return int(cells.size());
    case 1: return int(shapes.size());
    case 2: return int(lattice_meta.size());
    case 3: return int(part_meta.size());
    case 6: return int(trim_table.size());
    case 7: return int(trim_data.size());
    case 9: return int(trim_table_exact.size());
    case 12: return int(nodes.size());
    case 13: return int(patches.size());
    case 14: return int(flat_meta.size());
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
    const tconv::TeapotPart& shown = *engine->teapot().parts[engine->scene_parts()[std::size_t(part)]];
    const tconv::Surface& surface = engine->geometry == tconv::Geometry::bezier && !shown.sampled ? static_cast<const tconv::Surface&>(shown.bezier)
                                    : engine->geometry == tconv::Geometry::control_mesh ? static_cast<const tconv::Surface&>(shown.linear)
                                                                                       : static_cast<const tconv::Surface&>(shown.conv);
    const tconv::SurfacePoint point = surface.eval(u, v);
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
// The body as the engine's own solver has it, in the engine's frame (metres, z up,
// origin at the centre of mass): mass, radius, centre of mass in the input frame (3),
// inertia (9), friction, restitution, rolling resistance, frame time, number of
// hull vertices; then each vertex with the index of its hull.
EMSCRIPTEN_KEEPALIVE const double* tp_solver_body() {
    const zc::phys::Shape& shape = engine->sim.shape();
    body = {shape.mass, shape.radius, shape.com_offset.x, shape.com_offset.y, shape.com_offset.z};
    body.insert(body.end(), shape.inertia.begin(), shape.inertia.end());
    body.insert(body.end(), {shape.friction, shape.restitution, shape.rolling_resistance, engine->sim.world_params().frame_dt});
    std::size_t count = 0;
    for (const auto& hull : shape.hulls) { count += hull.vertices.size(); }
    body.push_back(double(count));
    for (std::size_t h = 0; h < shape.hulls.size(); ++h) {
        for (const zc::phys::Vec3& v : shape.hulls[h].vertices) { body.insert(body.end(), {v.x, v.y, v.z, double(h)}); }
    }
    return body.data();
}
// The faces of the engine's convex hulls as triangles, in the frame the hulls
// were given in (metres, z up), for integrating the body's mass, centre of
// mass and inertia elsewhere: the number of triangles, then nine numbers each.
// The point sets the collision hulls are taken of, for making the hulls elsewhere (metres, z up):
// friction, restitution, rolling resistance, the frame's length, density, the number of sets; then each set's count and points.
EMSCRIPTEN_KEEPALIVE const double* tp_hull_points() {
    const zc::phys::ShapeDesc& desc = engine->sim.description();
    body = {desc.friction, desc.restitution, desc.rolling_resistance, tconv::TeapotSim::frame_dt, desc.density, double(desc.hulls.size())};
    // Each set twice: as the engine's hull builder takes it, its points moved apart by a micron, and before that.
    for (std::size_t h = 0; h < desc.hulls.size(); ++h) {
        body.push_back(double(desc.hulls[h].points.size()));
        for (const zc::phys::Vec3& p : desc.hulls[h].points) { body.insert(body.end(), {p.x, p.y, p.z}); }
        for (const zc::phys::Vec3& p : engine->sim.plain_points()[h]) { body.insert(body.end(), {p.x, p.y, p.z}); }
    }
    return body.data();
}
// Whether the engine has made its own hulls (it does when its simulation is first used).
EMSCRIPTEN_KEEPALIVE int tp_cooked() { return engine->sim.cooked() ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE const double* tp_hull_triangles() {
    const zc::phys::Shape& shape = engine->sim.shape();
    body.assign(1, 0.0);
    for (const zc::phys::HullData& hull : shape.hulls) {
        for (std::size_t f = 0; f + 1 < hull.face_first.size(); ++f) {
            const std::int32_t first = hull.face_first[f], last = hull.face_first[f + 1];
            for (std::int32_t k = first + 1; k + 1 < last; ++k) {
                for (std::int32_t corner : {hull.face_loop[std::size_t(first)], hull.face_loop[std::size_t(k)], hull.face_loop[std::size_t(k + 1)]}) {
                    const zc::phys::Vec3 p = hull.vertices[std::size_t(corner)] + shape.com_offset;
                    body.insert(body.end(), {p.x, p.y, p.z});
                }
                body[0] += 1.0;
            }
        }
    }
    return body.data();
}
// The source of a sheet of the unified teapot whose rows are placed by CONV's
// own error (0: body, 1: lid), for placing them elsewhere: the meridian the
// sampled patches give, as the engine chains it. Numbers, in order: 1 if there
// is such a sheet; rows asked for; nodes around; length; number of runs; points
// a run; number of creases; sizes of the fine table, the coarse table and the
// engine's own plan; then the creases' arc lengths, the runs' samples (x, y),
// the fine table (s, x, y, t, run), the coarse table (s, x, y), and the
// engine's row arc lengths, to compare with.
// Rows placed on the GPU, handed back: arc lengths then crease flags, for the
// body and then the lid, written by the page into tp_rows_buffer().
static double rows_buffer[1024];
EMSCRIPTEN_KEEPALIVE int tp_rows_wanted() { return engine->rows_wanted() ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE int tp_rows_given() { return engine->rows_given() ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE int tp_rows_scale() { return engine->blinn_scale ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE double* tp_rows_buffer() { return rows_buffer; }
EMSCRIPTEN_KEEPALIVE void tp_rows(int body_rows, int lid_rows) {
    tconv::SheetPlan* plans = engine->given_rows[engine->blinn_scale ? 1 : 0];
    const double* from = rows_buffer;
    for (int which = 0; which < 2; ++which) {
        const int n = which == 0 ? body_rows : lid_rows;
        plans[which].row_arc.assign(from, from + n);
        plans[which].row_seam.clear();
        for (int k = 0; k < n; ++k) { plans[which].row_seam.push_back(from[n + k] > 0.5 ? 1 : 0); }
        from += 2 * n;
    }
    engine->rebuild();
}
// What a hook lays out for the page (0 nodes, 1 shapes, 2 lattices, 3 data: floats; 4 arguments, 5 the result, 6 a sheet: doubles).
EMSCRIPTEN_KEEPALIVE const void* tp_hook_buffer(int kind) {
    switch (kind) {
    case 0: return hook_nodes.data();
    case 1: return hook_shapes.data();
    case 2: return hook_meta.data();
    case 3: return hook_data.data();
    case 4: return hook_args.data();
    case 5: return hook_out.data();
    default: return hook_sheet.data();
    }
}
EMSCRIPTEN_KEEPALIVE int tp_hook_length(int kind) {
    switch (kind) {
    case 0: return int(hook_nodes.size());
    case 1: return int(hook_shapes.size());
    case 2: return int(hook_meta.size());
    case 3: return int(hook_data.size());
    case 4: return int(hook_args.size());
    case 5: return int(hook_out.size());
    case 7: return int(hook_cells);
    default: return int(hook_sheet.size());
    }
}
EMSCRIPTEN_KEEPALIVE const double* tp_sheet(int which, int compare) {
    using namespace tconv;
    body.assign(1, 0.0);
    if (engine->version != Version::v2026_unified || which < 0 || which > 1) { return body.data(); }
    const TeapotOptions defaults;
    const bool is_body = which == 0;
    const double y_scale = engine->blinn_scale ? 1.0 : 4.0 / 3.0;
    std::vector<ProfileCurve::Run> runs;
    for (int r = 0; r < (is_body ? utah::dense_body_runs : utah::dense_lid_runs); ++r) {
        ProfileCurve::Run run;
        for (int k = 0; k < utah::dense_profile_points; ++k) {
            const float* p = is_body ? utah::dense_body_profile[r][k] : utah::dense_lid_profile[r][k];
            run.x.push_back(double(p[0]));
            run.y.push_back(double(p[1]) * y_scale);
        }
        runs.push_back(run);
    }
    const ProfileCurve coarse = chain_runs(runs);
    ProfileCurve fine = coarse;
    fine.build_table(32);
    std::vector<double> creases;
    for (std::size_t run = 1; run < fine.runs.size(); ++run) {
        if (fine.turn_at_joint(run) > 4.0) { creases.push_back(fine.run_start[run]); }
    }
    body = {1.0, double(is_body ? defaults.sheet_rows : defaults.sheet_rows * 3 / 4), double(defaults.sheet_around), fine.length, double(fine.runs.size()),
            double(utah::dense_profile_points), double(creases.size()), double(fine.table_s.size()), double(coarse.table_s.size()), double(compare)};
    body.insert(body.end(), creases.begin(), creases.end());
    for (const ProfileCurve::Run& run : fine.runs) {
        for (std::size_t k = 0; k < run.x.size(); ++k) { body.insert(body.end(), {run.x[k], run.y[k]}); }
    }
    for (std::size_t k = 0; k < fine.table_s.size(); ++k) {
        body.insert(body.end(), {fine.table_s[k], fine.table_x[k], fine.table_y[k], fine.table_t[k], double(fine.table_run[k])});
    }
    for (std::size_t k = 0; k < coarse.table_s.size(); ++k) { body.insert(body.end(), {coarse.table_s[k], coarse.table_x[k], coarse.table_y[k]}); }
    if (compare == 0) { return body.data(); }
    // To compare with: the engine's own plan and its error, and its first plan (by curvature).
    const std::size_t rows = std::size_t(body[1]);
    double error = 0.0;
    const SheetPlan own = plan_rows_by_error(coarse, rows, true, 0.2, 8, &error);
    const SheetPlan first = plan_rows(fine, rows, fine.length / (40.0 * double(rows)), 4.0 * fine.length / double(rows), 0.2);
    body.push_back(double(own.row_arc.size()));
    body.insert(body.end(), own.row_arc.begin(), own.row_arc.end());
    body.push_back(error);
    body.push_back(double(first.row_arc.size()));
    body.insert(body.end(), first.row_arc.begin(), first.row_arc.end());
    // The error, measured here, of the rows the page gave (0 if it gave none).
    {
        const TeapotPart& part = *engine->teapot().parts[std::size_t(which)];
        SheetPlan used;
        used.row_arc = part.row_param;
        used.row_seam = part.conv.lattice.seams_v;
        body.push_back(plan_error(coarse, used, true));
    }
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
