// Headless capture: renders the teapot to a PNG and prints the work counters.
// Arguments are key=value pairs; see `usage` below.
#include "../src/image.hpp"
#include "../src/sim.hpp"
#include "../src/teapot.hpp"
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
using namespace tconv;

static const char* usage =
    "capture out=FILE.png [size=WxH] [geometry=conv|bezier|linear] [method=transport|independent]\n"
    "        [per=N] [seams=0|1] [closure=cubic|paper] [bubble=none|standard|continuous] [cap=0|1]\n"
    "        [wire=0|1|2] [texture=0..3] [filter=nearest|bilinear|conv] [mapping=part|patch]\n"
    "        [lighting=0|1] [shadows=0|1] [ground=0|1] [work=0|1] [k=N] [threads=N] [repeat=N]\n"
    "        [azimuth=DEG] [elevation=DEG] [distance=D] [fov=DEG] [version=1975|1976|1987|2026] [blinn=0|1]\n"
    "        [hide=handle,spout,lid,body,interior] [material=0|1|2] [gloss=0|1] [floor=0|1] [aa=0|1] [round=0|1] [drop=SECONDS]\n";

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int k = 1; k < argc; ++k) {
        const char* eq = std::strchr(argv[k], '=');
        if (eq == nullptr) { std::fputs(usage, stderr); return 2; }
        args[std::string(argv[k], std::size_t(eq - argv[k]))] = eq + 1;
    }
    const auto get = [&](const char* key, const char* fallback) { return args.count(key) != 0 ? args[key] : std::string(fallback); };
    const auto number = [&](const char* key, double fallback) { return args.count(key) != 0 ? std::atof(args[key].c_str()) : fallback; };

    Rigid pose; // the teapot's motion from rest, applied to its surfaces
    TeapotOptions teapot_options;
    teapot_options.per_u = teapot_options.per_v = std::size_t(number("per", 4));
    teapot_options.seams = number("seams", 1) != 0;
    teapot_options.bottom_cap = number("cap", 1) != 0;
    teapot_options.circular = number("round", 0) != 0;
    const std::string version = get("version", "1976");
    teapot_options.version = version == "1975" ? Version::v1975 : version == "1987" ? Version::v1987 : version == "2026" ? Version::v2026 : version == "2026u" ? Version::v2026_unified : Version::v1976;
    teapot_options.sheet_rows = std::size_t(number("rows", 72));
    teapot_options.sheet_around = std::size_t(number("around", 12));
    teapot_options.joins = number("joins", 1) != 0;
    teapot_options.join_two_sided = number("twosided", 1) != 0;
    teapot_options.blinn_scale = number("blinn", version == "1975" ? 0 : 1) != 0;
    const std::string hide = get("hide", "");
    teapot_options.smooth = number("smooth", 0) != 0;
    teapot_options.closure = get("closure", "cubic") == "paper" ? Closure::paper : Closure::cubic;
    const std::string bubble = get("bubble", "continuous");
    teapot_options.bubble = bubble == "none" ? Bubble::none : bubble == "standard" ? Bubble::standard : Bubble::continuous;
    Teapot teapot = make_teapot(teapot_options);
    if (args.count("drop") != 0) {
        // Simulate a drop for the given number of seconds before rendering.
        TeapotSim sim;
        sim.move_nodes = number("movenodes", 0) != 0;
        sim.build(teapot);
        sim.reset(number("lift", 3.0), number("tilt", 0.45));
        const int steps = int(number("drop", 1.0) * 120.0);
        for (int k = 0; k < steps; ++k) { sim.step(1.0 / 120.0); }
        const zc::phys::SolveStats stats = sim.stats();
        std::printf("drop: %d steps, %zu hulls, mass %.3f kg, kinetic %.3e J, asleep %d, contacts %zu, newton iterations %lld\n", steps, sim.hulls(),
                    sim.mass(), sim.kinetic(), sim.asleep() ? 1 : 0, sim.contacts(), static_cast<long long>(stats.newton_iterations));
        if (!sim.move_nodes) { pose = sim.pose(); }
    }

    if (args.count("tip") != 0) {
        // Tip the teapot over by its handle, as a drag in the demo does, and let it settle.
        TeapotSim sim;
        sim.build(teapot);
        Vec3 held = teapot.parts[2]->conv.lattice.nodes[0];
        for (const Vec3& node : teapot.parts[2]->conv.lattice.nodes) {
            if (node.x < held.x) { held = node; }
        }
        sim.grab(held);
        for (int k = 0; k < 300; ++k) {
            const double t = std::min(1.0, k / 180.0);
            sim.drag(held + Vec3{2.5 * t, 4.0 * t, 1.0 * t});
            sim.step(1.0 / 120.0);
        }
        sim.release();
        for (int k = 0; k < 1200; ++k) { sim.step(1.0 / 120.0); }
        std::printf("tip: asleep %d\n", sim.asleep() ? 1 : 0);
        if (!sim.move_nodes) { pose = sim.pose(); }
    }
    if (args.count("roll") != 0) {
        // Lay the teapot over: a turn about the z axis, the lowest node resting on the floor.
        // movenodes=1 turns the lattice nodes themselves and rebuilds CONV through them instead.
        const double a = number("roll", 90.0) * M_PI / 180.0;
        Rigid turn;
        turn.a[0] = std::cos(a); turn.a[1] = -std::sin(a);
        turn.a[3] = std::sin(a); turn.a[4] = std::cos(a);
        turn.b = Vec3{0, 0, 0} - turn.direction({0.0, 1.5, 0.0});
        double low = 1e30;
        for (auto& part : teapot.parts) {
            if (part->conv.on != nullptr) { continue; } // a join's nodes are parameters, not points
            for (const Vec3& node : part->conv.lattice.nodes) { low = std::min(low, turn.point(node).y); }
        }
        turn.b.y -= low;
        if (number("movenodes", 0) != 0) {
            for (auto& part : teapot.parts) {
                if (part->conv.on != nullptr) { continue; }
                for (Vec3& node : part->conv.lattice.nodes) { node = turn.point(node); }
                part->conv.compile();
            }
        } else {
            pose = turn;
        }
    }
    const auto textures = make_textures();
    Scene scene;
    std::vector<PosedSurface> posed(teapot.parts.size());
    const std::string geometry = get("geometry", "conv");
    std::vector<long> scene_index(teapot.parts.size(), -1);
    for (std::size_t part_index = 0; part_index < teapot.parts.size(); ++part_index) {
        const auto& part = teapot.parts[part_index];
        if (hide.find(part->group) != std::string::npos || (part->interior && hide.find("interior") != std::string::npos)) { continue; }
        RenderPart render_part;
        render_part.surface = geometry == "bezier" && !part->sampled ? static_cast<const Surface*>(&part->bezier)
                              : geometry == "linear" ? static_cast<const Surface*>(&part->linear)
                                                     : static_cast<const Surface*>(&part->conv);
        if (!pose.identity() || args.count("shift") != 0 || args.count("turn") != 0) {
            posed[part_index].base = render_part.surface;
            posed[part_index].pose = &pose;
            render_part.surface = &posed[part_index];
        }
        render_part.orientation = part->orientation;
        render_part.per_u = part->per_u;
        render_part.per_v = part->per_v;
        render_part.two_sided = part->sampled;
        render_part.interior_from = part->interior ? 0.0 : part->interior_from;
        render_part.solid = teapot_options.version == Version::v2026_unified; // the unified sheets close a solid: wall, interior and joins
        scene_index[part_index] = long(scene.parts.size());
        scene.parts.push_back(render_part);
    }
    for (const TeapotJoin& join : teapot.joins) {
        if (scene_index[join.body_part] < 0 || scene_index[join.tube_part] < 0) { continue; }
        scene.parts[std::size_t(scene_index[join.body_part])].trims.push_back(&join.body_trim);
        scene.parts[std::size_t(scene_index[join.tube_part])].trims.push_back(&join.tube_trim);
    }
    for (const auto& texture : textures) { scene.textures.push_back(texture.get()); }

    // side=left|right: cut the teapot along its plane of symmetry and show one half.
    const std::string side = get("side", "both");
    if (side != "both") {
        scene.cut = true;
        scene.cut_normal = pose.direction({0.0, 0.0, side == "left" ? 1.0 : -1.0});
        scene.cut_offset = dot(scene.cut_normal, pose.b);
    }
    scene.axis = pose.direction({0.0, 1.0, 0.0});
    scene.pose = pose;
    if (args.count("tip") != 0 || args.count("drop") != 0) {
        // How far the surface itself is from the floor where the body rests.
        for (const auto& part : teapot.parts) {
            double low = 1e30, at_v = 0.0;
            for (int j = 0; j <= 200; ++j) {
                for (int i = 0; i < 96; ++i) {
                    const double v = double(part->conv.cells_v()) * j / 200.0;
                    const double y = pose.point(part->conv.position(double(part->conv.cells_u()) * i / 96.0, v)).y;
                    if (y < low) {
                        low = y;
                        at_v = v;
                    }
                }
            }
            if (low < 0.05) { std::printf("lowest point of %s: y = %.4f at row %.1f of %zu\n", part->name.c_str(), low, at_v, part->conv.cells_v()); }
        }
    }
    scene.stamp = number("cache", 1) != 0 ? 1 : 0; // the scene does not change between repeats
    scene.open_top = hide.find("lid") != std::string::npos;
    RenderOptions options;
    const std::string method = get("method", "transport");
    options.method = method == "independent" ? Method::independent : method == "raster" ? Method::raster : method == "seeded" ? Method::seeded : Method::transport;
    options.supersample = int(number("ss", 2));
    options.mesh_lines = number("lines", 0) != 0;
    options.live_reflection = number("live", 1) != 0;
    options.ambient = int(number("ambient", 3));
    options.plain_inside = number("plaininside", 0) != 0;
    options.exact_shadows = number("exact", 1) != 0;
    options.wire = Wire(int(number("wire", 0)));
    options.texture = int(number("texture", 0));
    const std::string filter = get("filter", "conv");
    options.filter = filter == "nearest" ? TextureFilter::nearest : filter == "bilinear" ? TextureFilter::bilinear : TextureFilter::conv;
    options.mapping = get("mapping", "part") == "patch" ? TextureMap::patch : TextureMap::part;
    options.lighting = number("lighting", 1) != 0;
    options.shadows = number("shadows", 1) != 0;
    options.ground = number("ground", 1) != 0;
    options.show_work = number("work", 0) != 0;
    options.edge_samples = number("aa", 1) != 0;
    options.gloss = number("gloss", 0) != 0;
    options.floor_reflection = number("floor", 0) != 0;
    options.material = int(number("material", 0));
    options.subdivision = int(number("k", 4));
    options.threads = int(number("threads", 0));
    options.one_step_px = number("onestep", 0.35);

    int width = 960, height = 600;
    std::sscanf(get("size", "960x600").c_str(), "%dx%d", &width, &height);
    Camera camera;
    const double azimuth = number("azimuth", 68.0) * M_PI / 180.0, elevation = number("elevation", 22.0) * M_PI / 180.0;
    const double distance = number("distance", 11.0);
    camera.target = {number("lookx", 0.15), number("lookat", 1.35), 0.0};
    camera.eye = camera.target + Vec3{std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth)} * distance;
    camera.fov_y = number("fov", 34.0) * M_PI / 180.0;

    Frame frame;
    frame.resize(width, height);
    Renderer renderer;
    Mesh mesh;
    if (options.method == Method::raster || options.method == Method::seeded) {
        // Bake the surfaces to vertices, and with light=1 the lighting to light maps.
        MeshOptions mesh_options;
        const std::string layout = get("layout", "tree");
        mesh_options.layout = layout == "uniform" ? Layout::uniform : layout == "bounds" ? Layout::bounds : Layout::tree;
        mesh_options.tolerance = number("tolerance", 1e-2);
        mesh_options.uniform = int(number("cuts", 4));
        mesh_options.angle = number("angle", 20.0);
        std::vector<const Surface*> surfaces;
        std::vector<std::vector<const Trim*>> trims;
        for (const RenderPart& part : scene.parts) {
            surfaces.push_back(part.surface);
            trims.push_back(part.trims);
        }
        const double start = detail::now_ms();
        mesh = build_mesh(surfaces, trims, mesh_options);
        mesh.ms_build = detail::now_ms() - start;
        scene.mesh = &mesh;
        std::printf("mesh: layout %s, %zu vertices, %zu triangles, %zu surface evaluations, %.2f ms\n", layout.c_str(), mesh.vertices.size(), mesh.triangle_count(),
                    mesh.evaluations, mesh.ms_build);
        if (number("light", 0) != 0) {
            // bakeaz: bake from another azimuth than the picture is taken from.
            Camera bake_camera = camera;
            if (args.count("bakeaz") != 0) {
                const double az = number("bakeaz", 68.0) * M_PI / 180.0;
                bake_camera.eye = camera.target + Vec3{std::cos(elevation) * std::cos(az), std::sin(elevation), std::cos(elevation) * std::sin(az)} * distance;
            }
            renderer.bake_light(scene, bake_camera, options, width, height, number("texel", 0.016));
            std::printf("light: %zu texels, %.2f ms\n", mesh.texels, mesh.ms_light);
        }
    }
    if (args.count("shift") != 0 || args.count("turn") != 0) {
        // Move the body after the bake: shift=X,Y,Z, and turn=DEG about the vertical through its origin.
        // relight=1 then finds the baked lit fraction again, as the demo does when the body comes to rest.
        double dx = 0, dy = 0, dz = 0;
        if (args.count("shift") != 0) { std::sscanf(args["shift"].c_str(), "%lf,%lf,%lf", &dx, &dy, &dz); }
        const double angle = number("turn", 0.0) * M_PI / 180.0;
        Rigid turn;
        turn.a[0] = std::cos(angle); turn.a[2] = std::sin(angle);
        turn.a[6] = -std::sin(angle); turn.a[8] = std::cos(angle);
        Rigid next;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) { next.a[r * 3 + c] = turn.a[r * 3] * pose.a[c] + turn.a[r * 3 + 1] * pose.a[3 + c] + turn.a[r * 3 + 2] * pose.a[6 + c]; }
        }
        next.b = turn.direction(pose.b) + Vec3{dx, dy, dz};
        pose = next;
        scene.pose = pose;
        scene.axis = pose.direction({0.0, 1.0, 0.0});
        scene.stamp += 1;
        if (scene.mesh != nullptr) {
            std::vector<const Surface*> surfaces;
            for (const RenderPart& part : scene.parts) { surfaces.push_back(part.surface); }
            refresh_mesh(mesh, surfaces);
            if (number("relight", 0) != 0) {
                const double before = mesh.ms_light;
                renderer.bake_light(scene, camera, options, width, height, number("texel", 0.016), true);
                std::printf("relight: %.2f ms (first bake %.2f ms)\n", mesh.ms_light, before);
            }
        }
    }
    const int repeat = int(number("repeat", 1));
    std::vector<double> total, lattice, predict, solve;
    for (int k = 0; k < repeat; ++k) {
        renderer.render(scene, camera, options, frame);
        total.push_back(renderer.stats.ms_total);
        lattice.push_back(renderer.stats.ms_lattice);
        predict.push_back(renderer.stats.ms_predict);
        solve.push_back(renderer.stats.ms_solve);
    }
    const auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const RenderStats& s = renderer.stats;
    std::printf("%s %s %dx%d nodes=%zu lattice=%zu candidates=%zu solves=%zu evaluations=%zu hits=%zu visits=%zu edge_pixels=%zu\n",
                geometry.c_str(), method.c_str(), width, height,
                teapot.node_count(), s.lattice_points, s.candidates, s.solves, s.iterations, s.hit_pixels, s.node_visits, s.edge_pixels);
    std::printf("  median of %d: total %.2f ms (lattice+shadow %.2f, predict/bvh %.2f, solve+shade %.2f); solves/hit %.3f evaluations/hit %.3f\n",
                repeat, median(total), median(lattice), median(predict), median(solve),
                double(s.solves) / double(std::max<std::size_t>(1, s.hit_pixels)), double(s.iterations) / double(std::max<std::size_t>(1, s.hit_pixels)));
    if (number("verify", 0) != 0) {
        // Check the frame against the surface itself and against the
        // independent renderer run with exact re-evaluation.
        const detail::View view(camera, width, height);
        double worst = 0.0, sum = 0.0;
        std::size_t count = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const Hit& hit = frame.hits[std::size_t(y) * std::size_t(width) + std::size_t(x)];
                if (hit.t <= 0.0f) { continue; }
                double sx, sy, sz;
                view.project(scene.parts[hit.part].surface->position(hit.u, hit.v), sx, sy, sz);
                const double residual = std::hypot(sx - (x + 0.5), sy - (y + 0.5));
                worst = std::max(worst, residual);
                sum += residual;
                ++count;
            }
        }
        RenderOptions reference = options;
        reference.method = Method::independent;
        reference.one_step_px = 0.0;
        reference.subdivision = 16;
        Frame other;
        other.resize(width, height);
        Renderer second;
        second.render(scene, camera, reference, other);
        std::size_t only_frame = 0, only_reference = 0, deeper = 0, nearer = 0, both = 0;
        for (std::size_t k = 0; k < frame.hits.size(); ++k) {
            const bool a = frame.hits[k].t > 0.0f, b = other.hits[k].t > 0.0f;
            if (a && !b) { ++only_frame; }
            if (!a && b) { ++only_reference; }
            if (a && b) {
                ++both;
                const double d = double(frame.hits[k].t) - double(other.hits[k].t);
                if (d > 2e-3) { ++deeper; }
                if (d < -2e-3) { ++nearer; }
            }
        }
        if (args.count("diff") != 0) {
            std::vector<std::uint8_t> mask(frame.rgba);
            for (std::size_t k = 0; k < frame.hits.size(); ++k) {
                const bool a = frame.hits[k].t > 0.0f, b = other.hits[k].t > 0.0f;
                const double d = double(frame.hits[k].t) - double(other.hits[k].t);
                std::uint8_t* p = &mask[k * 4];
                for (int ch = 0; ch < 3; ++ch) { p[ch] = std::uint8_t(160 + p[ch] / 3); }
                if (a != b) { p[0] = 20; p[1] = 60; p[2] = 255; }
                else if (a && d > 2e-3) { p[0] = 255; p[1] = 20; p[2] = 20; }
                else if (a && d < -2e-3) { p[0] = 0; p[1] = 170; p[2] = 40; }
            }
            write_png(args["diff"], width, height, mask.data());
        }
        std::printf("  verify: reprojection residual of %zu hits: max %.4f px, mean %.5f px\n", count, worst, sum / double(std::max<std::size_t>(1, count)));
        std::printf("  against a 16x-lattice independent reference: agree %zu, hit only here %zu, hit only in reference %zu, deeper here %zu, nearer here %zu\n",
                    both - deeper - nearer, only_frame, only_reference, deeper, nearer);
    }
    if (args.count("probe") != 0) {
        // The hits down a short column of pixels: probe=X,Y.
        int px = 0, py = 0;
        std::sscanf(args["probe"].c_str(), "%d,%d", &px, &py);
        for (int y = py - 8; y <= py + 8; ++y) {
            const Hit& hit = frame.hits[std::size_t(y) * std::size_t(width) + std::size_t(px)];
            const std::uint8_t* c = &frame.rgba[(std::size_t(y) * std::size_t(width) + std::size_t(px)) * 4];
            std::printf("  pixel %d,%d: part %d u %.4f v %.4f depth %.5f colour %d %d %d\n", px, y, int(hit.part), double(hit.u), double(hit.v), double(hit.t), c[0], c[1], c[2]);
        }
    }
    const std::string out = get("out", "");
    if (!out.empty() && !write_png(out, width, height, frame.rgba.data())) { std::fputs("cannot write output\n", stderr); return 1; }
    return 0;
}
