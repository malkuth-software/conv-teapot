// The teapot engine: geometry, simulation, renderer and camera behind one
// small interface. The web front-end (web/main.cpp) and the capture tool
// drive the same object.
#pragma once
#include "image.hpp"
#include "sim.hpp"
#include "teapot.hpp"
#include <memory>

namespace tconv {

enum class Geometry : int {
    conv_seamed = 0,   // CONV lattice with seams at the patch boundaries (exact at rest)
    conv_seamless = 1, // CONV lattice without seams, paper closure
    bezier = 2,        // the 28 original patches (and the cap); rest pose only
    control_mesh = 3,  // the CONV nodes joined bilinearly
    conv_round = 4     // exact profile, every ring of the body and lid a true circle
};

class Engine {
public:
    Geometry geometry = Geometry::conv_seamed;
    std::size_t per = 4;
    Bubble bubble = Bubble::continuous;
    bool bottom_cap = true;
    Version version = Version::v1976;
    bool blinn_scale = true;
    bool show_handle = true, show_spout = true, show_lid = true, show_body = true, show_interior = true;
    RenderOptions render;
    TeapotSim sim;
    bool running = false;
    double azimuth = 1.2, elevation = 0.36, distance = 11.0;
    Vec3 target{0.1, 1.3, 0.0};
    Frame frame;
    Renderer renderer;
    double ms_sim = 0.0;
    double time_scale = 0.5;
    // Baked geometry for Method::raster: built on demand, kept until the
    // teapot or the parts shown change, moved with the body when it moves.
    MeshOptions mesh_options{Layout::tree, 1e-2, 4, 20.0};
    Mesh mesh;
    double light_texel = 0.016;

    Engine() {
        textures_ = make_textures();
        rebuild();
    }

    void resize(int width, int height) { frame.resize(width, height); }

    // Rebuild the teapot from the current geometry options (rest pose).
    void rebuild() {
        TeapotOptions options;
        options.per_u = options.per_v = per;
        options.bottom_cap = bottom_cap;
        options.version = version;
        options.blinn_scale = blinn_scale;
        options.bubble = bubble;
        options.seams = geometry != Geometry::conv_seamless;
        options.circular = geometry == Geometry::conv_round;
        options.closure = geometry == Geometry::conv_seamless ? Closure::paper : Closure::cubic;
        teapot_ = make_teapot(options);
        sim.move_nodes = false; // the lattice stays at rest; the pose carries the motion
        sim.build(teapot_);
        pose_now_ = Rigid{};
        running = false;
        mesh = Mesh{};
    }

    // Bake the surfaces now shown to vertices.
    void bake_vertices() {
        const Scene scene = build_scene();
        std::vector<const Surface*> surfaces;
        std::vector<std::vector<const Trim*>> trims;
        for (const RenderPart& part : scene.parts) {
            surfaces.push_back(part.surface);
            trims.push_back(part.trims);
        }
        const double start = detail::now_ms();
        mesh = build_mesh(surfaces, trims, mesh_options);
        mesh.ms_build = detail::now_ms() - start;
        mesh_signature_ = signature();
        mesh_pose_ = pose_;
    }

    // Bake the lighting, as the ray tracer sees it from the camera now, into the mesh's light maps.
    void bake_light() {
        if (mesh.vertices.empty() || mesh_signature_ != signature()) { bake_vertices(); }
        Scene scene = build_scene();
        sync_mesh(scene);
        scene.mesh = &mesh;
        renderer.bake_light(scene, camera(), render, frame.width, frame.height, light_texel);
    }

    Camera camera() const {
        Camera c;
        c.target = target;
        c.eye = target + Vec3{std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth)} * distance;
        c.fov_y = 34.0 * 3.14159265358979 / 180.0;
        return c;
    }

    const Teapot& teapot() const { return teapot_; }

    // Advance the simulation and render one frame.
    const std::uint8_t* draw(double dt) {
        if (running) {
            const double start = detail::now_ms();
            // Shown at half speed: at teapot scale a drop is over in half a second.
            if (sim.step((dt > 1.0 / 30.0 ? 1.0 / 30.0 : dt) * time_scale) > 0) { moved(); }
            ms_sim = detail::now_ms() - start;
            // The engine puts a settled body to sleep; stop drawing then.
            if (sim.asleep() && !sim.grabbing()) { running = false; }
        }
        Scene scene = build_scene();
        if (render.method == Method::raster || render.method == Method::seeded) {
            if (mesh.vertices.empty() || mesh_signature_ != signature()) { bake_vertices(); }
            sync_mesh(scene);
            scene.mesh = &mesh;
        }
        // Sampled versions carry twice the nodes per patch: halve the lattice
        // samples per node interval so the transported lattice stays the same.
        RenderOptions options = render;
        if ((version == Version::v2026 && per >= 8) || version == Version::v2026_unified) { options.subdivision = std::max(1, render.subdivision / 2); }
        renderer.render(scene, camera(), options, frame);
        return frame.rgba.data();
    }

    // Pointer: a press on the teapot grabs it, elsewhere it orbits.
    void pointer_down(double x, double y, bool force_orbit) {
        last_x_ = x;
        last_y_ = y;
        orbiting_ = true;
        const Hit* hit = hit_at(x, y);
        if (!force_orbit && hit != nullptr) {
            const Vec3 point = pose_now_.point(teapot_.parts[scene_parts_[hit->part]]->conv.position(hit->u, hit->v));
            grab_depth_ = hit->t;
            sim.grab(point);
            running = true;
            orbiting_ = false;
        }
    }
    void pointer_move(double x, double y) {
        if (sim.grabbing()) {
            const detail::View view(camera(), frame.width, frame.height);
            sim.drag(view.eye + view.ray(x, y) * grab_depth_);
        } else if (orbiting_) {
            azimuth += (x - last_x_) * 0.008;
            elevation += (y - last_y_) * 0.008;
            elevation = elevation < -1.2 ? -1.2 : elevation > 1.45 ? 1.45 : elevation;
        }
        last_x_ = x;
        last_y_ = y;
    }
    void pointer_up() {
        sim.release();
        orbiting_ = false;
    }
    void zoom(double factor) {
        distance *= factor;
        distance = distance < 3.5 ? 3.5 : distance > 40.0 ? 40.0 : distance;
    }
    void poke(double x, double y) {
        const Hit* hit = hit_at(x, y);
        if (hit == nullptr) { return; }
        const detail::View view(camera(), frame.width, frame.height);
        const Vec3 point = pose_now_.point(teapot_.parts[scene_parts_[hit->part]]->conv.position(hit->u, hit->v));
        sim.poke(point, normalize(view.ray(x, y)));
        running = true;
    }
    void drop() {
        sim.reset(6.0, 1.2);
        moved();
        running = true;
    }
    void reset() {
        sim.reset();
        moved();
        running = false;
    }

private:
    // The scene as the options have it now; fills scene_parts_.
    Scene build_scene() {
        Scene scene;
        scene_parts_.clear();
        posed_.assign(teapot_.parts.size(), PosedSurface{});
        for (std::size_t index = 0; index < teapot_.parts.size(); ++index) {
            const auto& part = teapot_.parts[index];
            const bool shown = part->group == "handle" ? show_handle : part->group == "spout" ? show_spout : part->group == "lid" ? show_lid : show_body;
            if (!shown || (part->interior && !show_interior)) { continue; }
            RenderPart render_part;
            // The surface at rest, carried by the body's pose.
            posed_[index].base = geometry == Geometry::bezier && !part->sampled ? static_cast<const Surface*>(&part->bezier)
                                 : geometry == Geometry::control_mesh ? static_cast<const Surface*>(&part->linear)
                                                                      : static_cast<const Surface*>(&part->conv);
            posed_[index].pose = &pose_now_;
            render_part.surface = &posed_[index];
            render_part.orientation = part->orientation;
            render_part.per_u = part->per_u;
            render_part.per_v = part->per_v;
            render_part.two_sided = part->sampled;
            scene.parts.push_back(render_part);
            scene_parts_.push_back(index);
        }
        for (const TeapotJoin& join : teapot_.joins) {
            long body = -1, tube = -1;
            for (std::size_t k = 0; k < scene_parts_.size(); ++k) {
                if (scene_parts_[k] == join.body_part) { body = long(k); }
                if (scene_parts_[k] == join.tube_part) { tube = long(k); }
            }
            if (body < 0 || tube < 0) { continue; }
            scene.parts[std::size_t(body)].trims.push_back(&join.body_trim);
            scene.parts[std::size_t(tube)].trims.push_back(&join.tube_trim);
        }
        for (const auto& texture : textures_) { scene.textures.push_back(texture.get()); }
        return scene;
    }

    // What the mesh was built from: a change means it must be built again.
    std::uint64_t signature() const {
        std::uint64_t value = std::uint64_t(geometry) | std::uint64_t(show_handle) << 8 | std::uint64_t(show_spout) << 9 | std::uint64_t(show_lid) << 10 |
                              std::uint64_t(show_body) << 11 | std::uint64_t(show_interior) << 12 | std::uint64_t(mesh_options.layout) << 16 |
                              std::uint64_t(mesh_options.uniform) << 20;
        value = value * 1099511628211ull ^ std::uint64_t(mesh_options.tolerance * 1e7);
        value = value * 1099511628211ull ^ std::uint64_t(mesh_options.angle * 1e3);
        return value + 1;
    }

    // Move the mesh's vertices with the body when it has moved since they were placed.
    void sync_mesh(const Scene& scene) {
        if (mesh_pose_ == pose_) { return; }
        std::vector<const Surface*> surfaces;
        for (const RenderPart& part : scene.parts) { surfaces.push_back(part.surface); }
        refresh_mesh(mesh, surfaces);
        mesh_pose_ = pose_;
    }

    // The body moved: take its pose.
    void moved() {
        pose_now_ = sim.pose();
        ++pose_;
    }

    const Hit* hit_at(double x, double y) const {
        const int ix = int(x), iy = int(y);
        if (ix < 0 || iy < 0 || ix >= frame.width || iy >= frame.height) { return nullptr; }
        const Hit& hit = frame.hits[std::size_t(iy) * std::size_t(frame.width) + std::size_t(ix)];
        return hit.t > 0.0f && hit.part < scene_parts_.size() ? &hit : nullptr;
    }

    Teapot teapot_;
    std::vector<std::size_t> scene_parts_; // scene part -> teapot part
    std::vector<std::unique_ptr<Texture>> textures_;
    std::uint64_t mesh_signature_ = 0, mesh_pose_ = 0, pose_ = 0;
    Rigid pose_now_;
    std::vector<PosedSurface> posed_; // one per teapot part
    double last_x_ = 0.0, last_y_ = 0.0, grab_depth_ = 0.0;
    bool orbiting_ = false;
};

} // namespace tconv
