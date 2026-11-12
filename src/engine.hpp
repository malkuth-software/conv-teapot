// The teapot engine: geometry, simulation, renderer and camera behind one
// small interface. The web front-end (web/main.cpp) and the capture tool
// drive the same object.
#pragma once
#include "image.hpp"
#include "sim.hpp"
#include "teapot.hpp"
#include <memory>
#include <string>

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
    int side = 0;      // 0: the whole teapot; 1, 2: cut along its plane of symmetry, one half shown
    bool joins = true; // unified version: blend the handle and spout into the body
    RenderOptions render;
    TeapotSim sim;
    bool running = false;
    double azimuth = 1.2, elevation = 0.36, distance = 11.0;
    Vec3 target{0.1, 1.3, 0.0};
    Frame frame;
    Renderer renderer;
    double ms_sim = 0.0;
    // While the picture is changing (physics, a drag, an orbit) it is drawn at
    // half the size and enlarged; the full picture is drawn when it rests.
    bool quick_while_moving = true;
    double time_scale = 0.5;
    // Baked geometry for Method::raster: built on demand, kept until the
    // teapot or the parts shown change, moved with the body when it moves.
    MeshOptions mesh_options = [] {
        MeshOptions options;
        options.layout = Layout::tree;
        options.tolerance = 1e-2;
        options.angle = 20.0;
        return options;
    }();
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
        options.joins = joins;
        options.seams = geometry != Geometry::conv_seamless;
        options.circular = geometry == Geometry::conv_round;
        options.closure = geometry == Geometry::conv_seamless ? Closure::paper : Closure::cubic;
        teapot_ = make_teapot(options);
        sim.move_nodes = false; // the lattice stays at rest; the pose carries the motion
        sim.build(teapot_);
        pose_now_ = Rigid{};
        ++build_count_;
        running = false;
        mesh = Mesh{};
    }

    // Bake the surfaces now shown to vertices. The mesh is made from the
    // surfaces at rest and carried by the body's pose, like the surfaces.
    void bake_vertices() {
        const Scene scene = build_scene();
        std::vector<const Surface*> surfaces;
        std::vector<std::vector<const Trim*>> trims;
        for (std::size_t k = 0; k < scene.parts.size(); ++k) {
            surfaces.push_back(posed_[scene_parts_[k]].base);
            trims.push_back(scene.parts[k].trims);
        }
        const double start = detail::now_ms();
        mesh = build_mesh(surfaces, trims, mesh_options);
        mesh.ms_build = detail::now_ms() - start;
        rest_p_.clear();
        rest_n_.clear();
        for (const MeshVertex& vertex : mesh.vertices) {
            rest_p_.push_back(vertex.p);
            rest_n_.push_back(vertex.n);
        }
        mesh_signature_ = signature();
        mesh_pose_ = pose_ + 1; // not yet posed
        sync_mesh(scene);
    }

    // The baked mesh in its rest pose as a Wavefront OBJ file: positions,
    // light map coordinates and normals. z_up turns it for programs whose
    // vertical axis is z.
    std::string export_obj(bool z_up) {
        Scene scene = build_scene();
        std::vector<const Surface*> surfaces;
        std::vector<std::vector<const Trim*>> trims;
        for (std::size_t k = 0; k < scene.parts.size(); ++k) {
            surfaces.push_back(posed_[scene_parts_[k]].base); // at rest, not as posed
            trims.push_back(scene.parts[k].trims);
        }
        const Mesh rest = build_mesh(surfaces, trims, mesh_options);
        std::string out = "# The Utah Teapot in CONV geometry: a mesh baked from the lattice.\n";
        char line[160];
        const auto turned = [&](Vec3 p) { return z_up ? Vec3{p.x, -p.z, p.y} : p; };
        for (const MeshVertex& vertex : rest.vertices) {
            const Vec3 p = turned(vertex.p);
            std::snprintf(line, sizeof line, "v %.6f %.6f %.6f\n", p.x, p.y, p.z);
            out += line;
        }
        for (const MeshVertex& vertex : rest.vertices) {
            std::snprintf(line, sizeof line, "vt %.6f %.6f\n", double(vertex.s), double(vertex.t));
            out += line;
        }
        for (std::size_t k = 0; k < rest.vertices.size(); ++k) {
            // Outward where the part has a known outside.
            std::size_t part = 0;
            while (part + 1 < rest.parts.size() && k >= rest.parts[part].first_vertex + rest.parts[part].vertex_count) { ++part; }
            const Vec3 n = turned(rest.vertices[k].n * scene.parts[part].orientation);
            std::snprintf(line, sizeof line, "vn %.5f %.5f %.5f\n", n.x, n.y, n.z);
            out += line;
        }
        std::size_t part = 0;
        for (std::size_t t = 0; t < rest.triangle_count(); ++t) {
            while (part < rest.parts.size() && t >= rest.parts[part].first_triangle + rest.parts[part].triangle_count) { ++part; }
            if (t == (part < rest.parts.size() ? rest.parts[part].first_triangle : 0)) {
                std::snprintf(line, sizeof line, "g part%zu\n", part);
                out += line;
            }
            const unsigned a = rest.triangles[t * 3] + 1, b = rest.triangles[t * 3 + 1] + 1, c = rest.triangles[t * 3 + 2] + 1;
            std::snprintf(line, sizeof line, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", a, a, a, b, b, b, c, c, c);
            out += line;
        }
        return out;
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
        // One visibility sample per pixel while the picture is changing; the full count when it rests.
        const bool moving = running || sim.grabbing() || orbiting_ || flying_;
        flying_ = false;
        if (!moving && render.method == Method::raster && render.baked_light && mesh.lit && mesh.lit_diffuse_only) {
            // At rest after turning: the baked lit fraction is found again for the new rotation.
            bool turned = false;
            for (int k = 0; k < 9; ++k) { turned = turned || std::fabs(mesh.light_turn[k] - pose_now_.a[k]) > 1e-7; }
            if (turned) { renderer.bake_light(scene, camera(), render, frame.width, frame.height, light_texel, true); }
        }
        if (moving && quick_while_moving) {
            // Quick frames: one visibility sample per pixel and a smaller shadow map as well as half the size.
            // With quick frames off a moving frame is drawn exactly as a resting one.
            options.supersample = 1;
            options.shadow_size = 512;
        }
        hit_scale_ = 1;
        if (moving && quick_while_moving && frame.width >= 320) {
            hit_scale_ = 2;
            if (quick_.width != frame.width / 2 || quick_.height != frame.height / 2) { quick_.resize(frame.width / 2, frame.height / 2); }
            renderer.render(scene, camera(), options, quick_);
            // Enlarge: each quick pixel fills a block of two by two.
            for (int y = 0; y < frame.height; ++y) {
                const std::uint8_t* from = &quick_.rgba[std::size_t(std::min(y / 2, quick_.height - 1)) * std::size_t(quick_.width) * 4];
                std::uint8_t* to = &frame.rgba[std::size_t(y) * std::size_t(frame.width) * 4];
                for (int x = 0; x < frame.width; ++x) {
                    const std::uint8_t* pixel = from + std::size_t(std::min(x / 2, quick_.width - 1)) * 4;
                    to[0] = pixel[0];
                    to[1] = pixel[1];
                    to[2] = pixel[2];
                    to[3] = 255;
                    to += 4;
                }
            }
            return frame.rgba.data();
        }
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
        } else if (flown_) {
            // After a flight the orbit turns about what is at the centre of the picture, the eye staying where it is.
            const Hit* centre = hit_at(frame.width * 0.5, frame.height * 0.5);
            if (centre != nullptr) {
                const Camera now = camera();
                const Vec3 point = pose_now_.point(teapot_.parts[scene_parts_[centre->part]]->conv.position(centre->u, centre->v));
                const double away = length(point - now.eye);
                if (away > 0.05) {
                    target = now.eye + (target - now.eye) * (away / distance);
                    distance = away;
                }
            }
            flown_ = false;
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
        distance = distance < 0.3 ? 0.3 : distance > 40.0 ? 40.0 : distance;
    }
    // Fly: the eye and what it looks at move together, by these distances
    // along the view direction, to its right, and up.
    void fly(double forward, double right, double up) {
        const Camera now = camera();
        const Vec3 ahead = normalize(now.target - now.eye);
        const Vec3 side = normalize(cross(ahead, Vec3{0.0, 1.0, 0.0}));
        Vec3 step = ahead * forward + side * right + Vec3{0.0, 1.0, 0.0} * up;
        // The eye stays above the floor.
        const double floor = 0.05 - now.eye.y;
        if (step.y < floor) { step.y = floor; }
        target = target + step;
        flying_ = true;
        flown_ = true;
    }
    void home() {
        azimuth = 1.2;
        elevation = 0.36;
        distance = 11.0;
        target = Vec3{0.1, 1.3, 0.0};
        flown_ = false;
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
            render_part.interior_from = part->interior ? 0.0 : part->interior_from;
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
        scene.axis = pose_now_.direction({0.0, 1.0, 0.0});
        scene.pose = pose_now_;
        scene.open_top = !show_lid;
        // The surfaces and where they are: the renderer keeps what depends only on these.
        scene.stamp = (build_count_ * 1000003ull + pose_) * 1099511628211ull + signature();
        if (side != 0) {
            // The cutting plane is the teapot's own plane of symmetry and moves with it.
            scene.cut = true;
            scene.cut_normal = pose_now_.direction({0.0, 0.0, side == 1 ? 1.0 : -1.0});
            scene.cut_offset = dot(scene.cut_normal, pose_now_.b);
        }
        return scene;
    }

    // What the mesh was built from: a change means it must be built again.
    std::uint64_t signature() const {
        std::uint64_t value = std::uint64_t(geometry) | std::uint64_t(show_handle) << 8 | std::uint64_t(show_spout) << 9 | std::uint64_t(show_lid) << 10 |
                              std::uint64_t(show_body) << 11 | std::uint64_t(show_interior) << 12 | std::uint64_t(mesh_options.layout) << 16 |
                              std::uint64_t(mesh_options.uniform) << 20 | std::uint64_t(mesh_options.match_edge_normals) << 28 |
                              std::uint64_t(mesh_options.diagonal) << 29;
        value = value * 1099511628211ull ^ std::uint64_t(mesh_options.tolerance * 1e7);
        value = value * 1099511628211ull ^ std::uint64_t(mesh_options.angle * 1e3);
        return value + 1;
    }

    // Carry the mesh's vertices with the body when it has moved since they were placed.
    void sync_mesh(const Scene&) {
        if (mesh_pose_ == pose_) { return; }
        for (std::size_t k = 0; k < mesh.vertices.size(); ++k) {
            mesh.vertices[k].p = pose_now_.point(rest_p_[k]);
            mesh.vertices[k].n = pose_now_.direction(rest_n_[k]);
        }
        // A bake of the body's own colour and lit fraction goes with the body; one that holds the view's light does not.
        mesh.lit = mesh.lit && (mesh.lit_diffuse_only || mesh_pose_ == pose_);
        ++mesh.version;
        mesh_pose_ = pose_;
    }

    // The body moved: take its pose.
    void moved() {
        pose_now_ = sim.pose();
        ++pose_;
    }

    const Hit* hit_at(double x, double y) const {
        // The last frame drawn may have been a quick one at half the size.
        const Frame& drawn = hit_scale_ == 2 ? quick_ : frame;
        const int ix = int(x) / hit_scale_, iy = int(y) / hit_scale_;
        if (ix < 0 || iy < 0 || ix >= drawn.width || iy >= drawn.height) { return nullptr; }
        const Hit& hit = drawn.hits[std::size_t(iy) * std::size_t(drawn.width) + std::size_t(ix)];
        return hit.t > 0.0f && hit.part < scene_parts_.size() ? &hit : nullptr;
    }

    Teapot teapot_;
    std::vector<std::size_t> scene_parts_; // scene part -> teapot part
    std::vector<std::unique_ptr<Texture>> textures_;
    std::uint64_t mesh_signature_ = 0, mesh_pose_ = 0, pose_ = 0, build_count_ = 0;
    Rigid pose_now_;
    Frame quick_;
    std::vector<Vec3> rest_p_, rest_n_; // the mesh's vertices at rest
    int hit_scale_ = 1;
    std::vector<PosedSurface> posed_; // one per teapot part
    double last_x_ = 0.0, last_y_ = 0.0, grab_depth_ = 0.0;
    bool orbiting_ = false, flying_ = false, flown_ = false;
};

} // namespace tconv
