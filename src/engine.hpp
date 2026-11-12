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
        sim.build(teapot_);
        running = false;
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
            sim.step((dt > 1.0 / 30.0 ? 1.0 / 30.0 : dt) * time_scale);
            ms_sim = detail::now_ms() - start;
            // The engine puts a settled body to sleep; stop drawing then.
            if (sim.asleep() && !sim.grabbing()) { running = false; }
        }
        Scene scene;
        scene_parts_.clear();
        for (std::size_t index = 0; index < teapot_.parts.size(); ++index) {
            const auto& part = teapot_.parts[index];
            const bool shown = part->group == "handle" ? show_handle : part->group == "spout" ? show_spout : part->group == "lid" ? show_lid : show_body;
            if (!shown || (part->interior && !show_interior)) { continue; }
            RenderPart render_part;
            const bool deformed = sim.moved();
            render_part.surface = geometry == Geometry::bezier && !deformed && !part->sampled ? static_cast<const Surface*>(&part->bezier)
                                  : geometry == Geometry::control_mesh ? static_cast<const Surface*>(&part->linear)
                                                                       : static_cast<const Surface*>(&part->conv);
            render_part.orientation = part->orientation;
            render_part.per_u = part->per_u;
            render_part.per_v = part->per_v;
            render_part.two_sided = part->sampled;
            scene.parts.push_back(render_part);
            scene_parts_.push_back(index);
        }
        for (const auto& texture : textures_) { scene.textures.push_back(texture.get()); }
        // Sampled versions carry twice the nodes per patch: halve the lattice
        // samples per node interval so the transported lattice stays the same.
        RenderOptions options = render;
        if (version == Version::v2026 && per >= 8) { options.subdivision = std::max(1, render.subdivision / 2); }
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
            const Vec3 point = teapot_.parts[scene_parts_[hit->part]]->conv.position(hit->u, hit->v);
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
        const Vec3 point = teapot_.parts[scene_parts_[hit->part]]->conv.position(hit->u, hit->v);
        sim.poke(point, normalize(view.ray(x, y)));
        running = true;
    }
    void drop() {
        sim.reset(6.0, 1.2);
        running = true;
    }
    void reset() {
        sim.reset();
        running = false;
    }

private:
    const Hit* hit_at(double x, double y) const {
        const int ix = int(x), iy = int(y);
        if (ix < 0 || iy < 0 || ix >= frame.width || iy >= frame.height) { return nullptr; }
        const Hit& hit = frame.hits[std::size_t(iy) * std::size_t(frame.width) + std::size_t(ix)];
        return hit.t > 0.0f && hit.part < scene_parts_.size() ? &hit : nullptr;
    }

    Teapot teapot_;
    std::vector<std::size_t> scene_parts_; // scene part -> teapot part
    std::vector<std::unique_ptr<Texture>> textures_;
    double last_x_ = 0.0, last_y_ = 0.0, grab_depth_ = 0.0;
    bool orbiting_ = false;
};

} // namespace tconv
