// Ray tracing of parametric surfaces by correspondence transport.
//
// A camera ray meets a surface S(u, v) where two equations vanish. The
// classical route searches for that point ray by ray. Transport goes the
// other way: it sends a lattice of known parameter values forward through
// S, projects them, and lets every projected lattice cell tell the pixels
// it covers where their rays must meet the surface. Each pixel then
// verifies and corrects that prediction on the true surface with a Newton
// step. No ray searches anything.
//
// The idea is the anchor transport of Plan Paint's CONV cloth example
// (examples/conv_cloth/FLOOD.md), restated here for closed parts with a
// two-pass depth order. An independent per-ray renderer (a BVH over the same
// lattice cells) is kept as the comparison.
#pragma once
#include "surface.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace tconv {

struct Camera {
    Vec3 eye{6.0, 4.0, 9.0};
    Vec3 target{0.0, 1.4, 0.0};
    Vec3 up{0.0, 1.0, 0.0};
    double fov_y = 0.6; // radians
};

struct RenderPart {
    const Surface* surface = nullptr;
    double orientation = 1.0;
    std::size_t per_u = 1, per_v = 1; // node intervals per source patch (for patch outlines)
    bool two_sided = false;           // no outward side is known: never shade as an inside
};

enum class Method : unsigned char { transport, independent };
enum class Wire : unsigned char { off, overlay, only };
enum class TextureFilter : unsigned char { nearest, bilinear, conv };
enum class TextureMap : unsigned char { part, patch };

struct RenderOptions {
    Method method = Method::transport;
    Wire wire = Wire::off;
    int texture = 0;          // 0: porcelain, 1..: index into Scene::textures
    TextureFilter filter = TextureFilter::conv;
    TextureMap mapping = TextureMap::part;
    bool lighting = true;
    bool shadows = true;
    bool ground = true;
    bool show_work = false;   // colour pixels by the solver work they needed
    bool gloss = false;       // glaze: one traced reflection bounce and an area-emitter window
    bool floor_reflection = false; // polished floor, by transport from the mirrored eye
    int material = 0;         // 0: porcelain, 1: chrome, 2: reflection lines (when untextured)
    bool edge_samples = true; // four more samples for pixels on a silhouette or part boundary
    int subdivision = 4;      // lattice samples per node interval
    int threads = 0;          // 0: all hardware threads
    double tolerance_px = 0.01;
    // A prediction whose first residual is below this many pixels takes one
    // Newton step and is accepted without a second evaluation; the step's
    // own error is second order in that residual. 0 always re-evaluates.
    double one_step_px = 0.35;
};

struct RenderStats {
    std::size_t lattice_points = 0;
    std::size_t candidates = 0;   // predictions gathered (transport)
    std::size_t solves = 0;       // Newton solves started
    std::size_t iterations = 0;   // surface evaluations inside solves
    std::size_t hit_pixels = 0;
    std::size_t node_visits = 0;  // BVH nodes visited (independent)
    std::size_t edge_pixels = 0;  // pixels given extra samples
    double ms_lattice = 0.0, ms_predict = 0.0, ms_solve = 0.0, ms_total = 0.0;
};

// A texture: an RGB lattice and its CONV reconstruction.
struct Texture {
    ConvSurface conv; // nodes are colours in [0,1]
    std::size_t width() const { return conv.lattice.nu; }
    std::size_t height() const { return conv.lattice.nv; }
    Vec3 sample(double x, double y, TextureFilter filter) const {
        // x, y in [0,1); x wraps, y clamps. Texel centres sit on nodes.
        const double u = (x - std::floor(x)) * double(width());
        const double v = (y < 0.0 ? 0.0 : y > 1.0 ? 1.0 : y) * double(height() - 1);
        Vec3 c;
        if (filter == TextureFilter::conv) {
            c = conv.position(u, v);
        } else if (filter == TextureFilter::bilinear) {
            LinearSurface linear;
            linear.lattice = &conv.lattice;
            c = linear.eval(u, v).p;
        } else {
            const std::size_t iu = std::size_t(std::floor(u + 0.5)) % width();
            const std::size_t iv = std::min(height() - 1, std::size_t(std::floor(v + 0.5)));
            c = conv.lattice.at(iu, iv);
        }
        const auto clamp01 = [](double t) { return t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t; };
        return {clamp01(c.x), clamp01(c.y), clamp01(c.z)};
    }
};

struct Scene {
    std::vector<RenderPart> parts;
    std::vector<const Texture*> textures;
    double ground_y = 0.0;
    Vec3 light = normalize(Vec3{-0.45, 0.85, 0.35}); // direction towards the light
};

struct Hit {
    float u = 0, v = 0;   // node units
    float t = 0;          // depth along the view axis; 0 means no hit
    std::uint16_t part = 0;
    std::uint16_t solves = 0;
};

struct Frame {
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgba;
    std::vector<Hit> hits;
    void resize(int w, int h) {
        width = w;
        height = h;
        rgba.assign(std::size_t(w) * std::size_t(h) * 4, 255);
        hits.assign(std::size_t(w) * std::size_t(h), Hit{});
    }
};

namespace detail {

inline double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct View {
    Vec3 eye, forward, right, up;
    double focal = 1.0;
    int width = 0, height = 0;
    View(const Camera& camera, int w, int h) : eye(camera.eye), width(w), height(h) {
        forward = normalize(camera.target - camera.eye);
        right = normalize(cross(forward, camera.up));
        up = cross(right, forward);
        focal = 0.5 * double(h) / std::tan(0.5 * camera.fov_y);
    }
    // Screen position and view depth of a point.
    void project(Vec3 p, double& x, double& y, double& z) const {
        const Vec3 d = p - eye;
        z = dot(d, forward);
        const double scale = focal / (z > 1e-9 ? z : 1e-9);
        x = 0.5 * double(width) + dot(d, right) * scale;
        y = 0.5 * double(height) - dot(d, up) * scale;
    }
    // Direction of the ray through a pixel centre, with unit forward part,
    // so the ray parameter is the view depth.
    Vec3 ray(double px, double py) const {
        return forward + right * ((px - 0.5 * double(width)) / focal) + up * ((0.5 * double(height) - py) / focal);
    }
};

// Forward-transported lattice of one part.
struct PartLattice {
    int nu = 0, nv = 0;       // sample counts (nu includes the closing column)
    double step = 0.25;       // node units between samples
    std::vector<Vec3> p;      // world positions
    std::vector<float> sx, sy, sz; // screen x, y and view depth
    std::vector<float> mx, my, mz; // the same for the eye mirrored in the floor
};

template<class Body> void parallel_rows(int rows, int threads, Body body) {
    int count = threads > 0 ? threads : int(std::thread::hardware_concurrency());
    count = std::max(1, std::min(count, 64));
#ifdef TCONV_SINGLE_THREAD
    count = 1;
#endif
    if (count == 1) {
        body(0, rows, 0);
        return;
    }
    std::atomic<int> next{0};
    const int chunk = 4;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < count; ++worker) {
        workers.emplace_back([&, worker] {
            for (;;) {
                const int begin = next.fetch_add(chunk);
                if (begin >= rows) { return; }
                body(begin, std::min(rows, begin + chunk), worker);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

// One Newton solve of ray and surface. Returns true with the corrected
// (u, v), the depth and the surface point when the residual is below the
// tolerance.
struct Solve {
    double u = 0, v = 0, t = 0;
    SurfacePoint point;
    double grad_u = 0, grad_v = 0; // |d(u)/d(pixel)|, |d(v)/d(pixel)|
    int evaluations = 0;
};

inline bool solve_ray(const Surface& surface, const View& view, Vec3 dir, double tolerance_px, double one_step_px,
                      double reach, Solve& s) {
    const Vec3 unit = normalize(dir);
    const Vec3 helper = std::fabs(unit.y) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    const Vec3 e1 = normalize(cross(unit, helper));
    const Vec3 e2 = cross(unit, e1);
    const double u0 = s.u, v0 = s.v;
    const double cu = double(surface.cells_u()), cv = double(surface.cells_v());
    double previous = 1e300;
    for (int iteration = 0; iteration < 7; ++iteration) {
        s.point = surface.eval(s.u, s.v);
        ++s.evaluations;
        const Vec3 r = s.point.p - view.eye;
        const double f1 = dot(e1, r), f2 = dot(e2, r);
        const double depth = dot(r, view.forward);
        if (!(depth > 1e-6)) { return false; }
        const double a = dot(e1, s.point.du), b = dot(e1, s.point.dv);
        const double c = dot(e2, s.point.du), d = dot(e2, s.point.dv);
        const double pixel = depth / view.focal; // world size of a pixel at this depth
        const double residual = std::hypot(f1, f2);
        // A ray that misses the surface stalls instead of converging: stop
        // as soon as an iteration fails to halve the residual.
        if (iteration >= 2 && residual > 0.5 * previous) { return false; }
        previous = residual;
        const double jacobian = a * d - b * c;
        const auto finish = [&] {
            s.t = depth;
            if (std::fabs(jacobian) > 1e-300) {
                s.grad_u = std::hypot(d, b) / std::fabs(jacobian) * pixel;
                s.grad_v = std::hypot(c, a) / std::fabs(jacobian) * pixel;
            } else {
                s.grad_u = s.grad_v = 1e30;
            }
        };
        if (residual <= tolerance_px * pixel) {
            finish();
            return true;
        }
        if (iteration == 0 && residual <= one_step_px * pixel && std::fabs(jacobian) > 1e-12 * (std::fabs(a * d) + std::fabs(b * c))) {
            // Close prediction: one exact Newton step, no second evaluation.
            const double du = -(d * f1 - b * f2) / jacobian;
            const double dv = -(a * f2 - c * f1) / jacobian;
            // Not at a grazing angle: there the first-order step is long for
            // its residual and proves nothing about a hit.
            const double travel = length(s.point.du * du + s.point.dv * dv);
            // And inside one cell: derivatives may break at a node line.
            const bool same_cell = std::floor(s.u + du) == std::floor(s.u) && std::floor(s.v + dv) == std::floor(s.v);
            if (same_cell && travel <= 3.0 * residual && s.v + dv >= 0.0 && s.v + dv <= cv &&
                (surface.periodic_u() || (s.u + du >= 0.0 && s.u + du <= cu))) {
                s.u += du;
                s.v += dv;
                // First-order update of the point, so depth and position follow.
                s.point.p = s.point.p + s.point.du * du + s.point.dv * dv;
                const double moved = dot(s.point.p - view.eye, view.forward);
                finish();
                s.t = moved;
                return true;
            }
        }
        // Damped least squares, so a pole (a zero column) stays solvable.
        const double m00 = a * a + c * c, m01 = a * b + c * d, m11 = b * b + d * d;
        const double lambda = 1e-10 * (m00 + m11) + 1e-300;
        const double g0 = a * f1 + c * f2, g1 = b * f1 + d * f2;
        const double det = (m00 + lambda) * (m11 + lambda) - m01 * m01;
        if (!(std::fabs(det) > 0.0)) { return false; }
        double du = -((m11 + lambda) * g0 - m01 * g1) / det;
        double dv = -((m00 + lambda) * g1 - m01 * g0) / det;
        const double size = std::fmax(std::fabs(du), std::fabs(dv));
        if (size > 1.0) {
            du /= size;
            dv /= size;
        }
        s.u += du;
        s.v += dv;
        if (s.v < 0.0) { s.v = 0.0; }
        if (s.v > cv) { s.v = cv; }
        if (!surface.periodic_u()) {
            if (s.u < 0.0) { s.u = 0.0; }
            if (s.u > cu) { s.u = cu; }
        }
        if (std::fabs(s.u - u0) > reach || std::fabs(s.v - v0) > reach) { return false; }
    }
    return false;
}

// Shadow map: depth along the light of the transported lattice.
struct ShadowMap {
    int size = 0;
    Vec3 x, y, z, centre;
    double radius = 1.0;
    std::vector<float> depth;
    void project(Vec3 p, double& sx, double& sy, double& sz) const {
        const Vec3 d = p - centre;
        sx = (dot(d, x) / radius * 0.5 + 0.5) * double(size);
        sy = (dot(d, y) / radius * 0.5 + 0.5) * double(size);
        sz = -dot(d, z); // grows away from the light
    }
    double lit(Vec3 p, double bias) const {
        if (size == 0) { return 1.0; }
        double sx, sy, sz;
        project(p, sx, sy, sz);
        double sum = 0.0;
        for (int k = 0; k < 4; ++k) {
            const int ix = int(std::floor(sx + (k & 1 ? 0.35 : -0.35)));
            const int iy = int(std::floor(sy + (k & 2 ? 0.35 : -0.35)));
            if (ix < 0 || iy < 0 || ix >= size || iy >= size) {
                sum += 1.0;
                continue;
            }
            sum += sz - bias <= double(depth[std::size_t(iy) * std::size_t(size) + std::size_t(ix)]) ? 1.0 : 0.0;
        }
        return 0.25 * sum;
    }
};

inline void raster_depth(ShadowMap& map, const double* a, const double* b, const double* c) {
    const double area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (std::fabs(area) < 1e-12) { return; }
    const int x0 = std::max(0, int(std::floor(std::min({a[0], b[0], c[0]}))));
    const int x1 = std::min(map.size - 1, int(std::ceil(std::max({a[0], b[0], c[0]}))));
    const int y0 = std::max(0, int(std::floor(std::min({a[1], b[1], c[1]}))));
    const int y1 = std::min(map.size - 1, int(std::ceil(std::max({a[1], b[1], c[1]}))));
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const double px = x + 0.5, py = y + 0.5;
            const double w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area;
            const double w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area;
            const double w2 = 1.0 - w0 - w1;
            if (w0 < -0.02 || w1 < -0.02 || w2 < -0.02) { continue; }
            const float z = float(w0 * a[2] + w1 * b[2] + w2 * c[2]);
            float& slot = map.depth[std::size_t(y) * std::size_t(map.size) + std::size_t(x)];
            if (z < slot) { slot = z; }
        }
    }
}

// A studio environment: soft sky, a bright window band, a dark floor.
inline Vec3 environment(Vec3 d) {
    const double t = 0.5 * (d.y + 1.0);
    Vec3 sky = Vec3{0.20, 0.24, 0.30} * (1.0 - t) + Vec3{0.78, 0.84, 0.92} * t;
    if (d.y < 0.0) { sky = Vec3{0.16, 0.15, 0.14} * (1.0 + d.y) + Vec3{0.08, 0.08, 0.08} * (-d.y); }
    const double az = std::atan2(d.z, d.x);
    const double window = std::exp(-18.0 * (d.y - 0.55) * (d.y - 0.55)) * (0.5 + 0.5 * std::cos(3.0 * az + 0.6));
    return sky + Vec3{1.0, 0.98, 0.92} * (0.9 * window * window);
}

inline double smoothstep(double a, double b, double x) {
    const double t = x <= a ? 0.0 : x >= b ? 1.0 : (x - a) / (b - a);
    return t * t * (3.0 - 2.0 * t);
}

inline std::uint8_t to_byte(double linear) {
    const double c = linear <= 0.0 ? 0.0 : linear >= 1.0 ? 1.0 : std::sqrt(linear); // gamma 2
    return std::uint8_t(c * 255.0 + 0.5);
}

} // namespace detail

class Renderer {
public:
    RenderStats stats;

    void render(const Scene& scene, const Camera& camera, const RenderOptions& options, Frame& frame) {
        using namespace detail;
        stats = RenderStats{};
        const double start = now_ms();
        const View view(camera, frame.width, frame.height);
        // The eye mirrored in the floor: its view of the scene is the floor's reflection.
        View mirror = view;
        mirror.eye.y = 2.0 * scene.ground_y - view.eye.y;
        mirror.forward.y = -view.forward.y;
        mirror.right.y = -view.right.y;
        mirror.up.y = -view.up.y;
        const bool reflect_floor = options.floor_reflection && options.ground && options.lighting && view.eye.y > scene.ground_y;
        const bool mirror_ready = reflect_floor && options.method == Method::transport;
        transport_lattices(scene, view, mirror_ready ? &mirror : nullptr, options);
        build_shadow_map(scene, options);
        stats.ms_lattice = now_ms() - start;
        std::fill(frame.hits.begin(), frame.hits.end(), Hit{});
        const double mid = now_ms();
        nodes_.clear();
        if (options.method == Method::transport) {
            predict(scene, view, options, false);
            if (mirror_ready) { predict(scene, view, options, true); }
        }
        // Per-ray tracing serves the independent method and every incoherent ray.
        if (options.method == Method::independent || (options.gloss && options.lighting)) { build_bvh(); }
        stats.ms_predict = now_ms() - mid;
        const double solve_start = now_ms();
        resolve_and_shade(scene, view, mirror, mirror_ready, options, frame);
        stats.ms_solve = now_ms() - solve_start;
        stats.ms_total = now_ms() - start;
    }

private:
    // ---- transport: forward lattice ------------------------------------
    void transport_lattices(const Scene& scene, const detail::View& view, const detail::View* mirror, const RenderOptions& options) {
        lattices_.resize(scene.parts.size());
        const int k = std::max(1, options.subdivision);
        for (std::size_t part = 0; part < scene.parts.size(); ++part) {
            const Surface& surface = *scene.parts[part].surface;
            detail::PartLattice& lattice = lattices_[part];
            lattice.nu = int(surface.cells_u()) * k + 1;
            lattice.nv = int(surface.cells_v()) * k + 1;
            lattice.step = 1.0 / double(k);
            const std::size_t count = std::size_t(lattice.nu) * std::size_t(lattice.nv);
            lattice.p.resize(count);
            lattice.sx.resize(count);
            lattice.sy.resize(count);
            lattice.sz.resize(count);
            if (mirror != nullptr) {
                lattice.mx.resize(count);
                lattice.my.resize(count);
                lattice.mz.resize(count);
            }
            detail::parallel_rows(lattice.nv, options.threads, [&](int begin, int end, int) {
                for (int j = begin; j < end; ++j) {
                    for (int i = 0; i < lattice.nu; ++i) {
                        const std::size_t index = std::size_t(j) * std::size_t(lattice.nu) + std::size_t(i);
                        const Vec3 p = surface.position(double(i) * lattice.step, double(j) * lattice.step);
                        double x, y, z;
                        view.project(p, x, y, z);
                        lattice.p[index] = p;
                        lattice.sx[index] = float(x);
                        lattice.sy[index] = float(y);
                        lattice.sz[index] = float(z);
                        if (mirror != nullptr) {
                            mirror->project(p, x, y, z);
                            lattice.mx[index] = float(x);
                            lattice.my[index] = float(y);
                            lattice.mz[index] = float(z);
                        }
                    }
                }
            });
            stats.lattice_points += count;
        }
    }

    void build_shadow_map(const Scene& scene, const RenderOptions& options) {
        shadow_.size = 0;
        if (!options.shadows || !options.lighting) { return; }
        Vec3 low{1e30, 1e30, 1e30}, high{-1e30, -1e30, -1e30};
        for (const detail::PartLattice& lattice : lattices_) {
            for (const Vec3& p : lattice.p) {
                low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
                high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
            }
        }
        shadow_.centre = (low + high) * 0.5;
        shadow_.radius = 0.5 * length(high - low) * 1.02 + 1e-6;
        shadow_.z = scene.light;
        const Vec3 helper = std::fabs(scene.light.y) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        shadow_.x = normalize(cross(helper, shadow_.z));
        shadow_.y = cross(shadow_.z, shadow_.x);
        shadow_.size = 512;
        shadow_.depth.assign(std::size_t(shadow_.size) * std::size_t(shadow_.size), 1e30f);
        for (const detail::PartLattice& lattice : lattices_) {
            for (int j = 0; j + 1 < lattice.nv; ++j) {
                for (int i = 0; i + 1 < lattice.nu; ++i) {
                    double corner[4][3];
                    for (int c = 0; c < 4; ++c) {
                        const std::size_t index = std::size_t(j + (c >> 1)) * std::size_t(lattice.nu) + std::size_t(i + (c & 1));
                        shadow_.project(lattice.p[index], corner[c][0], corner[c][1], corner[c][2]);
                    }
                    detail::raster_depth(shadow_, corner[0], corner[1], corner[3]);
                    detail::raster_depth(shadow_, corner[0], corner[3], corner[2]);
                }
            }
        }
    }

    // ---- transport: predictions ----------------------------------------
    static constexpr int max_candidates = 6;
    struct Candidate {
        float u, v, z_low;
        std::uint16_t part;
    };

    static void add_candidate(Candidate* all, std::uint8_t* counts, std::size_t pixel, const Candidate& candidate, float step,
                              std::size_t& added) {
        Candidate* list = &all[pixel * max_candidates];
        std::uint8_t& count = counts[pixel];
        for (int k = 0; k < count; ++k) {
            if (list[k].part == candidate.part && std::fabs(list[k].u - candidate.u) < 0.75f * step &&
                std::fabs(list[k].v - candidate.v) < 0.75f * step) {
                if (candidate.z_low < list[k].z_low) { list[k].z_low = candidate.z_low; }
                return;
            }
        }
        int slot = count;
        if (count == max_candidates) {
            if (candidate.z_low >= list[count - 1].z_low) { return; }
            slot = count - 1;
        } else {
            ++count;
        }
        while (slot > 0 && list[slot - 1].z_low > candidate.z_low) {
            list[slot] = list[slot - 1];
            --slot;
        }
        list[slot] = candidate;
        ++added;
    }

    static void predict_triangle(Candidate* all, std::uint8_t* counts, int width, int row_begin, int row_end, std::uint16_t part,
                                 float step, const float* a, const float* b, const float* c, std::size_t& added) {
        // Each corner: screen x, y, depth, u, v.
        const double area = double(b[0] - a[0]) * double(c[1] - a[1]) - double(b[1] - a[1]) * double(c[0] - a[0]);
        if (std::fabs(area) < 1e-9) { return; }
        const double dilate = 0.6; // pixels: the prediction may sit just outside its cell
        const int x0 = std::max(0, int(std::floor(std::min({a[0], b[0], c[0]}) - dilate)));
        const int x1 = std::min(width - 1, int(std::ceil(std::max({a[0], b[0], c[0]}) + dilate)));
        const int y0 = std::max(row_begin, int(std::floor(std::min({a[1], b[1], c[1]}) - dilate)));
        const int y1 = std::min(row_end - 1, int(std::ceil(std::max({a[1], b[1], c[1]}) + dilate)));
        if (x0 > x1 || y0 > y1) { return; }
        const double sign = area > 0.0 ? 1.0 : -1.0;
        const double len_a = std::hypot(double(c[0] - b[0]), double(c[1] - b[1])); // edge opposite a
        const double len_b = std::hypot(double(a[0] - c[0]), double(a[1] - c[1]));
        const double len_c = std::hypot(double(b[0] - a[0]), double(b[1] - a[1]));
        const float z_min = std::min({a[2], b[2], c[2]});
        const float z_max = std::max({a[2], b[2], c[2]});
        const float z_low = z_min - 0.25f * (z_max - z_min) - 1e-3f * z_min;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;
                const double e_a = (double(b[0]) - px) * (double(c[1]) - py) - (double(b[1]) - py) * (double(c[0]) - px);
                const double e_b = (double(c[0]) - px) * (double(a[1]) - py) - (double(c[1]) - py) * (double(a[0]) - px);
                const double e_c = area - e_a - e_b;
                if (sign * e_a < -dilate * len_a || sign * e_b < -dilate * len_b || sign * e_c < -dilate * len_c) { continue; }
                const double w_a = e_a / area, w_b = e_b / area, w_c = e_c / area;
                Candidate candidate;
                candidate.u = float(w_a * a[3] + w_b * b[3] + w_c * c[3]);
                candidate.v = float(w_a * a[4] + w_b * b[4] + w_c * c[4]);
                candidate.z_low = z_low;
                candidate.part = part;
                add_candidate(all, counts, std::size_t(y) * std::size_t(width) + std::size_t(x), candidate, step, added);
            }
        }
    }

    void predict(const Scene& scene, const detail::View& view, const RenderOptions& options, bool mirrored) {
        const std::size_t pixels = std::size_t(view.width) * std::size_t(view.height);
        std::vector<Candidate>& all = mirrored ? mirror_candidates_ : candidates_;
        std::vector<std::uint8_t>& counts = mirrored ? mirror_counts_ : counts_;
        all.resize(pixels * max_candidates);
        counts.assign(pixels, 0);
        // Each worker owns a band of rows and writes only there, so the
        // candidate lists need no locks and their order is deterministic.
        std::vector<std::size_t> added(64, 0);
        int workers = options.threads > 0 ? options.threads : int(std::thread::hardware_concurrency());
        workers = std::max(1, std::min(workers, 64));
#ifdef TCONV_SINGLE_THREAD
        workers = 1;
#endif
        const int bands = workers == 1 ? 1 : workers * 4;
        detail::parallel_rows(bands, workers, [&](int band_begin, int band_end, int worker) {
            for (int band = band_begin; band < band_end; ++band) {
                const int row_begin = int(std::int64_t(view.height) * band / bands);
                const int row_end = int(std::int64_t(view.height) * (band + 1) / bands);
                predict_band(scene, view, mirrored, all.data(), counts.data(), row_begin, row_end, added[std::size_t(worker)]);
            }
        });
        for (std::size_t count : added) { stats.candidates += count; }
    }

    void predict_band(const Scene& scene, const detail::View& view, bool mirrored, Candidate* all, std::uint8_t* counts, int row_begin,
                      int row_end, std::size_t& added) const {
        for (std::size_t part = 0; part < scene.parts.size(); ++part) {
            const detail::PartLattice& lattice = lattices_[part];
            const float step = float(lattice.step);
            for (int j = 0; j + 1 < lattice.nv; ++j) {
                const std::vector<float>& screen_x = mirrored ? lattice.mx : lattice.sx;
                const std::vector<float>& screen_y = mirrored ? lattice.my : lattice.sy;
                const std::vector<float>& screen_z = mirrored ? lattice.mz : lattice.sz;
                const float* sy0 = &screen_y[std::size_t(j) * std::size_t(lattice.nu)];
                const float* sy1 = sy0 + lattice.nu;
                for (int i = 0; i + 1 < lattice.nu; ++i) {
                    const float low = std::min(std::min(sy0[i], sy0[i + 1]), std::min(sy1[i], sy1[i + 1]));
                    const float high = std::max(std::max(sy0[i], sy0[i + 1]), std::max(sy1[i], sy1[i + 1]));
                    if (high < float(row_begin) - 1.0f || low > float(row_end) + 1.0f) { continue; }
                    float corner[4][5];
                    bool visible = true;
                    for (int c = 0; c < 4; ++c) {
                        const int ci = i + (c & 1), cj = j + (c >> 1);
                        const std::size_t index = std::size_t(cj) * std::size_t(lattice.nu) + std::size_t(ci);
                        corner[c][0] = screen_x[index];
                        corner[c][1] = screen_y[index];
                        corner[c][2] = screen_z[index];
                        corner[c][3] = float(ci) * step;
                        corner[c][4] = float(cj) * step;
                        visible = visible && screen_z[index] > 0.05f;
                    }
                    if (!visible) { continue; }
                    predict_triangle(all, counts, view.width, row_begin, row_end, std::uint16_t(part), step, corner[0], corner[1], corner[3], added);
                    predict_triangle(all, counts, view.width, row_begin, row_end, std::uint16_t(part), step, corner[0], corner[3], corner[2], added);
                }
            }
        }
    }

    // ---- independent: a BVH over the same lattice cells -----------------
    struct Box {
        float low[3], high[3];
    };
    struct Node {
        Box box;
        std::uint32_t left = 0, right = 0; // children, or first/count of a leaf
        bool leaf = false;
    };
    struct Quad {
        Box box;
        std::uint16_t part;
        std::uint16_t i, j;
    };

    void build_bvh() {
        quads_.clear();
        for (std::size_t part = 0; part < lattices_.size(); ++part) {
            const detail::PartLattice& lattice = lattices_[part];
            for (int j = 0; j + 1 < lattice.nv; ++j) {
                for (int i = 0; i + 1 < lattice.nu; ++i) {
                    Quad quad;
                    quad.part = std::uint16_t(part);
                    quad.i = std::uint16_t(i);
                    quad.j = std::uint16_t(j);
                    for (int axis = 0; axis < 3; ++axis) {
                        quad.box.low[axis] = 1e30f;
                        quad.box.high[axis] = -1e30f;
                    }
                    for (int c = 0; c < 4; ++c) {
                        const Vec3& p = lattice.p[std::size_t(j + (c >> 1)) * std::size_t(lattice.nu) + std::size_t(i + (c & 1))];
                        const float xyz[3] = {float(p.x), float(p.y), float(p.z)};
                        for (int axis = 0; axis < 3; ++axis) {
                            quad.box.low[axis] = std::min(quad.box.low[axis], xyz[axis]);
                            quad.box.high[axis] = std::max(quad.box.high[axis], xyz[axis]);
                        }
                    }
                    // The surface bulges past the chord of its cell: pad the box.
                    float diagonal = 0.0f;
                    for (int axis = 0; axis < 3; ++axis) { diagonal += (quad.box.high[axis] - quad.box.low[axis]) * (quad.box.high[axis] - quad.box.low[axis]); }
                    const float pad = 0.15f * std::sqrt(diagonal) + 1e-4f;
                    for (int axis = 0; axis < 3; ++axis) {
                        quad.box.low[axis] -= pad;
                        quad.box.high[axis] += pad;
                    }
                    quads_.push_back(quad);
                }
            }
        }
        nodes_.clear();
        nodes_.reserve(quads_.size() * 2);
        if (!quads_.empty()) { build_node(0, std::uint32_t(quads_.size())); }
    }

    std::uint32_t build_node(std::uint32_t first, std::uint32_t count) {
        const std::uint32_t index = std::uint32_t(nodes_.size());
        nodes_.push_back(Node{});
        Box box;
        for (int axis = 0; axis < 3; ++axis) {
            box.low[axis] = 1e30f;
            box.high[axis] = -1e30f;
        }
        for (std::uint32_t k = first; k < first + count; ++k) {
            for (int axis = 0; axis < 3; ++axis) {
                box.low[axis] = std::min(box.low[axis], quads_[k].box.low[axis]);
                box.high[axis] = std::max(box.high[axis], quads_[k].box.high[axis]);
            }
        }
        nodes_[index].box = box;
        if (count <= 2) {
            nodes_[index].leaf = true;
            nodes_[index].left = first;
            nodes_[index].right = count;
            return index;
        }
        int axis = 0;
        for (int k = 1; k < 3; ++k) {
            if (box.high[k] - box.low[k] > box.high[axis] - box.low[axis]) { axis = k; }
        }
        const std::uint32_t half = count / 2;
        std::nth_element(quads_.begin() + first, quads_.begin() + first + half, quads_.begin() + first + count,
                         [axis](const Quad& a, const Quad& b) {
                             return a.box.low[axis] + a.box.high[axis] < b.box.low[axis] + b.box.high[axis];
                         });
        const std::uint32_t left = build_node(first, half);
        const std::uint32_t right = build_node(first + half, count - half);
        nodes_[index].left = left;
        nodes_[index].right = right;
        return index;
    }

    static bool hit_box(const Box& box, const Vec3& origin, const Vec3& inverse, double t_max, double& t_near) {
        const double o[3] = {origin.x, origin.y, origin.z};
        const double inv[3] = {inverse.x, inverse.y, inverse.z};
        double t0 = 0.0, t1 = t_max;
        for (int axis = 0; axis < 3; ++axis) {
            double a = (double(box.low[axis]) - o[axis]) * inv[axis];
            double b = (double(box.high[axis]) - o[axis]) * inv[axis];
            if (a > b) { std::swap(a, b); }
            if (a > t0) { t0 = a; }
            if (b < t1) { t1 = b; }
            if (t0 > t1) { return false; }
        }
        t_near = t0;
        return true;
    }

    // Möller-Trumbore with a small margin; returns barycentric weights.
    static bool hit_triangle(const Vec3& o, const Vec3& d, const Vec3& a, const Vec3& b, const Vec3& c, double& t, double& wb, double& wc) {
        const Vec3 ab = b - a, ac = c - a;
        const Vec3 p = cross(d, ac);
        const double det = dot(ab, p);
        if (std::fabs(det) < 1e-14) { return false; }
        const Vec3 s = o - a;
        wb = dot(s, p) / det;
        const Vec3 q = cross(s, ab);
        wc = dot(d, q) / det;
        const double margin = 0.12;
        if (wb < -margin || wc < -margin || wb + wc > 1.0 + margin) { return false; }
        t = dot(ac, q) / det;
        return t > 1e-6;
    }

    bool trace_independent(const Scene& scene, const detail::View& view, Vec3 dir, double tolerance, double one_step, detail::Solve& best,
                           std::uint16_t& best_part, std::size_t& solves, std::size_t& evaluations, std::size_t& visits) const {
        if (nodes_.empty()) { return false; }
        const Vec3 inverse{1.0 / (std::fabs(dir.x) > 1e-300 ? dir.x : 1e-300), 1.0 / (std::fabs(dir.y) > 1e-300 ? dir.y : 1e-300),
                           1.0 / (std::fabs(dir.z) > 1e-300 ? dir.z : 1e-300)};
        bool found = false;
        double best_t = 1e30;
        std::uint32_t stack[64];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node& node = nodes_[stack[--top]];
            ++visits;
            double t_near;
            if (!hit_box(node.box, view.eye, inverse, best_t, t_near)) { continue; }
            if (!node.leaf) {
                // Near child last on the stack, so it is taken first.
                double tl = 0, tr = 0;
                const bool hl = hit_box(nodes_[node.left].box, view.eye, inverse, best_t, tl);
                const bool hr = hit_box(nodes_[node.right].box, view.eye, inverse, best_t, tr);
                if (hl && hr) {
                    if (tl < tr) {
                        stack[top++] = node.right;
                        stack[top++] = node.left;
                    } else {
                        stack[top++] = node.left;
                        stack[top++] = node.right;
                    }
                } else if (hl) {
                    stack[top++] = node.left;
                } else if (hr) {
                    stack[top++] = node.right;
                }
                continue;
            }
            for (std::uint32_t k = node.left; k < node.left + node.right; ++k) {
                const Quad& quad = quads_[k];
                const detail::PartLattice& lattice = lattices_[quad.part];
                const auto corner = [&](int c) -> const Vec3& {
                    return lattice.p[std::size_t(quad.j + (c >> 1)) * std::size_t(lattice.nu) + std::size_t(quad.i + (c & 1))];
                };
                for (int half = 0; half < 2; ++half) {
                    const Vec3& a = corner(0);
                    const Vec3& b = half == 0 ? corner(1) : corner(3);
                    const Vec3& c = half == 0 ? corner(3) : corner(2);
                    double t, wb, wc;
                    if (!hit_triangle(view.eye, dir, a, b, c, t, wb, wc)) { continue; }
                    // Parameter guess from the triangle's barycentric weights.
                    const double ub = half == 0 ? 1.0 : 1.0, vb = half == 0 ? 0.0 : 1.0;
                    const double uc = half == 0 ? 1.0 : 0.0, vc = half == 0 ? 1.0 : 1.0;
                    detail::Solve attempt;
                    attempt.u = (double(quad.i) + wb * ub + wc * uc) * lattice.step;
                    attempt.v = (double(quad.j) + wb * vb + wc * vc) * lattice.step;
                    ++solves;
                    const bool ok = detail::solve_ray(*scene.parts[quad.part].surface, view, dir, tolerance, one_step, 3.0 * lattice.step + 0.5, attempt);
                    evaluations += std::size_t(attempt.evaluations);
                    if (ok && attempt.t < best_t) {
                        best_t = attempt.t;
                        best = attempt;
                        best_part = quad.part;
                        found = true;
                    }
                }
            }
        }
        return found;
    }

    // ---- solve and shade -------------------------------------------------
    struct Work {
        std::size_t solves = 0, evaluations = 0, visits = 0, hits = 0, edges = 0;
    };

    // The area emitter and sky seen along a direction: what a mirror shows
    // where it reflects nothing else. The emitter is a rectangle facing the
    // scene; a reflected ray either passes through that window or does not,
    // so a glossy highlight has the emitter's shape, not a point light's lobe.
    static Vec3 sky(const Scene& scene, Vec3 origin, Vec3 dir) {
        const Vec3 d = normalize(dir);
        const double t = 0.5 * (d.y + 1.0);
        Vec3 colour = d.y >= 0.0 ? Vec3{0.50, 0.56, 0.66} * (1.0 - t) + Vec3{0.80, 0.86, 0.95} * t
                                 : Vec3{0.30, 0.31, 0.33} * (1.0 + d.y) + Vec3{0.12, 0.12, 0.13} * (-d.y);
        const Vec3 centre = Vec3{0.0, 1.3, 0.0} + scene.light * 14.0;
        const double facing = -dot(d, scene.light);
        if (facing < -1e-6) {
            const double reach = dot(centre - origin, scene.light) / dot(d, scene.light);
            if (reach > 0.0) {
                const Vec3 local = origin + d * reach - centre;
                const Vec3 helper = std::fabs(scene.light.y) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
                const Vec3 ax = normalize(cross(helper, scene.light)), ay = cross(scene.light, ax);
                const double x = std::fabs(dot(local, ax)), y = std::fabs(dot(local, ay));
                const double window = (1.0 - detail::smoothstep(3.6, 4.2, x)) * (1.0 - detail::smoothstep(3.6, 4.2, y));
                // Two mullions make the window read as a window.
                const double bars = detail::smoothstep(0.10, 0.22, std::fabs(dot(local, ax))) * detail::smoothstep(0.10, 0.22, std::fabs(dot(local, ay)));
                colour = colour + Vec3{4.2, 4.0, 3.7} * (window * (0.25 + 0.75 * bars));
            }
        }
        return colour;
    }

    bool trace_secondary(const Scene& scene, const RenderOptions& options, const detail::View& primary, Vec3 origin, Vec3 dir,
                         detail::Solve& solve, std::uint16_t& part, Work& work) const {
        detail::View view = primary;
        view.eye = origin;
        view.forward = normalize(dir);
        return trace_independent(scene, view, view.forward, options.tolerance_px * 2.0, options.one_step_px, solve, part, work.solves,
                                 work.evaluations, work.visits);
    }

    // What a mirror at `origin` shows along `dir`: the teapot, the floor, or the sky.
    Vec3 reflected_radiance(const Scene& scene, const RenderOptions& options, const detail::View& primary, Vec3 origin, Vec3 dir,
                            Work& work) const {
        detail::Solve solve;
        std::uint16_t part = 0;
        const bool hit = !nodes_.empty() && trace_secondary(scene, options, primary, origin, dir, solve, part, work);
        const Vec3 unit = normalize(dir);
        double ground_t = -1.0;
        if (options.ground && unit.y < -1e-9 && origin.y > scene.ground_y) { ground_t = (scene.ground_y - origin.y) / unit.y; }
        if (hit && !(ground_t > 0.0 && ground_t < solve.t)) { return shade_surface(scene, options, primary, unit, solve, part, 1, work); }
        if (ground_t > 0.0) { return floor_colour(scene, options, origin + unit * ground_t, ground_t); }
        return sky(scene, origin, unit);
    }

    Vec3 floor_colour(const Scene& scene, const RenderOptions& options, Vec3 p, double t) const {
        const long ix = long(std::floor(p.x)), iz = long(std::floor(p.z));
        const Vec3 base = ((ix + iz) & 1) != 0 ? Vec3{0.50, 0.52, 0.55} : Vec3{0.66, 0.68, 0.71};
        double light = 1.0;
        if (options.lighting) {
            const double lit = options.shadows ? shadow_.lit(p, 0.02) : 1.0;
            light = 0.55 + 0.45 * lit * std::fmax(0.0, scene.light.y);
        }
        const double fade = detail::smoothstep(18.0, 60.0, t);
        return base * light * (1.0 - fade) + Vec3{0.80, 0.84, 0.90} * fade;
    }

    Vec3 shade_surface(const Scene& scene, const RenderOptions& options, const detail::View& primary, Vec3 dir, const detail::Solve& solve,
                       std::uint16_t part_index, int depth, Work& work) const {
        const RenderPart& part = scene.parts[part_index];
        const Surface& surface = *part.surface;
        Vec3 normal = cross(solve.point.du, solve.point.dv);
        if (length(normal) < 1e-9) {
            // A pole: take the normal a little way along v.
            const double cv = double(surface.cells_v());
            const double v = solve.v < 0.5 * cv ? solve.v + 0.05 : solve.v - 0.05;
            const SurfacePoint near = surface.eval(solve.u, v);
            normal = cross(near.du, near.dv);
        }
        normal = normalize(normal) * part.orientation;
        const Vec3 incoming = normalize(dir);
        const Vec3 to_eye = incoming * -1.0;
        const bool back = dot(normal, to_eye) < 0.0;
        if (back) { normal = normal * -1.0; }
        const bool inside = back && !part.two_sided;

        // Base colour.
        Vec3 base{0.93, 0.92, 0.89};
        if (options.texture > 0 && std::size_t(options.texture) <= scene.textures.size()) {
            const Texture& texture = *scene.textures[std::size_t(options.texture - 1)];
            double tx, ty;
            if (options.mapping == TextureMap::patch) {
                tx = -solve.u / double(part.per_u);
                ty = solve.v / double(part.per_v);
                ty = ty - std::floor(ty);
            } else {
                // Around the part (three times for the round parts), reading
                // left to right from outside; along it from top to bottom.
                tx = 0.70 - solve.u / double(surface.cells_u()) * (surface.cells_u() >= 3 * part.per_u ? 3.0 : 1.0);
                ty = solve.v / double(surface.cells_v());
            }
            base = texture.sample(tx, ty, options.filter);
        }
        const bool chrome = options.material == 1 && options.texture == 0;
        if (options.material == 2 && options.texture == 0 && options.wire != Wire::only) {
            // Reflection lines: stripes of the mirror direction, the standard
            // test of curvature continuity. A kink in a stripe is a jump in
            // curvature; a break is a jump in the normal.
            const Vec3 reflected = incoming - normal * (2.0 * dot(incoming, normal));
            const double stripe = std::sin(14.0 * std::atan2(reflected.y, std::hypot(reflected.x, reflected.z)) + 9.0 * std::atan2(reflected.z, reflected.x));
            const double ink = detail::smoothstep(-0.12, 0.12, stripe);
            const double shade = 0.55 + 0.45 * std::fmax(0.0, dot(normal, to_eye));
            return (Vec3{0.06, 0.07, 0.09} * (1.0 - ink) + Vec3{0.96, 0.96, 0.94} * ink) * shade;
        }
        if (inside) { base = base * 0.55; }
        if (options.wire == Wire::only) { base = Vec3{0.97, 0.97, 0.96}; }

        Vec3 colour = base;
        if (options.lighting) {
            const double n_l = dot(normal, scene.light);
            const double lit = options.shadows && n_l > 0.0 ? shadow_.lit(solve.point.p + normal * 0.02, 0.035) : 1.0;
            const double diffuse = std::fmax(0.0, n_l) * lit;
            const double up = 0.5 * (normal.y + 1.0);
            const Vec3 ambient = Vec3{0.30, 0.28, 0.26} * (1.0 - up) + Vec3{0.46, 0.50, 0.56} * up;
            colour = Vec3{base.x * (ambient.x + 0.80 * diffuse), base.y * (ambient.y + 0.78 * diffuse), base.z * (ambient.z + 0.74 * diffuse)};
            if (options.wire != Wire::only) {
                const double n_v = std::fmax(0.0, dot(normal, to_eye));
                const double f0 = chrome ? 0.86 : 0.05;
                const double fresnel = f0 + (1.0 - f0) * std::pow(1.0 - n_v, 5.0);
                const Vec3 reflected = incoming - normal * (2.0 * dot(incoming, normal));
                if (options.gloss) {
                    // A glaze over the body colour: what the mirror direction shows,
                    // weighted by Fresnel reflectance. One bounce is traced.
                    const Vec3 origin = solve.point.p + normal * 2e-3;
                    const Vec3 mirror = depth == 0 ? reflected_radiance(scene, options, primary, origin, reflected, work)
                                                   : sky(scene, origin, reflected);
                    const Vec3 tint = chrome ? Vec3{0.95, 0.96, 0.98} : Vec3{1, 1, 1};
                    const Vec3 under = chrome ? Vec3{0, 0, 0} : colour;
                    colour = under * (1.0 - fresnel) + Vec3{mirror.x * tint.x, mirror.y * tint.y, mirror.z * tint.z} * (fresnel * (lit * 0.5 + 0.5));
                } else {
                    const Vec3 half = normalize(scene.light + to_eye);
                    const double specular = std::pow(std::fmax(0.0, dot(normal, half)), 90.0) * lit;
                    colour = colour * (1.0 - 0.6 * fresnel) + detail::environment(reflected) * (0.6 * fresnel) + Vec3{1, 1, 1} * (0.55 * specular);
                }
            }
        }
        if (options.wire != Wire::off && depth == 0) {
            // Node lines and patch outlines from the hit's own parameters.
            const auto line = [](double coordinate, double period, double gradient, double half_width) {
                const double scaled = coordinate / period;
                const double distance = std::fabs(scaled - std::floor(scaled + 0.5)) * period / (gradient > 1e-12 ? gradient : 1e-12);
                return 1.0 - detail::smoothstep(half_width - 0.5, half_width + 0.5, distance);
            };
            const double node = std::fmax(line(solve.u, 1.0, solve.grad_u, 0.55), line(solve.v, 1.0, solve.grad_v, 0.55));
            const double patch = std::fmax(line(solve.u, double(part.per_u), solve.grad_u, 1.1), line(solve.v, double(part.per_v), solve.grad_v, 1.1));
            const Vec3 node_ink = inside ? Vec3{0.45, 0.50, 0.58} : Vec3{0.16, 0.30, 0.52};
            const Vec3 patch_ink{0.08, 0.09, 0.12};
            colour = colour * (1.0 - 0.75 * node) + node_ink * (0.75 * node);
            colour = colour * (1.0 - patch) + patch_ink * patch;
        }
        return colour;
    }

    // Nearest hit among a pixel's transported predictions.
    bool resolve_candidates(const Scene& scene, const RenderOptions& options, const detail::View& view, Vec3 dir, const Candidate* list,
                            int count, detail::Solve& best, std::uint16_t& best_part, Work& work) const {
        bool found = false;
        for (int k = 0; k < count; ++k) {
            if (found && double(list[k].z_low) > best.t) { break; }
            detail::Solve attempt;
            attempt.u = list[k].u;
            attempt.v = list[k].v;
            const double reach = 3.0 * lattices_[list[k].part].step + 0.5;
            ++work.solves;
            const bool ok = detail::solve_ray(*scene.parts[list[k].part].surface, view, dir, options.tolerance_px, options.one_step_px, reach, attempt);
            work.evaluations += std::size_t(attempt.evaluations);
            if (ok && (!found || attempt.t < best.t)) {
                best = attempt;
                best_part = list[k].part;
                found = true;
            }
        }
        return found;
    }

    // One camera sample at screen position (sx, sy), using the predictions
    // gathered for `pixel`.
    Vec3 sample(const Scene& scene, const detail::View& view, const detail::View& mirror, bool mirror_ready, const RenderOptions& options,
                double sx, double sy, std::size_t pixel, Work& work, Hit& hit) const {
        using namespace detail;
        const Vec3 dir = view.ray(sx, sy);
        Solve best;
        std::uint16_t best_part = 0;
        const std::size_t solves_before = work.solves;
        bool found;
        if (options.method == Method::transport) {
            found = resolve_candidates(scene, options, view, dir, &candidates_[pixel * max_candidates], counts_[pixel], best, best_part, work);
        } else {
            found = trace_independent(scene, view, dir, options.tolerance_px, options.one_step_px, best, best_part, work.solves,
                                      work.evaluations, work.visits);
        }
        hit = Hit{};
        hit.solves = std::uint16_t(std::min<std::size_t>(work.solves - solves_before, 65535));
        // The floor may be nearer than the surface.
        double ground_t = -1.0;
        if (options.ground && std::fabs(dir.y) > 1e-9) { ground_t = (scene.ground_y - view.eye.y) / dir.y; }
        if (found && !(ground_t > 0.0 && ground_t < best.t - 1e-4 && view.eye.y > scene.ground_y)) {
            hit.u = float(best.u);
            hit.v = float(best.v);
            hit.t = float(best.t);
            hit.part = best_part;
            return shade_surface(scene, options, view, dir, best, best_part, 0, work);
        }
        if (!(ground_t > 0.0)) { return options.gloss || options.floor_reflection ? sky(scene, view.eye, dir) : background(dir); }
        const Vec3 p = view.eye + dir * ground_t;
        Vec3 colour = floor_colour(scene, options, p, ground_t);
        if (options.floor_reflection && options.lighting && view.eye.y > scene.ground_y) {
            // A polished floor: the reflected view is the view from the eye
            // mirrored in the floor, so its rays are as coherent as the
            // camera's and transport serves them too.
            const Vec3 unit = normalize(dir);
            const Vec3 reflected{unit.x, -unit.y, unit.z};
            Vec3 seen;
            if (mirror_ready) {
                Solve image;
                std::uint16_t image_part = 0;
                const Vec3 mirror_dir = mirror.ray(sx, sy);
                if (resolve_candidates(scene, options, mirror, mirror_dir, &mirror_candidates_[pixel * max_candidates], mirror_counts_[pixel],
                                       image, image_part, work)) {
                    seen = shade_surface(scene, options, view, mirror_dir, image, image_part, 1, work);
                } else {
                    seen = sky(scene, p, reflected);
                }
            } else {
                RenderOptions plain = options;
                plain.ground = false;
                seen = reflected_radiance(scene, plain, view, p + Vec3{0.0, 2e-3, 0.0}, reflected, work);
            }
            const double fresnel = 0.07 + 0.93 * std::pow(1.0 - std::fabs(unit.y), 5.0);
            const double fade = smoothstep(18.0, 60.0, ground_t);
            colour = colour * (1.0 - fresnel * (1.0 - fade)) + seen * (fresnel * (1.0 - fade));
        }
        return colour;
    }

    void resolve_and_shade(const Scene& scene, const detail::View& view, const detail::View& mirror, bool mirror_ready,
                           const RenderOptions& options, Frame& frame) {
        using namespace detail;
        const int width = frame.width, height = frame.height;
        std::vector<Work> totals(64);
        const auto write = [&](std::size_t pixel, Vec3 colour) {
            std::uint8_t* out = &frame.rgba[pixel * 4];
            out[0] = to_byte(colour.x);
            out[1] = to_byte(colour.y);
            out[2] = to_byte(colour.z);
            out[3] = 255;
        };
        parallel_rows(height, options.threads, [&](int begin, int end, int worker) {
            Work work;
            for (int y = begin; y < end; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t pixel = std::size_t(y) * std::size_t(width) + std::size_t(x);
                    Hit hit;
                    Vec3 colour = sample(scene, view, mirror, mirror_ready, options, x + 0.5, y + 0.5, pixel, work, hit);
                    if (hit.t > 0.0f) { ++work.hits; }
                    if (options.show_work) {
                        const int amount = hit.solves;
                        colour = amount == 0 ? colour * 0.35 : amount <= 1 ? Vec3{0.20, 0.45, 0.85} : amount <= 2 ? Vec3{0.95, 0.65, 0.15} : Vec3{0.85, 0.15, 0.2};
                    }
                    frame.hits[pixel] = hit;
                    write(pixel, colour);
                }
            }
            Work& total = totals[std::size_t(worker)];
            total.solves += work.solves;
            total.evaluations += work.evaluations;
            total.visits += work.visits;
            total.hits += work.hits;
        });
        if (options.edge_samples && !options.show_work) {
            // Edges: a pixel whose neighbour shows another part, another sheet
            // or no surface gets four more samples on a rotated grid. Their
            // rays stay within the reach of the pixel's own predictions.
            // The predictions were made for the pixel centre; a sub-pixel ray
            // starts up to 0.4 pixel further from them.
            RenderOptions sub = options;
            sub.one_step_px = options.one_step_px > 0.0 ? options.one_step_px + 0.45 : 0.0;
            parallel_rows(height, options.threads, [&](int begin, int end, int worker) {
                Work work;
                std::size_t edges = 0;
                for (int y = begin; y < end; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const std::size_t pixel = std::size_t(y) * std::size_t(width) + std::size_t(x);
                        const Hit& here = frame.hits[pixel];
                        bool edge = false;
                        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                        for (const auto& offset : offsets) {
                            const int nx = x + offset[0], ny = y + offset[1];
                            if (nx < 0 || ny < 0 || nx >= width || ny >= height) { continue; }
                            const Hit& there = frame.hits[std::size_t(ny) * std::size_t(width) + std::size_t(nx)];
                            const bool a = here.t > 0.0f, b = there.t > 0.0f;
                            if (a != b || (a && (here.part != there.part || std::fabs(here.t - there.t) > 0.004f * here.t + 0.02f))) { edge = true; }
                        }
                        if (!edge) { continue; }
                        ++edges;
                        static constexpr double grid[4][2] = {{-0.375, -0.125}, {0.125, -0.375}, {0.375, 0.125}, {-0.125, 0.375}};
                        Vec3 sum{};
                        for (const auto& g : grid) {
                            Hit ignored;
                            sum = sum + sample(scene, view, mirror, mirror_ready, sub, x + 0.5 + g[0], y + 0.5 + g[1], pixel, work, ignored);
                        }
                        write(pixel, sum * 0.25);
                    }
                }
                Work& total = totals[std::size_t(worker)];
                total.solves += work.solves;
                total.evaluations += work.evaluations;
                total.visits += work.visits;
                total.edges += edges;
            });
        }
        for (const Work& total : totals) {
            stats.solves += total.solves;
            stats.iterations += total.evaluations;
            stats.hit_pixels += total.hits;
            stats.node_visits += total.visits;
            stats.edge_pixels += total.edges;
        }
    }

    static Vec3 background(Vec3 dir) {
        const double t = 0.5 * (normalize(dir).y + 1.0);
        return Vec3{0.84, 0.86, 0.89} * (1.0 - t) + Vec3{0.93, 0.95, 0.98} * t;
    }

    std::vector<detail::PartLattice> lattices_;
    detail::ShadowMap shadow_;
    std::vector<Candidate> candidates_, mirror_candidates_;
    std::vector<std::uint8_t> counts_, mirror_counts_;
    std::vector<Quad> quads_;
    std::vector<Node> nodes_;
};

} // namespace tconv
