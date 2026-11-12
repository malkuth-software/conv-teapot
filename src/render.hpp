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
#include "blend.hpp"
#include "raster.hpp"
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
    std::vector<const Trim*> trims;   // regions of this surface that are not shown
    // Rows that face the teapot's cavity: v >= interior_from, or, when it is
    // negative, v <= -interior_from - 1. A point there sees the floor and the
    // sky only through an opening.
    double interior_from = 1e30;
    bool faces_cavity(double v) const { return interior_from >= 0.0 ? v >= interior_from : v <= -interior_from - 1.0; }
};

// transport and independent trace the surface itself; raster draws the baked
// mesh; seeded traces the surface itself, each ray starting from where the
// rasterized mesh says it lands.
enum class Method : unsigned char { transport, independent, raster, seeded };
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
    // Method::raster: the baked mesh drawn by the CPU rasterizer.
    int supersample = 2;      // visibility samples per pixel, each way
    bool baked_light = true;  // show the mesh's light maps when it has them
    bool mesh_lines = false;  // draw the triangle edges
    // Where the shadow map's texels disagree, trace the light ray to the
    // surface itself: a shadow's edge is then as sharp as the surface makes
    // it, whatever the map's resolution.
    bool exact_shadows = true;
    int shadow_size = 0;      // side of the shadow map in texels; 0: 512 from lattices, 1024 from the mesh
    // Baked light: keep in the light map only what does not depend on the
    // view, and add reflections per frame from emitters along each pixel's
    // mirror direction (the sky and window above, a stand-in for the floor
    // below), so highlights move with the camera.
    bool live_reflection = true;
    bool diffuse_only = false; // set by the bake: shade_surface stops before the view-dependent terms
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
    // Changes whenever the surfaces or their poses change; 0: unknown. What
    // depends only on the surfaces (lattice points, the shadow map, the
    // tree for incoherent rays) is kept from frame to frame while it holds.
    std::uint64_t stamp = 0;
    Mesh* mesh = nullptr; // baked geometry, for Method::raster and for light baking
    // Cutaway: everything with dot(cut_normal, p) > cut_offset is removed,
    // and where the cut passes through solid material the cut face is filled.
    bool cut = false;
    Vec3 cut_normal{0, 0, 1};
    double cut_offset = 0.0;
    bool removed(Vec3 p) const { return cut && dot(cut_normal, p) > cut_offset; }
    // The teapot's own up direction and whether its top is open (the lid is not shown).
    Vec3 axis{0, 1, 0};
    bool open_top = false;
    // Does a mirror ray along `direction` from a point that faces the
    // cavity leave it? Only through the cut or the open top. The test is in
    // the teapot's frame, so it holds when the teapot is turned over.
    bool leaves_cavity(Vec3 direction) const {
        return (cut && dot(direction, cut_normal) > 0.15) || (open_top && dot(direction, axis) > 0.6);
    }
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
    // Cells that lie wholly under a join (a tube's end inside the body, the
    // body under a band). They are not part of the teapot's surface: no ray
    // is predicted from them, they cast no shadow and nothing reflects them.
    std::vector<unsigned char> buried; // [(nv - 1) cells of (nu - 1)]
    bool cell_buried(int i, int j) const { return !buried.empty() && buried[std::size_t(j) * std::size_t(nu - 1) + std::size_t(i)] != 0; }
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
    // What each texel saw nearest the light: the part and where on it.
    // A shadow's edge is settled on the surface itself from these.
    std::vector<float> u, v;
    std::vector<std::uint16_t> part; // 0xffff: nothing
    void clear(int n) {
        size = n;
        depth.assign(std::size_t(n) * std::size_t(n), 1e30f);
        u.assign(depth.size(), 0.0f);
        v.assign(depth.size(), 0.0f);
        part.assign(depth.size(), std::uint16_t(0xffff));
    }
    // Do the texels around p agree? 1: all say lit, 0: all say shadowed,
    // -1: they differ. `own(part, u, v)` is true for a texel that saw the
    // receiving surface's own neighbourhood, which cannot shade it; such a
    // texel counts as lit, so the band of doubt along a terminator is not
    // mistaken for a shadow's edge.
    template<class Own> int settled(Vec3 p, double bias, Own own) const {
        double sx, sy, sz;
        project(p, sx, sy, sz);
        const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
        int open = 0, shut = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int tx = ix + dx, ty = iy + dy;
                if (tx < 0 || ty < 0 || tx >= size || ty >= size) {
                    ++open;
                    continue;
                }
                const std::size_t index = std::size_t(ty) * std::size_t(size) + std::size_t(tx);
                if (sz - bias <= double(depth[index]) || own(part[index], u[index], v[index])) {
                    ++open;
                } else {
                    ++shut;
                }
            }
        }
        return shut == 0 ? 1 : open == 0 ? 0 : -1;
    }
    void project(Vec3 p, double& sx, double& sy, double& sz) const {
        const Vec3 d = p - centre;
        sx = (dot(d, x) / radius * 0.5 + 0.5) * double(size);
        sy = (dot(d, y) / radius * 0.5 + 0.5) * double(size);
        sz = -dot(d, z); // grows away from the light
    }
    // The lit fraction at p. The four depth texels around p are each tested
    // and the results blended by p's position between their centres, so the
    // answer changes continuously as p moves: a shadow's edge is a ramp one
    // texel wide, not a staircase of texels.
    // Is p outside everything the map covers? Most of the floor is.
    bool beyond(Vec3 p) const {
        const Vec3 d = p - centre;
        const double reach = radius * (1.0 + 4.0 / double(size > 0 ? size : 1));
        return std::fabs(dot(d, x)) > reach || std::fabs(dot(d, y)) > reach;
    }
    double lit(Vec3 p, double bias) const {
        if (size == 0 || beyond(p)) { return 1.0; }
        double sx, sy, sz;
        project(p, sx, sy, sz);
        const double x = sx - 0.5, y = sy - 0.5;
        const double fx = std::floor(x), fy = std::floor(y);
        const double ax = x - fx, ay = y - fy;
        const int ix = int(fx), iy = int(fy);
        const auto open = [&](int tx, int ty) {
            if (tx < 0 || ty < 0 || tx >= size || ty >= size) { return 1.0; }
            return sz - bias <= double(depth[std::size_t(ty) * std::size_t(size) + std::size_t(tx)]) ? 1.0 : 0.0;
        };
        return (open(ix, iy) * (1.0 - ax) + open(ix + 1, iy) * ax) * (1.0 - ay) + (open(ix, iy + 1) * (1.0 - ax) + open(ix + 1, iy + 1) * ax) * ay;
    }
};

// Corners are (x, y, depth, u, v).
inline void raster_depth(ShadowMap& map, const double* a, const double* b, const double* c, std::uint16_t part) {
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
            const std::size_t index = std::size_t(y) * std::size_t(map.size) + std::size_t(x);
            if (z < map.depth[index]) {
                map.depth[index] = z;
                map.u[index] = float(w0 * a[3] + w1 * b[3] + w2 * c[3]);
                map.v[index] = float(w0 * a[4] + w1 * b[4] + w2 * c[4]);
                map.part[index] = part;
            }
        }
    }
}

// A studio environment: soft sky, a bright window band, a dark floor.
inline Vec3 environment(Vec3 d) {
    const double t = 0.5 * (d.y + 1.0);
    Vec3 sky = Vec3{0.20, 0.24, 0.30} * (1.0 - t) + Vec3{0.78, 0.84, 0.92} * t;
    if (d.y < 0.0) { sky = Vec3{0.16, 0.15, 0.14} * (1.0 + d.y) + Vec3{0.08, 0.08, 0.08} * (-d.y); }
    // cos(3 az + 0.6) for the azimuth az of d, from the triple-angle
    // identities: no atan2 and no cos per pixel.
    const double r = std::sqrt(d.x * d.x + d.z * d.z);
    const double c = r > 1e-12 ? d.x / r : 1.0, sn = r > 1e-12 ? d.z / r : 0.0;
    const double cos3 = 4.0 * c * c * c - 3.0 * c, sin3 = 3.0 * sn - 4.0 * sn * sn * sn;
    const double lobe = cos3 * 0.8253356149096783 - sin3 * 0.5646424733950354; // cos 0.6, sin 0.6
    const double window = std::exp(-18.0 * (d.y - 0.55) * (d.y - 0.55)) * (0.5 + 0.5 * lobe);
    return sky + Vec3{1.0, 0.98, 0.92} * (0.9 * window * window);
}

// Fixed powers by multiplication: pow() is the slowest thing a pixel does.
inline double fifth(double x) {
    const double x2 = x * x;
    return x2 * x2 * x;
}
inline double ninetieth(double x) {
    const double x2 = x * x, x4 = x2 * x2, x8 = x4 * x4, x16 = x8 * x8, x32 = x16 * x16, x64 = x32 * x32;
    return x64 * x16 * x8 * x2;
}

inline double smoothstep(double a, double b, double x) {
    const double t = x <= a ? 0.0 : x >= b ? 1.0 : (x - a) / (b - a);
    return t * t * (3.0 - 2.0 * t);
}

// Linear light to a display byte (gamma 2), by table: 65536 steps of light
// are finer than a byte everywhere.
inline std::uint8_t to_byte(double linear) {
    static const std::vector<std::uint8_t> table = [] {
        std::vector<std::uint8_t> t(65536);
        for (int k = 0; k < 65536; ++k) { t[std::size_t(k)] = std::uint8_t(std::sqrt((k + 0.5) / 65536.0) * 255.0 + 0.5); }
        return t;
    }();
    return linear <= 0.0 ? std::uint8_t(0) : linear >= 1.0 ? std::uint8_t(255) : table[std::size_t(linear * 65536.0)];
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
        if ((options.method == Method::raster || options.method == Method::seeded) && scene.mesh != nullptr) {
            render_mesh(scene, view, options, frame);
            stats.ms_total = now_ms() - start;
            return;
        }
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
        if (options.method == Method::transport) {
            predict(scene, view, options, false);
            if (mirror_ready) { predict(scene, view, options, true); }
        }
        // Per-ray tracing serves the independent method and every incoherent ray.
        need_bvh(scene, options.method == Method::independent || (options.gloss && options.lighting));
        count_crossings(scene, view, options, frame.width, frame.height);
        stats.ms_predict = now_ms() - mid;
        const double solve_start = now_ms();
        resolve_and_shade(scene, view, mirror, mirror_ready, options, frame);
        stats.ms_solve = now_ms() - solve_start;
        stats.ms_total = now_ms() - start;
    }

private:
    // ---- transport: forward lattice ------------------------------------
    void transport_lattices(const Scene& scene, const detail::View& view, const detail::View* mirror, const RenderOptions& options) {
        const int k = std::max(1, options.subdivision);
        // The lattice points do not depend on the camera: only their projections are redone.
        const bool same = scene.stamp != 0 && scene.stamp == lattice_stamp_ && k == lattice_k_ && lattices_.size() == scene.parts.size();
        lattice_stamp_ = scene.stamp;
        lattice_k_ = k;
        lattices_.resize(scene.parts.size());
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
                        const Vec3 p = same ? lattice.p[index] : surface.position(double(i) * lattice.step, double(j) * lattice.step);
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
            if (!same) {
                lattice.buried.clear();
                if (!scene.parts[part].trims.empty()) {
                    lattice.buried.assign(std::size_t(lattice.nu - 1) * std::size_t(lattice.nv - 1), 0);
                    for (int j = 0; j + 1 < lattice.nv; ++j) {
                        for (int i = 0; i + 1 < lattice.nu; ++i) {
                            bool all = hidden(scene.parts[part], (i + 0.5) * lattice.step, (j + 0.5) * lattice.step);
                            for (int c = 0; c < 4 && all; ++c) { all = hidden(scene.parts[part], (i + (c & 1)) * lattice.step, (j + (c >> 1)) * lattice.step); }
                            lattice.buried[std::size_t(j) * std::size_t(lattice.nu - 1) + std::size_t(i)] = all ? 1 : 0;
                        }
                    }
                }
            }
        }
    }

    void build_shadow_map(const Scene& scene, const RenderOptions& options) {
        const double cut_key = scene.cut ? scene.cut_offset + 7.0 * scene.cut_normal.x + 13.0 * scene.cut_normal.y + 17.0 * scene.cut_normal.z + 1e3 : 0.0;
        const int wanted_size = options.shadow_size > 0 ? options.shadow_size : 512;
        if (options.shadows && options.lighting && scene.stamp != 0 && scene.stamp == shadow_stamp_ && shadow_version_ == 0 && shadow_.size == wanted_size &&
            cut_key == shadow_cut_) {
            return; // the same surfaces as last frame: the map stands
        }
        shadow_stamp_ = scene.stamp;
        shadow_cut_ = cut_key;
        shadow_.size = 0;
        shadow_version_ = 0;
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
        shadow_.clear(options.shadow_size > 0 ? options.shadow_size : 512);
        for (std::size_t part = 0; part < lattices_.size(); ++part) {
            const detail::PartLattice& lattice = lattices_[part];
            for (int j = 0; j + 1 < lattice.nv; ++j) {
                for (int i = 0; i + 1 < lattice.nu; ++i) {
                    double corner[4][5];
                    int gone = 0;
                    for (int c = 0; c < 4; ++c) {
                        const std::size_t index = std::size_t(j + (c >> 1)) * std::size_t(lattice.nu) + std::size_t(i + (c & 1));
                        gone += scene.removed(lattice.p[index]) ? 1 : 0;
                        shadow_.project(lattice.p[index], corner[c][0], corner[c][1], corner[c][2]);
                        corner[c][3] = double(i + (c & 1)) * lattice.step;
                        corner[c][4] = double(j + (c >> 1)) * lattice.step;
                    }
                    if (gone == 4 || lattice.cell_buried(i, j)) { continue; } // removed by the cutaway, or under a join: casts no shadow
                    detail::raster_depth(shadow_, corner[0], corner[1], corner[3], std::uint16_t(part));
                    detail::raster_depth(shadow_, corner[0], corner[3], corner[2], std::uint16_t(part));
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
                    if (high < float(row_begin) - 1.0f || low > float(row_end) + 1.0f || lattice.cell_buried(i, j)) { continue; }
                    float corner[4][5];
                    bool visible = true;
                    int gone = 0;
                    for (int c = 0; c < 4; ++c) {
                        const int ci = i + (c & 1), cj = j + (c >> 1);
                        const std::size_t index = std::size_t(cj) * std::size_t(lattice.nu) + std::size_t(ci);
                        gone += scene.removed(lattice.p[index]) ? 1 : 0;
                        corner[c][0] = screen_x[index];
                        corner[c][1] = screen_y[index];
                        corner[c][2] = screen_z[index];
                        corner[c][3] = float(ci) * step;
                        corner[c][4] = float(cj) * step;
                        visible = visible && screen_z[index] > 0.05f;
                    }
                    // A cell wholly in the half a cutaway removed predicts nothing:
                    // it would only take a place in the pixel's short list.
                    if (!visible || gone == 4) { continue; }
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

    // Cutaway: how often each pixel's ray crosses the surface in the removed
    // half, by drawing the lattice cells there as triangles and counting
    // them. Counting among a pixel's few solved predictions is not
    // reliable: the list is short and a missed or repeated hit flips the
    // answer along every silhouette.
    void count_crossings(const Scene& scene, const detail::View& view, const RenderOptions& options, int width, int height) {
        cut_visibility_.samples.clear();
        if (!scene.cut) { return; }
        Mesh& mesh = cut_mesh_;
        mesh.vertices.clear();
        mesh.triangles.clear();
        mesh.triangle_part.clear();
        mesh.triangle_trimmed.clear();
        for (std::size_t part = 0; part < lattices_.size(); ++part) {
            const detail::PartLattice& lattice = lattices_[part];
            const std::uint32_t first = std::uint32_t(mesh.vertices.size());
            for (int j = 0; j < lattice.nv; ++j) {
                for (int i = 0; i < lattice.nu; ++i) {
                    MeshVertex vertex;
                    vertex.p = lattice.p[std::size_t(j) * std::size_t(lattice.nu) + std::size_t(i)];
                    vertex.u = float(i * lattice.step);
                    vertex.v = float(j * lattice.step);
                    vertex.s = vertex.t = 0.0f;
                    mesh.vertices.push_back(vertex);
                }
            }
            for (int j = 0; j + 1 < lattice.nv; ++j) {
                for (int i = 0; i + 1 < lattice.nu; ++i) {
                    if (lattice.cell_buried(i, j)) { continue; }
                    bool near = false;
                    for (const Trim* trim : scene.parts[part].trims) {
                        near = near || detail::trim_near(*trim, i * lattice.step, (i + 1) * lattice.step, j * lattice.step, (j + 1) * lattice.step);
                    }
                    const std::uint32_t a = first + std::uint32_t(j * lattice.nu + i), b = a + 1, c = a + std::uint32_t(lattice.nu), d = c + 1;
                    for (const auto& corners : {std::array<std::uint32_t, 3>{a, b, d}, std::array<std::uint32_t, 3>{a, d, c}}) {
                        mesh.triangles.insert(mesh.triangles.end(), corners.begin(), corners.end());
                        mesh.triangle_part.push_back(std::uint16_t(part));
                        mesh.triangle_trimmed.push_back(near ? 1 : 0);
                    }
                }
            }
        }
        std::vector<float> clip(mesh.vertices.size());
        for (std::size_t k = 0; k < clip.size(); ++k) { clip[k] = float(dot(scene.cut_normal, mesh.vertices[k].p) - scene.cut_offset); }
        cut_visibility_.resize(width, height, 1);
        rasterize(
            mesh, [&](Vec3 p, double& x, double& y, double& z) { view.project(p, x, y, z); },
            [&](std::uint16_t part, double u, double v, bool exact) { return hidden(scene.parts[part], u, v, exact); },
            [&](int rows, const std::function<void(int, int)>& body) { detail::parallel_rows(rows, options.threads, [&](int begin, int end, int) { body(begin, end); }); },
            cut_visibility_, 0.05, clip.data());
        despeckle_crossings(cut_visibility_);
    }

    // The tree over the lattice quads, kept while the surfaces are the same.
    void need_bvh(const Scene& scene, bool wanted) {
        if (!wanted) {
            nodes_.clear();
            bvh_stamp_ = 0;
            return;
        }
        if (scene.stamp != 0 && scene.stamp == bvh_stamp_ && !nodes_.empty()) { return; }
        nodes_.clear();
        build_bvh();
        bvh_stamp_ = scene.stamp;
    }

    void build_bvh() {
        quads_.clear();
        for (std::size_t part = 0; part < lattices_.size(); ++part) {
            const detail::PartLattice& lattice = lattices_[part];
            for (int j = 0; j + 1 < lattice.nv; ++j) {
                for (int i = 0; i + 1 < lattice.nu; ++i) {
                    if (lattice.cell_buried(i, j)) { continue; }
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
                    if (ok && attempt.t < best_t && !hidden(scene.parts[quad.part], attempt.u, attempt.v) && !scene.removed(attempt.point.p)) {
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
        // Where the floor was reached by the mirror rays of the pixel to the
        // left and of the pixel above: their distance from this pixel's is
        // the size of the pixel's image on the floor. x < 0: not tracked.
        int x = -1;
        Vec3 left;
        bool has_left = false;
        std::vector<Vec3> up;
        std::vector<unsigned char> has_up;
        void start_row(int width) {
            has_left = false;
            if (int(up.size()) != width) {
                up.assign(std::size_t(width), Vec3{});
                has_up.assign(std::size_t(width), 0);
            }
        }
        // The footprint of this pixel's mirror ray on the floor at q.
        double footprint(Vec3 q, double fallback) {
            if (x < 0 || x >= int(up.size())) { return fallback; }
            double size = 0.0;
            if (has_left) { size = std::max(size, length(q - left)); }
            if (has_up[std::size_t(x)] != 0) { size = std::max(size, length(q - up[std::size_t(x)])); }
            left = q;
            has_left = true;
            up[std::size_t(x)] = q;
            has_up[std::size_t(x)] = 1;
            return size > 0.0 ? std::min(size, 40.0) : fallback;
        }
        void no_floor() {
            has_left = false;
            if (x >= 0 && x < int(has_up.size())) { has_up[std::size_t(x)] = 0; }
        }
    };

    // The floor's checker averaged over a square of side `width` about p: the
    // product of two square waves, each averaged exactly. A point sample of
    // the checker aliases wherever its image is smaller than a pixel, which
    // is most of what a curved mirror shows of it.
    static Vec3 checker(Vec3 p, double width) {
        const auto wave = [](double x, double w) {
            // The integral of the square wave of period 2 is a triangle wave.
            const auto integral = [](double t) {
                const double m = t - 2.0 * std::floor(0.5 * t);
                return m < 1.0 ? m : 2.0 - m;
            };
            if (!(w > 1e-6)) { return (x - 2.0 * std::floor(0.5 * x)) < 1.0 ? 1.0 : -1.0; }
            return (integral(x + 0.5 * w) - integral(x - 0.5 * w)) / w;
        };
        const double mix = 0.5 + 0.5 * wave(p.x, width) * wave(p.z, width); // 1: the light square
        return Vec3{0.50, 0.52, 0.55} * (1.0 - mix) + Vec3{0.66, 0.68, 0.71} * mix;
    }

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
    // What the inside of the teapot shows of itself where no ray is traced.
    static Vec3 cavity() { return {0.27, 0.26, 0.25}; }

    Vec3 reflected_radiance(const Scene& scene, const RenderOptions& options, const detail::View& primary, Vec3 origin, Vec3 dir,
                            Work& work, bool from_cavity = false) const {
        detail::Solve solve;
        std::uint16_t part = 0;
        const bool hit = !nodes_.empty() && trace_secondary(scene, options, primary, origin, dir, solve, part, work);
        const Vec3 unit = normalize(dir);
        if (!hit && nodes_.empty() && from_cavity && !scene.leaves_cavity(unit)) {
            // Nothing is traced here, and the floor and sky are not in view from inside.
            work.no_floor();
            return cavity();
        }
        double ground_t = -1.0;
        if (options.ground && unit.y < -1e-9 && origin.y > scene.ground_y) { ground_t = (scene.ground_y - origin.y) / unit.y; }
        if (hit && !(ground_t > 0.0 && ground_t < solve.t)) {
            work.no_floor();
            return shade_surface(scene, options, primary, unit, solve, part, 1, work);
        }
        if (ground_t > 0.0) {
            const Vec3 q = origin + unit * ground_t;
            return floor_colour(scene, options, q, ground_t, nullptr, nullptr, work.footprint(q, 4.0 * ground_t / primary.focal / std::fmax(0.1, -unit.y)));
        }
        work.no_floor();
        return sky(scene, origin, unit);
    }

    // Is the light visible from q? The shadow map's texels around q name
    // the surfaces that may be in the way and where; each is tried by a
    // Newton solve of the light ray against the surface itself.
    template<class Own>
    bool light_ray_open(const Scene& scene, const RenderOptions& options, const detail::View& primary, Vec3 q, Own own, Work& work) const {
        double sx, sy, sz;
        shadow_.project(q, sx, sy, sz);
        const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
        struct Seed {
            std::uint16_t part;
            float u, v;
        };
        Seed tried[9];
        int count = 0;
        detail::View view = primary;
        view.eye = q;
        view.forward = scene.light;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int tx = ix + dx, ty = iy + dy;
                if (tx < 0 || ty < 0 || tx >= shadow_.size || ty >= shadow_.size) { continue; }
                const std::size_t index = std::size_t(ty) * std::size_t(shadow_.size) + std::size_t(tx);
                const Seed seed{shadow_.part[index], shadow_.u[index], shadow_.v[index]};
                if (seed.part >= scene.parts.size() || !(double(shadow_.depth[index]) < sz - 0.01) || own(seed.part, seed.u, seed.v)) { continue; }
                bool repeat = false;
                for (int k = 0; k < count; ++k) {
                    repeat = repeat || (tried[k].part == seed.part && std::fabs(tried[k].u - seed.u) < 0.75f && std::fabs(tried[k].v - seed.v) < 0.75f);
                }
                if (repeat) { continue; }
                tried[count++] = seed;
                detail::Solve solve;
                solve.u = double(seed.u);
                solve.v = double(seed.v);
                ++work.solves;
                const bool hit = detail::solve_ray(*scene.parts[seed.part].surface, view, scene.light, options.tolerance_px * 4.0, options.one_step_px, 2.5, solve);
                work.evaluations += std::size_t(solve.evaluations);
                if (hit && solve.t > 0.03 && !hidden(scene.parts[seed.part], solve.u, solve.v) && !scene.removed(solve.point.p)) { return false; }
            }
        }
        return true;
    }

    // The lit fraction at p. `pixel` is the size of a pixel there and t1, t2
    // two directions in the surface. Where the shadow map's texels differ,
    // light rays are traced from a rotated grid inside the pixel: two
    // first, and the other two only if those disagree. `self` is the part p
    // lies on (0xffff: none, the floor) and (self_u, self_v) where.
    double shadow_lit(const Scene& scene, const RenderOptions& options, const detail::View& primary, Vec3 p, double bias, double pixel, Vec3 t1, Vec3 t2,
                      std::uint16_t self, double self_u, double self_v, Work& work) const {
        if (shadow_.size == 0 || shadow_.beyond(p)) { return 1.0; }
        if (!options.exact_shadows || !(pixel > 0.0) || shadow_.part.empty()) { return shadow_.lit(p, bias); }
        const double turn = self < scene.parts.size() && scene.parts[self].surface->periodic_u() ? double(scene.parts[self].surface->cells_u()) : 0.0;
        const auto own = [&](std::uint16_t part, float u, float v) {
            if (part != self) { return false; }
            double du = std::fabs(double(u) - self_u);
            if (turn > 0.0) {
                du = std::fmod(du, turn);
                du = std::min(du, turn - du);
            }
            return du < 0.75 && std::fabs(double(v) - self_v) < 0.75;
        };
        const int agreed = shadow_.settled(p, bias, own);
        if (agreed >= 0) { return double(agreed); }
        static constexpr double grid[4][2] = {{-0.375, -0.125}, {0.375, 0.125}, {0.125, -0.375}, {-0.125, 0.375}};
        const auto ray = [&](int k) { return light_ray_open(scene, options, primary, p + t1 * (grid[k][0] * pixel) + t2 * (grid[k][1] * pixel), own, work) ? 1.0 : 0.0; };
        const double first = ray(0), second = ray(1);
        if (first == second) { return first; }
        return 0.25 * (first + second + ray(2) + ray(3));
    }

    // `width`: the size of the pixel's image on the floor, for a mirror ray;
    // for a camera ray it is found here.
    Vec3 floor_colour(const Scene& scene, const RenderOptions& options, Vec3 p, double t, const detail::View* primary = nullptr, Work* work = nullptr,
                      double width = 0.0) const {
        if (primary != nullptr) {
            const double far = length(p - primary->eye);
            width = far / primary->focal / std::fmax(0.08, std::fabs(primary->eye.y - p.y) / (far > 1e-9 ? far : 1e-9));
        }
        const Vec3 base = checker(p, width);
        double light = 1.0;
        if (options.lighting) {
            const double lit = !options.shadows ? 1.0
                               : primary != nullptr ? shadow_lit(scene, options, *primary, p + Vec3{0.0, 0.004, 0.0}, 0.02, t / primary->focal, Vec3{1, 0, 0}, Vec3{0, 0, 1}, std::uint16_t(0xffff), 0.0, 0.0, *work)
                                                    : shadow_.lit(p, 0.02);
            light = 0.55 + 0.45 * lit * std::fmax(0.0, scene.light.y);
        }
        const double fade = detail::smoothstep(18.0, 60.0, t);
        return base * light * (1.0 - fade) + Vec3{0.80, 0.84, 0.90} * fade;
    }

    // How much a shadow dims what a glossy surface mirrors. A shadow is only
    // looked up on surface that faces the light, so its dimming fades to
    // nothing toward the terminator; a step there would outline every
    // shadowed strip that happens to face the light.
    static double mirror_dim(double lit, double n_l) { return 1.0 - 0.5 * (1.0 - lit) * detail::smoothstep(0.0, 0.3, n_l); }

    Vec3 shade_surface(const Scene& scene, const RenderOptions& options, const detail::View& primary, Vec3 dir, const detail::Solve& solve,
                       std::uint16_t part_index, int depth, Work& work, double* lit_out = nullptr) const {
        const RenderPart& part = scene.parts[part_index];
        const Surface& surface = *part.surface;
        Vec3 normal = cross(solve.point.du, solve.point.dv);
        if (length(normal) < 1e-9) {
            // A pole: one normal for every u, the mean a little way along v all the way round.
            const double cv = double(surface.cells_v()), cu = double(surface.cells_u());
            const double v = solve.v < 0.5 * cv ? solve.v + 0.05 : solve.v - 0.05;
            normal = {};
            for (int k = 0; k < 8; ++k) {
                const SurfacePoint near = surface.eval(cu * k / 8.0, v);
                normal = normal + normalize(cross(near.du, near.dv));
            }
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
                tx = -solve.u / double(std::max<std::size_t>(1, part.per_u));
                ty = solve.v / double(std::max<std::size_t>(1, part.per_v));
                ty = ty - std::floor(ty);
            } else {
                // Around the part (three times for the round parts), reading
                // left to right from outside; along it from top to bottom.
                tx = 0.70 - solve.u / double(surface.cells_u()) * (surface.cells_u() >= 16 ? 3.0 : 1.0);
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
            double lit = 1.0;
            if (options.shadows && n_l > 0.0) {
                const double lu = length(solve.point.du), lv = length(solve.point.dv);
                // Where the light only grazes the surface a traced light ray
                // meets it at a glancing angle and its answer is unreliable,
                // so the traced edge is used in proportion to how squarely
                // the surface faces the light, and the shadow map's own
                // smooth answer takes over toward the terminator.
                // The shadow map is drawn from flat pieces, which stray from the
                // surface by more along the light the more the light grazes
                // it: the depth allowance grows with the tangent of that angle.
                const double bias = 0.02 + 0.02 * std::fmin(12.0, std::sqrt(std::fmax(0.0, 1.0 - n_l * n_l)) / n_l);
                const double coarse = shadow_.lit(solve.point.p + normal * 0.02, bias);
                const double trust = detail::smoothstep(0.2, 0.5, n_l);
                lit = coarse;
                if (depth == 0 && trust > 0.0 && lu > 1e-9 && lv > 1e-9) {
                    const double traced = shadow_lit(scene, options, primary, solve.point.p + normal * 0.02, bias, solve.t / primary.focal,
                                                     solve.point.du * (1.0 / lu), solve.point.dv * (1.0 / lv), part_index, solve.u, solve.v, work);
                    lit = coarse + (traced - coarse) * trust;
#ifdef TCONV_SHADOW_DEBUG
                    if (std::fabs(traced - coarse) > 0.5) { std::fprintf(stderr, "part %d u %.3f v %.3f n_l %.3f coarse %.2f traced %.2f\n", int(part_index), solve.u, solve.v, n_l, coarse, traced); }
#endif
                }
            }
            const double diffuse = std::fmax(0.0, n_l) * lit;
            const double up = 0.5 * (normal.y + 1.0);
            const Vec3 ambient = Vec3{0.30, 0.28, 0.26} * (1.0 - up) + Vec3{0.46, 0.50, 0.56} * up;
            colour = Vec3{base.x * (ambient.x + 0.80 * diffuse), base.y * (ambient.y + 0.78 * diffuse), base.z * (ambient.z + 0.74 * diffuse)};
            if (lit_out != nullptr) { *lit_out = lit; }
            if (options.wire != Wire::only && !options.diffuse_only) {
                const double n_v = std::fmax(0.0, dot(normal, to_eye));
                const double f0 = chrome ? 0.86 : 0.05;
                const double fresnel = f0 + (1.0 - f0) * detail::fifth(1.0 - n_v);
                const Vec3 reflected = incoming - normal * (2.0 * dot(incoming, normal));
                if (options.gloss) {
                    // A glaze over the body colour: what the mirror direction shows,
                    // weighted by Fresnel reflectance. One bounce is traced.
                    const Vec3 origin = solve.point.p + normal * 2e-3;
                    const Vec3 mirror = depth == 0 ? reflected_radiance(scene, options, primary, origin, reflected, work, part.faces_cavity(solve.v))
                                                   : sky(scene, origin, reflected);
                    const Vec3 tint = chrome ? Vec3{0.95, 0.96, 0.98} : Vec3{1, 1, 1};
                    const Vec3 under = chrome ? Vec3{0, 0, 0} : colour;
                    colour = under * (1.0 - fresnel) + Vec3{mirror.x * tint.x, mirror.y * tint.y, mirror.z * tint.z} * (fresnel * mirror_dim(lit, n_l));
                } else {
                    const Vec3 half = normalize(scene.light + to_eye);
                    const double specular = detail::ninetieth(std::fmax(0.0, dot(normal, half))) * lit;
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
            const double patch = part.per_u == 0 ? 0.0 : std::fmax(line(solve.u, double(part.per_u), solve.grad_u, 1.1), line(solve.v, double(part.per_v), solve.grad_v, 1.1));
            const Vec3 node_ink = inside ? Vec3{0.45, 0.50, 0.58} : Vec3{0.16, 0.30, 0.52};
            const Vec3 patch_ink{0.08, 0.09, 0.12};
            colour = colour * (1.0 - 0.75 * node) + node_ink * (0.75 * node);
            colour = colour * (1.0 - patch) + patch_ink * patch;
        }
        return colour;
    }

    static bool hidden(const RenderPart& part, double u, double v, bool exact = false) {
        for (const Trim* trim : part.trims) {
            if (trim->hides(u, v, exact)) { return true; }
        }
        return false;
    }

    // Nearest hit among a pixel's transported predictions.
    bool resolve_candidates(const Scene& scene, const RenderOptions& options, const detail::View& view, Vec3 dir, const Candidate* list,
                            int count, detail::Solve& best, std::uint16_t& best_part, Work& work, int* crossings = nullptr) const {
        bool found = false;
        // With a cutaway every prediction is solved: the hits in the removed
        // half are counted (each once), for the cut face.
        struct Seen {
            std::uint16_t part;
            double u, v;
        };
        Seen seen[max_candidates];
        int seen_count = 0;
        for (int k = 0; k < count; ++k) {
            if (!scene.cut && found && double(list[k].z_low) > best.t) { break; }
            detail::Solve attempt;
            attempt.u = list[k].u;
            attempt.v = list[k].v;
            const double reach = 3.0 * lattices_[list[k].part].step + 0.5;
            ++work.solves;
            const bool ok = detail::solve_ray(*scene.parts[list[k].part].surface, view, dir, options.tolerance_px, options.one_step_px, reach, attempt);
            work.evaluations += std::size_t(attempt.evaluations);
            if (ok && scene.removed(attempt.point.p)) {
                if (hidden(scene.parts[list[k].part], attempt.u, attempt.v, true)) { continue; }
                bool repeat = false;
                for (int j = 0; j < seen_count; ++j) {
                    repeat = repeat || (seen[j].part == list[k].part && std::fabs(seen[j].u - attempt.u) < 0.05 && std::fabs(seen[j].v - attempt.v) < 0.05);
                }
                if (!repeat) {
                    seen[seen_count++] = {list[k].part, attempt.u, attempt.v};
                    if (crossings != nullptr) { ++*crossings; }
                }
                continue;
            }
            if (ok && hidden(scene.parts[list[k].part], attempt.u, attempt.v)) { continue; }
            if (ok && (!found || attempt.t < best.t)) {
                best = attempt;
                best_part = list[k].part;
                found = true;
            }
        }
        return found;
    }

    // The cut face. A ray that starts in the removed half and has crossed
    // the surface an odd number of times there is inside solid material
    // when it reaches the cutting plane, and shows the cut face.
    static bool cut_face(const Scene& scene, const RenderOptions& options, const detail::View& view, Vec3 dir, bool odd, double ground_t, Hit& hit, Vec3& colour) {
        if (!scene.cut || !odd || !(dot(scene.cut_normal, view.eye) > scene.cut_offset)) { return false; }
        const double along = dot(scene.cut_normal, dir);
        if (!(along < -1e-12)) { return false; }
        const double t = (scene.cut_offset - dot(scene.cut_normal, view.eye)) / along;
        if (!(t > 0.0) || (ground_t > 0.0 && ground_t < t && view.eye.y > scene.ground_y)) { return false; }
        hit.t = float(t);
        hit.part = 0xffff; // not a surface: it cannot be picked
        const Vec3 clay{0.36, 0.26, 0.22};
        const double light = options.lighting ? 0.55 + 0.6 * std::fmax(0.0, dot(scene.cut_normal, scene.light)) : 1.0;
        colour = clay * light;
        return true;
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
        int crossings = 0;
        if (options.method == Method::transport) {
            found = resolve_candidates(scene, options, view, dir, &candidates_[pixel * max_candidates], counts_[pixel], best, best_part, work, &crossings);
        } else {
            found = trace_independent(scene, view, dir, options.tolerance_px, options.one_step_px, best, best_part, work.solves,
                                      work.evaluations, work.visits);
        }
        if (!cut_visibility_.samples.empty()) { crossings = int(cut_visibility_.samples[pixel].crossings); }
        hit = Hit{};
        hit.solves = std::uint16_t(std::min<std::size_t>(work.solves - solves_before, 65535));
        // The floor may be nearer than the surface.
        double ground_t = -1.0;
        if (options.ground && std::fabs(dir.y) > 1e-9) { ground_t = (scene.ground_y - view.eye.y) / dir.y; }
        Vec3 face;
        if (cut_face(scene, options, view, dir, (crossings & 1) != 0, ground_t, hit, face)) { return face; }
        if (found && !(ground_t > 0.0 && ground_t < best.t - 1e-4 && view.eye.y > scene.ground_y)) {
            hit.u = float(best.u);
            hit.v = float(best.v);
            hit.t = float(best.t);
            hit.part = best_part;
            return shade_surface(scene, options, view, dir, best, best_part, 0, work);
        }
        if (!(ground_t > 0.0)) { return options.gloss || options.floor_reflection ? sky(scene, view.eye, dir) : background(dir); }
        const Vec3 p = view.eye + dir * ground_t;
        Vec3 colour = floor_colour(scene, options, p, ground_t, &view, &work);
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
            const double fresnel = 0.07 + 0.93 * detail::fifth(1.0 - std::fabs(unit.y));
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
                work.start_row(width);
                for (int x = 0; x < width; ++x) {
                    const std::size_t pixel = std::size_t(y) * std::size_t(width) + std::size_t(x);
                    Hit hit;
                    work.x = x;
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

    // ---- baked mesh ------------------------------------------------------
public:
    // Paint the mesh's light maps: every texel is shaded as the ray tracer
    // shades a hit there, seen from the camera, and the result is stored
    // with its chain of half-size levels. A texel that faces away from the
    // camera is shaded for the mirror image of the view direction, so the
    // far side holds its own outward lighting. `texel` is the texel size in
    // model units.
    void bake_light(const Scene& scene, const Camera& camera, const RenderOptions& options, int width, int height, double texel = 0.016) {
        using namespace detail;
        Mesh& mesh = *scene.mesh;
        const double start = now_ms();
        const View view(camera, width, height);
        RenderOptions bake = options;
        bake.diffuse_only = options.live_reflection && options.material != 2 && options.wire != Wire::only && options.lighting;
        mesh.lit_diffuse_only = bake.diffuse_only;
        transport_lattices(scene, view, nullptr, options);
        build_shadow_map(scene, options);
        need_bvh(scene, options.gloss && options.lighting);
        mesh.texels = 0;
        for (std::size_t k = 0; k < mesh.parts.size() && k < scene.parts.size(); ++k) {
            MeshPart& part = mesh.parts[k];
            const Surface& surface = *scene.parts[k].surface;
            // Both sides of every part are baked, each as it is seen from its
            // own side: which side a pixel shows is known only when it is drawn.
            LightMap::Level level, back;
            level.width = std::min(2048, std::max(32, int(part.world_around / texel + 15.0) / 16 * 16));
            level.height = std::min(2048, std::max(16, int(part.world_along / texel + 0.5)));
            level.texel.resize(std::size_t(level.width) * std::size_t(level.height));
            back = level;
            const double cu = double(surface.cells_u()), cv = double(surface.cells_v());
            parallel_rows(level.height, options.threads, [&](int begin, int end, int) {
                Work work;
                for (int y = begin; y < end; ++y) {
                    const double v = part.v_of((y + 0.5) / double(level.height));
                    for (int x = 0; x < level.width; ++x) {
                        Solve solve;
                        solve.u = (x + 0.5) / double(level.width) * cu;
                        solve.v = v;
                        solve.point = surface.eval(solve.u, solve.v);
                        solve.grad_u = cu / double(level.width);
                        solve.grad_v = cv / double(level.height);
                        const Vec3 towards_point = solve.point.p - view.eye;
                        solve.t = dot(towards_point, view.forward);
                        const Vec3 raw = normalize(cross(solve.point.du, solve.point.dv));
                        for (int side = 0; side < 2; ++side) {
                            // This side seen from the camera, or, when it faces
                            // away, from the camera's mirror image in the surface.
                            const Vec3 facing = raw * (side == 0 ? 1.0 : -1.0);
                            Vec3 dir = towards_point;
                            const double away = dot(facing, dir);
                            if (away > 0.0) { dir = dir - facing * (2.0 * away); }
                            double lit = 1.0;
                            const Vec3 colour = shade_surface(scene, bake, view, dir, solve, std::uint16_t(k), 0, work, &lit);
                            (side == 0 ? level : back).texel[std::size_t(y) * std::size_t(level.width) + std::size_t(x)] =
                                std::uint32_t(to_byte(colour.x)) | std::uint32_t(to_byte(colour.y)) << 8 | std::uint32_t(to_byte(colour.z)) << 16 |
                                std::uint32_t(lit * 255.0 + 0.5) << 24;
                        }
                    }
                }
            });
            mesh.texels += 2 * level.texel.size();
            for (LightMap* map : {&part.light, &part.light_back}) {
                map->wrap = surface.periodic_u();
                map->levels.clear();
                map->levels.push_back(std::move(map == &part.light ? level : back));
                map->build_levels();
            }
        }
        // The floor emitter: the shadow on the floor, over the square it can reach.
        mesh.floor = FloorEmitter{};
        if (options.shadows && options.lighting && options.ground && shadow_.size != 0 && scene.light.y > 0.05) {
            FloorEmitter& floor = mesh.floor;
            const Vec3 foot = shadow_.centre - scene.light * ((shadow_.centre.y - scene.ground_y) / scene.light.y);
            const double half = shadow_.radius / scene.light.y + 0.25;
            floor.size = 384;
            floor.extent = 2.0 * half;
            floor.x0 = foot.x - half;
            floor.z0 = foot.z - half;
            floor.lit.assign(std::size_t(floor.size) * std::size_t(floor.size), 1.0f);
            const double texel_size = floor.extent / double(floor.size);
            parallel_rows(floor.size, options.threads, [&](int begin, int end, int) {
                Work work;
                for (int y = begin; y < end; ++y) {
                    for (int x = 0; x < floor.size; ++x) {
                        const Vec3 p{floor.x0 + (x + 0.5) * texel_size, scene.ground_y + 0.004, floor.z0 + (y + 0.5) * texel_size};
                        floor.lit[std::size_t(y) * std::size_t(floor.size) + std::size_t(x)] =
                            float(shadow_lit(scene, options, view, p, 0.02, texel_size, Vec3{1, 0, 0}, Vec3{0, 0, 1}, std::uint16_t(0xffff), 0.0, 0.0, work));
                    }
                }
            });
            mesh.texels += floor.lit.size();
        }
        mesh.lit = true;
        mesh.ms_light = now_ms() - start;
    }

private:
    void build_shadow_map_mesh(const Scene& scene, const RenderOptions& options) {
        const Mesh& mesh = *scene.mesh;
        if (!options.shadows || !options.lighting || mesh.vertices.empty()) {
            shadow_.size = 0;
            shadow_version_ = 0;
            return;
        }
        const double cut_key = scene.cut ? scene.cut_offset + 7.0 * scene.cut_normal.x + 13.0 * scene.cut_normal.y + 17.0 * scene.cut_normal.z + 1e3 : 0.0;
        const int wanted_size = options.shadow_size > 0 ? options.shadow_size : 1024;
        if (shadow_version_ == mesh.version && shadow_.size == wanted_size && cut_key == shadow_cut_) { return; }
        shadow_cut_ = cut_key;
        Vec3 low{1e30, 1e30, 1e30}, high{-1e30, -1e30, -1e30};
        for (const MeshVertex& vertex : mesh.vertices) {
            low = {std::min(low.x, vertex.p.x), std::min(low.y, vertex.p.y), std::min(low.z, vertex.p.z)};
            high = {std::max(high.x, vertex.p.x), std::max(high.y, vertex.p.y), std::max(high.z, vertex.p.z)};
        }
        shadow_.centre = (low + high) * 0.5;
        shadow_.radius = 0.5 * length(high - low) * 1.02 + 1e-6;
        shadow_.z = scene.light;
        const Vec3 helper = std::fabs(scene.light.y) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        shadow_.x = normalize(cross(helper, shadow_.z));
        shadow_.y = cross(shadow_.z, shadow_.x);
        shadow_.clear(options.shadow_size > 0 ? options.shadow_size : 1024); // drawn once per pose, not per frame
        std::vector<double> projected(mesh.vertices.size() * 5);
        for (std::size_t k = 0; k < mesh.vertices.size(); ++k) {
            shadow_.project(mesh.vertices[k].p, projected[k * 5], projected[k * 5 + 1], projected[k * 5 + 2]);
            projected[k * 5 + 3] = double(mesh.vertices[k].u);
            projected[k * 5 + 4] = double(mesh.vertices[k].v);
        }
        for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
            // What the cutaway removed casts no shadow, so light reaches the inside.
            if (scene.removed(mesh.vertices[mesh.triangles[t * 3]].p) && scene.removed(mesh.vertices[mesh.triangles[t * 3 + 1]].p) &&
                scene.removed(mesh.vertices[mesh.triangles[t * 3 + 2]].p)) {
                continue;
            }
            detail::raster_depth(shadow_, &projected[std::size_t(mesh.triangles[t * 3]) * 5], &projected[std::size_t(mesh.triangles[t * 3 + 1]) * 5],
                                 &projected[std::size_t(mesh.triangles[t * 3 + 2]) * 5], mesh.triangle_part[t]);
        }
        shadow_version_ = mesh.version;
    }

    // One sample of the visibility buffer, shaded. (sx, sy) is its position in pixels.
    Vec3 shade_visible(const Scene& scene, const RenderOptions& options, const detail::View& view, const VisibilitySample& sample, double sx, double sy,
                       Hit& hit, Work& work) const {
        using namespace detail;
        const Mesh& mesh = *scene.mesh;
        const Vec3 dir = view.ray(sx, sy);
        hit = Hit{};
        double ground_t = -1.0;
        if (options.ground && std::fabs(dir.y) > 1e-9) { ground_t = (scene.ground_y - view.eye.y) / dir.y; }
        const double depth = sample.triangle != 0xffffffffu ? 1.0 / double(sample.iz) : 0.0;
        Vec3 face;
        if (cut_face(scene, options, view, dir, sample.crossings % 2 != 0, ground_t, hit, face)) { return face; }
        if (sample.triangle == 0xffffffffu || (ground_t > 0.0 && ground_t < depth - 1e-4 && view.eye.y > scene.ground_y)) {
            if (!(ground_t > 0.0)) { return options.gloss || options.floor_reflection ? sky(scene, view.eye, dir) : background(dir); }
            return floor_colour(scene, options, view.eye + dir * ground_t, ground_t, &view, &work);
        }
        float w[3];
        corner_weights(visibility_, mesh, sample, w);
        const std::uint32_t* corner = &mesh.triangles[std::size_t(sample.triangle) * 3];
        const MeshVertex &a = mesh.vertices[corner[0]], &b = mesh.vertices[corner[1]], &c = mesh.vertices[corner[2]];
        const std::uint16_t part = mesh.triangle_part[sample.triangle];
        hit.u = w[0] * a.u + w[1] * b.u + w[2] * c.u;
        hit.v = w[0] * a.v + w[1] * b.v + w[2] * c.v;
        hit.t = float(depth);
        hit.part = part;
        hit.solves = 0;
        Vec3 colour;
        if (options.method == Method::seeded) {
            // The mesh says where the ray lands; Newton puts it on the surface itself.
            Solve solve;
            solve.u = double(hit.u);
            solve.v = double(hit.v);
            ++work.solves;
            const bool ok = solve_ray(*scene.parts[part].surface, view, dir, options.tolerance_px, options.one_step_px, 1.5, solve);
            work.evaluations += std::size_t(solve.evaluations);
            if (ok && !hidden(scene.parts[part], solve.u, solve.v)) {
                hit.u = float(solve.u);
                hit.v = float(solve.v);
                hit.t = float(solve.t);
                hit.solves = 1;
                return shade_surface(scene, options, view, dir, solve, part, 0, work);
            }
            // A ray the mesh catches and the surface does not (the mesh's
            // tolerance at a silhouette): the mesh's own point is shown.
        }
        if (options.method == Method::raster && options.baked_light && mesh.lit && !mesh.parts[part].light.levels.empty()) {
            float rgb[4];
            // The side the eye is on.
            const Vec3 raw_normal = a.n * double(w[0]) + b.n * double(w[1]) + c.n * double(w[2]);
            (dot(raw_normal, dir) > 0.0 ? mesh.parts[part].light_back : mesh.parts[part].light).sample(w[0] * a.s + w[1] * b.s + w[2] * c.s, w[0] * a.t + w[1] * b.t + w[2] * c.t, triangle_lod_[sample.triangle], rgb);
            // Stored display-encoded (gamma 2): back to linear.
            colour = {double(rgb[0] * rgb[0]) / 65025.0, double(rgb[1] * rgb[1]) / 65025.0, double(rgb[2] * rgb[2]) / 65025.0};
            if (mesh.lit_diffuse_only) {
                // What depends on the view, as shade_surface has it, with the
                // mirror direction answered by emitters instead of traced.
                const double lit = double(rgb[3]) / 255.0;
                Vec3 normal = normalize(a.n * double(w[0]) + b.n * double(w[1]) + c.n * double(w[2])) * scene.parts[part].orientation;
                const Vec3 incoming = normalize(dir), to_eye = incoming * -1.0;
                if (dot(normal, to_eye) < 0.0) { normal = normal * -1.0; }
                const bool chrome = options.material == 1 && options.texture == 0;
                const double f0 = chrome ? 0.86 : 0.05;
                const double fresnel = f0 + (1.0 - f0) * detail::fifth(1.0 - std::fmax(0.0, dot(normal, to_eye)));
                const Vec3 reflected = incoming - normal * (2.0 * dot(incoming, normal));
                if (options.gloss) {
                    const Vec3 p = a.p * double(w[0]) + b.p * double(w[1]) + c.p * double(w[2]) + normal * 2e-3;
                    Vec3 mirror;
                    if (scene.parts[part].faces_cavity(double(hit.v)) && !scene.leaves_cavity(reflected)) {
                        work.no_floor();
                        mirror = cavity();
                    } else if (options.ground && reflected.y < -1e-9 && p.y > scene.ground_y) {
                        // The stand-in floor: its pattern in closed form, its shadow from the bake.
                        const double reach = (scene.ground_y - p.y) / reflected.y;
                        const Vec3 q = p + reflected * reach;
                        const Vec3 base = checker(q, work.footprint(q, 4.0 * reach / view.focal / std::fmax(0.1, -reflected.y)));
                        const double light = 0.55 + 0.45 * mesh.floor.lit_at(q.x, q.z) * std::fmax(0.0, scene.light.y);
                        const double fade = smoothstep(18.0, 60.0, reach);
                        mirror = base * light * (1.0 - fade) + Vec3{0.80, 0.84, 0.90} * fade;
                    } else {
                        work.no_floor();
                        mirror = sky(scene, p, reflected);
                    }
                    const Vec3 tint = chrome ? Vec3{0.95, 0.96, 0.98} : Vec3{1, 1, 1};
                    const Vec3 under = chrome ? Vec3{0, 0, 0} : colour;
                    colour = under * (1.0 - fresnel) + Vec3{mirror.x * tint.x, mirror.y * tint.y, mirror.z * tint.z} * (fresnel * mirror_dim(lit, dot(normal, scene.light)));
                } else {
                    const Vec3 half = normalize(scene.light + to_eye);
                    const double specular = detail::ninetieth(std::fmax(0.0, dot(normal, half))) * lit;
                    colour = colour * (1.0 - 0.6 * fresnel) + environment(reflected) * (0.6 * fresnel) + Vec3{1, 1, 1} * (0.55 * specular);
                }
            }
        } else {
            Solve solve;
            solve.u = double(hit.u);
            solve.v = double(hit.v);
            solve.t = depth;
            solve.point.p = a.p * double(w[0]) + b.p * double(w[1]) + c.p * double(w[2]);
            // The interpolated normal, as a pair of tangents whose cross product it is.
            const Vec3 normal = normalize(a.n * double(w[0]) + b.n * double(w[1]) + c.n * double(w[2]));
            const Vec3 helper = std::fabs(normal.y) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
            solve.point.du = normalize(cross(helper, normal));
            solve.point.dv = cross(normal, solve.point.du);
            // Parameter change per pixel, from the triangle's own slopes.
            const double scale = double(visibility_.scale);
            const double x0 = visibility_.sx[corner[0]], y0 = visibility_.sy[corner[0]];
            const double x1 = visibility_.sx[corner[1]] - x0, y1 = visibility_.sy[corner[1]] - y0, x2 = visibility_.sx[corner[2]] - x0, y2 = visibility_.sy[corner[2]] - y0;
            const double det = x1 * y2 - x2 * y1;
            if (std::fabs(det) > 1e-9) {
                const auto slope = [&](double f1, double f2) { return std::hypot((f1 * y2 - f2 * y1) / det, (f2 * x1 - f1 * x2) / det) * scale; };
                solve.grad_u = slope(double(b.u) - double(a.u), double(c.u) - double(a.u));
                solve.grad_v = slope(double(b.v) - double(a.v), double(c.v) - double(a.v));
            }
            colour = shade_surface(scene, options, view, dir, solve, part, 0, work);
        }
        if (options.mesh_lines) {
            // Distance to the nearest edge of the triangle, in pixels.
            const double scale = double(visibility_.scale);
            const double x[3] = {visibility_.sx[corner[0]], visibility_.sx[corner[1]], visibility_.sx[corner[2]]};
            const double y[3] = {visibility_.sy[corner[0]], visibility_.sy[corner[1]], visibility_.sy[corner[2]]};
            const double area2 = std::fabs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]));
            const double l[3] = {1.0 - double(sample.l1) - double(sample.l2), double(sample.l1), double(sample.l2)};
            double nearest = 1e30;
            for (int k = 0; k < 3; ++k) {
                const double edge = std::hypot(x[(k + 1) % 3] - x[(k + 2) % 3], y[(k + 1) % 3] - y[(k + 2) % 3]);
                nearest = std::min(nearest, l[k] * area2 / (edge > 1e-9 ? edge : 1e-9) / scale);
            }
            const double ink = 1.0 - smoothstep(0.2, 0.9, nearest);
            colour = colour * (1.0 - 0.7 * ink) + Vec3{0.10, 0.22, 0.42} * (0.7 * ink);
        }
        return colour;
    }

    void render_mesh(const Scene& scene, const detail::View& view, const RenderOptions& options, Frame& frame) {
        using namespace detail;
        const Mesh& mesh = *scene.mesh;
        const double start = now_ms();
        build_shadow_map_mesh(scene, options);
        if (options.method == Method::seeded && options.gloss && options.lighting) {
            // The traced bounce needs the lattices and their tree.
            transport_lattices(scene, view, nullptr, options);
            need_bvh(scene, true);
        } else {
            need_bvh(scene, false);
        }
        stats.ms_lattice = now_ms() - start;
        const double raster_start = now_ms();
        const int scale = std::max(1, std::min(3, options.supersample));
        visibility_.resize(frame.width, frame.height, scale);
        std::vector<float> clip;
        if (scene.cut) {
            clip.resize(mesh.vertices.size());
            for (std::size_t k = 0; k < clip.size(); ++k) { clip[k] = float(dot(scene.cut_normal, mesh.vertices[k].p) - scene.cut_offset); }
        }
        rasterize(
            mesh, [&](Vec3 p, double& x, double& y, double& z) { view.project(p, x, y, z); },
            [&](std::uint16_t part, double u, double v, bool exact) { return hidden(scene.parts[part], u, v, exact); },
            [&](int rows, const std::function<void(int, int)>& body) { parallel_rows(rows, options.threads, [&](int begin, int end, int) { body(begin, end); }); },
            visibility_, 0.05, clip.empty() ? nullptr : clip.data());
        if (scene.cut) { despeckle_crossings(visibility_); }
        // Level of the light map for every triangle: texels per pixel.
        const bool textured = options.method == Method::raster && options.baked_light && mesh.lit;
        if (textured) {
            triangle_lod_.assign(mesh.triangle_count(), 0.0f);
            for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
                const std::uint32_t* corner = &mesh.triangles[t * 3];
                if (visibility_.iz[corner[0]] == 0.0f || visibility_.iz[corner[1]] == 0.0f || visibility_.iz[corner[2]] == 0.0f) { continue; }
                const LightMap& light = mesh.parts[mesh.triangle_part[t]].light;
                if (light.levels.empty()) { continue; }
                const MeshVertex &a = mesh.vertices[corner[0]], &b = mesh.vertices[corner[1]], &c = mesh.vertices[corner[2]];
                const double x1 = visibility_.sx[corner[1]] - visibility_.sx[corner[0]], y1 = visibility_.sy[corner[1]] - visibility_.sy[corner[0]];
                const double x2 = visibility_.sx[corner[2]] - visibility_.sx[corner[0]], y2 = visibility_.sy[corner[2]] - visibility_.sy[corner[0]];
                const double pixels = std::fabs(x1 * y2 - x2 * y1) / double(scale * scale);
                const double s1 = (double(b.s) - double(a.s)) * light.levels[0].width, t1 = (double(b.t) - double(a.t)) * light.levels[0].height;
                const double s2 = (double(c.s) - double(a.s)) * light.levels[0].width, t2 = (double(c.t) - double(a.t)) * light.levels[0].height;
                const double texels = std::fabs(s1 * t2 - s2 * t1);
                triangle_lod_[t] = pixels > 1e-9 && texels > 0.0 ? float(0.5 * std::log2(texels / pixels)) : 0.0f;
            }
        }
        stats.ms_predict = now_ms() - raster_start;
        const double shade_start = now_ms();
        const int width = frame.width, height = frame.height;
        std::vector<Work> totals(64);
        parallel_rows(height, options.threads, [&](int begin, int end, int worker) {
            Work work;
            std::size_t hits = 0, edges = 0;
            for (int y = begin; y < end; ++y) {
                work.start_row(width);
                for (int x = 0; x < width; ++x) {
                    const std::size_t pixel = std::size_t(y) * std::size_t(width) + std::size_t(x);
                    // A pixel whose samples all show one part at one depth is shaded once.
                    const VisibilitySample& first = visibility_.at(x * scale, y * scale);
                    bool flat = true;
                    for (int k = 1; k < scale * scale && flat; ++k) {
                        const VisibilitySample& other = visibility_.at(x * scale + k % scale, y * scale + k / scale);
                        const bool a = first.triangle != 0xffffffffu, b = other.triangle != 0xffffffffu;
                        if (a != b || (first.crossings & 1) != (other.crossings & 1)) {
                            flat = false;
                        } else if (a && other.triangle != first.triangle) {
                            flat = mesh.triangle_part[other.triangle] == mesh.triangle_part[first.triangle] && std::fabs(other.iz - first.iz) < 0.004f * first.iz;
                        }
                    }
                    Hit hit;
                    Vec3 colour;
                    work.x = flat ? x : -1;
                    if (flat && !options.mesh_lines) {
                        // Seeded rays go through the pixel's centre; the mesh alone is read at its first sample.
                        const double offset = options.method == Method::seeded ? 0.5 : 0.5 / scale;
                        colour = shade_visible(scene, options, view, first, x + offset, y + offset, hit, work);
                    } else {
                        Vec3 sum{};
                        for (int k = 0; k < scale * scale; ++k) {
                            Hit other;
                            sum = sum + shade_visible(scene, options, view, visibility_.at(x * scale + k % scale, y * scale + k / scale),
                                                      x + (k % scale + 0.5) / scale, y + (k / scale + 0.5) / scale, k == 0 ? hit : other, work);
                            if (k > 0 && hit.t <= 0.0f && other.t > 0.0f) { hit = other; }
                        }
                        colour = sum * (1.0 / double(scale * scale));
                        if (!flat) { ++edges; }
                    }
                    if (hit.t > 0.0f) { ++hits; }
                    frame.hits[pixel] = hit;
                    std::uint8_t* out = &frame.rgba[pixel * 4];
                    out[0] = to_byte(colour.x);
                    out[1] = to_byte(colour.y);
                    out[2] = to_byte(colour.z);
                    out[3] = 255;
                }
            }
            Work& total = totals[std::size_t(worker)];
            total.hits += hits;
            total.edges += edges;
            total.solves += work.solves;
            total.evaluations += work.evaluations;
        });
        for (const Work& total : totals) {
            stats.hit_pixels += total.hits;
            stats.edge_pixels += total.edges;
            stats.solves += total.solves;
            stats.iterations += total.evaluations;
        }
        stats.lattice_points = mesh.vertices.size();
        stats.ms_solve = now_ms() - shade_start;
    }

    static Vec3 background(Vec3 dir) {
        const double t = 0.5 * (normalize(dir).y + 1.0);
        return Vec3{0.84, 0.86, 0.89} * (1.0 - t) + Vec3{0.93, 0.95, 0.98} * t;
    }

    std::vector<detail::PartLattice> lattices_;
    detail::ShadowMap shadow_;
    double shadow_cut_ = 0.0;
    std::uint64_t lattice_stamp_ = 0, shadow_stamp_ = 0, bvh_stamp_ = 0;
    int lattice_k_ = 0;
    std::uint64_t shadow_version_ = 0; // mesh version the shadow map was drawn from; 0: from the lattices
    Visibility visibility_, cut_visibility_;
    Mesh cut_mesh_;
    std::vector<float> triangle_lod_;
    std::vector<Candidate> candidates_, mirror_candidates_;
    std::vector<std::uint8_t> counts_, mirror_counts_;
    std::vector<Quad> quads_;
    std::vector<Node> nodes_;
};

} // namespace tconv
