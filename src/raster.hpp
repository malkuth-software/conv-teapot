// Baked geometry: a CONV surface sampled to vertices, and a CPU rasterizer
// for the triangles between them.
//
// The ray tracer evaluates the surface at every pixel. Baking evaluates it
// once, at vertices chosen so that flat triangles between them stay within
// a tolerance of the surface, and from then on a frame costs no surface
// evaluation at all. Three ways of choosing the vertices are kept, because
// which is best is an experiment (tools/mesh_study.cpp):
//
//   uniform  every lattice cell cut k by k
//   bounds   a tensor grid; each column and row of cells is cut as often as
//            the second differences of the lattice nodes say a chord needs.
//            Only the CONV data is read: no surface evaluation decides it.
//   tree     every lattice cell is halved, along u or v or both, until the
//            surface at the middle of each edge and of the cell is within
//            the tolerance of the flat piece. Neighbours of different depth
//            are joined without cracks: a piece is triangulated as a fan
//            that takes in every vertex its neighbours left on its border.
//
// The rasterizer writes a visibility buffer (triangle and barycentric
// weights per sample, nearest first by 1/z); shading happens afterwards,
// once per pixel. Edge functions are integers with 8 bits below the pixel
// and a top-left fill rule, so triangles that share an edge neither overlap
// nor leave a gap.
#pragma once
#include "blend.hpp"
#include "surface.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace tconv {

enum class Layout : int { uniform = 0, bounds = 1, tree = 2 };

struct MeshOptions {
    Layout layout = Layout::tree;
    double tolerance = 2e-3; // model units: bounds and tree
    int uniform = 4;         // cuts per cell: uniform
    double angle = 0.0;      // bounds and tree: also cut until the normal turns no more than this many degrees along an edge (0: off)
};

struct MeshVertex {
    Vec3 p, n;  // position and unit normal (the surface's own, cross(du, dv))
    float u, v; // node units; u is not wrapped, so the closing column has u = cells
    float s, t; // light map coordinates in [0, 1]
    float side = 0.0f; // on a crease: which side's normal this vertex carries (-1 below its row, +1 above)
};

// A light map: colours over a part's (s, t), with a chain of half-size levels.
struct LightMap {
    struct Level {
        int width = 0, height = 0;
        std::vector<std::uint32_t> texel; // r | g << 8 | b << 16 display-encoded, lit fraction << 24
    };
    std::vector<Level> levels;
    bool wrap = true;

    static void unpack(std::uint32_t c, float* rgb) {
        rgb[0] = float(c & 255u);
        rgb[1] = float((c >> 8) & 255u);
        rgb[2] = float((c >> 16) & 255u);
        rgb[3] = float(c >> 24);
    }
    void build_levels() {
        while (levels.back().width > 8 && levels.back().height > 8) {
            const Level& fine = levels.back();
            Level coarse;
            coarse.width = fine.width / 2;
            coarse.height = fine.height / 2;
            coarse.texel.resize(std::size_t(coarse.width) * std::size_t(coarse.height));
            for (int y = 0; y < coarse.height; ++y) {
                for (int x = 0; x < coarse.width; ++x) {
                    float sum[4] = {0, 0, 0, 0}, c[4];
                    for (int k = 0; k < 4; ++k) {
                        unpack(fine.texel[std::size_t(2 * y + (k >> 1)) * std::size_t(fine.width) + std::size_t(2 * x + (k & 1))], c);
                        for (int e = 0; e < 4; ++e) { sum[e] += c[e]; }
                    }
                    coarse.texel[std::size_t(y) * std::size_t(coarse.width) + std::size_t(x)] =
                        std::uint32_t(sum[0] * 0.25f + 0.5f) | std::uint32_t(sum[1] * 0.25f + 0.5f) << 8 | std::uint32_t(sum[2] * 0.25f + 0.5f) << 16 |
                        std::uint32_t(sum[3] * 0.25f + 0.5f) << 24;
                }
            }
            levels.push_back(std::move(coarse));
        }
    }
    // Bilinear within one level; texel centres at (i + 0.5) / width.
    void sample_level(int index, float s, float t, float* rgb) const {
        const Level& level = levels[std::size_t(index)];
        const float x = s * float(level.width) - 0.5f, y = t * float(level.height) - 0.5f;
        const float fx = std::floor(x), fy = std::floor(y);
        const float ax = x - fx, ay = y - fy;
        int x0 = int(fx), x1 = x0 + 1, y0 = int(fy), y1 = y0 + 1;
        if (wrap) {
            x0 = ((x0 % level.width) + level.width) % level.width;
            x1 = ((x1 % level.width) + level.width) % level.width;
        } else {
            x0 = std::min(level.width - 1, std::max(0, x0));
            x1 = std::min(level.width - 1, std::max(0, x1));
        }
        y0 = std::min(level.height - 1, std::max(0, y0));
        y1 = std::min(level.height - 1, std::max(0, y1));
        float a[4], b[4], c[4], d[4];
        unpack(level.texel[std::size_t(y0) * std::size_t(level.width) + std::size_t(x0)], a);
        unpack(level.texel[std::size_t(y0) * std::size_t(level.width) + std::size_t(x1)], b);
        unpack(level.texel[std::size_t(y1) * std::size_t(level.width) + std::size_t(x0)], c);
        unpack(level.texel[std::size_t(y1) * std::size_t(level.width) + std::size_t(x1)], d);
        for (int k = 0; k < 4; ++k) { rgb[k] = (a[k] * (1 - ax) + b[k] * ax) * (1 - ay) + (c[k] * (1 - ax) + d[k] * ax) * ay; }
    }
    // Between the two levels around `lod` (0 is the finest). Four values, 0..255:
    // the display-encoded colour and the lit fraction.
    void sample(float s, float t, float lod, float* rgb) const {
        const float top = float(levels.size() - 1);
        lod = lod < 0.0f ? 0.0f : lod > top ? top : lod;
        const int low = int(lod);
        const float blend = lod - float(low);
        sample_level(low, s, t, rgb);
        if (blend > 0.02f && low + 1 < int(levels.size())) {
            float coarse[4];
            sample_level(low + 1, s, t, coarse);
            for (int k = 0; k < 4; ++k) { rgb[k] += (coarse[k] - rgb[k]) * blend; }
        }
    }
};

struct MeshPart {
    std::size_t first_vertex = 0, vertex_count = 0;
    std::size_t first_triangle = 0, triangle_count = 0;
    std::vector<float> arc;                 // light map t at every node row
    double world_around = 0, world_along = 0; // sizes in model units, for the light map's resolution
    LightMap light;
    double t_of(double v) const {
        const double x = v < 0.0 ? 0.0 : v > double(arc.size() - 1) ? double(arc.size() - 1) : v;
        const std::size_t k = std::min(arc.size() - 2, std::size_t(x));
        return double(arc[k]) + (double(arc[k + 1]) - double(arc[k])) * (x - double(k));
    }
    double v_of(double t) const {
        const auto upper = std::upper_bound(arc.begin(), arc.end(), float(t));
        const std::size_t k = upper == arc.begin() ? 0 : std::min(arc.size() - 2, std::size_t(upper - arc.begin()) - 1);
        const double span = double(arc[k + 1]) - double(arc[k]);
        return double(k) + (span > 0.0 ? std::min(1.0, std::max(0.0, (t - double(arc[k])) / span)) : 0.0);
    }
};

// The floor as an emitter for reflections: not the floor that is drawn, but
// what a mirror direction that points down receives from it. Its pattern is
// the floor's own, in closed form; what is baked is the shadow lying on it.
struct FloorEmitter {
    int size = 0;
    double x0 = 0, z0 = 0, extent = 1; // the square [x0, x0 + extent] x [z0, z0 + extent]
    std::vector<float> lit;
    double lit_at(double x, double z) const {
        if (size == 0) { return 1.0; }
        const double fx = (x - x0) / extent * double(size) - 0.5, fy = (z - z0) / extent * double(size) - 0.5;
        if (fx < 0.0 || fy < 0.0 || fx >= double(size - 1) || fy >= double(size - 1)) { return 1.0; }
        const int ix = int(fx), iy = int(fy);
        const double ax = fx - ix, ay = fy - iy;
        const float* row = &lit[std::size_t(iy) * std::size_t(size) + std::size_t(ix)];
        return (double(row[0]) * (1 - ax) + double(row[1]) * ax) * (1 - ay) + (double(row[size]) * (1 - ax) + double(row[size + 1]) * ax) * ay;
    }
};

struct Mesh {
    std::vector<MeshVertex> vertices;
    std::vector<std::uint32_t> triangles;      // three vertex indices each
    std::vector<std::uint16_t> triangle_part;
    std::vector<std::uint8_t> triangle_trimmed; // crosses the edge of a trim: tested per sample
    std::vector<MeshPart> parts;
    bool lit = false;            // the light maps hold a bake
    bool lit_diffuse_only = false; // the bake left out what depends on the view; reflections are added per frame
    FloorEmitter floor;
    std::uint64_t version = 0;   // changes whenever positions change
    std::size_t evaluations = 0; // surface evaluations spent building it
    double ms_build = 0.0, ms_light = 0.0;
    std::size_t texels = 0;
    std::size_t triangle_count() const { return triangles.size() / 3; }
};

namespace detail {

// Cuts per cell must divide this; it halves six times.
constexpr long mesh_resolution = 960;

inline long cuts_allowed(double wanted) {
    static constexpr long allowed[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64};
    for (long a : allowed) {
        if (double(a) >= wanted) { return a; }
    }
    return 64;
}

inline SurfacePoint mesh_point(const Surface& surface, double u, double v) {
    SurfacePoint point = surface.eval(u, v);
    Vec3 normal = cross(point.du, point.dv);
    if (length(normal) < 1e-9) {
        // A pole: every u is the same point there, so there is one normal,
        // the mean of the normals a little way along v all the way round.
        // Taking it at this u alone gives the coincident vertices of a pole
        // normals that differ slightly, which shows when a sharp edge of
        // light crosses the pole.
        const double cv = double(surface.cells_v()), cu = double(surface.cells_u());
        const double inward = v < 0.5 * cv ? v + 0.05 : v - 0.05;
        normal = {};
        for (int k = 0; k < 8; ++k) {
            const SurfacePoint near = surface.eval(cu * k / 8.0, inward);
            normal = normal + normalize(cross(near.du, near.dv));
        }
    }
    point.du = normalize(normal); // du carries the unit normal from here on
    return point;
}

// Can the edge of a trim pass through this box of (u, v)?
inline bool trim_near(const Trim& trim, double u0, double u1, double v0, double v1) {
    if (!trim.cut.empty()) {
        const auto [low, high] = std::minmax_element(trim.cut.begin(), trim.cut.end());
        return v0 <= *high && v1 >= *low;
    }
    if (trim.loop.empty()) { return false; }
    double shift = 0.0;
    if (trim.period_u > 0.0) { shift = trim.period_u * std::round((0.5 * (u0 + u1) - trim.centre_u) / trim.period_u); }
    double lu = 1e30, hu = -1e30, lv = 1e30, hv = -1e30;
    for (const auto& point : trim.loop) {
        lu = std::min(lu, point[0]);
        hu = std::max(hu, point[0]);
        lv = std::min(lv, point[1]);
        hv = std::max(hv, point[1]);
    }
    return u0 - shift <= hu && u1 - shift >= lu && v0 <= hv && v1 >= lv;
}

struct MeshBuilder {
    const Surface& surface;
    const std::vector<const Trim*>& trims;
    const MeshOptions& options;
    Mesh& mesh;
    long cu, cv;
    struct Rect {
        long u0, v0, u1, v1;
    };
    std::vector<Rect> rects;
    std::unordered_map<std::uint64_t, SurfacePoint> cache;
    std::unordered_map<std::uint64_t, std::uint32_t> index;
    std::map<long, std::vector<long>> on_row, on_column; // vertices on a line of constant v, of constant u

    static std::uint64_t key(long iu, long iv) { return std::uint64_t(iu) << 32 | std::uint64_t(std::uint32_t(iv)); }
    const SurfacePoint& point(long iu, long iv) {
        const auto found = cache.find(key(iu, iv));
        if (found != cache.end()) { return found->second; }
        ++mesh.evaluations;
        return cache.emplace(key(iu, iv), mesh_point(surface, double(iu) / double(mesh_resolution), double(iv) / double(mesh_resolution))).first->second;
    }

    void uniform_rects(const std::vector<long>& cuts_u, const std::vector<long>& cuts_v) {
        for (long j = 0; j < cv; ++j) {
            for (long i = 0; i < cu; ++i) {
                const long nu = cuts_u[std::size_t(i)], nv = cuts_v[std::size_t(j)];
                for (long b = 0; b < nv; ++b) {
                    for (long a = 0; a < nu; ++a) {
                        rects.push_back({i * mesh_resolution + a * mesh_resolution / nu, j * mesh_resolution + b * mesh_resolution / nv,
                                         i * mesh_resolution + (a + 1) * mesh_resolution / nu, j * mesh_resolution + (b + 1) * mesh_resolution / nv});
                    }
                }
            }
        }
    }

    // Cuts from the lattice nodes alone. A chord over a step h of a curve
    // whose second derivative is at most M strays by at most M h^2 / 8; the
    // second difference of the nodes is that M in node units. A node on a
    // crease has no second derivative and is left out.
    void bounds_rects() {
        const auto node = [&](long i, long j) { return surface.position(double(i), double(j)); };
        const bool wrap = surface.periodic_u();
        std::vector<double> bend_u(std::size_t(cu), 0.0), bend_v(std::size_t(cv), 0.0), turn_u(std::size_t(cu), 0.0), turn_v(std::size_t(cv), 0.0);
        // How far the polyline of nodes turns at b, in degrees.
        const auto turn = [](Vec3 a, Vec3 b, Vec3 c) {
            const Vec3 d0 = b - a, d1 = c - b;
            const double l0 = length(d0), l1 = length(d1);
            return l0 > 1e-12 && l1 > 1e-12 ? std::acos(std::max(-1.0, std::min(1.0, dot(d0, d1) / (l0 * l1)))) * 180.0 / 3.14159265358979 : 0.0;
        };
        const auto creased = [](Vec3 a, Vec3 b, Vec3 c) {
            const Vec3 d0 = b - a, d1 = c - b;
            const double l0 = length(d0), l1 = length(d1);
            return l0 > 1e-12 && l1 > 1e-12 && dot(d0, d1) < 0.82 * l0 * l1; // more than about 35 degrees at one node
        };
        for (long j = 0; j <= cv; ++j) {
            for (long i = 0; i < cu + (wrap ? 0 : 1); ++i) {
                // Along u at node (i, j): feeds cells i-1 and i.
                if (wrap || (i > 0 && i < cu)) {
                    const long a = wrap ? (i + cu - 1) % cu : i - 1, c = wrap ? (i + 1) % cu : i + 1;
                    const Vec3 pa = node(a, j), pb = node(i, j), pc = node(c, j);
                    if (!creased(pa, pb, pc)) {
                        const double bend = length(pa - pb * 2.0 + pc), angle = turn(pa, pb, pc);
                        bend_u[std::size_t(wrap ? a : i - 1)] = std::max(bend_u[std::size_t(wrap ? a : i - 1)], bend);
                        turn_u[std::size_t(wrap ? a : i - 1)] = std::max(turn_u[std::size_t(wrap ? a : i - 1)], angle);
                        if (i < cu) {
                            bend_u[std::size_t(i)] = std::max(bend_u[std::size_t(i)], bend);
                            turn_u[std::size_t(i)] = std::max(turn_u[std::size_t(i)], angle);
                        }
                    }
                }
                // Along v at node (i, j): feeds cells j-1 and j.
                if (j > 0 && j < cv) {
                    const Vec3 pa = node(i, j - 1), pb = node(i, j), pc = node(i, j + 1);
                    if (!creased(pa, pb, pc)) {
                        const double bend = length(pa - pb * 2.0 + pc), angle = turn(pa, pb, pc);
                        bend_v[std::size_t(j - 1)] = std::max(bend_v[std::size_t(j - 1)], bend);
                        bend_v[std::size_t(j)] = std::max(bend_v[std::size_t(j)], bend);
                        turn_v[std::size_t(j - 1)] = std::max(turn_v[std::size_t(j - 1)], angle);
                        turn_v[std::size_t(j)] = std::max(turn_v[std::size_t(j)], angle);
                    }
                }
            }
        }
        std::vector<long> cuts_u(static_cast<std::size_t>(cu)), cuts_v(static_cast<std::size_t>(cv));
        // The turn at a node is the turn across one cell: the angle limit asks for that many pieces of it.
        const double per_turn = options.angle > 0.0 ? 1.0 / options.angle : 0.0;
        for (long i = 0; i < cu; ++i) {
            cuts_u[std::size_t(i)] = cuts_allowed(std::max(std::sqrt(bend_u[std::size_t(i)] / (8.0 * options.tolerance)), turn_u[std::size_t(i)] * per_turn));
        }
        for (long j = 0; j < cv; ++j) {
            cuts_v[std::size_t(j)] = cuts_allowed(std::max(std::sqrt(bend_v[std::size_t(j)] / (8.0 * options.tolerance)), turn_v[std::size_t(j)] * per_turn));
        }
        uniform_rects(cuts_u, cuts_v);
    }

    void split(Rect r) {
        const long mu = (r.u0 + r.u1) / 2, mv = (r.v0 + r.v1) / 2;
        const bool can_u = (r.u1 - r.u0) % 2 == 0 && r.u1 - r.u0 >= 30, can_v = (r.v1 - r.v0) % 2 == 0 && r.v1 - r.v0 >= 30;
        const Vec3 p00 = point(r.u0, r.v0).p, p10 = point(r.u1, r.v0).p, p01 = point(r.u0, r.v1).p, p11 = point(r.u1, r.v1).p;
        double err_u = 0.0, err_v = 0.0, err_c = 0.0;
        if (can_u) {
            err_u = std::max(length(point(mu, r.v0).p - (p00 + p10) * 0.5), length(point(mu, r.v1).p - (p01 + p11) * 0.5));
        }
        if (can_v) {
            err_v = std::max(length(point(r.u0, mv).p - (p00 + p01) * 0.5), length(point(r.u1, mv).p - (p10 + p11) * 0.5));
        }
        if (can_u && can_v) { err_c = length(point(mu, mv).p - (p00 + p10 + p01 + p11) * 0.25); }
        bool cut_u = err_u > options.tolerance, cut_v = err_v > options.tolerance;
        if (options.angle > 0.0) {
            const double limit = std::cos(options.angle * 3.14159265358979 / 180.0);
            const Vec3 n00 = point(r.u0, r.v0).du, n10 = point(r.u1, r.v0).du, n01 = point(r.u0, r.v1).du, n11 = point(r.u1, r.v1).du;
            if (can_u && (dot(n00, n10) < limit || dot(n01, n11) < limit)) { cut_u = true; }
            if (can_v && (dot(n00, n01) < limit || dot(n10, n11) < limit)) { cut_v = true; }
        }
        if (!cut_u && !cut_v && err_c > options.tolerance) { cut_u = cut_v = true; }
        if (cut_u && cut_v) {
            split({r.u0, r.v0, mu, mv});
            split({mu, r.v0, r.u1, mv});
            split({r.u0, mv, mu, r.v1});
            split({mu, mv, r.u1, r.v1});
        } else if (cut_u) {
            split({r.u0, r.v0, mu, r.v1});
            split({mu, r.v0, r.u1, r.v1});
        } else if (cut_v) {
            split({r.u0, r.v0, r.u1, mv});
            split({r.u0, mv, r.u1, r.v1});
        } else {
            rects.push_back(r);
        }
    }

    // Node rows where the surface is creased: the normal differs on the two sides.
    std::vector<unsigned char> crease;
    void find_creases() {
        crease.assign(std::size_t(cv) + 1, 0);
        for (long j = 1; j < cv; ++j) {
            for (long k = 0; k < 4 && crease[std::size_t(j)] == 0; ++k) {
                const double u = (double(k) + 0.37) * double(cu) / 4.0;
                const Vec3 below = mesh_point(surface, u, double(j) - 1e-4).du, above = mesh_point(surface, u, double(j) + 1e-4).du;
                if (dot(below, above) < 0.9994) { crease[std::size_t(j)] = 1; } // more than about 2 degrees
            }
        }
    }

    // `side`: -1 for a piece that lies below the vertex's row, +1 above.
    std::uint32_t vertex(long iu, long iv, const MeshPart& part, int side = 0) {
        const bool on_crease = iv % mesh_resolution == 0 && crease[std::size_t(iv / mesh_resolution)] != 0 && side != 0;
        if (on_crease) {
            // A crease vertex exists once per side, with that side's normal.
            const std::uint64_t side_key = key(iu, iv) ^ (side < 0 ? 0x4000000000000000ull : 0x2000000000000000ull);
            const auto found = index.find(side_key);
            if (found != index.end()) { return found->second; }
            const std::uint32_t base = vertex(iu, iv, part, 0);
            MeshVertex copy = mesh.vertices[base];
            const bool closing = surface.periodic_u() && iu == cu * mesh_resolution;
            ++mesh.evaluations;
            copy.side = float(side);
            copy.n = mesh_point(surface, closing ? 0.0 : double(iu) / double(mesh_resolution), double(iv) / double(mesh_resolution) + side * 1e-4).du;
            const std::uint32_t id = std::uint32_t(mesh.vertices.size());
            mesh.vertices.push_back(copy);
            index.emplace(side_key, id);
            return id;
        }
        const auto found = index.find(key(iu, iv));
        if (found != index.end()) { return found->second; }
        // The closing column is the first column again, one turn on.
        const bool closing = surface.periodic_u() && iu == cu * mesh_resolution;
        const SurfacePoint& sp = point(closing ? 0 : iu, iv);
        MeshVertex vertex;
        vertex.p = sp.p;
        vertex.n = sp.du;
        vertex.u = float(double(iu) / double(mesh_resolution));
        vertex.v = float(double(iv) / double(mesh_resolution));
        vertex.s = float(double(iu) / double(cu * mesh_resolution));
        vertex.t = float(part.t_of(double(vertex.v)));
        const std::uint32_t id = std::uint32_t(mesh.vertices.size());
        mesh.vertices.push_back(vertex);
        index.emplace(key(iu, iv), id);
        return id;
    }

    bool hidden(const MeshVertex& vertex) const {
        for (const Trim* trim : trims) {
            if (trim->hides(double(vertex.u), double(vertex.v))) { return true; }
        }
        return false;
    }

    void triangle(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint16_t part) {
        const MeshVertex &va = mesh.vertices[a], &vb = mesh.vertices[b], &vc = mesh.vertices[c];
        const int gone = int(hidden(va)) + int(hidden(vb)) + int(hidden(vc));
        bool near = false;
        for (const Trim* trim : trims) {
            near = near || trim_near(*trim, double(std::min({va.u, vb.u, vc.u})), double(std::max({va.u, vb.u, vc.u})), double(std::min({va.v, vb.v, vc.v})),
                                     double(std::max({va.v, vb.v, vc.v})));
        }
        if (gone == 3 && !near) { return; }
        if (gone == 3) {
            // Hidden at every corner, but the trim's edge is close: keep it only if its middle shows.
            MeshVertex middle = va;
            middle.u = (va.u + vb.u + vc.u) / 3.0f;
            middle.v = (va.v + vb.v + vc.v) / 3.0f;
            if (hidden(middle)) { return; }
        }
        mesh.triangles.push_back(a);
        mesh.triangles.push_back(b);
        mesh.triangles.push_back(c);
        mesh.triangle_part.push_back(part);
        mesh.triangle_trimmed.push_back(gone > 0 || near ? 1 : 0);
    }

    void run(std::uint16_t part_index) {
        cu = long(surface.cells_u());
        cv = long(surface.cells_v());
        MeshPart part;
        part.first_vertex = mesh.vertices.size();
        part.first_triangle = mesh.triangle_count();
        // Light map t: length along v, averaged over a few columns.
        std::vector<double> along(std::size_t(cv) + 1, 0.0);
        for (long j = 0; j <= cv; ++j) {
            double ring = 0.0;
            for (long i = 0; i < cu; ++i) { ring += length(surface.position(double(i + 1), double(j)) - surface.position(double(i), double(j))); }
            part.world_around = std::max(part.world_around, ring);
            if (j == 0) { continue; }
            double step = 0.0;
            const long columns = std::min(cu, 8L);
            for (long k = 0; k < columns; ++k) {
                const double u = double(k * cu / columns);
                step += length(surface.position(u, double(j)) - surface.position(u, double(j - 1)));
            }
            along[std::size_t(j)] = along[std::size_t(j - 1)] + step / double(columns) + 1e-9;
        }
        part.world_along = along.back();
        for (double a : along) { part.arc.push_back(float(a / along.back())); }

        find_creases();
        if (options.layout == Layout::uniform) {
            const long k = cuts_allowed(double(options.uniform));
            uniform_rects(std::vector<long>(std::size_t(cu), k), std::vector<long>(std::size_t(cv), k));
        } else if (options.layout == Layout::bounds) {
            bounds_rects();
        } else {
            for (long j = 0; j < cv; ++j) {
                for (long i = 0; i < cu; ++i) { split({i * mesh_resolution, j * mesh_resolution, (i + 1) * mesh_resolution, (j + 1) * mesh_resolution}); }
            }
        }
        // Every corner, filed under its row and its column. A vertex on the
        // seam of a closed part is on both the first and the closing column.
        const long turn = cu * mesh_resolution;
        const auto file = [&](long iu, long iv) {
            on_row[iv].push_back(iu);
            on_column[iu].push_back(iv);
            if (surface.periodic_u() && (iu == 0 || iu == turn)) {
                on_row[iv].push_back(turn - iu);
                on_column[turn - iu].push_back(iv);
            }
        };
        for (const Rect& r : rects) {
            file(r.u0, r.v0);
            file(r.u1, r.v0);
            file(r.u0, r.v1);
            file(r.u1, r.v1);
        }
        for (auto* lines : {&on_row, &on_column}) {
            for (auto& [line, list] : *lines) {
                std::sort(list.begin(), list.end());
                list.erase(std::unique(list.begin(), list.end()), list.end());
            }
        }
        std::vector<std::uint32_t> ring;
        for (const Rect& r : rects) {
            // The border, counter-clockwise in (u, v), with every vertex on it.
            ring.clear();
            const auto side = [&](long iv) { return iv == r.v0 ? 1 : iv == r.v1 ? -1 : 0; };
            const auto walk = [&](const std::vector<long>& list, long from, long to, bool along_u, long fixed) {
                if (from < to) {
                    for (auto it = std::lower_bound(list.begin(), list.end(), from); it != list.end() && *it < to; ++it) {
                        ring.push_back(along_u ? vertex(*it, fixed, part, side(fixed)) : vertex(fixed, *it, part, side(*it)));
                    }
                } else {
                    auto it = std::upper_bound(list.begin(), list.end(), from);
                    while (it != list.begin() && *(it - 1) > to) {
                        --it;
                        ring.push_back(along_u ? vertex(*it, fixed, part, side(fixed)) : vertex(fixed, *it, part, side(*it)));
                    }
                }
            };
            walk(on_row[r.v0], r.u0, r.u1, true, r.v0);
            walk(on_column[r.u1], r.v0, r.v1, false, r.u1);
            walk(on_row[r.v1], r.u1, r.u0, true, r.v1);
            walk(on_column[r.u0], r.v1, r.v0, false, r.u0);
            if (ring.size() == 4) {
                // The shorter diagonal.
                const Vec3 a = mesh.vertices[ring[0]].p, b = mesh.vertices[ring[1]].p, c = mesh.vertices[ring[2]].p, d = mesh.vertices[ring[3]].p;
                if (length(a - c) <= length(b - d)) {
                    triangle(ring[0], ring[1], ring[2], part_index);
                    triangle(ring[0], ring[2], ring[3], part_index);
                } else {
                    triangle(ring[0], ring[1], ring[3], part_index);
                    triangle(ring[1], ring[2], ring[3], part_index);
                }
            } else {
                const std::uint32_t centre = vertex((r.u0 + r.u1) / 2, (r.v0 + r.v1) / 2, part);
                for (std::size_t k = 0; k < ring.size(); ++k) { triangle(centre, ring[k], ring[(k + 1) % ring.size()], part_index); }
            }
        }
        part.vertex_count = mesh.vertices.size() - part.first_vertex;
        part.triangle_count = mesh.triangle_count() - part.first_triangle;
        mesh.parts.push_back(std::move(part));
    }
};

} // namespace detail

// Bake surfaces to one mesh. `trims[k]` are the trims of surface k.
inline Mesh build_mesh(const std::vector<const Surface*>& surfaces, const std::vector<std::vector<const Trim*>>& trims, const MeshOptions& options) {
    Mesh mesh;
    for (std::size_t k = 0; k < surfaces.size(); ++k) {
        detail::MeshBuilder builder{*surfaces[k], trims[k], options, mesh, 0, 0, {}, {}, {}, {}, {}, {}};
        builder.run(std::uint16_t(k));
    }
    mesh.version = 1;
    return mesh;
}

// Move the vertices to where the surfaces are now (the surfaces moved
// rigidly or otherwise; the layout is kept).
inline void refresh_mesh(Mesh& mesh, const std::vector<const Surface*>& surfaces) {
    for (std::size_t k = 0; k < mesh.parts.size() && k < surfaces.size(); ++k) {
        const MeshPart& part = mesh.parts[k];
        const double cells = double(surfaces[k]->cells_u());
        for (std::size_t index = part.first_vertex; index < part.first_vertex + part.vertex_count; ++index) {
            MeshVertex& vertex = mesh.vertices[index];
            const double u = surfaces[k]->periodic_u() && double(vertex.u) >= cells ? 0.0 : double(vertex.u);
            const SurfacePoint point = detail::mesh_point(*surfaces[k], u, double(vertex.v));
            vertex.p = point.p;
            vertex.n = vertex.side == 0.0f ? point.du : detail::mesh_point(*surfaces[k], u, double(vertex.v) + double(vertex.side) * 1e-4).du;
        }
    }
    mesh.lit = false;
    ++mesh.version;
}

// ---- rasterizer ------------------------------------------------------------

struct VisibilitySample {
    float iz = 0.0f;                    // 1 / view depth; 0: nothing
    std::uint32_t triangle = 0xffffffffu;
    float l1 = 0.0f, l2 = 0.0f;         // screen-affine weights of the second and third corner
};

struct Visibility {
    int width = 0, height = 0, scale = 1; // samples = pixels * scale in each direction
    std::vector<VisibilitySample> samples;
    std::vector<float> sx, sy, iz;        // projected vertices, in samples
    void resize(int w, int h, int s) {
        width = w * s;
        height = h * s;
        scale = s;
        samples.assign(std::size_t(width) * std::size_t(height), VisibilitySample{});
    }
    const VisibilitySample& at(int x, int y) const { return samples[std::size_t(y) * std::size_t(width) + std::size_t(x)]; }
};

// Perspective-correct weights of the three corners from a sample.
inline void corner_weights(const Visibility& vis, const Mesh& mesh, const VisibilitySample& sample, float* w) {
    const std::uint32_t* corner = &mesh.triangles[std::size_t(sample.triangle) * 3];
    const float a = (1.0f - sample.l1 - sample.l2) * vis.iz[corner[0]], b = sample.l1 * vis.iz[corner[1]], c = sample.l2 * vis.iz[corner[2]];
    const float sum = a + b + c;
    w[0] = a / sum;
    w[1] = b / sum;
    w[2] = c / sum;
}

// Draw the mesh into the visibility buffer. `project(p, x, y, z)` gives a
// pixel position and view depth; `hidden(part, u, v)` is asked for samples
// of triangles that cross a trim; `rows(count, body)` runs body(begin, end)
// over ranges of bands of rows, possibly on several threads.
template<class Project, class Hidden, class Rows>
void rasterize(const Mesh& mesh, Project project, Hidden hidden, Rows rows, Visibility& vis, double near = 0.05) {
    const std::size_t count = mesh.vertices.size();
    vis.sx.resize(count);
    vis.sy.resize(count);
    vis.iz.resize(count);
    std::vector<std::int32_t> fx(count), fy(count);
    for (std::size_t k = 0; k < count; ++k) {
        double x, y, z;
        project(mesh.vertices[k].p, x, y, z);
        if (!(z > near)) {
            vis.iz[k] = 0.0f;
            continue;
        }
        x *= double(vis.scale);
        y *= double(vis.scale);
        x = x < -30000.0 ? -30000.0 : x > 30000.0 ? 30000.0 : x;
        y = y < -30000.0 ? -30000.0 : y > 30000.0 ? 30000.0 : y;
        vis.sx[k] = float(x);
        vis.sy[k] = float(y);
        vis.iz[k] = float(1.0 / z);
        fx[k] = std::int32_t(std::lrint(x * 256.0));
        fy[k] = std::int32_t(std::lrint(y * 256.0));
    }
    const std::size_t triangles = mesh.triangle_count();
    // Triangles filed by the bands of rows they touch, so a row range reads
    // only its own.
    const int band = 8, bands = (vis.height + band - 1) / band;
    std::vector<std::vector<std::uint32_t>> filed(static_cast<std::size_t>(bands));
    for (std::size_t t = 0; t < triangles; ++t) {
        const std::uint32_t ia = mesh.triangles[t * 3], ib = mesh.triangles[t * 3 + 1], ic = mesh.triangles[t * 3 + 2];
        if (vis.iz[ia] == 0.0f || vis.iz[ib] == 0.0f || vis.iz[ic] == 0.0f) { continue; }
        const int y0 = std::max(0, int((std::min({fy[ia], fy[ib], fy[ic]}) + 127) >> 8));
        const int y1 = std::min(vis.height - 1, int((std::max({fy[ia], fy[ib], fy[ic]}) - 128) >> 8));
        for (int b = y0 / band; y0 <= y1 && b <= y1 / band; ++b) { filed[std::size_t(b)].push_back(std::uint32_t(t)); }
    }
    rows(bands, [&](int band_begin, int band_end) {
      for (int b = band_begin; b < band_end; ++b) {
        const int row_begin = b * band, row_end = std::min(vis.height, row_begin + band);
        for (const std::uint32_t t : filed[std::size_t(b)]) {
            std::uint32_t ia = mesh.triangles[t * 3], ib = mesh.triangles[t * 3 + 1], ic = mesh.triangles[t * 3 + 2];
            const std::int32_t min_y = std::min({fy[ia], fy[ib], fy[ic]}), max_y = std::max({fy[ia], fy[ib], fy[ic]});
            // Sample centres are at (k + 0.5): rows whose centre lies in [min, max].
            int y0 = int((min_y + 127) >> 8), y1 = int((max_y - 128) >> 8);
            y0 = std::max(y0, row_begin);
            y1 = std::min(y1, row_end - 1);
            if (y0 > y1) { continue; }
            const std::int32_t min_x = std::min({fx[ia], fx[ib], fx[ic]}), max_x = std::max({fx[ia], fx[ib], fx[ic]});
            const int x0 = std::max(0, int((min_x + 127) >> 8)), x1 = std::min(vis.width - 1, int((max_x - 128) >> 8));
            if (x0 > x1) { continue; }
            std::int64_t area = std::int64_t(fx[ib] - fx[ia]) * (fy[ic] - fy[ia]) - std::int64_t(fy[ib] - fy[ia]) * (fx[ic] - fx[ia]);
            if (area == 0) { continue; }
            bool swapped = false;
            if (area < 0) {
                std::swap(ib, ic);
                area = -area;
                swapped = true;
            }
            // Edge functions, positive inside; w0 belongs to corner a, and so on.
            const std::int64_t ax = fx[ia], ay = fy[ia], bx = fx[ib], by = fy[ib], cx = fx[ic], cy = fy[ic];
            const std::int64_t e0x = by - cy, e0y = cx - bx; // edge b -> c
            const std::int64_t e1x = cy - ay, e1y = ax - cx; // edge c -> a
            const std::int64_t e2x = ay - by, e2y = bx - ax; // edge a -> b
            // Top-left rule: an edge that is neither a top nor a left edge does not own its own line.
            const auto bias = [](std::int64_t ex, std::int64_t ey) { return (ex > 0 || (ex == 0 && ey < 0)) ? 0 : -1; };
            const std::int64_t px = std::int64_t(x0) * 256 + 128, py = std::int64_t(y0) * 256 + 128;
            std::int64_t row0 = e0x * (px - bx) + e0y * (py - by) + bias(e0x, e0y);
            std::int64_t row1 = e1x * (px - cx) + e1y * (py - cy) + bias(e1x, e1y);
            std::int64_t row2 = e2x * (px - ax) + e2y * (py - ay) + bias(e2x, e2y);
            const float inverse = 1.0f / float(area);
            const float iza = vis.iz[ia], dzb = vis.iz[ib] - iza, dzc = vis.iz[ic] - iza;
            const bool trimmed = mesh.triangle_trimmed[t] != 0;
            for (int y = y0; y <= y1; ++y) {
                std::int64_t w0 = row0, w1 = row1, w2 = row2;
                VisibilitySample* out = &vis.samples[std::size_t(y) * std::size_t(vis.width) + std::size_t(x0)];
                for (int x = x0; x <= x1; ++x, ++out, w0 += e0x * 256, w1 += e1x * 256, w2 += e2x * 256) {
                    if ((w0 | w1 | w2) < 0) { continue; }
                    const float lb = float(w1) * inverse, lc = float(w2) * inverse;
                    const float iz = iza + lb * dzb + lc * dzc;
                    if (!(iz > out->iz)) { continue; }
                    const float l1 = swapped ? lc : lb, l2 = swapped ? lb : lc;
                    if (trimmed) {
                        const std::uint32_t* corner = &mesh.triangles[t * 3];
                        const float wa = (1.0f - l1 - l2) * vis.iz[corner[0]], wb = l1 * vis.iz[corner[1]], wc = l2 * vis.iz[corner[2]];
                        const float sum = wa + wb + wc;
                        const MeshVertex &va = mesh.vertices[corner[0]], &vb = mesh.vertices[corner[1]], &vc = mesh.vertices[corner[2]];
                        if (hidden(mesh.triangle_part[t], double((wa * va.u + wb * vb.u + wc * vc.u) / sum), double((wa * va.v + wb * vb.v + wc * vc.v) / sum))) { continue; }
                    }
                    out->iz = iz;
                    out->triangle = std::uint32_t(t);
                    out->l1 = l1;
                    out->l2 = l2;
                }
                row0 += e0y * 256;
                row1 += e1y * 256;
                row2 += e2y * 256;
            }
        }
      }
    });
}

} // namespace tconv
