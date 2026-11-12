// Vertex layouts for the baked mesh: how many vertices each needs for a
// given closeness to the surface and a given smoothness of shading.
//
// Layouts: the three of src/raster.hpp (uniform, bounds, tree) and one more
// tried only here:
//   clusters  k-means over dense surface samples in position and normal
//             (so clusters shrink where the normal turns), run with the
//             mini-batch moving-average updates of e-means (Bessa et al.,
//             arXiv 2611.09501); each cluster gives the sample nearest its
//             centre as a vertex, and the triangles are the dual of the
//             clusters on the sample grid.
// Measures, over points spread inside every triangle:
//   distance  from the flat triangle to the surface at the same (u, v)
//   normal    angle between the interpolated vertex normal and the surface's
// and for the whole picture, against the ray-traced frame: pixels whose
// coverage differs (silhouette) and the mean colour difference.
#include "../src/image.hpp"
#include "../src/teapot.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
using namespace tconv;

namespace {

struct Quality {
    double dev_max = 0, dev_rms = 0, normal_max = 0, normal_rms = 0;
};

// The surface point nearest `flat`, starting from (u, v) and sliding along
// the surface (Gauss-Newton). v stays within `slack` rows of [v_low, v_high].
SurfacePoint nearest_on(const Surface& surface, Vec3 flat, double& u, double& v, double v_low, double v_high) {
    // The slide stays within a cell of where it started: further off, a thin
    // wall's other face or the far side of a pole can be nearer in space.
    const double u_low = u - 1.0, u_high = u + 1.0;
    v_low = std::max(v_low, v - 1.0);
    v_high = std::min(v_high, v + 1.0);
    // Nor does it cross a crease: the face beyond is another surface.
    if (const auto* conv = dynamic_cast<const ConvSurface*>(&surface)) {
        const auto& seams = conv->lattice.seams_v;
        for (std::size_t row = 1; row + 1 < seams.size(); ++row) {
            if (seams[row] == 0) { continue; }
            if (double(row) <= v) { v_low = std::max(v_low, double(row) + 1e-4); }
            if (double(row) > v) { v_high = std::min(v_high, double(row) - 1e-4); }
        }
    }
    SurfacePoint near = surface.eval(u, v);
    for (int iteration = 0; iteration < 6; ++iteration) {
        const Vec3 r = near.p - flat;
        const double e = dot(near.du, near.du), f = dot(near.du, near.dv), g = dot(near.dv, near.dv), det = e * g - f * f;
        if (det < 1e-14) { break; }
        const double ru = dot(r, near.du), rv = dot(r, near.dv);
        double step_u = -(g * ru - f * rv) / det, step_v = -(e * rv - f * ru) / det;
        const double size = std::max(std::fabs(step_u), std::fabs(step_v));
        if (size > 1.0) {
            step_u /= size;
            step_v /= size;
        }
        u = std::min(u_high, std::max(u_low, u + step_u));
        v = std::min(v_high, std::max(v_low, v + step_v));
        near = surface.eval(u, v);
        if (size < 1e-6) { break; }
    }
    return detail::mesh_point(surface, u, v);
}

Quality measure(const Mesh& mesh, const std::vector<const Surface*>& surfaces) {
    Quality q;
    double dev2 = 0, normal2 = 0;
    std::size_t count = 0;
    static const double points[7][3] = {{1 / 3.0, 1 / 3.0, 1 / 3.0}, {0.6, 0.2, 0.2}, {0.2, 0.6, 0.2}, {0.2, 0.2, 0.6}, {0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}};
    for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
        // The four sheets only. The join bands are small blends whose own
        // surface folds in places (the interior spout join), which is a fault
        // of that join and would be charged to whichever layout meets it.
        if (mesh.triangle_part[t] >= 4) { continue; }
        const Surface& surface = *surfaces[mesh.triangle_part[t]];
        const MeshVertex& a = mesh.vertices[mesh.triangles[t * 3]];
        const MeshVertex& b = mesh.vertices[mesh.triangles[t * 3 + 1]];
        const MeshVertex& c = mesh.vertices[mesh.triangles[t * 3 + 2]];
        // u of the three corners on one turn.
        const double cells = double(surface.cells_u());
        double ub = b.u, uc = c.u;
        if (surface.periodic_u()) {
            ub += std::round((double(a.u) - ub) / cells) * cells;
            uc += std::round((double(a.u) - uc) / cells) * cells;
        }
        if (length(cross(b.p - a.p, c.p - a.p)) < 1e-12) { continue; } // collapsed at a pole
        for (const auto& w : points) {
            const Vec3 flat = a.p * w[0] + b.p * w[1] + c.p * w[2];
            const double u = double(a.u) * w[0] + ub * w[1] + uc * w[2], v = double(a.v) * w[0] + double(b.v) * w[1] + double(c.v) * w[2];
            // The surface point nearest the flat one, on the triangle's own side
            // of a crease when an edge runs along one, and otherwise free to
            // slide a few rows: a triangle that spans unevenly spaced rows is
            // not charged for the unevenness.
            const double middle_v = (double(a.v) + double(b.v) + double(c.v)) / 3.0;
            double pu = u, pv = v + (middle_v - v) * 1e-3;
            const double v_min = std::min({double(a.v), double(b.v), double(c.v)}), v_max = std::max({double(a.v), double(b.v), double(c.v)});
            const bool on_crease = a.side != 0.0f || b.side != 0.0f || c.side != 0.0f;
            const SurfacePoint truth = nearest_on(surface, flat, pu, pv, on_crease ? v_min + 1e-4 : std::max(0.0, v_min - 3.0),
                                                  on_crease ? v_max - 1e-4 : std::min(double(surface.cells_v()), v_max + 3.0));
            const double d = length(flat - truth.p);
            const Vec3 n = normalize(a.n * w[0] + b.n * w[1] + c.n * w[2]);
            const double angle = std::acos(std::max(-1.0, std::min(1.0, dot(n, truth.du)))) * 180.0 / M_PI;
            if (std::getenv("WORST") != nullptr && (d > q.dev_max || angle > q.normal_max)) {
                std::fprintf(stderr, "worst so far: part %d dev %.3e angle %.1f | uv (%.2f %.2f) (%.2f %.2f) (%.2f %.2f) of cells %zu x %zu | p (%.2f %.2f %.2f)\n",
                             int(mesh.triangle_part[t]), d, angle, double(a.u), double(a.v), ub, double(b.v), uc, double(c.v), surface.cells_u(), surface.cells_v(), flat.x, flat.y, flat.z);
            }
            q.dev_max = std::max(q.dev_max, d);
            dev2 += d * d;
            q.normal_max = std::max(q.normal_max, angle);
            normal2 += angle * angle;
            ++count;
        }
    }
    q.dev_rms = std::sqrt(dev2 / double(std::max<std::size_t>(1, count)));
    q.normal_rms = std::sqrt(normal2 / double(std::max<std::size_t>(1, count)));
    return q;
}

// k-means over dense samples, e-means style; `share` of the samples become vertices.
Mesh cluster_mesh(const std::vector<const Surface*>& surfaces, std::size_t wanted, double normal_weight, int epochs) {
    Mesh mesh;
    // Vertices are shared out among the parts by their sample counts weighted by how much the normal turns.
    struct Sample {
        float f[6];
        float u, v;
    };
    const int per = 6;
    std::vector<std::vector<Sample>> samples(surfaces.size());
    std::vector<double> weight(surfaces.size(), 0.0);
    std::vector<std::vector<Vec3>> normals(surfaces.size());
    for (std::size_t k = 0; k < surfaces.size(); ++k) {
        const Surface& s = *surfaces[k];
        const long nu = long(s.cells_u()) * per + (s.periodic_u() ? 0 : 1), nv = long(s.cells_v()) * per + 1;
        for (long j = 0; j < nv; ++j) {
            for (long i = 0; i < nu; ++i) {
                const SurfacePoint point = detail::mesh_point(s, double(i) / per, double(j) / per);
                Sample sample;
                sample.f[0] = float(point.p.x);
                sample.f[1] = float(point.p.y);
                sample.f[2] = float(point.p.z);
                sample.f[3] = float(point.du.x * normal_weight);
                sample.f[4] = float(point.du.y * normal_weight);
                sample.f[5] = float(point.du.z * normal_weight);
                sample.u = float(double(i) / per);
                sample.v = float(double(j) / per);
                samples[k].push_back(sample);
                normals[k].push_back(point.du);
            }
        }
        // Spread in feature space along the grid: the part's share of vertices.
        for (long j = 0; j + 1 < nv; ++j) {
            for (long i = 0; i + 1 < nu; ++i) {
                const Sample &a = samples[k][std::size_t(j * nu + i)], &b = samples[k][std::size_t(j * nu + i + 1)], &c = samples[k][std::size_t((j + 1) * nu + i)];
                double du = 0, dv = 0;
                for (int d = 0; d < 6; ++d) {
                    du += double(b.f[d] - a.f[d]) * double(b.f[d] - a.f[d]);
                    dv += double(c.f[d] - a.f[d]) * double(c.f[d] - a.f[d]);
                }
                weight[k] += std::sqrt(du * dv);
            }
        }
    }
    double total = 0;
    for (double w : weight) { total += w; }
    std::uint32_t state = 2463534242u;
    const auto random = [&] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (std::size_t part_index = 0; part_index < surfaces.size(); ++part_index) {
        const Surface& s = *surfaces[part_index];
        const std::vector<Sample>& data = samples[part_index];
        const long nu = long(s.cells_u()) * per + (s.periodic_u() ? 0 : 1), nv = long(s.cells_v()) * per + 1;
        const std::size_t n = data.size();
        const std::size_t k = std::max<std::size_t>(8, std::min(n / 3, std::size_t(double(wanted) * weight[part_index] / total)));
        std::vector<float> centre(k * 6), sum(k * 6, 0.0f), mass(k, 0.0f);
        for (std::size_t c = 0; c < k; ++c) {
            // Start on a jittered regular pick of the samples.
            const std::size_t pick = std::min(n - 1, c * n / k + random() % std::max<std::size_t>(1, n / k));
            std::memcpy(&centre[c * 6], data[pick].f, sizeof(float) * 6);
        }
        std::vector<std::uint32_t> label(n, 0);
        const auto nearest = [&](const float* f) {
            std::size_t best = 0;
            float best_d = 1e30f;
            for (std::size_t c = 0; c < k; ++c) {
                const float* q = &centre[c * 6];
                float d = 0;
                for (int e = 0; e < 6; ++e) { d += (f[e] - q[e]) * (f[e] - q[e]); }
                if (d < best_d) {
                    best_d = d;
                    best = c;
                }
            }
            return best;
        };
        // Mini-batches with moving-average sums and counts; the rate follows a cosine from alpha to a tenth of it.
        const std::size_t batch = std::max<std::size_t>(256, std::min(n, 4 * k));
        const std::size_t steps = std::size_t(epochs) * n / batch;
        std::vector<float> batch_sum(k * 6), batch_mass(k);
        for (std::size_t step = 0; step < steps; ++step) {
            const double alpha0 = 0.5, progress = double(step) / double(std::max<std::size_t>(1, steps - 1));
            const float alpha = float(alpha0 * (0.1 + 0.9 * 0.5 * (1.0 + std::cos(M_PI * progress))));
            std::fill(batch_sum.begin(), batch_sum.end(), 0.0f);
            std::fill(batch_mass.begin(), batch_mass.end(), 0.0f);
            for (std::size_t b = 0; b < batch; ++b) {
                const Sample& sample = data[random() % n];
                const std::size_t c = nearest(sample.f);
                for (int e = 0; e < 6; ++e) { batch_sum[c * 6 + std::size_t(e)] += sample.f[e]; }
                batch_mass[c] += 1.0f;
            }
            for (std::size_t c = 0; c < k; ++c) {
                for (int e = 0; e < 6; ++e) { sum[c * 6 + std::size_t(e)] = (1 - alpha) * sum[c * 6 + std::size_t(e)] + alpha * batch_sum[c * 6 + std::size_t(e)]; }
                mass[c] = (1 - alpha) * mass[c] + alpha * batch_mass[c];
                if (mass[c] > 1e-6f) {
                    for (int e = 0; e < 6; ++e) { centre[c * 6 + std::size_t(e)] = sum[c * 6 + std::size_t(e)] / mass[c]; }
                }
            }
        }
        // Finish on exact means, then one more assignment.
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<double> exact(k * 6, 0.0), members(k, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                label[i] = std::uint32_t(nearest(data[i].f));
                for (int e = 0; e < 6; ++e) { exact[label[i] * 6 + std::size_t(e)] += double(data[i].f[e]); }
                members[label[i]] += 1.0;
            }
            if (pass == 1) { break; }
            for (std::size_t c = 0; c < k; ++c) {
                if (members[c] > 0) {
                    for (int e = 0; e < 6; ++e) { centre[c * 6 + std::size_t(e)] = float(exact[c * 6 + std::size_t(e)] / members[c]); }
                }
            }
        }
        // A cluster's vertex: its member nearest the centre.
        std::vector<long> member(k, -1);
        std::vector<float> member_d(k, 1e30f);
        for (std::size_t i = 0; i < n; ++i) {
            const float* q = &centre[label[i] * 6];
            float d = 0;
            for (int e = 0; e < 6; ++e) { d += (data[i].f[e] - q[e]) * (data[i].f[e] - q[e]); }
            if (d < member_d[label[i]]) {
                member_d[label[i]] = d;
                member[label[i]] = long(i);
            }
        }
        MeshPart part;
        part.first_vertex = mesh.vertices.size();
        part.first_triangle = mesh.triangle_count();
        std::vector<long> vertex_of(k, -1);
        for (std::size_t c = 0; c < k; ++c) {
            if (member[c] < 0) { continue; }
            MeshVertex vertex;
            const Sample& sample = data[std::size_t(member[c])];
            vertex.p = {double(sample.f[0]), double(sample.f[1]), double(sample.f[2])};
            vertex.n = normals[part_index][std::size_t(member[c])];
            vertex.u = sample.u;
            vertex.v = sample.v;
            vertex.s = vertex.t = 0.0f;
            vertex_of[c] = long(mesh.vertices.size());
            mesh.vertices.push_back(vertex);
        }
        // The dual: a grid square whose corners lie in three clusters is a
        // triangle, in four two triangles.
        std::set<std::array<std::uint32_t, 3>> seen;
        const auto emit = [&](long a, long b, long c) {
            if (a == b || b == c || a == c) { return; }
            std::array<std::uint32_t, 3> sorted{std::uint32_t(a), std::uint32_t(b), std::uint32_t(c)};
            std::sort(sorted.begin(), sorted.end());
            if (!seen.insert(sorted).second) { return; }
            mesh.triangles.push_back(std::uint32_t(a));
            mesh.triangles.push_back(std::uint32_t(b));
            mesh.triangles.push_back(std::uint32_t(c));
            mesh.triangle_part.push_back(std::uint16_t(part_index));
            mesh.triangle_trimmed.push_back(0);
        };
        const long columns = s.periodic_u() ? nu : nu - 1;
        for (long j = 0; j + 1 < nv; ++j) {
            for (long i = 0; i < columns; ++i) {
                const long i1 = (i + 1) % nu;
                const long a = vertex_of[label[std::size_t(j * nu + i)]], b = vertex_of[label[std::size_t(j * nu + i1)]];
                const long c = vertex_of[label[std::size_t((j + 1) * nu + i1)]], d = vertex_of[label[std::size_t((j + 1) * nu + i)]];
                std::set<long> distinct{a, b, c, d};
                if (distinct.size() == 4) {
                    emit(a, b, c);
                    emit(a, c, d);
                } else if (distinct.size() == 3) {
                    if (a == b) { emit(a, c, d); }
                    else if (b == c) { emit(a, b, d); }
                    else if (c == d) { emit(a, b, c); }
                    else { emit(a, b, c); }
                }
            }
        }
        part.vertex_count = mesh.vertices.size() - part.first_vertex;
        part.triangle_count = mesh.triangle_count() - part.first_triangle;
        part.arc = {0.0f, 1.0f};
        mesh.parts.push_back(part);
    }
    mesh.version = 1;
    return mesh;
}

// ---- a layout from the curvature metric, read from the lattice nodes ---------
//
// The metric. At every node the first and second derivatives of the surface
// are taken from the nodes alone, with the centred five-sample stencils CONV
// itself uses for its jets (three samples, or one-sided, next to an end or
// a crease). They give the first and second fundamental forms I and II and
// the shape operator W = I^-1 II. A step d in (u, v) then
//   strays from its chord by about |II|(d, d) / 8, and
//   turns the normal by about sqrt(III(d, d)),  III = II I^-1 II,
// so M = |II| / (8 e) + III / a^2 + I / L^2 measures a step in units of
// "one allowed edge": tolerance e, normal turn a, longest edge L. A mesh
// whose edges have length one in M is the mesh wanted, and it needs about
// Int sqrt(det M) du dv / (sqrt 3 / 4) triangles.
struct Metric {
    double a = 0, b = 0, c = 0;
    bool ok = false;
};

struct NodeForms {
    double E, F, G, L, M, N;
    bool ok;
};

// First and second derivative at index k of a line, from samples within [lo, hi].
template<class Get> void line_derivatives(Get get, long k, long lo, long hi, Vec3& d1, Vec3& d2) {
    if (k - 2 >= lo && k + 2 <= hi) {
        d1 = (get(k - 2) - get(k - 1) * 8.0 + get(k + 1) * 8.0 - get(k + 2)) * (1.0 / 12.0);
        d2 = (get(k - 2) * -1.0 + get(k - 1) * 16.0 - get(k) * 30.0 + get(k + 1) * 16.0 - get(k + 2)) * (1.0 / 12.0);
    } else if (k - 1 >= lo && k + 1 <= hi) {
        d1 = (get(k + 1) - get(k - 1)) * 0.5;
        d2 = get(k - 1) - get(k) * 2.0 + get(k + 1);
    } else if (k + 2 <= hi) {
        d1 = (get(k) * -3.0 + get(k + 1) * 4.0 - get(k + 2)) * 0.5;
        d2 = get(k) - get(k + 1) * 2.0 + get(k + 2);
    } else if (k - 2 >= lo) {
        d1 = (get(k) * 3.0 - get(k - 1) * 4.0 + get(k - 2)) * 0.5;
        d2 = get(k) - get(k - 1) * 2.0 + get(k - 2);
    } else {
        d1 = k + 1 <= hi ? get(k + 1) - get(k) : get(k) - get(k - 1);
        d2 = {};
    }
}

NodeForms forms_at(const Lattice& l, long i, long j, long lo, long hi) {
    const long nu = long(l.nu);
    const auto node = [&](long a, long b) { return l.at(std::size_t(((a % nu) + nu) % nu), std::size_t(b)); };
    Vec3 su, suu, sv, svv, sv_left, sv_right, unused;
    line_derivatives([&](long a) { return node(a, j); }, i, -1000000, 1000000, su, suu);
    line_derivatives([&](long b) { return node(i, b); }, j, lo, hi, sv, svv);
    line_derivatives([&](long b) { return node(i - 1, b); }, j, lo, hi, sv_left, unused);
    line_derivatives([&](long b) { return node(i + 1, b); }, j, lo, hi, sv_right, unused);
    const Vec3 suv = (sv_right - sv_left) * 0.5;
    NodeForms f;
    f.E = dot(su, su);
    f.F = dot(su, sv);
    f.G = dot(sv, sv);
    const Vec3 n = normalize(cross(su, sv));
    f.L = dot(suu, n);
    f.M = dot(suv, n);
    f.N = dot(svv, n);
    f.ok = f.E * f.G - f.F * f.F > 1e-10;
    return f;
}

Metric metric_of(const NodeForms& f, double tolerance, double angle, double longest) {
    Metric m;
    if (!f.ok) { return m; }
    const double det = f.E * f.G - f.F * f.F;
    // The shape operator and its absolute value.
    const double w00 = (f.G * f.L - f.F * f.M) / det, w01 = (f.G * f.M - f.F * f.N) / det;
    const double w10 = (f.E * f.M - f.F * f.L) / det, w11 = (f.E * f.N - f.F * f.M) / det;
    const double K = w00 * w11 - w01 * w10, trace = w00 + w11;
    double a00, a01, a10, a11;
    if (K >= 0.0) {
        const double sign = trace < 0.0 ? -1.0 : 1.0;
        a00 = sign * w00;
        a01 = sign * w01;
        a10 = sign * w10;
        a11 = sign * w11;
    } else {
        const double root = std::sqrt(0.25 * trace * trace - K), l1 = 0.5 * trace + root, l2 = 0.5 * trace - root;
        const double p = (std::fabs(l1) - std::fabs(l2)) / (l1 - l2), q = (std::fabs(l2) * l1 - std::fabs(l1) * l2) / (l1 - l2);
        a00 = p * w00 + q;
        a01 = p * w01;
        a10 = p * w10;
        a11 = p * w11 + q;
    }
    // |II| = I |W| and III = I W W, both symmetric.
    const double ii_a = f.E * a00 + f.F * a10, ii_b = f.E * a01 + f.F * a11, ii_c = f.F * a01 + f.G * a11;
    const double s00 = w00 * w00 + w01 * w10, s01 = w00 * w01 + w01 * w11, s10 = w10 * w00 + w11 * w10, s11 = w10 * w01 + w11 * w11;
    const double iii_a = f.E * s00 + f.F * s10, iii_b = f.E * s01 + f.F * s11, iii_c = f.F * s01 + f.G * s11;
    const double k1 = 1.0 / (8.0 * tolerance), k2 = 1.0 / (angle * angle), k3 = 1.0 / (longest * longest);
    m.a = k1 * ii_a + k2 * iii_a + k3 * f.E;
    m.b = k1 * ii_b + k2 * iii_b + k3 * f.F;
    m.c = k1 * ii_c + k2 * iii_c + k3 * f.G;
    m.ok = true;
    return m;
}

struct MetricOptions {
    double tolerance = 1e-2, angle_degrees = 20.0, longest = 1.0;
    bool nucleate = true; // seeds by farthest point in the metric; otherwise a jittered regular pick
    bool bounded = false;    // a seed's cell may not leave the lattice cells it belongs to: those round its node, or the one it was put in
    bool from_nodes = false; // seeds are the lattice's own nodes: CONV's rows already crowd where the shape is hard
    int epochs = 8;       // e-means epochs; 0: seeds are used where they were placed
    double density = 1.0; // multiplies the vertex count the metric asks for
    int refine = 0;       // rounds of checking every triangle against the surface and seeding where it fails
};

struct MetricReport {
    double root_k = 0, abs_k = 0; // Int sqrt|K| dA and Int |K| dA, from the nodes alone
};

Mesh metric_mesh(const std::vector<const ConvSurface*>& surfaces, const MetricOptions& options, MetricReport* report = nullptr) {
    Mesh mesh;
    const int per = 6;
    const double angle = options.angle_degrees * M_PI / 180.0;
    std::uint32_t state = 88172645u;
    const auto random = [&] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (std::size_t part_index = 0; part_index < surfaces.size(); ++part_index) {
        const ConvSurface& surface = *surfaces[part_index];
        const Lattice& l = surface.lattice;
        const long cu = long(l.cells_u()), cv = long(l.cells_v());
        const long nu = cu * per, nv = cv * per + 1; // samples; u closes on itself
        // Stretches of rows between cuts; a cut row is pinned.
        std::vector<long> cuts{0};
        for (long j = 1; j < cv; ++j) {
            if (!l.seams_v.empty() && l.seams_v[std::size_t(j)] != 0) { cuts.push_back(j); }
        }
        cuts.push_back(cv);
        std::vector<long> stretch_of_cell(static_cast<std::size_t>(cv));
        for (std::size_t k = 0; k + 1 < cuts.size(); ++k) {
            for (long j = cuts[k]; j < cuts[k + 1]; ++j) { stretch_of_cell[std::size_t(j)] = long(k); }
        }
        // Metric at every sample, from the corners of its cell.
        std::vector<Metric> metric(std::size_t(nu * nv));
        std::vector<long> stretch(std::size_t(nu * nv));
        for (long j = 0; j < cv; ++j) {
            const long lo = cuts[std::size_t(stretch_of_cell[std::size_t(j)])], hi = cuts[std::size_t(stretch_of_cell[std::size_t(j)]) + 1];
            for (long i = 0; i < cu; ++i) {
                NodeForms corner[4] = {forms_at(l, i, j, lo, hi), forms_at(l, i + 1, j, lo, hi), forms_at(l, i, j + 1, lo, hi), forms_at(l, i + 1, j + 1, lo, hi)};
                Metric m[4];
                for (int c = 0; c < 4; ++c) { m[c] = metric_of(corner[c], options.tolerance, angle, options.longest); }
                // A pole has no metric of its own: it takes the other row's.
                for (int c = 0; c < 4; ++c) {
                    if (!m[c].ok) { m[c] = m[c ^ 2]; }
                }
                if (report != nullptr) {
                    for (int c = 0; c < 4; ++c) {
                        if (!corner[c].ok) { continue; }
                        const double det = corner[c].E * corner[c].G - corner[c].F * corner[c].F;
                        const double K = (corner[c].L * corner[c].N - corner[c].M * corner[c].M) / det;
                        report->root_k += 0.25 * std::sqrt(std::fabs(K)) * std::sqrt(det);
                        report->abs_k += 0.25 * std::fabs(K) * std::sqrt(det);
                    }
                }
                const long last_b = j + 1 == cv ? per : per - 1;
                for (long b = 0; b <= last_b; ++b) {
                    for (long a = 0; a < per; ++a) {
                        const double fa = double(a) / per, fb = double(b) / per;
                        Metric& out = metric[std::size_t((j * per + b) * nu + i * per + a)];
                        const double w[4] = {(1 - fa) * (1 - fb), fa * (1 - fb), (1 - fa) * fb, fa * fb};
                        for (int c = 0; c < 4; ++c) {
                            out.a += w[c] * m[c].a;
                            out.b += w[c] * m[c].b;
                            out.c += w[c] * m[c].c;
                        }
                        out.ok = true;
                        stretch[std::size_t((j * per + b) * nu + i * per + a)] = stretch_of_cell[std::size_t(j)];
                    }
                }
            }
        }
        // Seeds. Pinned rows first: equal steps of metric length along the row.
        struct Seed {
            double u, v;  // sample units
            Metric m;
            long row;     // pinned to this sample row, or -1
            long stretch; // for a free seed
            // The box of samples its cell may reach (bounded cells): centre and half-width round u, a range of rows.
            double box_u = 0, box_half = 1e30;
            long box_low = -1000000000, box_high = 1000000000;
        };
        std::vector<Seed> seeds;
        std::vector<std::uint32_t> label(std::size_t(nu * nv), 0);
        std::vector<unsigned char> pinned_row(std::size_t(nv), 0);
        for (long cut : cuts) {
            const long row = cut * per;
            pinned_row[std::size_t(row)] = 1;
            // A pinned row is a curve in its own right (a crease, an end): a
            // chord of it strays by the curve's own curvature, whatever the
            // surface's normal curvature along it, so the row is measured by
            // the larger of the surface metric and the curve's.
            std::vector<double> along(std::size_t(nu) + 1, 0.0);
            std::vector<Vec3> on_row(static_cast<std::size_t>(nu));
            for (long i = 0; i < nu; ++i) { on_row[std::size_t(i)] = surface.position(double(i) / per, double(cut)); }
            for (long i = 0; i < nu; ++i) {
                const Vec3 before = on_row[std::size_t((i + nu - 1) % nu)], here = on_row[std::size_t(i)], after = on_row[std::size_t((i + 1) % nu)];
                const double step = length(after - here);
                double curve = 0.0;
                if (step > 1e-9 && length(here - before) > 1e-9) {
                    const double turn = std::acos(std::max(-1.0, std::min(1.0, dot(normalize(here - before), normalize(after - here)))));
                    const double curvature = turn / step;
                    curve = std::max(step * std::sqrt(curvature / (8.0 * options.tolerance)), turn / angle);
                }
                along[std::size_t(i) + 1] = along[std::size_t(i)] + std::max(std::sqrt(std::max(0.0, metric[std::size_t(row * nu + i)].a)), curve * per);
            }
            const double total = along.back() / per;
            const long count = total < 1.5 ? 1 : std::max(3L, long(std::lround(total * std::sqrt(options.density))));
            const std::size_t first = seeds.size();
            for (long k = 0; k < count; ++k) {
                const double want = along.back() * double(k) / double(count);
                const long i = std::min(nu - 1, long(std::upper_bound(along.begin(), along.end(), want) - along.begin()) - 1);
                seeds.push_back({double(std::max(0L, i)), double(row), metric[std::size_t(row * nu + std::max(0L, i))], row, -1});
                if (options.bounded) {
                    seeds.back().box_u = seeds.back().u;
                    seeds.back().box_half = double(nu); // a row seed reaches along its row; the rows beside it bound it
                    seeds.back().box_low = row - per;
                    seeds.back().box_high = row + per;
                }
            }
            // Row samples belong to the nearest row seed along the row.
            for (long i = 0; i < nu; ++i) {
                std::size_t best = first;
                double best_d = 1e30;
                for (std::size_t k = first; k < seeds.size(); ++k) {
                    double d = std::fabs(seeds[k].u - double(i));
                    d = std::min(d, double(nu) - d);
                    if (d < best_d) {
                        best_d = d;
                        best = k;
                    }
                }
                label[std::size_t(row * nu + i)] = std::uint32_t(best);
            }
        }
        // How many free seeds the metric asks for.
        double metric_area = 0.0;
        for (long j = 0; j < nv; ++j) {
            for (long i = 0; i < nu; ++i) {
                const Metric& m = metric[std::size_t(j * nu + i)];
                metric_area += std::sqrt(std::max(0.0, m.a * m.c - m.b * m.b));
            }
        }
        metric_area /= double(per * per);
        const long wanted = std::max(0L, long(options.density * 0.5 * metric_area / (std::sqrt(3.0) / 4.0)) - long(seeds.size()));
        // Cells are grown over the sample grid by shortest paths in the
        // metric (each sample joined to its eight neighbours), so a cell is
        // always one connected piece: nearest-seed cells under a metric that
        // varies can come apart, and their dual then folds. Pinned rows are
        // sources that nothing crosses.
        const long total = nu * nv;
        std::vector<double> reach(std::size_t(total), 1e30);
        std::vector<std::pair<double, long>> heap;
        const auto push = [&](double d, long sample) {
            heap.push_back({d, sample});
            std::push_heap(heap.begin(), heap.end(), std::greater<>());
        };
        const auto grow = [&] {
            while (!heap.empty()) {
                std::pop_heap(heap.begin(), heap.end(), std::greater<>());
                const auto [d, sample] = heap.back();
                heap.pop_back();
                if (d > reach[std::size_t(sample)]) { continue; }
                const long i = sample % nu, j = sample / nu;
                const Seed& owner = seeds[label[std::size_t(sample)]];
                const Metric& m = metric[std::size_t(sample)];
                for (long dj = -1; dj <= 1; ++dj) {
                    const long jj = j + dj;
                    if (jj < 0 || jj >= nv || pinned_row[std::size_t(jj)] != 0) { continue; }
                    for (long di = -1; di <= 1; ++di) {
                        if (di == 0 && dj == 0) { continue; }
                        const long other = jj * nu + ((i + di) % nu + nu) % nu;
                        if (options.bounded) {
                            double du = double(other % nu) - owner.box_u;
                            du -= double(nu) * std::round(du / double(nu));
                            if (std::fabs(du) > owner.box_half || jj < owner.box_low || jj > owner.box_high) { continue; }
                        }
                        const Metric& o = metric[std::size_t(other)];
                        const double a = 0.5 * (m.a + o.a), b = 0.5 * (m.b + o.b), c = 0.5 * (m.c + o.c);
                        const double step = std::sqrt(std::max(0.0, a * di * di + 2.0 * b * di * dj + c * dj * dj)) / per;
                        if (d + step < reach[std::size_t(other)]) {
                            reach[std::size_t(other)] = d + step;
                            label[std::size_t(other)] = label[std::size_t(sample)];
                            push(d + step, other);
                        }
                    }
                }
            }
        };
        std::vector<long> free_samples;
        for (long j = 0; j < nv; ++j) {
            for (long i = 0; i < nu; ++i) {
                if (pinned_row[std::size_t(j)] != 0) {
                    reach[std::size_t(j * nu + i)] = 0.0;
                    push(0.0, j * nu + i);
                } else {
                    free_samples.push_back(j * nu + i);
                }
            }
        }
        grow();
        const std::size_t first_free = seeds.size();
        std::vector<long> seed_sample(seeds.size(), -1);
        std::size_t unreached_report = 0;
        // `at_node`: the seed is a lattice node and may reach the cells round
        // it; otherwise it was put inside a cell and stays in that cell.
        const auto plant = [&](long sample, bool at_node = false) {
            seeds.push_back({double(sample % nu), double(sample / nu), metric[std::size_t(sample)], -1, stretch[std::size_t(sample)]});
            seed_sample.push_back(sample);
            if (options.bounded) {
                Seed& seed = seeds.back();
                const long i = sample % nu, j = sample / nu;
                if (at_node) {
                    seed.box_u = double(i);
                    seed.box_half = double(per);
                    seed.box_low = j - per;
                    seed.box_high = j + per;
                } else {
                    seed.box_u = double((i / per) * per) + 0.5 * per;
                    seed.box_half = 0.5 * per;
                    seed.box_low = (std::min(j, nv - 2) / per) * per;
                    seed.box_high = seed.box_low + per;
                }
            }
        };
        if (options.from_nodes) {
            for (long j = 1; j < cv; ++j) {
                if (pinned_row[std::size_t(j * per)] != 0) { continue; }
                for (long i = 0; i < cu; ++i) { plant(j * per * nu + i * per, true); }
            }
        } else if (options.nucleate) {
            // Nucleation: each new seed goes where the surface is furthest, in
            // the metric, from every seed so far (the bottom of the deepest
            // basin still uncovered), and its cell grows from there.
            for (long added = 0; added < wanted && !free_samples.empty(); ++added) {
                long far = free_samples[0];
                for (const long sample : free_samples) {
                    if (reach[std::size_t(sample)] > reach[std::size_t(far)]) { far = sample; }
                }
                plant(far);
                reach[std::size_t(far)] = 0.0;
                label[std::size_t(far)] = std::uint32_t(seeds.size() - 1);
                push(0.0, far);
                grow();
            }
        } else {
            std::set<long> taken;
            for (long added = 0; added < wanted && !free_samples.empty(); ++added) {
                const long pick = free_samples[std::min(free_samples.size() - 1, std::size_t(added) * free_samples.size() / std::size_t(wanted) +
                                                                                     random() % std::max<std::size_t>(1, free_samples.size() / std::size_t(wanted)))];
                if (taken.insert(pick).second) { plant(pick); }
            }
        }
        // All cells again from the seeds as they stand.
        const auto regrow = [&] {
            for (const long sample : free_samples) { reach[std::size_t(sample)] = 1e30; }
            for (long j = 0; j < nv; ++j) {
                if (pinned_row[std::size_t(j)] == 0) { continue; }
                for (long i = 0; i < nu; ++i) { push(0.0, j * nu + i); }
            }
            for (std::size_t c = first_free; c < seeds.size(); ++c) {
                reach[std::size_t(seed_sample[c])] = 0.0;
                label[std::size_t(seed_sample[c])] = std::uint32_t(c);
                push(0.0, seed_sample[c]);
            }
            grow();
        };
        regrow();
        // e-means: every seed moves toward the mean of its cell, the sums and
        // counts kept as moving averages with a cosine rate. The batch here is
        // the whole sheet, because the cells come from one sweep of the grid.
        if (seeds.size() > first_free) {
            const std::size_t k = seeds.size();
            std::vector<double> sum_u(k, 0.0), sum_v(k, 0.0), mass(k, 0.0), batch_u(k), batch_v(k), batch_mass(k), want_u(k), want_v(k), best(k);
            for (int epoch = 0; epoch < options.epochs; ++epoch) {
                const double progress = options.epochs > 1 ? double(epoch) / double(options.epochs - 1) : 0.0;
                const double alpha = 0.1 + 0.9 * 0.5 * (1.0 + std::cos(M_PI * progress));
                std::fill(batch_u.begin(), batch_u.end(), 0.0);
                std::fill(batch_v.begin(), batch_v.end(), 0.0);
                std::fill(batch_mass.begin(), batch_mass.end(), 0.0);
                for (const long sample : free_samples) {
                    const std::size_t c = label[std::size_t(sample)];
                    if (c < first_free) { continue; }
                    double du = double(sample % nu) - seeds[c].u;
                    du -= double(nu) * std::round(du / double(nu));
                    // Weighted by metric area, so the mean is the cell's centre of work.
                    const Metric& m = metric[std::size_t(sample)];
                    const double weight = std::sqrt(std::max(1e-12, m.a * m.c - m.b * m.b));
                    batch_u[c] += weight * du;
                    batch_v[c] += weight * (double(sample / nu) - seeds[c].v);
                    batch_mass[c] += weight;
                }
                for (std::size_t c = first_free; c < k; ++c) {
                    sum_u[c] = (1 - alpha) * sum_u[c] + alpha * batch_u[c];
                    sum_v[c] = (1 - alpha) * sum_v[c] + alpha * batch_v[c];
                    mass[c] = (1 - alpha) * mass[c] + alpha * batch_mass[c];
                    want_u[c] = seeds[c].u + (mass[c] > 1e-12 ? sum_u[c] / mass[c] : 0.0);
                    want_v[c] = seeds[c].v + (mass[c] > 1e-12 ? sum_v[c] / mass[c] : 0.0);
                    best[c] = 1e30;
                }
                // Each seed onto the sample of its own cell nearest where it wants to be.
                std::vector<long> next(seed_sample);
                for (const long sample : free_samples) {
                    const std::size_t c = label[std::size_t(sample)];
                    if (c < first_free) { continue; }
                    double du = double(sample % nu) - want_u[c];
                    du -= double(nu) * std::round(du / double(nu));
                    const double dv = double(sample / nu) - want_v[c];
                    if (du * du + dv * dv < best[c]) {
                        best[c] = du * du + dv * dv;
                        next[c] = sample;
                    }
                }
                for (std::size_t c = first_free; c < k; ++c) {
                    double moved_u = double(next[c] % nu) - seeds[c].u;
                    moved_u -= double(nu) * std::round(moved_u / double(nu));
                    const double moved_v = double(next[c] / nu) - seeds[c].v;
                    sum_u[c] -= moved_u * mass[c]; // the sums are relative to the seed's place: carry them along
                    sum_v[c] -= moved_v * mass[c];
                    seed_sample[c] = next[c];
                    seeds[c].u = double(next[c] % nu);
                    seeds[c].v = double(next[c] / nu);
                    seeds[c].m = metric[std::size_t(next[c])];
                }
                regrow();
            }
        }
        // Vertices and triangles from the cells as they stand. A seed on a
        // crease row is two vertices, one per side; the triangles are the
        // dual of the cells on the sample grid.
        struct Built {
            std::vector<MeshVertex> vertices;
            std::vector<std::array<long, 3>> triangles;
        };
        std::size_t evaluations = 0;
        const auto build = [&] {
            Built out;
            std::vector<long> vertex_of(seeds.size() * 2, -1);
            const auto make = [&](double su, double sv, double side) {
                MeshVertex vertex;
                const double u = su / per, v = sv / per;
                const SurfacePoint point = detail::mesh_point(surface, u, v);
                vertex.p = point.p;
                vertex.n = side == 0.0 ? point.du : detail::mesh_point(surface, u, v + side * 1e-4).du;
                vertex.u = float(u);
                vertex.v = float(v);
                vertex.s = vertex.t = 0.0f;
                vertex.side = float(side);
                out.vertices.push_back(vertex);
                return long(out.vertices.size()) - 1;
            };
            for (std::size_t c = 0; c < seeds.size(); ++c) {
                if (seeds[c].row >= 0) {
                    const bool crease = seeds[c].row != 0 && seeds[c].row != nv - 1;
                    vertex_of[c * 2] = make(seeds[c].u, seeds[c].v, crease ? -1.0 : 0.0);
                    vertex_of[c * 2 + 1] = crease ? make(seeds[c].u, seeds[c].v, 1.0) : vertex_of[c * 2];
                } else {
                    vertex_of[c * 2] = vertex_of[c * 2 + 1] = make(double(seed_sample[c] % nu), double(seed_sample[c] / nu), 0.0);
                }
            }
            std::set<std::array<long, 3>> seen;
            const auto emit = [&](long a, long b, long c) {
                if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c) { return; }
                std::array<long, 3> sorted{a, b, c};
                std::sort(sorted.begin(), sorted.end());
                if (seen.insert(sorted).second) { out.triangles.push_back({a, b, c}); }
            };
            for (long j = 0; j + 1 < nv; ++j) {
                for (long i = 0; i < nu; ++i) {
                    const long i1 = (i + 1) % nu;
                    // A square above a pinned row reads that row's upper vertices; below, its lower ones.
                    const auto at = [&](long si, long sj) { return vertex_of[std::size_t(label[std::size_t(sj * nu + si)]) * 2 + (sj == j ? 1 : 0)]; };
                    const long a = at(i, j), b = at(i1, j), c = at(i1, j + 1), d = at(i, j + 1);
                    const std::set<long> distinct{a, b, c, d};
                    if (distinct.size() == 4) {
                        emit(a, b, c);
                        emit(a, c, d);
                    } else if (distinct.size() == 3) {
                        if (a == b) { emit(a, c, d); }
                        else if (b == c) { emit(a, b, d); }
                        else if (c == d) { emit(a, b, c); }
                        else if (d == a) { emit(a, b, c); }
                        else if (a == c) { emit(a, b, d); }
                        else { emit(a, b, c); }
                    }
                }
            }
            return out;
        };
        for (const long sample : free_samples) {
            if (reach[std::size_t(sample)] > 1e29) { ++unreached_report; }
        }
        if (unreached_report > 0) { std::fprintf(stderr, "part %zu: %zu samples no seed may reach\n", part_index, unreached_report); }
        Built built = build();
        // Check and refine. The metric is second order: it cannot see a chord
        // that leaves a line of zero curvature sideways (the top of a lip, a
        // disc near a pole). So every triangle is tried against the surface
        // at its middle and the middles of its edges, and one that fails gets
        // a seed where it fails worst.
        std::set<long> is_seed(seed_sample.begin() + long(first_free), seed_sample.end());
        for (int round = 0; round < (part_index < 4 ? options.refine : 0); ++round) {
            std::set<long> add;
            static const double probes[4][3] = {{1 / 3.0, 1 / 3.0, 1 / 3.0}, {0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}};
            for (const auto& t : built.triangles) {
                const MeshVertex &a = built.vertices[std::size_t(t[0])], &b = built.vertices[std::size_t(t[1])], &c = built.vertices[std::size_t(t[2])];
                if (length(cross(b.p - a.p, c.p - a.p)) < 1e-12) { continue; }
                double ub = b.u, uc = c.u;
                ub += std::round((double(a.u) - ub) / double(cu)) * double(cu);
                uc += std::round((double(a.u) - uc) / double(cu)) * double(cu);
                const double v_min = std::min({double(a.v), double(b.v), double(c.v)}), v_max = std::max({double(a.v), double(b.v), double(c.v)});
                const bool on_crease = a.side != 0.0f || b.side != 0.0f || c.side != 0.0f;
                // Where it fails, worst first; the seed goes to the first of those places that is free.
                std::vector<std::array<double, 3>> failing;
                for (const auto& w : probes) {
                    const Vec3 flat = a.p * w[0] + b.p * w[1] + c.p * w[2];
                    double pu = double(a.u) * w[0] + ub * w[1] + uc * w[2], pv = double(a.v) * w[0] + double(b.v) * w[1] + double(c.v) * w[2];
                    pv += ((double(a.v) + double(b.v) + double(c.v)) / 3.0 - pv) * 1e-3;
                    const double start_u = pu, start_v = pv;
                    const SurfacePoint truth = nearest_on(surface, flat, pu, pv, on_crease ? v_min + 1e-4 : std::max(0.0, v_min - 3.0),
                                                          on_crease ? v_max - 1e-4 : std::min(double(cv), v_max + 3.0));
                    evaluations += 4;
                    const Vec3 n = normalize(a.n * w[0] + b.n * w[1] + c.n * w[2]);
                    const double turn = std::acos(std::max(-1.0, std::min(1.0, dot(n, truth.du)))) * 180.0 / M_PI;
                    const double score = std::max(length(flat - truth.p) / (2.0 * options.tolerance), turn / (0.5 * options.angle_degrees));
                    if (score > 1.0) { failing.push_back({score, start_u, start_v}); }
                }
                std::sort(failing.begin(), failing.end(), [](const auto& x, const auto& y) { return x[0] > y[0]; });
                for (const auto& place : failing) {
                    bool placed = false;
                    // The sample at that place, or one of its neighbours.
                    for (long dj : {0L, 1L, -1L}) {
                        for (long di : {0L, 1L, -1L}) {
                            const long i = ((long(std::lround(place[1] * per)) + di) % nu + nu) % nu, j = long(std::lround(place[2] * per)) + dj;
                            if (placed || j < 0 || j >= nv || pinned_row[std::size_t(j)] != 0 || is_seed.count(j * nu + i) != 0 || add.count(j * nu + i) != 0) { continue; }
                            add.insert(j * nu + i);
                            placed = true;
                        }
                    }
                    if (placed) { break; }
                }
            }
            if (add.empty()) { break; }
            for (const long sample : add) {
                plant(sample);
                is_seed.insert(sample);
                reach[std::size_t(sample)] = 0.0;
                label[std::size_t(sample)] = std::uint32_t(seeds.size() - 1);
                push(0.0, sample);
                grow();
            }
            built = build();
        }
        MeshPart part;
        part.first_vertex = mesh.vertices.size();
        part.first_triangle = mesh.triangle_count();
        for (const MeshVertex& vertex : built.vertices) { mesh.vertices.push_back(vertex); }
        for (const auto& t : built.triangles) {
            for (int k = 0; k < 3; ++k) { mesh.triangles.push_back(std::uint32_t(long(part.first_vertex) + t[std::size_t(k)])); }
            mesh.triangle_part.push_back(std::uint16_t(part_index));
            mesh.triangle_trimmed.push_back(0);
        }
        mesh.evaluations += built.vertices.size() + evaluations;
        part.vertex_count = mesh.vertices.size() - part.first_vertex;
        part.triangle_count = mesh.triangle_count() - part.first_triangle;
        part.arc = {0.0f, 1.0f};
        mesh.parts.push_back(part);
    }
    mesh.version = 1;
    return mesh;
}

struct Picture {
    std::size_t coverage = 0; // pixels covered in one frame and not the other
    double colour = 0;        // mean absolute difference, 0..255
    double ms = 0;
};

} // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "quick";
    TeapotOptions o;
    o.version = Version::v2026_unified;
    Teapot teapot = make_teapot(o);
    Scene scene;
    std::vector<long> scene_index(teapot.parts.size(), -1);
    for (std::size_t k = 0; k < teapot.parts.size(); ++k) {
        RenderPart part;
        part.surface = &teapot.parts[k]->conv;
        part.orientation = teapot.parts[k]->orientation;
        part.per_u = part.per_v = 0;
        part.two_sided = true;
        scene.parts.push_back(part);
    }
    for (const TeapotJoin& join : teapot.joins) {
        scene.parts[join.body_part].trims.push_back(&join.body_trim);
        scene.parts[join.tube_part].trims.push_back(&join.tube_trim);
    }
    std::vector<const Surface*> surfaces;
    std::vector<std::vector<const Trim*>> trims, no_trims;
    for (const RenderPart& part : scene.parts) {
        surfaces.push_back(part.surface);
        trims.push_back(part.trims);
        no_trims.push_back({});
    }
    const int width = 960, height = 600;
    Camera camera;
    const double azimuth = 68.0 * M_PI / 180.0, elevation = 22.0 * M_PI / 180.0;
    camera.target = {0.15, 1.35, 0.0};
    camera.eye = camera.target + Vec3{std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth)} * 11.0;
    camera.fov_y = 34.0 * M_PI / 180.0;
    RenderOptions options;
    Frame reference, frame;
    reference.resize(width, height);
    frame.resize(width, height);
    Renderer renderer;
    std::vector<double> times;
    for (int k = 0; k < 5; ++k) {
        renderer.render(scene, camera, options, reference);
        times.push_back(renderer.stats.ms_total);
    }
    std::sort(times.begin(), times.end());
    const double reference_ms = times[2];
    RenderOptions one = options;
    one.threads = 1;
    renderer.render(scene, camera, one, frame);
    const double reference_ms_one = renderer.stats.ms_total;

    const auto picture = [&](Mesh& mesh, bool lit, int threads) {
        Picture p;
        scene.mesh = &mesh;
        RenderOptions raster = options;
        raster.method = Method::raster;
        raster.threads = threads;
        raster.baked_light = lit;
        std::vector<double> ms;
        for (int k = 0; k < 5; ++k) {
            renderer.render(scene, camera, raster, frame);
            ms.push_back(renderer.stats.ms_total);
        }
        std::sort(ms.begin(), ms.end());
        p.ms = ms[2];
        double sum = 0;
        for (std::size_t k = 0; k < frame.hits.size(); ++k) {
            if ((frame.hits[k].t > 0.0f) != (reference.hits[k].t > 0.0f)) { ++p.coverage; }
            for (int c = 0; c < 3; ++c) { sum += std::fabs(double(frame.rgba[k * 4 + std::size_t(c)]) - double(reference.rgba[k * 4 + std::size_t(c)])); }
        }
        p.colour = sum / double(frame.hits.size() * 3);
        scene.mesh = nullptr;
        return p;
    };

    std::printf("Unified teapot, %zu nodes. Ray-traced reference at %dx%d: %.1f ms (all threads), %.1f ms (one thread).\n", teapot.node_count(), width, height,
                reference_ms, reference_ms_one);
    std::printf("A pixel at the teapot is about 0.011 model units.\n\n");
    std::printf("%-34s %8s %9s %8s %8s | %9s %9s | %8s %8s | %9s %7s | %7s %7s\n", "layout", "vertices", "triangles", "evals", "build ms", "dev max", "dev rms", "nrm max",
                "nrm rms", "coverage", "colour", "ms all", "ms one");
    const auto row = [&](const std::string& name, Mesh& mesh) {
        const Quality q = measure(mesh, surfaces);
        const Picture all = picture(mesh, false, 0), single = picture(mesh, false, 1);
        std::printf("%-34s %8zu %9zu %8zu %8.1f | %9.2e %9.2e | %8.2f %8.3f | %9zu %7.3f | %7.2f %7.2f\n", name.c_str(), mesh.vertices.size(), mesh.triangle_count(),
                    mesh.evaluations, mesh.ms_build, q.dev_max, q.dev_rms, q.normal_max, q.normal_rms, all.coverage, all.colour, all.ms, single.ms);
        std::fflush(stdout);
    };
    const auto built = [&](MeshOptions m) {
        const double start = detail::now_ms();
        Mesh mesh = build_mesh(surfaces, trims, m);
        mesh.ms_build = detail::now_ms() - start;
        return mesh;
    };
    if (argc > 1 && std::string(argv[1]) == "metric") {
        std::vector<const ConvSurface*> sheets;
        for (const auto& part : teapot.parts) { sheets.push_back(&part->conv); }
        MetricReport report;
        {
            MetricOptions probe;
            probe.density = 0.05;
            probe.epochs = 0;
            metric_mesh(sheets, probe, &report);
        }
        std::printf("From the lattice nodes alone: Int sqrt|K| dA = %.2f, Int |K| dA = %.2f (by differences of the evaluated surface: 76.57 and 167.05)\n\n", report.root_k,
                    report.abs_k);
        std::printf("%-34s %8s %9s %8s %8s | %9s %9s | %8s %8s | %9s %7s | %7s %7s\n", "layout", "vertices", "triangles", "evals", "build ms", "dev max", "dev rms",
                    "nrm max", "nrm rms", "coverage", "colour", "ms all", "ms one");
        for (const auto& [tolerance, angle] : std::vector<std::pair<double, double>>{{2e-2, 20.0}, {1e-2, 20.0}, {5e-3, 10.0}}) {
            char name[64];
            MeshOptions m;
            m.tolerance = tolerance;
            m.angle = angle;
            Mesh tree = built(m);
            std::snprintf(name, sizeof name, "tree %.0e + %.0f deg", tolerance, angle);
            row(name, tree);
        }
        struct Case {
            const char* name;
            MetricOptions o;
        };
        const auto with = [](double tolerance, double angle, double density, bool nucleate, int epochs, int refine) {
            MetricOptions o;
            o.tolerance = tolerance;
            o.angle_degrees = angle;
            o.density = density;
            o.nucleate = nucleate;
            o.epochs = epochs;
            o.refine = refine;
            return o;
        };
        const auto nodes = [&](double tolerance, double angle, int epochs, int refine) {
            MetricOptions o = with(tolerance, angle, 1, false, epochs, refine);
            o.from_nodes = true;
            return o;
        };
        const auto bounded = [&](double tolerance, double angle, int refine) {
            MetricOptions o = nodes(tolerance, angle, 0, refine);
            o.bounded = true;
            return o;
        };
        const std::vector<Case> cases{
            {"metric 2e-2 20deg, no check", with(2e-2, 20, 1, true, 8, 0)},
            {"metric 2e-2 20deg, checked", with(2e-2, 20, 1, true, 8, 8)},
            {"metric 2e-2 20deg, checked x24", with(2e-2, 20, 1, true, 8, 24)},
            {"  nucleate only, checked", with(2e-2, 20, 1, true, 0, 8)},
            {"  random + emeans, checked", with(2e-2, 20, 1, false, 8, 8)},
            {"  random only, checked", with(2e-2, 20, 1, false, 0, 8)},
            {"  half the seeds, checked", with(2e-2, 20, 0.5, true, 8, 8)},
            {"  lattice nodes, no check", nodes(2e-2, 20, 0, 0)},
            {"  lattice nodes, checked", nodes(2e-2, 20, 0, 8)},
            {"  lattice nodes + emeans, checked", nodes(2e-2, 20, 8, 8)},
            {"  nodes, checked, 1e-2 20deg", nodes(1e-2, 20, 0, 8)},
            {"  nodes bounded, no check", bounded(2e-2, 20, 0)},
            {"  nodes bounded, checked 2e-2 20deg", bounded(2e-2, 20, 8)},
            {"  nodes bounded, checked 1e-2 20deg", bounded(1e-2, 20, 8)},
            {"  nodes bounded, checked 5e-3 10deg", bounded(5e-3, 10, 8)},
            {"  nodes, checked, 5e-3 10deg", nodes(5e-3, 10, 0, 8)},
            {"metric 4e-2 30deg, checked", with(4e-2, 30, 1, true, 8, 8)},
            {"metric 1e-2 20deg, checked", with(1e-2, 20, 1, true, 8, 8)},
            {"metric 1e-2 15deg, checked", with(1e-2, 15, 1, true, 8, 8)},
            {"metric 5e-3 10deg, checked", with(5e-3, 10, 1, true, 8, 8)},
        };
        for (const Case& c : cases) {
            const double start = detail::now_ms();
            Mesh mesh = metric_mesh(sheets, c.o);
            mesh.ms_build = detail::now_ms() - start;
            row(c.name, mesh);
        }
        return 0;
    }
    for (int cuts : {1, 2, 4, 8}) {
        MeshOptions m;
        m.layout = Layout::uniform;
        m.uniform = cuts;
        Mesh mesh = built(m);
        row("uniform " + std::to_string(cuts) + "x" + std::to_string(cuts), mesh);
    }
    for (double tolerance : {2e-2, 1e-2, 5e-3, 2e-3, 1e-3}) {
        char name[64];
        MeshOptions m;
        m.tolerance = tolerance;
        m.layout = Layout::bounds;
        Mesh a = built(m);
        std::snprintf(name, sizeof name, "bounds %.0e", tolerance);
        row(name, a);
    }
    for (double tolerance : {2e-2, 1e-2, 5e-3, 2e-3, 1e-3}) {
        char name[64];
        MeshOptions m;
        m.tolerance = tolerance;
        m.layout = Layout::tree;
        Mesh b = built(m);
        std::snprintf(name, sizeof name, "tree %.0e", tolerance);
        row(name, b);
    }
    // Distance alone lets a tight bend be one flat piece; a limit on the turn of the normal along an edge does not.
    for (const Layout layout : {Layout::bounds, Layout::tree}) {
        for (const auto& [tolerance, angle] : std::vector<std::pair<double, double>>{{5e-2, 30.0}, {2e-2, 30.0}, {2e-2, 20.0}, {1e-2, 20.0}, {1e-2, 15.0}, {5e-3, 15.0}, {5e-3, 10.0}, {2e-3, 10.0}, {2e-3, 5.0}}) {
            char name[64];
            MeshOptions m;
            m.tolerance = tolerance;
            m.layout = layout;
            m.angle = angle;
            Mesh b = built(m);
            std::snprintf(name, sizeof name, "%s %.0e + %.0f deg", layout == Layout::tree ? "tree" : "bounds", tolerance, angle);
            row(name, b);
        }
    }
    if (!quick) {
        for (std::size_t wanted : {3000u, 6000u, 12000u, 24000u}) {
            for (double normal_weight : {0.3, 1.0}) {
                const double start = detail::now_ms();
                Mesh mesh = cluster_mesh(surfaces, wanted, normal_weight, 8);
                mesh.ms_build = detail::now_ms() - start;
                char name[64];
                std::snprintf(name, sizeof name, "clusters %zu w%.1f", wanted, normal_weight);
                row(name, mesh);
            }
        }
    }

    // Light maps: the lit picture against the ray-traced one, and what a frame costs.
    std::printf("\nLight maps on tree 5e-03 (texel size in model units): texels, bake ms, colour difference, frame ms (all threads / one)\n");
    for (double texel : {0.032, 0.016, 0.008}) {
        MeshOptions m;
        m.tolerance = 5e-3;
        Mesh mesh = built(m);
        scene.mesh = &mesh;
        renderer.bake_light(scene, camera, options, width, height, texel);
        scene.mesh = nullptr;
        const Picture all = picture(mesh, true, 0), single = picture(mesh, true, 1);
        std::printf("  texel %.3f: %8zu texels, bake %7.1f ms, colour %.3f, frame %.2f / %.2f ms\n", texel, mesh.texels, mesh.ms_light, all.colour, all.ms, single.ms);
    }
    return 0;
}
