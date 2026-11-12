// Node budget: how good a surface each representation makes from the same
// number of nodes.
//
// Every method is given the same lattice of points that lie on the target
// surface and must produce a surface from them. Three targets, none of them
// native to any of the methods:
//   vase   an analytic surface of revolution (trigonometric profile)
//   tube   half of a torus, a bent tube
//   body   the 2026 teapot body with its interior (dense sampled profile,
//          one crease at the rim, two poles)
// Methods:
//   facets        the nodes joined bilinearly (a polygon mesh)
//   catmull-rom   bicubic through the nodes, local, tangent-continuous
//   b-spline raw  uniform bicubic B-spline with the nodes as control points;
//                 on a regular grid this is also the Catmull-Clark limit
//                 surface of the node mesh
//   spline fit    the bicubic B-spline solved to pass through the nodes
//                 (the C2 interpolating spline; a global solve per line)
//   conv          CONV
// At a crease every method is cut, as CONV is by a seam.
#include "../src/teapot.hpp"
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
using namespace tconv;

namespace {

const double pi = 3.14159265358979323846;

struct Target {
    std::string name;
    // Point at `around` in [0,1) and `along` in [0,1].
    std::function<Vec3(double, double)> point;
    // Distance from p to the surface and the surface normal there; `along`
    // is where to look.
    std::function<double(Vec3, double, double, Vec3&)> nearest; // p, along, window, normal
    std::function<std::vector<double>(std::size_t)> rows_for; // `along` of every row
    std::function<std::vector<unsigned char>(std::size_t)> seams_for;
};

// A surface of revolution about y through a dense table of its profile.
struct Revolved {
    std::vector<double> r, y, tr, ty; // points and unit tangents
    void finish() {
        const std::size_t n = r.size();
        tr.resize(n);
        ty.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t a = k == 0 ? 0 : k - 1, b = k + 1 == n ? k : k + 1;
            const double dr = r[b] - r[a], dy = y[b] - y[a], len = std::hypot(dr, dy);
            tr[k] = dr / len;
            ty[k] = dy / len;
        }
    }
    double nearest(Vec3 p, double along, double window, Vec3& normal) const {
        const double rho = std::hypot(p.x, p.z);
        const long n = long(r.size());
        const long centre = long(along * double(n - 1)), reach = std::max(60L, long(window * double(n - 1)));
        double best = 1e30, best_t = 0.0;
        long best_k = 0;
        for (long k = std::max(0L, centre - reach); k < std::min(n - 1, centre + reach); ++k) {
            const double bx = r[std::size_t(k + 1)] - r[std::size_t(k)], by = y[std::size_t(k + 1)] - y[std::size_t(k)];
            const double len2 = bx * bx + by * by;
            const double t = len2 > 0.0 ? std::max(0.0, std::min(1.0, ((rho - r[std::size_t(k)]) * bx + (p.y - y[std::size_t(k)]) * by) / len2)) : 0.0;
            const double d = std::hypot(r[std::size_t(k)] + t * bx - rho, y[std::size_t(k)] + t * by - p.y);
            if (d < best) {
                best = d;
                best_k = k;
                best_t = t;
            }
        }
        const std::size_t k = std::size_t(best_k);
        double ar = tr[k] + (tr[k + 1] - tr[k]) * best_t, ay = ty[k] + (ty[k + 1] - ty[k]) * best_t;
        const double len = std::hypot(ar, ay);
        ar /= len;
        ay /= len;
        // The normal is the tangent turned a quarter, carried round the axis.
        const double c = rho > 1e-12 ? p.x / rho : 1.0, s = rho > 1e-12 ? p.z / rho : 0.0;
        normal = {ay * c, -ar, ay * s};
        return best;
    }
};

// One line of values under each method. `lo` and `hi` bound the stretch
// between cuts that holds the cell.
enum class Method { facets, catmull_rom, bspline_raw, spline_fit, conv };
const char* const method_name[] = {"facets", "catmull-rom", "b-spline raw", "spline fit", "conv"};

template<class Get> Vec3 local_line(Method method, long cell, double t, long lo, long hi, Get get) {
    if (method == Method::facets) { return get(cell) * (1.0 - t) + get(cell + 1) * t; }
    const auto at = [&](long k) {
        if (k < lo) { return method == Method::bspline_raw || hi - lo < 2 ? get(lo) * 2.0 - get(lo + 1) : get(lo) * 3.0 - get(lo + 1) * 3.0 + get(lo + 2); }
        if (k > hi) { return method == Method::bspline_raw || hi - lo < 2 ? get(hi) * 2.0 - get(hi - 1) : get(hi) * 3.0 - get(hi - 1) * 3.0 + get(hi - 2); }
        return get(k);
    };
    const Vec3 a = at(cell - 1), b = at(cell), c = at(cell + 1), d = at(cell + 2);
    const double t2 = t * t, t3 = t2 * t;
    if (method == Method::catmull_rom) {
        return (a * (-t3 + 2 * t2 - t) + b * (3 * t3 - 5 * t2 + 2) + c * (-3 * t3 + 4 * t2 + t) + d * (t3 - t2)) * 0.5;
    }
    const double s = 1.0 - t;
    return (a * (s * s * s) + b * (3 * t3 - 6 * t2 + 4) + c * (-3 * t3 + 3 * t2 + 3 * t + 1) + d * t3) * (1.0 / 6.0);
}

// Second derivatives of the C2 cubic through uniformly spaced values, with
// not-a-knot ends. Fewer than four values give a parabola or a line.
void spline_open(const Vec3* value, long n, Vec3* m) {
    for (long k = 0; k < n; ++k) { m[k] = {}; }
    if (n < 3) { return; }
    const auto second = [&](long k) { return (value[k - 1] - value[k] * 2.0 + value[k + 1]) * 6.0; };
    if (n == 3) {
        const Vec3 curve = second(1) * (1.0 / 6.0);
        m[0] = m[1] = m[2] = curve;
        return;
    }
    m[1] = second(1) * (1.0 / 6.0);
    m[n - 2] = second(n - 2) * (1.0 / 6.0);
    const long count = n - 4; // unknowns m[2..n-3]
    if (count > 0) {
        std::vector<double> diag(std::size_t(count), 4.0);
        std::vector<Vec3> rhs(static_cast<std::size_t>(count));
        for (long k = 0; k < count; ++k) { rhs[std::size_t(k)] = second(k + 2); }
        rhs[0] = rhs[0] - m[1];
        rhs[std::size_t(count - 1)] = rhs[std::size_t(count - 1)] - m[n - 2];
        for (long k = 1; k < count; ++k) {
            const double f = 1.0 / diag[std::size_t(k - 1)];
            diag[std::size_t(k)] -= f;
            rhs[std::size_t(k)] = rhs[std::size_t(k)] - rhs[std::size_t(k - 1)] * f;
        }
        for (long k = count - 1; k >= 0; --k) {
            Vec3 x = rhs[std::size_t(k)];
            if (k + 1 < count) { x = x - m[k + 3]; }
            m[k + 2] = x * (1.0 / diag[std::size_t(k)]);
        }
    }
    m[0] = m[1] * 2.0 - m[2];
    m[n - 1] = m[n - 2] * 2.0 - m[n - 3];
}

// The same for a closed line.
void spline_closed(const Vec3* value, long n, Vec3* m) {
    std::vector<double> a(std::size_t(n * n), 0.0);
    std::vector<Vec3> rhs(static_cast<std::size_t>(n));
    for (long k = 0; k < n; ++k) {
        a[std::size_t(k * n + k)] += 4.0;
        a[std::size_t(k * n + (k + 1) % n)] += 1.0;
        a[std::size_t(k * n + (k + n - 1) % n)] += 1.0;
        rhs[std::size_t(k)] = (value[(k + n - 1) % n] - value[k] * 2.0 + value[(k + 1) % n]) * 6.0;
    }
    for (long k = 0; k < n; ++k) { // diagonally dominant: no pivoting needed
        for (long row = k + 1; row < n; ++row) {
            const double f = a[std::size_t(row * n + k)] / a[std::size_t(k * n + k)];
            if (f == 0.0) { continue; }
            for (long col = k; col < n; ++col) { a[std::size_t(row * n + col)] -= f * a[std::size_t(k * n + col)]; }
            rhs[std::size_t(row)] = rhs[std::size_t(row)] - rhs[std::size_t(k)] * f;
        }
    }
    for (long k = n - 1; k >= 0; --k) {
        Vec3 x = rhs[std::size_t(k)];
        for (long col = k + 1; col < n; ++col) { x = x - m[col] * a[std::size_t(k * n + col)]; }
        m[k] = x * (1.0 / a[std::size_t(k * n + k)]);
    }
}

inline Vec3 spline_value(Vec3 a, Vec3 b, Vec3 ma, Vec3 mb, double t) {
    const double s = 1.0 - t;
    return a * s + b * t + ma * ((s * s * s - s) / 6.0) + mb * ((t * t * t - t) / 6.0);
}

struct Fitted {
    Method method = Method::conv;
    ConvSurface conv;
    std::vector<Vec3> row_m; // spline fit: second derivatives along u, per row
    mutable std::vector<Vec3> column, column_m;

    void build() {
        if (method == Method::conv) {
            conv.bubble = Bubble::continuous;
            conv.compile();
        }
        const Lattice& l = conv.lattice;
        if (method == Method::spline_fit) {
            row_m.resize(l.nodes.size());
            for (std::size_t v = 0; v < l.nv; ++v) { spline_closed(&l.nodes[v * l.nu], long(l.nu), &row_m[v * l.nu]); }
        }
        column.resize(l.nv);
        column_m.resize(l.nv);
    }

    Vec3 position(double u, double v) const {
        if (method == Method::conv) { return conv.position(u, v); }
        const Lattice& l = conv.lattice;
        std::size_t cu = 0, cv = 0;
        double s = 0.0, t = 0.0;
        locate(u, l.nu, true, cu, s);
        locate(v, l.nv - 1, false, cv, t);
        long lo = long(cv), hi = long(cv) + 1;
        while (lo > 0 && l.seams_v[std::size_t(lo)] == 0) { --lo; }
        while (hi < long(l.nv) - 1 && l.seams_v[std::size_t(hi)] == 0) { ++hi; }
        const long n = long(l.nu);
        // Local methods read at most three rows past the cell; the fit reads
        // the whole stretch.
        const long first = method == Method::spline_fit ? lo : std::max(lo, long(cv) - 3);
        const long last = method == Method::spline_fit ? hi : std::min(hi, long(cv) + 4);
        for (long row = first; row <= last; ++row) {
            const Vec3* nodes = &l.nodes[std::size_t(row) * l.nu];
            if (method == Method::spline_fit) {
                const Vec3* m = &row_m[std::size_t(row) * l.nu];
                const std::size_t next = (cu + 1) % l.nu;
                column[std::size_t(row)] = spline_value(nodes[cu], nodes[next], m[cu], m[next], s);
            } else {
                column[std::size_t(row)] = local_line(method, long(cu), s, -1000000, 1000000, [&](long k) { return nodes[std::size_t(((k % n) + n) % n)]; });
            }
        }
        if (method == Method::spline_fit) {
            spline_open(&column[std::size_t(lo)], hi - lo + 1, &column_m[std::size_t(lo)]);
            return spline_value(column[cv], column[cv + 1], column_m[cv], column_m[cv + 1], t);
        }
        return local_line(method, long(cv), t, lo, hi, [&](long k) { return column[std::size_t(k)]; });
    }
};

struct Score {
    std::size_t nodes = 0;
    double dev_max = 0.0, dev_rms = 0.0;       // distance from the target
    double normal_max = 0.0, normal_rms = 0.0; // degrees from the target's normal
    double kink_max = 0.0;                     // degrees the normal jumps across a node line
    double curvature_jump = 0.0;               // mean relative jump of curvature across node rows
};

double degrees_between(Vec3 a, Vec3 b) { return std::acos(std::min(1.0, std::fabs(dot(a, b)))) * 180.0 / pi; }

Score measure(const Target& target, Method method, std::size_t around, std::size_t rows, const std::vector<double>* placement = nullptr,
              const std::vector<unsigned char>* seams = nullptr) {
    Fitted fit;
    fit.method = method;
    Lattice& l = fit.conv.lattice;
    const std::vector<double> along = placement != nullptr ? *placement : target.rows_for(rows);
    l.nu = around;
    l.nv = along.size();
    l.wrap_u = true;
    l.closure = Closure::cubic;
    l.seams_v = seams != nullptr ? *seams : target.seams_for(rows);
    if (l.seams_v.size() != l.nv) { l.seams_v.assign(l.nv, 0); }
    l.nodes.resize(l.nu * l.nv);
    for (std::size_t v = 0; v < l.nv; ++v) {
        for (std::size_t u = 0; u < l.nu; ++u) { l.at(u, v) = target.point(double(u) / double(around), along[v]); }
    }
    fit.build();
    Score score;
    score.nodes = l.nu * l.nv;
    double window = 0.0;
    for (std::size_t v = 0; v + 1 < l.nv; ++v) { window = std::max(window, along[v + 1] - along[v]); }
    const auto normal_at = [&](double u, double v) {
        const double h = 2e-5;
        return normalize(cross(fit.position(u + h, v) - fit.position(u - h, v), fit.position(u, v + h) - fit.position(u, v - h)));
    };
    const std::size_t columns[3] = {0, around / 4, around / 2};
    const double fraction[5] = {0.1, 0.3, 0.5, 0.7, 0.9};
    double dev2 = 0.0, normal2 = 0.0, relative = 0.0;
    std::size_t count = 0, probes = 0;
    for (std::size_t column : columns) {
        for (double fu : fraction) {
            const double u = double(column) + fu;
            for (std::size_t cell = 0; cell + 1 < l.nv; ++cell) {
                for (double fv : fraction) {
                    const double v = double(cell) + fv;
                    const Vec3 p = fit.position(u, v);
                    Vec3 truth;
                    const double d = target.nearest(p, along[cell] + (along[cell + 1] - along[cell]) * fv, 2.0 * window, truth);
                    const double angle = degrees_between(normal_at(u, v), truth);
                    score.dev_max = std::max(score.dev_max, d);
                    dev2 += d * d;
                    score.normal_max = std::max(score.normal_max, angle);
                    normal2 += angle * angle;
                    ++count;
                }
            }
        }
        // Across node rows.
        const double u = double(column) + 0.37;
        for (std::size_t row = 1; row + 1 < l.nv; ++row) {
            if (l.seams_v[row] != 0) { continue; }
            const double gap = 1e-4;
            // What a smooth surface turns over the gap is taken out by a
            // second pair twice as far apart.
            const double near = degrees_between(normal_at(u, double(row) - gap), normal_at(u, double(row) + gap));
            const double far = degrees_between(normal_at(u, double(row) - 2 * gap), normal_at(u, double(row) + 2 * gap));
            score.kink_max = std::max(score.kink_max, std::max(0.0, 2.0 * near - far));
            const auto side = [&](double sign, double& curvature) {
                const double s = sign * 2e-3, t0 = double(row);
                const Vec3 p0 = fit.position(u, t0), p1 = fit.position(u, t0 + s), p2 = fit.position(u, t0 + 2 * s), p3 = fit.position(u, t0 + 3 * s);
                const Vec3 d1 = (p1 * 18.0 - p0 * 11.0 - p2 * 9.0 + p3 * 2.0) * (1.0 / (6.0 * s));
                const Vec3 d2 = (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (1.0 / (s * s));
                const double speed = length(d1);
                curvature = length(cross(d1, d2)) / (speed * speed * speed);
            };
            double ka, kb;
            side(-1.0, ka);
            side(1.0, kb);
            relative += std::fabs(ka - kb) / std::max(1e-6, std::max(ka, kb));
            ++probes;
        }
        // Across node columns.
        for (std::size_t cell = 0; cell + 1 < l.nv; cell += std::max<std::size_t>(1, l.nv / 24)) {
            const double v = double(cell) + 0.37, gap = 1e-4, line = double(column) + 1.0;
            const double near = degrees_between(normal_at(line - gap, v), normal_at(line + gap, v));
            const double far = degrees_between(normal_at(line - 2 * gap, v), normal_at(line + 2 * gap, v));
            score.kink_max = std::max(score.kink_max, std::max(0.0, 2.0 * near - far));
        }
    }
    score.dev_rms = std::sqrt(dev2 / double(count));
    score.normal_rms = std::sqrt(normal2 / double(count));
    score.curvature_jump = probes > 0 ? relative / double(probes) : 0.0;
    return score;
}

Target make_vase() {
    Target t;
    t.name = "vase (analytic revolution, 3.0 tall)";
    const auto radius = [](double h) { return 1.0 + 0.45 * std::sin(2.2 * h) + 0.12 * std::cos(5.0 * h); };
    auto table = std::make_shared<Revolved>();
    for (int k = 0; k <= 40000; ++k) {
        const double h = 3.0 * k / 40000.0;
        table->r.push_back(radius(h));
        table->y.push_back(h);
    }
    table->finish();
    t.point = [radius](double a, double s) {
        const double r = radius(3.0 * s);
        return Vec3{r * std::cos(2 * pi * a), 3.0 * s, -r * std::sin(2 * pi * a)};
    };
    t.nearest = [table](Vec3 p, double s, double w, Vec3& n) { return table->nearest(p, s, w, n); };
    t.rows_for = [](std::size_t rows) {
        std::vector<double> out;
        for (std::size_t k = 0; k <= rows; ++k) { out.push_back(double(k) / double(rows)); }
        return out;
    };
    t.seams_for = [](std::size_t rows) { return std::vector<unsigned char>(rows + 1, 0); };
    return t;
}

Target make_tube() {
    Target t;
    t.name = "tube (half torus, radii 1.2 and 0.35)";
    const double R = 1.2, r = 0.35;
    t.point = [=](double a, double s) {
        const double phi = pi * s, theta = 2 * pi * a;
        const double reach = R + r * std::cos(theta);
        return Vec3{reach * std::cos(phi), reach * std::sin(phi), r * std::sin(theta)};
    };
    t.nearest = [=](Vec3 p, double, double, Vec3& n) {
        const double rho = std::hypot(p.x, p.y), off = rho - R, d = std::hypot(off, p.z);
        n = normalize(Vec3{p.x / rho * off, p.y / rho * off, p.z});
        return std::fabs(d - r);
    };
    t.rows_for = [](std::size_t rows) {
        std::vector<double> out;
        for (std::size_t k = 0; k <= rows; ++k) { out.push_back(double(k) / double(rows)); }
        return out;
    };
    t.seams_for = [](std::size_t rows) { return std::vector<unsigned char>(rows + 1, 0); };
    return t;
}

struct Body {
    ProfileCurve profile;
    Revolved table;
    void point(double arc, double& r, double& y) const {
        std::size_t run = 0;
        double t = 0.0;
        profile.locate(std::min(profile.length, std::max(0.0, arc)), run, t);
        profile.point_on_run(run, t, r, y);
    }
    mutable std::map<std::size_t, SheetPlan> cache;
    // Rows by CONV's own error (the engine's placement).
    const SheetPlan& placed(std::size_t rows) const {
        auto found = cache.find(rows);
        if (found == cache.end()) { found = cache.emplace(rows, plan_rows_by_error(profile, rows, true)).first; }
        return found->second;
    }
    // Rows by the profile's radius of curvature (the earlier placement).
    SheetPlan plan(std::size_t rows) const {
        return plan_rows(profile, rows, profile.length / (40.0 * double(rows)), 4.0 * profile.length / double(rows));
    }
};

std::shared_ptr<Body> make_body_data() {
    auto body = std::make_shared<Body>();
    std::vector<ProfileCurve::Run> runs;
    for (int r = 0; r < utah::dense_body_runs; ++r) {
        ProfileCurve::Run run;
        for (int k = 0; k < utah::dense_profile_points; ++k) {
            run.x.push_back(double(utah::dense_body_profile[r][k][0]));
            run.y.push_back(double(utah::dense_body_profile[r][k][1]));
        }
        runs.push_back(run);
    }
    body->profile = chain_runs(runs);
    for (int k = 0; k <= 60000; ++k) {
        double r, y;
        body->point(body->profile.length * k / 60000.0, r, y);
        body->table.r.push_back(std::max(0.0, r));
        body->table.y.push_back(y);
    }
    body->table.finish();
    return body;
}

Target make_body(std::shared_ptr<Body> body) {
    Target t;
    t.name = "body (2026 teapot body with interior, 4.0 wide; rows placed by CONV error)";
    t.point = [body](double a, double s) {
        double r, y;
        body->point(s * body->profile.length, r, y);
        if (s <= 0.0 || s >= 1.0) { r = 0.0; }
        return Vec3{r * std::cos(2 * pi * a), y, -r * std::sin(2 * pi * a)};
    };
    t.nearest = [body](Vec3 p, double s, double w, Vec3& n) { return body->table.nearest(p, s, w, n); };
    t.rows_for = [body](std::size_t rows) {
        std::vector<double> out;
        for (double arc : body->placed(rows).row_arc) { out.push_back(arc / body->profile.length); }
        return out;
    };
    t.seams_for = [body](std::size_t rows) { return body->placed(rows).row_seam; };
    return t;
}

void report(const Target& target, const std::vector<std::size_t>& around_set, const std::vector<std::size_t>& rows_set) {
    std::printf("\n== %s ==\n", target.name.c_str());
    std::printf("%-13s %6s %5s %6s | %9s %9s | %9s %9s | %9s | %9s\n", "method", "around", "rows", "nodes", "dev max", "dev rms", "nrm max", "nrm rms", "kink max",
                "curv jump");
    std::vector<std::vector<Score>> all(5);
    std::vector<std::pair<std::size_t, std::size_t>> grid;
    for (std::size_t a : around_set) {
        for (std::size_t r : rows_set) { grid.push_back({a, r}); }
    }
    for (int m = 0; m < 5; ++m) {
        for (const auto& [a, r] : grid) { all[std::size_t(m)].push_back(measure(target, Method(m), a, r)); }
    }
    // A diagonal of the grid: node spacing halves in both directions.
    for (std::size_t step = 0; step < std::min(around_set.size(), rows_set.size()); ++step) {
        const std::size_t index = step * rows_set.size() + step;
        for (int m = 0; m < 5; ++m) {
            const Score& s = all[std::size_t(m)][index];
            std::printf("%-13s %6zu %5zu %6zu | %9.2e %9.2e | %9.3f %9.3f | %9.4f | %9.4f\n", method_name[m], grid[index].first, grid[index].second, s.nodes,
                        s.dev_max, s.dev_rms, s.normal_max, s.normal_rms, s.kink_max, s.curvature_jump);
        }
    }
    std::printf("fewest nodes that hold a tolerance (deviation in model units; normal in degrees; '-' = not reached by %zu nodes)\n",
                around_set.back() * (rows_set.back() + 1));
    std::printf("%-13s | %-14s %-14s %-14s | %-14s %-14s\n", "method", "dev <= 1e-2", "dev <= 1e-3", "dev <= 1e-4", "normal <= 1", "normal <= 0.1");
    for (int m = 0; m < 5; ++m) {
        std::printf("%-13s |", method_name[m]);
        const auto best = [&](const std::function<bool(const Score&)>& ok) {
            std::size_t fewest = 0, at = 0;
            for (std::size_t k = 0; k < grid.size(); ++k) {
                const Score& s = all[std::size_t(m)][k];
                if (ok(s) && (fewest == 0 || s.nodes < fewest)) {
                    fewest = s.nodes;
                    at = k;
                }
            }
            char text[32];
            if (fewest == 0) {
                std::snprintf(text, sizeof text, "-");
            } else {
                std::snprintf(text, sizeof text, "%zu (%zux%zu)", fewest, grid[at].first, all[std::size_t(m)][at].nodes / grid[at].first);
            }
            std::printf(" %-14s", text);
        };
        for (double tolerance : {1e-2, 1e-3, 1e-4}) {
            best([&](const Score& s) { return s.dev_max <= tolerance; });
        }
        std::printf(" |");
        for (double tolerance : {1.0, 0.1}) {
            best([&](const Score& s) { return s.normal_max <= tolerance; });
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    const bool place_only = argc > 1 && std::string(argv[1]) == "place";
    const std::vector<std::size_t> around{6, 8, 12, 16, 24, 32, 48};
    const std::vector<std::size_t> rows{6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    if (!place_only) {
        report(make_vase(), around, rows);
        report(make_tube(), around, rows);
    }
    const auto body = make_body_data();
    const Target target = make_body(body);
    if (!place_only) { report(target, around, {12, 16, 24, 32, 48, 64, 96, 128, 192}); }
    std::printf("\n== row placement on the body ==\n");
    std::printf("row placement on the body, conv, 16 around: dev max / normal max (deg) / kink max (deg)\n");
    std::printf("%5s | %-28s | %-28s | %-28s | %-28s\n", "rows", "radius graded", "error placed, growth 0.08", "growth 0.2", "growth 0.4");
    for (std::size_t r : {16u, 24u, 32u, 48u, 64u, 96u, 128u}) {
        std::printf("%5zu |", r);
        const SheetPlan radius = body->plan(r);
        std::vector<double> radius_along;
        for (double arc : radius.row_arc) { radius_along.push_back(arc / body->profile.length); }
        const Score base = measure(target, Method::conv, 16, r, &radius_along, &radius.row_seam);
        std::printf(" %9.2e %7.2f %7.2f    |", base.dev_max, base.normal_max, base.kink_max);
        for (double growth : {0.08, 0.2, 0.4}) {
            const SheetPlan plan = plan_rows_by_error(body->profile, r, true, growth);
            std::vector<double> along;
            for (double arc : plan.row_arc) { along.push_back(arc / body->profile.length); }
            const Score s = measure(target, Method::conv, 16, r, &along, &plan.row_seam);
            std::printf(" %9.2e %7.2f %7.2f %4zu |", s.dev_max, s.normal_max, s.kink_max, along.size() - 1);
        }
        std::printf("\n");
    }
    // Rows at even steps of each source run's own parameter, a cut at
    // every joint between runs: the placement that reproduces a profile
    // made of cubic pieces.
    std::printf("rows per source run, a cut at every joint (9 runs), all methods, 16 around: dev max / normal max / kink max\n");
    for (int per : {2, 3, 4, 6}) {
        std::vector<double> along;
        std::vector<unsigned char> cuts;
        const ProfileCurve& profile = body->profile;
        for (std::size_t k = 0; k < profile.table_s.size(); ++k) {
            const double steps = profile.table_t[k] * per / double(utah::dense_profile_points - 1);
            const bool run_start = k == 0 || profile.table_run[k] != profile.table_run[k - 1];
            const bool run_end = k + 1 == profile.table_s.size() || profile.table_run[k + 1] != profile.table_run[k];
            if (std::fabs(steps - std::round(steps)) < 1e-9 || run_end) {
                if (run_start && k > 0) { continue; }
                along.push_back(profile.table_s[k] / profile.length);
                cuts.push_back(run_end && k + 1 < profile.table_s.size() ? 1 : 0);
            }
        }
        std::printf("%d per run, %zu rows:", per, along.size() - 1);
        for (int m = 0; m < 5; ++m) {
            const Score s = measure(target, Method(m), 16, along.size() - 1, &along, &cuts);
            std::printf("  %s %.2e %.2f %.2f |", method_name[m], s.dev_max, s.normal_max, s.kink_max);
        }
        std::printf("\n");
    }
    return 0;
}
