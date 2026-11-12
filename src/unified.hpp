// Unified sheets: a whole part of the teapot as one lattice.
//
// The body of a teapot with its interior is one surface of revolution: a
// profile that starts at the outer bottom centre, climbs the outside, turns
// over the rim, descends the inside and ends at the inner bottom centre. It
// is a flat sheet, closed around, warped onto that shape. So is the lid.
//
// The sheet's rows are placed along the profile by a piecewise projective
// warp. Between stations the square root of the row spacing is linear in arc
// length, which is exactly the spacing a projective (Moebius) map of a
// uniform grid produces; the maps join with matching spacing, so the
// placement is smooth and rows gather where the profile turns tightly. CONV
// then interpolates in the row index, and what it interpolates is a smooth
// function of that index.
#pragma once
#include "conv_line.hpp"
#include "surface.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

namespace tconv {

// A planar curve through chained runs of samples (each run one source patch
// edge), evaluated run by run as an open CONV line with cubic ends.
struct ProfileCurve {
    struct Run {
        std::vector<double> x, y; // samples of one run
    };
    std::vector<Run> runs;
    // Dense arc-length table.
    std::vector<double> table_s, table_x, table_y, table_t;
    std::vector<std::size_t> table_run;
    std::vector<double> run_start; // arc length at the start of each run
    double length = 0.0;

    static double line(const std::vector<double>& samples, double t) {
        const std::size_t n = samples.size();
        std::size_t cell = std::size_t(t);
        if (cell + 1 >= n) { cell = n - 2; }
        double w[8] = {};
        for (int k = 0; k < 8; ++k) {
            const long index = long(cell) - 3 + k;
            if (index >= 0 && index < long(n)) { w[k] = samples[std::size_t(index)]; }
        }
        LineLayout layout;
        layout.nodes = n;
        layout.closure = Closure::cubic;
        return eval_cell(compile_cell(w, cell_shape(layout, cell), Bubble::continuous), t - double(cell));
    }

    void point_on_run(std::size_t run, double t, double& x, double& y) const {
        x = line(runs[run].x, t);
        y = line(runs[run].y, t);
    }

    void build_table(int per_interval = 8) {
        table_s.clear();
        table_x.clear();
        table_y.clear();
        table_t.clear();
        table_run.clear();
        run_start.clear();
        double s = 0.0, px = 0.0, py = 0.0;
        bool first = true;
        for (std::size_t run = 0; run < runs.size(); ++run) {
            run_start.push_back(s);
            const int intervals = int(runs[run].x.size()) - 1;
            for (int k = first ? 0 : 1; k <= intervals * per_interval; ++k) {
                double x, y;
                point_on_run(run, double(k) / per_interval, x, y);
                if (!first) { s += std::hypot(x - px, y - py); }
                first = false;
                table_s.push_back(s);
                table_x.push_back(x);
                table_y.push_back(y);
                table_t.push_back(double(k) / per_interval);
                table_run.push_back(run);
                px = x;
                py = y;
            }
        }
        length = s;
    }

    // Point at arc length s.
    void at(double s, double& x, double& y) const {
        const auto upper = std::upper_bound(table_s.begin(), table_s.end(), s);
        std::size_t k = upper == table_s.begin() ? 0 : std::size_t(upper - table_s.begin()) - 1;
        if (k + 1 >= table_s.size()) { k = table_s.size() - 2; }
        const double span = table_s[k + 1] - table_s[k];
        const double t = span > 0.0 ? (s - table_s[k]) / span : 0.0;
        x = table_x[k] + (table_x[k + 1] - table_x[k]) * t;
        y = table_y[k] + (table_y[k + 1] - table_y[k]) * t;
    }

    // The run and the parameter within it that hold arc length s.
    void locate(double s, std::size_t& run, double& t) const {
        const auto upper = std::upper_bound(table_s.begin(), table_s.end(), s);
        std::size_t k = upper == table_s.begin() ? 0 : std::size_t(upper - table_s.begin()) - 1;
        if (k + 1 >= table_s.size()) { k = table_s.size() - 2; }
        const double span = table_s[k + 1] - table_s[k];
        const double f = span > 0.0 ? std::min(1.0, std::max(0.0, (s - table_s[k]) / span)) : 0.0;
        if (table_run[k + 1] != table_run[k]) {
            // The step that crosses into the next run starts at its parameter 0.
            run = table_run[k + 1];
            t = table_t[k + 1] * f;
        } else {
            run = table_run[k];
            t = table_t[k] + (table_t[k + 1] - table_t[k]) * f;
        }
    }

    // Radius of curvature near arc length s, from three points `reach` apart.
    double radius(double s, double reach) const {
        const double a = std::max(0.0, s - reach), c = std::min(length, s + reach), b = 0.5 * (a + c);
        double ax, ay, bx, by, cx, cy;
        at(a, ax, ay);
        at(b, bx, by);
        at(c, cx, cy);
        const double area = std::fabs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
        const double ab = std::hypot(bx - ax, by - ay), bc = std::hypot(cx - bx, cy - by), ca = std::hypot(ax - cx, ay - cy);
        return area > 1e-14 ? ab * bc * ca / (2.0 * area) : 1e9;
    }

    // Turn of the tangent (degrees) across the joint that starts run `run`,
    // from the one-sided end tangents of the two runs that meet there.
    double turn_at_joint(std::size_t run) const {
        const double h = 1e-5;
        const double last = double(runs[run - 1].x.size() - 1);
        double ax, ay, bx, by, cx, cy, dx, dy;
        point_on_run(run - 1, last - h, ax, ay);
        point_on_run(run - 1, last, bx, by);
        point_on_run(run, 0.0, cx, cy);
        point_on_run(run, h, dx, dy);
        const double ux = bx - ax, uy = by - ay, vx = dx - cx, vy = dy - cy;
        const double c = (ux * vx + uy * vy) / (std::hypot(ux, uy) * std::hypot(vx, vy) + 1e-300);
        return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / 3.14159265358979;
    }
};

// Row placement by a piecewise projective warp.
//
// Stations sit at arc lengths `station`; at each the wanted row spacing is
// `spacing`. Inside a span sqrt(spacing) is linear, so the number of rows in
// a span of length d is d / (p0 p1), p = sqrt(spacing), and the arc length
// that holds c rows from the span's start is  c p0^2 / (1 - c p0 (p1 - p0)/d),
// a Moebius function of c.
struct ProjectiveGrading {
    std::vector<double> station, root; // root = sqrt(spacing)

    double rows_in_span(std::size_t k) const { return (station[k + 1] - station[k]) / (root[k] * root[k + 1]); }
    double rows_between(std::size_t first, std::size_t last) const {
        double total = 0.0;
        for (std::size_t k = first; k < last; ++k) { total += rows_in_span(k); }
        return total;
    }
    // Arc length holding `count` rows measured from station `first`.
    double arc_at(std::size_t first, double count) const {
        std::size_t k = first;
        while (k + 2 < station.size() && count > rows_in_span(k)) {
            count -= rows_in_span(k);
            ++k;
        }
        const double d = station[k + 1] - station[k], p0 = root[k], p1 = root[k + 1];
        const double x = count * p0 * p0 / (1.0 - count * p0 * (p1 - p0) / d);
        return station[k] + std::min(d, std::max(0.0, x));
    }
};

struct SheetPlan {
    std::vector<double> row_arc;          // arc length of every row
    std::vector<unsigned char> row_seam;  // rows on a crease
    std::size_t spans = 0;                // projective spans used
};

// Rows for a profile: about `rows` intervals in all, spacing proportional to
// the local radius of curvature within [finest, coarsest], and a row exactly
// on every crease.
//
// `shape`, when given, replaces the radius of curvature: it returns the
// wanted spacing at an arc length, up to one common scale. `growth` bounds
// how fast the spacing may change along the profile.
inline SheetPlan plan_rows(const ProfileCurve& profile, std::size_t rows, double finest, double coarsest, double growth = 0.08,
                           const std::function<double(double)>* shape = nullptr) {
    // Sections between creases (and the two ends).
    std::vector<double> breaks{0.0};
    for (std::size_t run = 1; run < profile.runs.size(); ++run) {
        if (profile.turn_at_joint(run) > 4.0) { breaks.push_back(profile.run_start[run]); }
    }
    breaks.push_back(profile.length);
    const int stations_total = 96;
    const auto grading_for = [&](double a, double b, double scale) {
        ProjectiveGrading g;
        const int n = std::max(2, int(std::ceil(stations_total * (b - a) / profile.length)));
        for (int k = 0; k <= n; ++k) {
            const double s = a + (b - a) * k / n;
            // The tightest radius within reach of the station sets its spacing.
            double radius = 1e9;
            for (int j = -2; j <= 2; ++j) {
                const double q = std::min(b, std::max(a, s + j * 0.25 * (b - a) / n));
                radius = std::min(radius, shape != nullptr ? (*shape)(q) : profile.radius(q, 0.001 * profile.length));
            }
            const double spacing = std::min(coarsest, std::max(finest, scale * radius));
            g.station.push_back(s);
            g.root.push_back(std::sqrt(spacing));
        }
        // Limit how fast the spacing may change between stations, both ways:
        // neighbouring rows then differ in spacing by at most about a sixth.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t k = 1; k < g.root.size(); ++k) {
                const double d = g.station[k] - g.station[k - 1];
                g.root[k] = std::min(g.root[k], g.root[k - 1] + growth * d / g.root[k - 1]);
            }
            for (std::size_t k = g.root.size() - 1; k-- > 0;) {
                const double d = g.station[k + 1] - g.station[k];
                g.root[k] = std::min(g.root[k], g.root[k + 1] + growth * d / g.root[k + 1]);
            }
        }
        return g;
    };
    // One scale for the whole profile, found by bisection on the row count.
    double low = 1e-9, high = 1e9;
    for (int iteration = 0; iteration < 120; ++iteration) {
        const double mid = std::sqrt(low * high);
        double count = 0.0;
        for (std::size_t k = 0; k + 1 < breaks.size(); ++k) {
            const ProjectiveGrading g = grading_for(breaks[k], breaks[k + 1], mid);
            count += g.rows_between(0, g.station.size() - 1);
        }
        if (count > double(rows)) { low = mid; } else { high = mid; }
    }
    const double scale = std::sqrt(low * high);
    SheetPlan plan;
    for (std::size_t k = 0; k + 1 < breaks.size(); ++k) {
        const ProjectiveGrading g = grading_for(breaks[k], breaks[k + 1], scale);
        const double exact = g.rows_between(0, g.station.size() - 1);
        const std::size_t n = std::max<std::size_t>(4, std::size_t(std::llround(exact)));
        plan.spans += g.station.size() - 1;
        for (std::size_t row = 0; row < n; ++row) {
            plan.row_arc.push_back(g.arc_at(0, exact * double(row) / double(n)));
            plan.row_seam.push_back(row == 0 && k > 0 ? 1 : 0);
        }
    }
    plan.row_arc.push_back(profile.length);
    plan.row_seam.push_back(0);
    return plan;
}

// One open CONV line through values with cuts.
inline double cut_line(const std::vector<double>& y, const std::vector<unsigned char>& seams, std::size_t cell, double t) {
    const LineLayout layout{y.size(), false, Closure::cubic, seams.data()};
    double w[8] = {};
    for (int k = 0; k < 8; ++k) {
        const long index = long(cell) - 3 + k;
        if (index >= 0 && index < long(y.size())) { w[k] = y[std::size_t(index)]; }
    }
    return eval_cell(compile_cell(w, cell_shape(layout, cell), Bubble::continuous), t);
}

// Rows placed by CONV's own error.
//
// The curve through the current rows is a pair of CONV lines. Each cell's
// distance from the profile, divided by the fourth power of the cell's
// length, measures how hard the profile is there; its fourth root is the
// density of rows that evens the error out. plan_rows turns that density
// into a smooth projective grading, and the step is repeated, keeping the
// plan with the smallest error. `error`, when given, receives that error.
// `on_axis` marks profiles whose two ends are poles (x = 0).
inline SheetPlan plan_rows_by_error(const ProfileCurve& source, std::size_t rows, bool on_axis, double growth = 0.2, int iterations = 8,
                                    double* error = nullptr) {
    ProfileCurve profile = source;
    profile.build_table(32);
    const double length = profile.length;
    const auto truth = [&](double arc, double& x, double& y) {
        std::size_t run = 0;
        double t = 0.0;
        profile.locate(std::min(length, std::max(0.0, arc)), run, t);
        profile.point_on_run(run, t, x, y);
    };
    const auto distance = [&](double x, double y, double arc, double reach) {
        const auto first = std::lower_bound(profile.table_s.begin(), profile.table_s.end(), arc - reach);
        std::size_t k = first == profile.table_s.begin() ? 0 : std::size_t(first - profile.table_s.begin()) - 1;
        double best = 1e30;
        for (; k + 1 < profile.table_s.size() && profile.table_s[k] <= arc + reach; ++k) {
            const double ax = profile.table_x[k], ay = profile.table_y[k];
            const double bx = profile.table_x[k + 1] - ax, by = profile.table_y[k + 1] - ay;
            const double len2 = bx * bx + by * by;
            const double t = len2 > 0.0 ? std::max(0.0, std::min(1.0, ((x - ax) * bx + (y - ay) * by) / len2)) : 0.0;
            best = std::min(best, std::hypot(ax + t * bx - x, ay + t * by - y));
        }
        return best;
    };
    const double finest = length / (400.0 * double(rows)), coarsest = 6.0 * length / double(rows);
    SheetPlan plan = plan_rows(profile, rows, length / (40.0 * double(rows)), 4.0 * length / double(rows), growth);
    SheetPlan best = plan;
    double best_error = 1e30;
    std::vector<double> hardness_s, hardness;
    for (int iteration = 0; iteration <= iterations; ++iteration) {
        const std::size_t n = plan.row_arc.size();
        std::vector<double> x(n), y(n);
        for (std::size_t k = 0; k < n; ++k) {
            truth(plan.row_arc[k], x[k], y[k]);
            if (on_axis && (k == 0 || k + 1 == n)) { x[k] = 0.0; }
        }
        double worst = 0.0;
        std::vector<double> cell_s, cell_g;
        for (std::size_t cell = 0; cell + 1 < n; ++cell) {
            const double h = plan.row_arc[cell + 1] - plan.row_arc[cell];
            double e = 0.0;
            for (double t : {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}) {
                e = std::max(e, distance(cut_line(x, plan.row_seam, cell, t), cut_line(y, plan.row_seam, cell, t), plan.row_arc[cell] + t * h, 2.0 * h));
            }
            worst = std::max(worst, e);
            cell_s.push_back(0.5 * (plan.row_arc[cell] + plan.row_arc[cell + 1]));
            cell_g.push_back(std::max(1e-7, e) / (h * h * h * h));
        }
        if (worst < best_error) {
            best_error = worst;
            best = plan;
        }
        if (iteration == iterations) { break; }
        const auto cell_of = [](const std::vector<double>& at, double s) {
            const auto upper = std::upper_bound(at.begin(), at.end(), s);
            return upper == at.begin() ? std::size_t(0) : std::min(at.size() - 1, std::size_t(upper - at.begin()) - 1);
        };
        // Blend with the previous estimate so the iteration settles.
        if (!hardness.empty()) {
            for (std::size_t k = 0; k < cell_g.size(); ++k) { cell_g[k] = std::sqrt(cell_g[k] * hardness[cell_of(hardness_s, cell_s[k])]); }
        }
        hardness_s = cell_s;
        hardness = cell_g;
        const std::function<double(double)> shape = [&](double s) {
            const std::size_t j = cell_of(hardness_s, s), j1 = std::min(hardness.size() - 1, j + 1);
            return std::pow(std::max(hardness[j], hardness[j1]), -0.25);
        };
        plan = plan_rows(profile, rows, finest, coarsest, growth, &shape);
    }
    if (error != nullptr) { *error = best_error; }
    return best;
}

// The error of a plan as plan_rows_by_error measures it: the largest distance
// from the profile of the CONV lines through the plan's rows.
inline double plan_error(const ProfileCurve& source, const SheetPlan& plan, bool on_axis) {
    ProfileCurve profile = source;
    profile.build_table(32);
    const std::size_t n = plan.row_arc.size();
    std::vector<double> x(n), y(n);
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t run = 0;
        double t = 0.0;
        profile.locate(std::min(profile.length, std::max(0.0, plan.row_arc[k])), run, t);
        profile.point_on_run(run, t, x[k], y[k]);
        if (on_axis && (k == 0 || k + 1 == n)) { x[k] = 0.0; }
    }
    double worst = 0.0;
    for (std::size_t cell = 0; cell + 1 < n; ++cell) {
        const double h = plan.row_arc[cell + 1] - plan.row_arc[cell];
        for (double t : {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}) {
            const double px = cut_line(x, plan.row_seam, cell, t), py = cut_line(y, plan.row_seam, cell, t), arc = plan.row_arc[cell] + t * h;
            const auto first = std::lower_bound(profile.table_s.begin(), profile.table_s.end(), arc - 2.0 * h);
            std::size_t k = first == profile.table_s.begin() ? 0 : std::size_t(first - profile.table_s.begin()) - 1;
            double best = 1e30;
            for (; k + 1 < profile.table_s.size() && profile.table_s[k] <= arc + 2.0 * h; ++k) {
                const double ax = profile.table_x[k], ay = profile.table_y[k];
                const double bx = profile.table_x[k + 1] - ax, by = profile.table_y[k + 1] - ay;
                const double len2 = bx * bx + by * by;
                const double f = len2 > 0.0 ? std::max(0.0, std::min(1.0, ((px - ax) * bx + (py - ay) * by) / len2)) : 0.0;
                best = std::min(best, std::hypot(ax + f * bx - px, ay + f * by - py));
            }
            worst = std::max(worst, best);
        }
    }
    return worst;
}

// Chains planar runs of samples (each an edge of one source patch) into one
// curve, starting from the end that lies on the axis x = 0 at the smallest
// height, or, when no end is on the axis, from the end `start_hint` picks.
inline ProfileCurve chain_runs(std::vector<ProfileCurve::Run> runs, bool start_on_axis = true) {
    ProfileCurve profile;
    if (runs.empty()) { return profile; }
    const auto end_x = [&](std::size_t k, int end) { return end == 0 ? runs[k].x.front() : runs[k].x.back(); };
    const auto end_y = [&](std::size_t k, int end) { return end == 0 ? runs[k].y.front() : runs[k].y.back(); };
    std::size_t start = 0;
    bool start_flipped = false;
    double best = 1e30;
    for (std::size_t k = 0; k < runs.size(); ++k) {
        for (int end = 0; end < 2; ++end) {
            // A free end: no other run shares it.
            bool shared = false;
            for (std::size_t other = 0; other < runs.size(); ++other) {
                for (int e = 0; e < 2; ++e) {
                    if ((other != k || e != end) && std::hypot(end_x(other, e) - end_x(k, end), end_y(other, e) - end_y(k, end)) < 1e-5) { shared = true; }
                }
            }
            if (shared) { continue; }
            const double score = start_on_axis ? (std::fabs(end_x(k, end)) < 1e-4 ? end_y(k, end) : 1e20) : end_y(k, end);
            if (score < best) {
                best = score;
                start = k;
                start_flipped = end == 1;
            }
        }
    }
    std::vector<unsigned char> used(runs.size(), 0);
    std::size_t current = start;
    bool flipped = start_flipped;
    for (;;) {
        used[current] = 1;
        ProfileCurve::Run run = runs[current];
        if (flipped) {
            std::reverse(run.x.begin(), run.x.end());
            std::reverse(run.y.begin(), run.y.end());
        }
        profile.runs.push_back(run);
        const double ex = run.x.back(), ey = run.y.back();
        bool found = false;
        for (std::size_t k = 0; k < runs.size() && !found; ++k) {
            if (used[k] != 0) { continue; }
            for (int end = 0; end < 2 && !found; ++end) {
                if (std::hypot(end_x(k, end) - ex, end_y(k, end) - ey) < 1e-5) {
                    current = k;
                    flipped = end == 1;
                    found = true;
                }
            }
        }
        if (!found) { break; }
    }
    profile.build_table(8);
    return profile;
}

// A sheet of revolution: every row a true circle of the profile's radius.
using SheetNodesOutside = std::function<bool(const ProfileCurve&, const SheetPlan&, Lattice&)>;
inline void revolve_sheet(const ProfileCurve& profile, const SheetPlan& plan, std::size_t around, Lattice& lattice, const SheetNodesOutside* nodes_outside = nullptr) {
    lattice.wrap_u = true;
    lattice.closure = Closure::cubic;
    lattice.nu = around;
    lattice.nv = plan.row_arc.size();
    lattice.nodes.resize(lattice.nu * lattice.nv);
    lattice.seams_v.assign(plan.row_seam.begin(), plan.row_seam.end());
    lattice.seams_u.clear();
    if (nodes_outside != nullptr && *nodes_outside && (*nodes_outside)(profile, plan, lattice)) { return; }
    for (std::size_t v = 0; v < lattice.nv; ++v) {
        double r, y;
        profile.at(plan.row_arc[v], r, y);
        if (v == 0 || v + 1 == lattice.nv) { r = 0.0; } // the two poles
        for (std::size_t u = 0; u < around; ++u) {
            const double angle = 6.283185307179586 * double(u) / double(around);
            lattice.at(u, v) = {r * std::cos(angle), y, -r * std::sin(angle)};
        }
    }
}

// A tube from mirror halves: each source patch is a grid of points, one half
// of the tube's cross-section by a stretch of its length, with two opposite
// edges in the symmetry plane z = 0. The patches are chained along the tube
// and the whole becomes one sheet, closed around.
// `plan_outside` and `nodes_outside`, when given, are asked first for the rows
// and for the nodes (the GPU page places and samples them in shaders); each
// returns false to leave the work here.
using RowsOutside = std::function<bool(const ProfileCurve&, std::size_t, bool, SheetPlan&)>;
using TubeNodesOutside = std::function<bool(const std::vector<ConvSurface>&, const ProfileCurve&, const SheetPlan&, std::size_t, Lattice&)>;
inline void tube_sheet(const float (*const* patches)[3], int count, int points, std::size_t rows, std::size_t around, Lattice& lattice,
                       std::vector<double>& row_arc, bool rows_by_error = false, const RowsOutside* plan_outside = nullptr,
                       const TubeNodesOutside* nodes_outside = nullptr) {
    const std::size_t n = std::size_t(points);
    struct Half {
        std::vector<Vec3> grid; // [b * n + a]: a around, b along
    };
    std::vector<Half> halves;
    for (int k = 0; k < count; ++k) {
        const float (*p)[3] = patches[k];
        const auto in_plane = [&](bool column, std::size_t index) {
            for (std::size_t q = 0; q < n; ++q) {
                const float z = column ? p[q * n + index][2] : p[index * n + q][2];
                if (std::fabs(z) > 1e-5f) { return false; }
            }
            return true;
        };
        const bool around_is_i = in_plane(true, 0) && in_plane(true, n - 1);
        Half half;
        half.grid.resize(n * n);
        for (std::size_t b = 0; b < n; ++b) {
            for (std::size_t a = 0; a < n; ++a) {
                const float* q = around_is_i ? p[b * n + a] : p[a * n + b];
                half.grid[b * n + a] = {double(q[0]), double(q[1]), double(q[2])};
            }
        }
        halves.push_back(half);
    }
    const auto ring_end = [&](const Half& h, bool last, bool far) { return h.grid[(last ? n - 1 : 0) * n + (far ? n - 1 : 0)]; };
    const auto flip_b = [&](Half& h) {
        for (std::size_t b = 0; b < n / 2; ++b) {
            for (std::size_t a = 0; a < n; ++a) { std::swap(h.grid[b * n + a], h.grid[(n - 1 - b) * n + a]); }
        }
    };
    const auto flip_a = [&](Half& h) {
        for (std::size_t b = 0; b < n; ++b) {
            for (std::size_t a = 0; a < n / 2; ++a) { std::swap(h.grid[b * n + a], h.grid[b * n + (n - 1 - a)]); }
        }
    };
    const auto same_ring = [&](const Half& x, bool x_last, const Half& y, bool y_last, bool& reversed) {
        const Vec3 x0 = ring_end(x, x_last, false), x1 = ring_end(x, x_last, true);
        const Vec3 y0 = ring_end(y, y_last, false), y1 = ring_end(y, y_last, true);
        if (length(x0 - y0) < 1e-4 && length(x1 - y1) < 1e-4) { reversed = false; return true; }
        if (length(x0 - y1) < 1e-4 && length(x1 - y0) < 1e-4) { reversed = true; return true; }
        return false;
    };
    // Start at a ring no other patch shares.
    std::size_t start = 0;
    bool start_last = false;
    for (std::size_t k = 0; k < halves.size(); ++k) {
        for (int end = 0; end < 2; ++end) {
            bool shared = false, reversed = false;
            for (std::size_t other = 0; other < halves.size(); ++other) {
                for (int e = 0; e < 2; ++e) {
                    if (other != k && same_ring(halves[k], end == 1, halves[other], e == 1, reversed)) { shared = true; }
                }
            }
            if (!shared) {
                start = k;
                start_last = end == 1;
                k = halves.size();
                break;
            }
        }
    }
    std::vector<Half> chain;
    std::vector<unsigned char> used(halves.size(), 0);
    Half current = halves[start];
    used[start] = 1;
    if (start_last) { flip_b(current); }
    for (;;) {
        chain.push_back(current);
        bool found = false;
        for (std::size_t k = 0; k < halves.size() && !found; ++k) {
            if (used[k] != 0) { continue; }
            for (int end = 0; end < 2 && !found; ++end) {
                bool reversed = false;
                if (same_ring(chain.back(), true, halves[k], end == 1, reversed)) {
                    current = halves[k];
                    if (end == 1) { flip_b(current); }
                    if (reversed) { flip_a(current); }
                    used[k] = 1;
                    found = true;
                }
            }
        }
        if (!found) { break; }
    }
    // Rows along the seam a = 0, a curve in the plane z = 0.
    ProfileCurve seam;
    std::vector<ConvSurface> surfaces(chain.size());
    for (std::size_t k = 0; k < chain.size(); ++k) {
        ProfileCurve::Run run;
        for (std::size_t b = 0; b < n; ++b) {
            run.x.push_back(chain[k].grid[b * n].x);
            run.y.push_back(chain[k].grid[b * n].y);
        }
        seam.runs.push_back(run);
        Lattice& source = surfaces[k].lattice;
        source.nu = source.nv = n;
        source.wrap_u = false;
        source.nodes = chain[k].grid;
        surfaces[k].compile();
    }
    seam.build_table(8);
    SheetPlan plan;
    if (!(rows_by_error && plan_outside != nullptr && *plan_outside && (*plan_outside)(seam, rows, false, plan))) {
        plan = rows_by_error ? plan_rows_by_error(seam, rows, false) : plan_rows(seam, rows, seam.length / 2000.0, seam.length / 24.0);
    }
    row_arc = plan.row_arc;
    lattice.wrap_u = true;
    lattice.closure = Closure::cubic;
    lattice.nu = around;
    lattice.nv = plan.row_arc.size();
    lattice.nodes.resize(lattice.nu * lattice.nv);
    lattice.seams_v.assign(plan.row_seam.begin(), plan.row_seam.end());
    lattice.seams_u.clear();
    const std::size_t half_count = around / 2;
    if (nodes_outside != nullptr && *nodes_outside && (*nodes_outside)(surfaces, seam, plan, around, lattice)) { return; }
    for (std::size_t v = 0; v < lattice.nv; ++v) {
        std::size_t run = 0;
        double t = 0.0;
        seam.locate(plan.row_arc[v], run, t);
        for (std::size_t u = 0; u < around; ++u) {
            const bool mirrored = u > half_count;
            const std::size_t step = mirrored ? around - u : u;
            Vec3 p = surfaces[run].position(double(n - 1) * double(step) / double(half_count), t);
            if (mirrored) { p.z = -p.z; }
            lattice.at(u, v) = p;
        }
    }
}

} // namespace tconv
