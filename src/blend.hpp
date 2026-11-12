// Blended joins between sheets.
//
// Where a tube (handle, spout) passes through the body, the two surfaces
// meet along a closed curve. The join is a third surface that interpolates
// between them: at parameter tau in [0,1] across a band,
//
//     C(k, tau) = (1 - beta(tau)) A(k, tau) + beta(tau) B(k, tau),
//
// where A walks on the body from the band's outer edge to the curve, B walks
// on the tube from the curve to the band's far edge, and beta rises from 0
// to 1 with zero first and second derivative at both ends. At tau = 0 the
// join is the body, at tau = 1 it is the tube, with matching tangent and
// curvature across the band's edges; on the curve the two surfaces are in
// unison. The buried parts of the tube and the covered part of the body are
// trimmed away.
#pragma once
#include "surface.hpp"
#include <array>
#include <cmath>
#include <vector>

namespace tconv {

// A region of a surface's parameter domain that is not shown.
struct Trim {
    // Inside this closed polygon of (u, v) points; u may run past the period.
    std::vector<std::array<double, 2>> loop;
    double centre_u = 0.0, period_u = 0.0;
    // Or beyond a cut that varies around a tube: v < cut(u) or v > cut(u).
    std::vector<double> cut; // one value per node column
    bool below = false;
    // The same boundaries at the band's own edge, with no overlap. Drawing
    // uses the ones above, which stop inside the band so no gap shows;
    // counting how often a ray crosses the surface (for a cutaway) needs
    // every point to belong to exactly one sheet, and uses these.
    std::vector<std::array<double, 2>> loop_exact;
    std::vector<double> cut_exact;

    bool hides(double u, double v, bool exact = false) const {
        const std::vector<double>& cut = exact && !cut_exact.empty() ? cut_exact : this->cut;
        const std::vector<std::array<double, 2>>& loop = exact && !loop_exact.empty() ? loop_exact : this->loop;
        if (!cut.empty()) {
            const double n = double(cut.size());
            double x = u - std::floor(u / n) * n;
            const std::size_t k = std::size_t(x) % cut.size();
            const double f = x - std::floor(x);
            const double c = cut[k] * (1.0 - f) + cut[(k + 1) % cut.size()] * f;
            return below ? v < c : v > c;
        }
        if (loop.empty()) { return false; }
        if (period_u > 0.0) { u -= period_u * std::round((u - centre_u) / period_u); }
        bool inside = false;
        for (std::size_t i = 0, j = loop.size() - 1; i < loop.size(); j = i++) {
            const double ui = loop[i][0], vi = loop[i][1], uj = loop[j][0], vj = loop[j][1];
            if ((vi > v) != (vj > v) && u < (uj - ui) * (v - vi) / (vj - vi) + ui) { inside = !inside; }
        }
        return inside;
    }
};

struct JoinResult {
    bool ok = false;
    Lattice blend;   // the join surface: closed around, `across` rows across the band
    Lattice toward;  // with `two_sided`: `blend` holds the body's side of the band and this the tube's
    Trim body_trim;  // the part of the body under the join
    Trim tube_trim;  // the buried part of the tube
};

// The body as a surface of revolution, queried by height on one branch.
struct BodyBranch {
    const Surface* body = nullptr;
    double v_low = 0.0, v_high = 0.0; // rows over which height is monotone
    bool rising = true;

    // Row coordinate at height y, by bisection.
    double row_at(double y) const {
        double a = v_low, b = v_high;
        for (int k = 0; k < 48; ++k) {
            const double m = 0.5 * (a + b);
            const bool above = body->position(0.0, m).y > y;
            if (above == rising) { b = m; } else { a = m; }
        }
        return 0.5 * (a + b);
    }
    double radius_at(double y) const {
        const Vec3 p = body->position(0.0, row_at(y));
        return std::hypot(p.x, p.z);
    }
};

// Join the end of `tube` at row `end_row` (0 or its last row) to the body.
// `across` is the number of rows across the band. The band's edge tangent is
// taken from its first four rows, over which the blend weight must still be
// small: where the two surfaces are far apart for the band's width (the
// spout's channel inside the body) nine rows are too few and the band
// leaves the body at an angle.
// `inside_is_smaller_radius` is true for the exterior branch (the tube is
// buried where its radius is below the body's) and false for the interior.
inline JoinResult make_join(const Surface& body, const BodyBranch& branch, bool inside_is_smaller_radius, const Surface& tube,
                            bool at_first_row, double width, std::size_t body_nu, int across = 9, bool two_sided = false) {
    JoinResult result;
    const std::size_t columns = tube.cells_u();
    const double rows = double(tube.cells_v());
    const double direction = at_first_row ? 1.0 : -1.0; // along the tube, away from its buried end
    const double end = at_first_row ? 0.0 : rows;
    const auto buried = [&](double u, double w) {
        const Vec3 p = tube.position(u, w);
        const double r = std::hypot(p.x, p.z);
        const double R = branch.radius_at(p.y);
        return inside_is_smaller_radius ? r < R : r > R;
    };
    std::vector<double> crossing(columns), body_u(columns), body_v(columns);
    // Where the end's ring is inside the body the join starts at the
    // intersection; where it only touches, at the tube's end.
    // An end that is nowhere near the body is not joined.
    for (std::size_t k = 0; k < columns; ++k) {
        const Vec3 p = tube.position(double(k), end);
        if (std::fabs(std::hypot(p.x, p.z) - branch.radius_at(p.y)) > 0.35) { return result; }
    }
    for (std::size_t k = 0; k < columns; ++k) {
        const double u = double(k);
        double a = end, b = end;
        bool found = !buried(u, end);
        for (double step = 0.25; !found && step <= 0.6 * rows; step += 0.25) {
            b = end + direction * step;
            if (!buried(u, b)) {
                found = true;
                break;
            }
            a = b;
        }
        if (!found) { return result; }
        for (int it = 0; it < 44; ++it) {
            const double m = 0.5 * (a + b);
            if (buried(u, m)) { a = m; } else { b = m; }
        }
        crossing[k] = 0.5 * (a + b);
        const Vec3 p = tube.position(u, crossing[k]);
        body_v[k] = branch.row_at(p.y);
        double theta = std::atan2(-p.z, p.x); // the sheet is (r cos, y, -r sin)
        double bu = theta / 6.283185307179586 * double(body_nu);
        if (k > 0) { bu -= double(body_nu) * std::round((bu - body_u[k - 1]) / double(body_nu)); }
        body_u[k] = bu;
    }
    // Centre of the loop on the body, in parameters.
    double centre_u = 0.0, centre_v = 0.0;
    for (std::size_t k = 0; k < columns; ++k) {
        centre_u += body_u[k];
        centre_v += body_v[k];
    }
    centre_u /= double(columns);
    centre_v /= double(columns);
    const auto beta = [](double t) { return t * t * t * (10.0 + t * (6.0 * t - 15.0)); };
    result.blend.wrap_u = true;
    result.blend.closure = Closure::cubic;
    result.blend.nu = columns;
    result.blend.nv = std::size_t(across);
    result.blend.nodes.resize(columns * std::size_t(across));
    if (two_sided) { result.toward = result.blend; }
    result.tube_trim.cut.resize(columns);
    result.tube_trim.below = at_first_row;
    result.body_trim.centre_u = centre_u;
    result.body_trim.period_u = double(body_nu);
    for (std::size_t k = 0; k < columns; ++k) {
        const double u = double(k);
        // Metric on the tube along its length, and on the body in both directions.
        const SurfacePoint on_tube = tube.eval(u, crossing[k]);
        const double tube_speed = length(on_tube.dv);
        const SurfacePoint on_body = body.eval(body_u[k], body_v[k]);
        const double speed_u = length(on_body.du), speed_v = length(on_body.dv);
        double out_u = (body_u[k] - centre_u) * speed_u, out_v = (body_v[k] - centre_v) * speed_v;
        const double norm = std::hypot(out_u, out_v);
        out_u /= norm;
        out_v /= norm;
        // A walk of a given length along each surface, in short steps that
        // read the surface's own speed as they go. One step from the speed
        // at the start is off where rows are graded or the surface turns,
        // and the band then misses the surface it should leave tangentially.
        const int steps = 8;
        const auto on_body_at = [&](double distance) {
            std::array<double, 2> q{body_u[k], body_v[k]};
            for (int step = 0; step < steps; ++step) {
                const SurfacePoint here = body.eval(q[0], q[1]);
                q[0] += distance / steps * out_u / std::max(1e-9, length(here.du));
                q[1] += distance / steps * out_v / std::max(1e-9, length(here.dv));
            }
            return q;
        };
        const auto on_tube_at = [&](double distance) {
            double w = crossing[k];
            for (int step = 0; step < steps; ++step) { w += direction * distance / steps / std::max(1e-9, length(tube.eval(u, w).dv)); }
            return w;
        };
        (void)speed_u;
        (void)speed_v;
        (void)tube_speed;
        for (int j = 0; j < across; ++j) {
            const double tau = double(j) / double(across - 1);
            const std::array<double, 2> q = on_body_at(width * (1.0 - tau));
            const Vec3 a = body.position(q[0], q[1]);
            const Vec3 b = tube.position(u, on_tube_at(width * tau));
            if (two_sided) {
                // The two sides are kept apart, as where they are on the body and
                // along the tube, and blended at evaluation (ConvSurface::toward, on).
                (void)a;
                (void)b;
                result.blend.at(k, std::size_t(j)) = Vec3{q[0], q[1], 0.0};
                result.toward.at(k, std::size_t(j)) = Vec3{on_tube_at(width * tau), 0.0, 0.0};
            } else {
                result.blend.at(k, std::size_t(j)) = a * (1.0 - beta(tau)) + b * beta(tau);
            }
        }
        // Trims stop inside the band so that no gap shows at its edges. The
        // band begins tangent to the surface it leaves, so the overlap is two
        // coincident sheets.
        const double keep = 0.93 * width;
        result.tube_trim.cut[k] = on_tube_at(keep);
        result.body_trim.loop.push_back(on_body_at(keep));
        result.tube_trim.cut_exact.push_back(on_tube_at(width));
        result.body_trim.loop_exact.push_back(on_body_at(width));
    }
    // The cut on the body is a smooth closed curve through those points, not
    // the polygon with straight sides between them: where the body's rows
    // are dense a straight side in (u, v) bows out past the band's edge and
    // leaves a sliver of the body cut away with nothing over it.
    for (std::vector<std::array<double, 2>>* outline : {&result.body_trim.loop, &result.body_trim.loop_exact}) {
        const std::vector<std::array<double, 2>> corner = *outline;
        const std::size_t n = corner.size();
        outline->clear();
        for (std::size_t k = 0; k < n; ++k) {
            const auto &p0 = corner[(k + n - 1) % n], &p1 = corner[k], &p2 = corner[(k + 1) % n], &p3 = corner[(k + 2) % n];
            for (int step = 0; step < 8; ++step) {
                const double t = step / 8.0, t2 = t * t, t3 = t2 * t;
                std::array<double, 2> q{};
                for (std::size_t c = 0; c < 2; ++c) {
                    q[c] = 0.5 * (p0[c] * (-t3 + 2 * t2 - t) + p1[c] * (3 * t3 - 5 * t2 + 2) + p2[c] * (-3 * t3 + 4 * t2 + t) + p3[c] * (t3 - t2));
                }
                outline->push_back(q);
            }
        }
    }
    result.ok = true;
    return result;
}

} // namespace tconv
