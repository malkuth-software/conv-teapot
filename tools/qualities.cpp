// The qualities the later Utah Teapot versions worked on, measured on the
// Bezier original and on the CONV teapots: roundness of the body, tangent
// and curvature continuity across joints, crease fidelity, and deviation.
#include "../src/teapot.hpp"
#include <cstdio>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
using namespace tconv;

using Curve = std::function<Vec3(double)>;

// Tangent direction and curvature of a curve on one side of a parameter.
static void one_sided(const Curve& c, double t, double h, Vec3& tangent, double& curvature) {
    const Vec3 p0 = c(t), p1 = c(t + h), p2 = c(t + 2 * h), p3 = c(t + 3 * h);
    const Vec3 d1 = (p1 * 18.0 - p0 * 11.0 - p2 * 9.0 + p3 * 2.0) * (1.0 / (6.0 * h));
    const Vec3 d2 = (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (1.0 / (h * h));
    tangent = normalize(d1);
    const double speed = length(d1);
    curvature = length(cross(d1, d2)) / (speed * speed * speed);
}

struct Joint { double angle = 0.0, jump = 0.0, relative = 0.0; };

static void probe(const Curve& c, double t, Joint& worst) {
    Vec3 ta, tb;
    double ka, kb;
    one_sided(c, t, -2e-3, ta, ka);
    one_sided(c, t, 2e-3, tb, kb);
    const double cosine = std::fmax(-1.0, std::fmin(1.0, dot(ta, tb)));
    worst.angle = std::fmax(worst.angle, std::acos(cosine) * 180.0 / M_PI);
    worst.jump = std::fmax(worst.jump, std::fabs(ka - kb));
    worst.relative = std::fmax(worst.relative, std::fabs(ka - kb) / std::fmax(1e-9, std::fmax(ka, kb)));
}

// Worst discontinuity across every interior node line of a surface, away
// from the rows listed as creases.
static void joints(const Surface& s, const std::vector<unsigned char>& crease_rows, std::size_t every_u, std::size_t every_v,
                   Joint& along_v, Joint& along_u) {
    const std::size_t cu = s.cells_u(), cv = s.cells_v();
    for (std::size_t row = every_v; row < cv; row += every_v) {
        if (!crease_rows.empty() && crease_rows[row] != 0) { continue; }
        for (int a = 0; a < 24; ++a) {
            const double u = double(cu) * (a + 0.31) / 24.0;
            probe([&](double v) { return s.position(u, v); }, double(row), along_v);
        }
    }
    for (std::size_t column = 0; column < cu; column += every_u) {
        for (int b = 0; b < 24; ++b) {
            const double v = double(cv) * (b + 0.43) / 24.0;
            // Stay clear of crease rows, where the cross derivative also breaks.
            bool near_crease = false;
            for (std::size_t row = 0; row < crease_rows.size(); ++row) {
                if (crease_rows[row] != 0 && std::fabs(v - double(row)) < 0.05) { near_crease = true; }
            }
            if (near_crease) { continue; }
            probe([&](double u) { return s.position(u, v); }, double(column), along_u);
        }
    }
}

static double out_of_round(const std::function<Vec3(double)>& ring) {
    double low = 1e30, high = 0.0;
    for (int k = 0; k < 2880; ++k) {
        const Vec3 p = ring(double(k) / 2880.0);
        const double r = std::hypot(p.x, p.z);
        low = std::fmin(low, r);
        high = std::fmax(high, r);
    }
    return (high - low) / (0.5 * (high + low));
}

int main() {
    std::puts("1. Roundness of the body at its widest ring: (max r - min r) / mean r");
    {
        const Teapot exact = make_teapot({});
        const TeapotPart& body = *exact.parts[0];
        const double v = 8.0; // joint of the upper and lower body, radius 2
        std::printf("   Bezier original                         %.3e\n", out_of_round([&](double a) { return body.bezier.position(a * 16.0, v); }));
        std::printf("   CONV exact (per 4)                      %.3e\n", out_of_round([&](double a) { return body.conv.position(a * 16.0, v); }));
        for (std::size_t per : {3u, 4u, 6u, 8u}) {
            TeapotOptions o;
            o.per_u = o.per_v = per;
            o.circular = true;
            const Teapot round = make_teapot(o);
            const TeapotPart& b = *round.parts[0];
            double worst = 0.0;
            for (const auto& part : round.parts) {
                if (!part->revolved) { continue; }
                for (double v = 0.0; v <= double(part->conv.cells_v()); v += 0.125) {
                    for (double u = 0.0; u < double(part->conv.cells_u()); u += 0.125) {
                        worst = std::fmax(worst, length(part->conv.position(u, v) - part->reference(u, v)));
                    }
                }
            }
            std::printf("   CONV round, %2zu nodes per turn               %.3e   (worst distance of body and lid from the exact revolved profile: %.3e)\n",
                        4 * per, out_of_round([&](double a) { return b.conv.position(a * double(b.conv.cells_u()), 2.0 * double(per)); }), worst);
        }
    }
    std::puts("\n2. Continuity across joints away from creases (worst over the teapot)");
    std::puts("   surface                       tangent turn (deg)   curvature jump (abs, relative)");
    for (int mode = 0; mode < 4; ++mode) {
        TeapotOptions o;
        o.smooth = mode == 3;
        o.circular = mode == 2;
        const Teapot teapot = make_teapot(o);
        Joint v_joint, u_joint;
        for (const auto& part : teapot.parts) {
            if (mode == 0) {
                // Bezier: its patch joints. Creases are the rows the smooth style detects.
                TeapotOptions so;
                so.smooth = true;
                const Teapot reference = make_teapot(so);
                std::vector<unsigned char> creases;
                for (const auto& other : reference.parts) {
                    if (other->name == part->name) { creases = other->conv.lattice.seams_v; }
                }
                joints(part->bezier, creases, part->per_u, part->per_v, v_joint, u_joint);
            } else {
                std::vector<unsigned char> creases = part->conv.lattice.seams_v;
                if (mode == 1 || mode == 2) {
                    // Exact style: seams sit at every patch joint; only the true creases are excluded.
                    TeapotOptions so;
                    so.smooth = true;
                    const Teapot reference = make_teapot(so);
                    for (const auto& other : reference.parts) {
                        if (other->name == part->name) { creases = other->conv.lattice.seams_v; }
                    }
                }
                joints(part->conv, creases, 1, 1, v_joint, u_joint);
            }
        }
        const char* names[4] = {"Bezier original", "CONV exact", "CONV round", "CONV smooth (experimental)"};
        std::printf("   %-28s  along v %8.4f   %.3e  %.3e\n", names[mode], v_joint.angle, v_joint.jump, v_joint.relative);
        std::printf("   %-28s  around  %8.4f   %.3e  %.3e\n", "", u_joint.angle, u_joint.jump, u_joint.relative);
    }
    std::puts("\n3. Deviation of the smooth CONV teapot (model units; body 4.0 wide)");
    std::puts("   per  nodes   from its reference (revolved, arc length)     from the Bezier original");
    for (std::size_t per : {3u, 4u, 6u, 8u}) {
        TeapotOptions o;
        o.per_u = o.per_v = per;
        o.smooth = true;
        const Teapot teapot = make_teapot(o);
        double worst = 0.0, sum2 = 0.0, worst_b = 0.0, sum2_b = 0.0;
        std::size_t count = 0;
        for (const auto& part : teapot.parts) {
            const std::size_t cu = part->conv.cells_u(), cv = part->conv.cells_v();
            for (std::size_t j = 0; j <= cv * 8; ++j) {
                for (std::size_t i = 0; i < cu * 8; ++i) {
                    const double u = double(i) / 8.0, v = double(j) / 8.0;
                    const Vec3 p = part->conv.position(u, v);
                    const double d = length(p - part->reference(u, v));
                    const double db = length(p - part->bezier.position(u, part->reference_v(v)));
                    worst = std::fmax(worst, d);
                    sum2 += d * d;
                    worst_b = std::fmax(worst_b, db);
                    sum2_b += db * db;
                    ++count;
                }
            }
        }
        std::printf("   %3zu  %5zu   max %.3e  rms %.3e              max %.3e  rms %.3e\n", per, teapot.node_count(), worst,
                    std::sqrt(sum2 / double(count)), worst_b, std::sqrt(sum2_b / double(count)));
    }
    std::puts("\n4. Creases kept (smooth style, per 4): turn of the tangent along v at each seam row");
    {
        TeapotOptions o;
        o.smooth = true;
        const Teapot teapot = make_teapot(o);
        for (const auto& part : teapot.parts) {
            const Lattice& lattice = part->conv.lattice;
            for (std::size_t row = 1; row + 1 < lattice.nv; ++row) {
                if (lattice.seams_v[row] == 0) { continue; }
                Joint conv, bezier;
                for (int a = 0; a < 16; ++a) {
                    const double u = double(lattice.nu) * (a + 0.31) / 16.0;
                    probe([&](double v) { return part->conv.position(u, v); }, double(row), conv);
                    probe([&](double v) { return part->bezier.position(u, v); }, part->row_param[row], bezier);
                }
                std::printf("   %-6s row %2zu: Bezier %.3f deg, CONV smooth %.3f deg\n", part->name.c_str(), row, bezier.angle, conv.angle);
            }
        }
        // Flatness of the cap: the lowest point of the body.
        double lowest = 1e30;
        const TeapotPart& body = *teapot.parts[0];
        for (double v = 0.0; v <= double(body.conv.cells_v()); v += 0.01) {
            for (double u = 0.0; u < 16.0; u += 0.37) { lowest = std::fmin(lowest, body.conv.position(u, v).y); }
        }
        std::printf("   lowest point of the body (the cap is the plane y = 0): %.3e\n", lowest);
    }
    std::puts("\n5. The 2026 version from surface samples (CONV lattice per patch)");
    {
        // Hold-out: lattices built from every other sample, checked at the samples left out.
        TeapotOptions coarse;
        coarse.version = Version::v2026;
        coarse.per_u = coarse.per_v = 4;
        const Teapot teapot = make_teapot(coarse);
        double worst[2] = {0.0, 0.0}, sum2[2] = {0.0, 0.0};
        std::size_t count[2] = {0, 0};
        for (std::size_t k = 0; k < teapot.parts.size(); ++k) {
            const TeapotPart& part = *teapot.parts[k];
            const int which = part.interior ? 1 : 0;
            for (int j = 0; j <= 8; ++j) {
                for (int i = 0; i <= 8; ++i) {
                    if (i % 2 == 0 && j % 2 == 0) { continue; }
                    const float* p = utah::utah2026_points[k][j * 9 + i];
                    const double d = length(part.conv.position(0.5 * i, 0.5 * j) - Vec3{p[0], p[1], p[2]});
                    worst[which] = std::fmax(worst[which], d);
                    sum2[which] += d * d;
                    ++count[which];
                }
            }
        }
        std::printf("   hold-out error of 5x5 nodes per patch at the 9x9 samples: exterior max %.3e rms %.3e; interior max %.3e rms %.3e\n",
                    worst[0], std::sqrt(sum2[0] / double(count[0])), worst[1], std::sqrt(sum2[1] / double(count[1])));
    }
    std::puts("   continuity across patch joints (exterior), 9x9 nodes per patch:");
    for (int which = 0; which < 2; ++which) {
        // Per-patch lattices of the 1976 teapot (from its Bezier patches) and of the 2026 samples.
        std::vector<std::unique_ptr<ConvSurface>> surfaces;
        if (which == 0) {
            TeapotOptions o;
            o.per_u = o.per_v = 8;
            o.bottom_cap = false;
            const Teapot source = make_teapot(o);
            for (const auto& part : source.parts) {
                for (std::size_t pv = 0; pv < part->bezier.patches_v; ++pv) {
                    for (std::size_t pu = 0; pu < part->bezier.patches_u; ++pu) {
                        auto s = std::make_unique<ConvSurface>();
                        s->lattice.nu = s->lattice.nv = 9;
                        s->lattice.nodes.resize(81);
                        for (std::size_t j = 0; j < 9; ++j) {
                            for (std::size_t i = 0; i < 9; ++i) { s->lattice.at(i, j) = part->bezier.position(double(pu * 8 + i), double(pv * 8 + j)); }
                        }
                        s->compile();
                        surfaces.push_back(std::move(s));
                    }
                }
            }
        } else {
            TeapotOptions o;
            o.version = Version::v2026;
            o.per_u = o.per_v = 8;
            Teapot source = make_teapot(o);
            for (auto& part : source.parts) {
                if (part->interior) { continue; }
                auto s = std::make_unique<ConvSurface>();
                s->lattice = part->conv.lattice;
                s->compile();
                surfaces.push_back(std::move(s));
            }
        }
        // An edge of a patch: a point at parameter a in [0,1] along it, and the inward cross direction.
        struct Edge { int surface, side; };
        const auto point = [&](const Edge& e, double a, double inward) {
            const ConvSurface& s = *surfaces[std::size_t(e.surface)];
            const double t = 8.0 * a;
            switch (e.side) {
            case 0: return s.eval(t, inward);
            case 1: return s.eval(t, 8.0 - inward);
            case 2: return s.eval(inward, t);
            default: return s.eval(8.0 - inward, t);
            }
        };
        std::vector<Edge> edges;
        for (int k = 0; k < int(surfaces.size()); ++k) {
            for (int side = 0; side < 4; ++side) { edges.push_back({k, side}); }
        }
        double worst_angle = 0.0, worst_jump = 0.0, worst_gap = 0.0, sum_angle = 0.0, sum_relative = 0.0;
        std::vector<double> relatives;
        std::size_t joints = 0, samples = 0;
        for (std::size_t a = 0; a < edges.size(); ++a) {
            const Vec3 a0 = point(edges[a], 0.0, 0.0).p, a1 = point(edges[a], 1.0, 0.0).p, am = point(edges[a], 0.5, 0.0).p;
            if (length(a1 - a0) < 1e-2) { continue; } // a pole
            for (std::size_t b = a + 1; b < edges.size(); ++b) {
                if (edges[b].surface == edges[a].surface) { continue; }
                const Vec3 b0 = point(edges[b], 0.0, 0.0).p, b1 = point(edges[b], 1.0, 0.0).p, bm = point(edges[b], 0.5, 0.0).p;
                const bool same = length(a0 - b0) < 1e-4 && length(a1 - b1) < 1e-4;
                const bool flipped = length(a0 - b1) < 1e-4 && length(a1 - b0) < 1e-4;
                if ((!same && !flipped) || length(am - bm) > 1e-3) { continue; }
                ++joints;
                for (int k = 1; k < 16; ++k) {
                    const double ta = k / 16.0, tb = same ? ta : 1.0 - ta;
                    const SurfacePoint pa = point(edges[a], ta, 1e-4), pb = point(edges[b], tb, 1e-4);
                    worst_gap = std::fmax(worst_gap, length(point(edges[a], ta, 0.0).p - point(edges[b], tb, 0.0).p));
                    const Vec3 na = normalize(cross(pa.du, pa.dv)), nb = normalize(cross(pb.du, pb.dv));
                    const double angle = std::acos(std::fmin(1.0, std::fabs(dot(na, nb)))) * 180.0 / M_PI;
                    worst_angle = std::fmax(worst_angle, angle);
                    sum_angle += angle;
                    ++samples;
                    // Curvature of the curve that crosses the joint, on each side.
                    Vec3 tangent;
                    double ka, kb;
                    one_sided([&](double x) { return point(edges[a], ta, x).p; }, 0.0, 4e-3, tangent, ka);
                    one_sided([&](double x) { return point(edges[b], tb, x).p; }, 0.0, 4e-3, tangent, kb);
                    worst_jump = std::fmax(worst_jump, std::fabs(ka - kb));
                    const double relative = std::fabs(ka - kb) / std::fmax(1e-6, std::fmax(ka, kb));
                    sum_relative += relative;
                    relatives.push_back(relative);
                }
            }
        }
        std::sort(relatives.begin(), relatives.end());
        std::printf("   %-22s %3zu patches, %3zu joints: gap max %.1e; normal turn max %.4f deg, mean %.4f deg;\n"
                    "   %-22s curvature jump across the joint: max %.3e; relative: median %.3f, mean %.3f\n",
                    which == 0 ? "1976 (Bezier samples)" : "2026 (page samples)", surfaces.size(), joints, worst_gap, worst_angle,
                    sum_angle / double(std::max<std::size_t>(1, samples)), "", worst_jump, relatives.empty() ? 0.0 : relatives[relatives.size() / 2],
                    sum_relative / double(std::max<std::size_t>(1, samples)));
    }
    return 0;
}
