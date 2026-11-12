// CONV through turned nodes against the turned CONV surface.
//
// Admission and the bubble act on each coordinate separately, so they do not
// commute with a rotation. This measures, for a turn about the z axis, how
// far the surface built through the turned nodes is from the turned surface,
// in position and in the direction of the normal.
#include "../src/teapot.hpp"
#include <cstdio>
using namespace tconv;

int main() {
    std::printf("%-22s %6s | %-22s | %-22s\n", "version", "turn", "distance max / rms", "normal max / rms (deg)");
    for (const Version version : {Version::v1976, Version::v2026_unified}) {
        for (const double degrees : {30.0, 60.0, 90.0, 120.0}) {
            TeapotOptions options;
            options.version = version;
            const Teapot rest = make_teapot(options);
            Teapot turned = make_teapot(options);
            const double a = degrees * M_PI / 180.0;
            Rigid turn;
            turn.a[0] = std::cos(a);
            turn.a[1] = -std::sin(a);
            turn.a[3] = std::sin(a);
            turn.a[4] = std::cos(a);
            double worst = 0, sum2 = 0, worst_angle = 0, angle2 = 0;
            std::size_t count = 0;
            for (std::size_t k = 0; k < rest.parts.size(); ++k) {
                for (Vec3& node : turned.parts[k]->conv.lattice.nodes) { node = turn.point(node); }
                turned.parts[k]->conv.compile();
                const ConvSurface &before = rest.parts[k]->conv, &after = turned.parts[k]->conv;
                for (std::size_t j = 0; j < before.cells_v(); ++j) {
                    for (std::size_t i = 0; i < before.cells_u(); ++i) {
                        for (double fv : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                            for (double fu : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                                const SurfacePoint p = before.eval(double(i) + fu, double(j) + fv), q = after.eval(double(i) + fu, double(j) + fv);
                                const double d = length(turn.point(p.p) - q.p);
                                worst = std::max(worst, d);
                                sum2 += d * d;
                                const Vec3 n0 = cross(p.du, p.dv), n1 = cross(q.du, q.dv);
                                if (length(n0) > 1e-9 && length(n1) > 1e-9) {
                                    const double angle = std::acos(std::max(-1.0, std::min(1.0, dot(normalize(turn.direction(n0)), normalize(n1))))) * 180.0 / M_PI;
                                    worst_angle = std::max(worst_angle, angle);
                                    angle2 += angle * angle;
                                }
                                ++count;
                            }
                        }
                    }
                }
            }
            std::printf("%-22s %6.0f | %10.2e %10.2e | %10.3f %10.4f\n", version == Version::v1976 ? "1976, exact" : "2026, unified sheets", degrees, worst,
                        std::sqrt(sum2 / double(count)), worst_angle, std::sqrt(angle2 / double(count)));
        }
    }
    return 0;
}
