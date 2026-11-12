// What the surface itself says a mesh must cost.
//
// For flat triangles through points of a smooth surface, the two classical
// asymptotic counts are set by two integrals of the Gaussian curvature K:
//   distance   triangles for a largest distance e:   T = (2 / (3 sqrt 3)) * Int sqrt|K| dA / e
//   normal     triangles whose edges turn the normal by at most a:  T = (4 / sqrt 3) * Int |K| dA / a^2
// Both assume triangles stretched along the principal directions by
// sqrt(k1 / k2). This tool integrates both over the unified teapot and sets
// the predicted counts beside what the tree layout actually used.
#include "../src/teapot.hpp"
#include <cstdio>
using namespace tconv;

int main() {
    TeapotOptions o;
    o.version = Version::v2026_unified;
    const Teapot teapot = make_teapot(o);
    double total_area = 0, total_root = 0, total_abs = 0, total_iso = 0;
    std::printf("%-12s %8s %10s %10s %9s %9s | %s\n", "part", "area", "Int sqrt|K|", "Int |K|", "elliptic", "ratio", "Int kmax^2 (isotropic normal cost)");
    for (const auto& part : teapot.parts) {
        const ConvSurface& s = part->conv;
        double area = 0, root = 0, absolute = 0, elliptic = 0, log_ratio = 0, weight = 0, iso = 0;
        const int per = 4;
        const double h = 1e-3;
        for (std::size_t j = 0; j < s.cells_v(); ++j) {
            for (std::size_t i = 0; i < s.cells_u(); ++i) {
                for (int b = 0; b < per; ++b) {
                    for (int a = 0; a < per; ++a) {
                        const double u = double(i) + (a + 0.5) / per, v = double(j) + (b + 0.5) / per;
                        const SurfacePoint p = s.eval(u, v);
                        const SurfacePoint pu0 = s.eval(u - h, v), pu1 = s.eval(u + h, v), pv0 = s.eval(u, v - h), pv1 = s.eval(u, v + h);
                        const Vec3 suu = (pu1.du - pu0.du) * (0.5 / h), svv = (pv1.dv - pv0.dv) * (0.5 / h);
                        const Vec3 suv = ((pv1.du - pv0.du) + (pu1.dv - pu0.dv)) * (0.25 / h);
                        const double E = dot(p.du, p.du), F = dot(p.du, p.dv), G = dot(p.dv, p.dv);
                        const double det = E * G - F * F;
                        if (det < 1e-14) { continue; }
                        const Vec3 n = normalize(cross(p.du, p.dv));
                        const double L = dot(suu, n), M = dot(suv, n), N = dot(svv, n);
                        const double K = (L * N - M * M) / det, H = 0.5 * (E * N - 2 * F * M + G * L) / det;
                        const double disc = std::sqrt(std::max(0.0, H * H - K));
                        const double k1 = std::fabs(H) + disc, k2 = std::fabs(std::fabs(H) - disc);
                        const double dA = std::sqrt(det) / double(per * per);
                        area += dA;
                        root += std::sqrt(std::fabs(K)) * dA;
                        absolute += std::fabs(K) * dA;
                        iso += k1 * k1 * dA;
                        if (K > 0) { elliptic += dA; }
                        if (k2 > 1e-6) {
                            log_ratio += std::log(k1 / k2) * std::fabs(K) * dA;
                            weight += std::fabs(K) * dA;
                        }
                    }
                }
            }
        }
        std::printf("%-12s %8.2f %10.2f %10.2f %8.0f%% %9.1f | %.1f\n", part->name.c_str(), area, root, absolute, 100.0 * elliptic / area,
                    std::exp(log_ratio / std::max(1e-12, weight)), iso);
        total_area += area;
        total_root += root;
        total_abs += absolute;
        total_iso += iso;
    }
    std::printf("%-12s %8.2f %10.2f %10.2f   (4 pi = %.2f)             | %.1f\n", "all", total_area, total_root, total_abs, 4 * M_PI, total_iso);
    std::printf("\ndistance: triangles = 0.385 * Int sqrt|K| dA / e;  vertices about half\n");
    for (double e : {2e-2, 1e-2, 5e-3, 2e-3, 1e-3}) { std::printf("  e = %.0e: %8.0f triangles, %8.0f vertices\n", e, 0.3849 * total_root / e, 0.5 * 0.3849 * total_root / e); }
    std::printf("normal turn along an edge: triangles = 2.309 * Int |K| dA / a^2 (anisotropic), 2.309 * Int kmax^2 dA / a^2 (equilateral)\n");
    for (double deg : {30.0, 20.0, 10.0, 5.0}) {
        const double a = deg * M_PI / 180.0;
        std::printf("  a = %2.0f deg: %8.0f triangles anisotropic, %8.0f equilateral\n", deg, 2.3094 * total_abs / (a * a), 2.3094 * total_iso / (a * a));
    }
    return 0;
}
