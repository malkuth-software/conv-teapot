// Times the surface evaluation alone: every part of a teapot at the same
// scattered parameters, with a checksum of what came back, so that a change
// to the evaluator can be shown to return the same numbers faster.
#include "../src/teapot.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    using namespace tconv;
    TeapotOptions options;
    options.version = argc > 1 && std::strcmp(argv[1], "1976") == 0 ? Version::v1976 : Version::v2026_unified;
    const Teapot teapot = make_teapot(options);
    const int count = argc > 2 ? std::atoi(argv[2]) : 400000;
    for (const auto& part : teapot.parts) {
        const Surface& surface = part->conv;
        const double cu = double(surface.cells_u()), cv = double(surface.cells_v());
        double best = 1e30, sum = 0.0;
        for (int round = 0; round < 7; ++round) {
            std::uint32_t state = 12345u;
            sum = 0.0;
            const auto start = std::chrono::steady_clock::now();
            for (int k = 0; k < count; ++k) {
                state = state * 1664525u + 1013904223u;
                const double u = cu * double(state >> 8) / 16777216.0;
                state = state * 1664525u + 1013904223u;
                const double v = cv * double(state >> 8) / 16777216.0;
                const SurfacePoint p = surface.eval(u, v);
                sum += p.p.x + p.p.y + p.p.z + p.du.x + p.du.y + p.du.z + p.dv.x + p.dv.y + p.dv.z;
            }
            best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / count);
        }
        std::printf("%-12s %7.1f ns an evaluation, sum %.17g\n", part->name.c_str(), best, sum);
    }
    return 0;
}
