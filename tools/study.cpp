// Geometry study: deviation of the CONV teapot from the Bezier teapot, and
// continuity of the surface under the two bubble limiters.
#include "../src/teapot.hpp"
#include <cstdio>
#include <random>
using namespace tconv;

struct Deviation { double worst = 0.0, sum2 = 0.0; std::size_t count = 0; };

static void measure(const Surface& a, const Surface& b, int fine, Deviation& out) {
    const std::size_t cu = a.cells_u(), cv = a.cells_v();
    for (std::size_t j = 0; j <= cv * std::size_t(fine); ++j) {
        for (std::size_t i = 0; i < cu * std::size_t(fine); ++i) {
            const double u = double(i) / fine, v = double(j) / fine;
            const double d = length(a.position(u, v) - b.position(u, v));
            out.worst = std::fmax(out.worst, d);
            out.sum2 += d * d;
            ++out.count;
        }
    }
}

// Largest position jump between samples `step` apart along u.
static double jump(const Surface& s, double step, std::mt19937_64& rng, int lines) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double worst = 0.0;
    for (int line = 0; line < lines; ++line) {
        const double v = uniform(rng) * double(s.cells_v());
        const double u0 = uniform(rng) * double(s.cells_u());
        Vec3 previous = s.position(u0, v);
        for (int k = 1; k <= 4000; ++k) {
            const Vec3 next = s.position(u0 + k * step, v);
            worst = std::fmax(worst, length(next - previous));
            previous = next;
        }
    }
    return worst;
}

int main() {
    std::puts("deviation from the Bezier teapot (model units; the body is 4.0 wide)");
    std::puts("per  nodes   bubble        max        rms     | linear max   linear rms");
    for (int mode = 0; mode < 3; ++mode) {
      std::printf("-- %s\n", mode == 0 ? "seamless, paper closure" : mode == 1 ? "seamless, cubic closure" : "seams at patch boundaries, cubic closure");
      for (std::size_t per : {2u, 3u, 4u, 6u, 8u, 12u}) {
        for (Bubble bubble : {Bubble::none, Bubble::continuous}) {
            TeapotOptions options;
            options.per_u = per;
            options.per_v = per;
            options.bubble = bubble;
            options.seams = mode == 2;
            options.closure = mode == 0 ? Closure::paper : Closure::cubic;
            if (mode == 0 && per < 2) { continue; }
            const Teapot teapot = make_teapot(options);
            Deviation conv, linear;
            for (const auto& part : teapot.parts) {
                measure(part->conv, part->bezier, 8, conv);
                measure(part->linear, part->bezier, 8, linear);
            }
            std::printf("%3zu  %5zu   %-10s  %.3e  %.3e | %.3e   %.3e\n", per, teapot.node_count(),
                        bubble == Bubble::none ? "none" : bubble == Bubble::standard ? "standard" : "continuous",
                        conv.worst, std::sqrt(conv.sum2 / double(conv.count)), linear.worst,
                        std::sqrt(linear.sum2 / double(linear.count)));
        }
      }
    }
    std::puts("\nper-part deviation at per = 4, continuous bubble");
    {
        const Teapot teapot = make_teapot({});
        for (const auto& part : teapot.parts) {
            Deviation conv;
            measure(part->conv, part->bezier, 8, conv);
            std::printf("  %-7s nodes %4zu  max %.3e  rms %.3e  orientation %+.0f\n", part->name.c_str(),
                        part->conv.lattice.nodes.size(), conv.worst, std::sqrt(conv.sum2 / double(conv.count)), part->orientation);
        }
    }
    std::puts("\ncontinuity: largest jump between samples one step apart along u");
    std::puts("surface            bubble       step 1e-4    step 1e-6    step 1e-8");
    for (int kind = 0; kind < 2; ++kind) {
        for (Bubble bubble : {Bubble::none, Bubble::standard, Bubble::continuous}) {
            TeapotOptions options;
            options.bubble = bubble;
            Teapot teapot = make_teapot(options);
            if (kind == 1) {
                // A rough sheet: the teapot body's nodes displaced by noise.
                std::mt19937_64 noise(7);
                std::normal_distribution<double> normal(0.0, 0.08);
                for (Vec3& node : teapot.parts[0]->conv.lattice.nodes) {
                    node = node + Vec3{normal(noise), normal(noise), normal(noise)};
                }
                teapot.parts[0]->conv.compile();
            }
            double jumps[3];
            const double steps[3] = {1e-4, 1e-6, 1e-8};
            for (int k = 0; k < 3; ++k) {
                std::mt19937_64 rng(11);
                jumps[k] = 0.0;
                for (const auto& part : teapot.parts) {
                    if (kind == 1 && part->name != "body") { continue; }
                    jumps[k] = std::fmax(jumps[k], jump(part->conv, steps[k], rng, kind == 1 ? 400 : 100));
                }
            }
            std::printf("%-18s %-10s   %.3e    %.3e    %.3e\n", kind == 0 ? "teapot" : "rough body",
                        bubble == Bubble::none ? "none" : bubble == Bubble::standard ? "standard" : "continuous", jumps[0],
                        jumps[1], jumps[2]);
        }
    }
    return 0;
}
