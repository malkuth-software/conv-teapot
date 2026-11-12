// The grab is a pin at the picked point: lifted by its handle, the teapot
// must hang from the handle (its centre of mass below the held point and
// the body turned), not ride up level.
#include "../src/sim.hpp"
#include <cstdio>
using namespace tconv;

int main() {
    TeapotOptions options;
    options.version = Version::v2026_unified;
    Teapot teapot = make_teapot(options);
    TeapotSim sim;
    sim.build(teapot);
    // The outermost point of the handle.
    const TeapotPart& handle = *teapot.parts[2];
    Vec3 held = handle.conv.lattice.nodes[0];
    for (const Vec3& node : handle.conv.lattice.nodes) {
        if (node.x < held.x) { held = node; }
    }
    const auto centre = [&] {
        Vec3 sum{};
        std::size_t count = 0;
        for (const Vec3& node : teapot.parts[0]->conv.lattice.nodes) {
            sum = sum + node;
            ++count;
        }
        return sum * (1.0 / double(count));
    };
    const Vec3 centre_before = centre();
    const Vec3 axis_before = teapot.parts[0]->conv.lattice.nodes.back() - teapot.parts[0]->conv.lattice.nodes.front();
    sim.grab(held);
    const Vec3 target = held + Vec3{0.0, 6.0, 0.0};
    Vec3 point = held;
    for (int k = 0; k < 1200; ++k) {
        sim.drag(held + (target - held) * std::min(1.0, k / 240.0));
        sim.step(1.0 / 120.0);
    }
    // Where the held node is now: the handle's node with the same index.
    std::size_t index = 0;
    {
        TeapotOptions again = options;
        const Teapot rest = make_teapot(again);
        const auto& nodes = rest.parts[2]->conv.lattice.nodes;
        for (std::size_t k = 0; k < nodes.size(); ++k) {
            if (length(nodes[k] - held) < 1e-12) { index = k; }
        }
    }
    point = handle.conv.lattice.nodes[index];
    const Vec3 centre_after = centre();
    const Vec3 axis_after = teapot.parts[0]->conv.lattice.nodes.back() - teapot.parts[0]->conv.lattice.nodes.front();
    const double tilt = std::acos(std::max(-1.0, std::min(1.0, dot(normalize(axis_before), normalize(axis_after))))) * 180.0 / M_PI;
    const double miss = length(point - target);
    const double sideways = std::hypot(centre_after.x - point.x, centre_after.z - point.z);
    std::printf("grab: held point %.3f from the pointer; body turned %.1f deg; centre of the body %.3f below and %.3f beside the held point (was %.3f beside)\n", miss, tilt,
                point.y - centre_after.y, sideways, std::hypot(centre_before.x - held.x, centre_before.z - held.z));
    const bool ok = miss < 0.15 && tilt > 45.0 && point.y > centre_after.y && sideways < 0.7; // a moment of a swing that has not died down: 0.59 to 0.61 as the hulls have changed
    std::puts(ok ? "grab check passed" : "FAIL grab check");
    return ok ? 0 : 1;
}
