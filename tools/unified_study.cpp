// Unified sheets of the 2026 teapot: node counts, distance from the sampled
// profile, and continuity along the profile.
#include "../src/teapot.hpp"
#include <cstdio>
using namespace tconv;

int main() {
    for (std::size_t rows : {32u, 48u, 64u, 96u}) {
        TeapotOptions o;
        o.version = Version::v2026_unified;
        o.sheet_rows = rows;
        const Teapot teapot = make_teapot(o);
        for (std::size_t index = 0; index < 2; ++index) {
            const TeapotPart& part = *teapot.parts[index];
            // The reference profile again, for distances.
            const bool is_body = part.group == "body";
            std::vector<ProfileCurve::Run> runs;
            for (int r = 0; r < (is_body ? utah::dense_body_runs : utah::dense_lid_runs); ++r) {
                ProfileCurve::Run run;
                for (int k = 0; k < utah::dense_profile_points; ++k) {
                    const float* p = is_body ? utah::dense_body_profile[r][k] : utah::dense_lid_profile[r][k];
                    run.x.push_back(double(p[0]));
                    run.y.push_back(double(p[1]));
                }
                runs.push_back(run);
            }
            const ProfileCurve profile = chain_runs(runs);
            const Lattice& lattice = part.conv.lattice;
            // Distance of the sheet's meridian from the profile.
            double worst = 0.0, sum2 = 0.0, worst_v = 0.0;
            std::size_t count = 0;
            for (double v = 0.0; v <= double(lattice.nv - 1); v += 0.05) {
                const Vec3 p = part.conv.position(0.37, v); // between two columns
                const double r = std::hypot(p.x, p.z);
                double best = 1e30;
                for (std::size_t k = 0; k + 1 < profile.table_s.size(); ++k) {
                    // Distance to the segment between two table points.
                    const double ax = profile.table_x[k], ay = profile.table_y[k];
                    const double bx = profile.table_x[k + 1] - ax, by = profile.table_y[k + 1] - ay;
                    const double len2 = bx * bx + by * by;
                    const double t = len2 > 0.0 ? std::max(0.0, std::min(1.0, ((r - ax) * bx + (p.y - ay) * by) / len2)) : 0.0;
                    best = std::min(best, std::hypot(ax + t * bx - r, ay + t * by - p.y));
                }
                if (best > worst) { worst_v = v; }
                worst = std::max(worst, best);
                sum2 += best * best;
                ++count;
            }
            // Continuity along the profile at every interior row.
            double turn = 0.0, jump = 0.0, relative_sum = 0.0;
            std::size_t probes = 0, seams = 0, turn_row = 0;
            for (std::size_t row = 1; row + 1 < lattice.nv; ++row) {
                if (!lattice.seams_v.empty() && lattice.seams_v[row] != 0) { ++seams; continue; }
                const auto curve = [&](double v) { return part.conv.position(0.37, v); };
                const double h = 2e-3;
                const auto side = [&](double sign, Vec3& tangent, double& curvature) {
                    const double s = sign * h, t0 = double(row);
                    const Vec3 p0 = curve(t0), p1 = curve(t0 + s), p2 = curve(t0 + 2 * s), p3 = curve(t0 + 3 * s);
                    const Vec3 d1 = (p1 * 18.0 - p0 * 11.0 - p2 * 9.0 + p3 * 2.0) * (1.0 / (6.0 * s));
                    const Vec3 d2 = (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (1.0 / (s * s));
                    tangent = normalize(d1);
                    const double speed = length(d1);
                    curvature = length(cross(d1, d2)) / (speed * speed * speed);
                };
                Vec3 ta, tb;
                double ka, kb;
                side(-1.0, ta, ka);
                side(1.0, tb, kb);
                const double this_turn = std::acos(std::min(1.0, dot(ta, tb))) * 180.0 / M_PI;
                if (this_turn > turn) { turn_row = row; }
                turn = std::max(turn, this_turn);
                jump = std::max(jump, std::fabs(ka - kb));
                relative_sum += std::fabs(ka - kb) / std::max(1e-6, std::max(ka, kb));
                ++probes;
            }
            if (rows == 64) {
                std::printf("   %s joint turns (deg):", part.name.c_str());
                for (std::size_t run = 1; run < profile.runs.size(); ++run) { std::printf(" %.2f", profile.turn_at_joint(run)); }
                const Vec3 p = part.conv.position(0.37, worst_v);
                const Vec3 q = part.conv.position(0.37, double(turn_row));
                std::printf("; worst distance at row %.2f, r %.3f y %.3f; worst turn at row %zu, r %.3f y %.3f, spacing before %.4f after %.4f\n", worst_v, std::hypot(p.x, p.z), p.y,
                            turn_row, std::hypot(q.x, q.z), q.y, part.row_param[turn_row] - part.row_param[turn_row - 1], part.row_param[turn_row + 1] - part.row_param[turn_row]);
            }
            std::printf("rows %3zu  %-5s runs %zu  length %.3f  lattice %zux%zu = %zu nodes, %zu crease rows | distance from profile max %.2e rms %.2e | "
                        "along profile: tangent turn max %.4f deg, curvature jump max %.3e, mean relative %.4f\n",
                        rows, part.name.c_str(), profile.runs.size(), profile.length, lattice.nu, lattice.nv, lattice.nodes.size(), seams, worst,
                        std::sqrt(sum2 / double(count)), turn, jump, relative_sum / double(std::max<std::size_t>(1, probes)));
        }
    }
    std::puts("\nJoins (unified version, default sizes): the band's edge on the body");
    {
        TeapotOptions o;
        o.version = Version::v2026_unified;
        const Teapot teapot = make_teapot(o);
        const TeapotPart& body = *teapot.parts[0];
        const Lattice& lattice = body.conv.lattice;
        std::size_t top = 0;
        for (std::size_t v = 0; v < lattice.nv; ++v) {
            if (lattice.at(0, v).y > lattice.at(0, top).y) { top = v; }
        }
        std::size_t total = 0;
        for (const auto& part : teapot.parts) { total += part->conv.lattice.nodes.size(); }
        std::printf("   parts %zu, nodes %zu (body %zu, lid %zu, handle %zu, spout %zu, joins %zu)\n", teapot.parts.size(), total,
                    teapot.parts[0]->conv.lattice.nodes.size(), teapot.parts[1]->conv.lattice.nodes.size(), teapot.parts[2]->conv.lattice.nodes.size(),
                    teapot.parts[3]->conv.lattice.nodes.size(), total - teapot.parts[0]->conv.lattice.nodes.size() - teapot.parts[1]->conv.lattice.nodes.size() -
                    teapot.parts[2]->conv.lattice.nodes.size() - teapot.parts[3]->conv.lattice.nodes.size());
        for (std::size_t index = 4; index < teapot.parts.size(); ++index) {
            const TeapotPart& join = *teapot.parts[index];
            // Is this join on the exterior or the interior branch? Compare with both.
            double best_gap = 1e30, best_turn = 0.0, best_mid = 0.0;
            for (int branch_index = 0; branch_index < 2; ++branch_index) {
                BodyBranch branch{&body.conv, branch_index == 0 ? 0.0 : double(top), branch_index == 0 ? double(top) : double(lattice.nv - 1), branch_index == 0};
                double gap = 0.0, turn = 0.0;
                for (int k = 0; k < 96; ++k) {
                    const double u = double(join.conv.cells_u()) * k / 96.0;
                    const SurfacePoint edge = join.conv.eval(u, 1e-4);
                    const double r = std::hypot(edge.p.x, edge.p.z);
                    const double v = branch.row_at(edge.p.y);
                    const double bu = std::atan2(-edge.p.z, edge.p.x) / 6.283185307179586 * double(lattice.nu);
                    const SurfacePoint on_body = body.conv.eval(bu, v);
                    gap = std::max(gap, std::fabs(r - std::hypot(on_body.p.x, on_body.p.z)));
                    const Vec3 na = normalize(cross(edge.du, edge.dv)), nb = normalize(cross(on_body.du, on_body.dv));
                    turn = std::max(turn, std::acos(std::min(1.0, std::fabs(dot(na, nb)))) * 180.0 / M_PI);
                }
                if (gap < best_gap) {
                    best_gap = gap;
                    best_turn = turn;
                    // The angle the join spans: its normal at the body edge against its normal at the tube edge.
                    best_mid = 0.0;
                    for (int k = 0; k < 96; ++k) {
                        const double u = double(join.conv.cells_u()) * k / 96.0;
                        const SurfacePoint a = join.conv.eval(u, 1e-4), b = join.conv.eval(u, double(join.conv.cells_v()) - 1e-4);
                        const Vec3 na = normalize(cross(a.du, a.dv)), nb = normalize(cross(b.du, b.dv));
                        best_mid = std::max(best_mid, std::acos(std::min(1.0, std::fabs(dot(na, nb)))) * 180.0 / M_PI);
                    }
                }
            }
            std::printf("   %-12s edge on the body: gap max %.2e, normal mismatch max %.3f deg; the join turns the normal by up to %.1f deg across the band\n",
                        join.name.c_str(), best_gap, best_turn, best_mid);
        }
    }
    return 0;
}
