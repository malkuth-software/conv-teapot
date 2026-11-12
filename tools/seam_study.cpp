// Where the nodes and the baked vertices of the unified 2026 teapot go:
// how many sit at creases, at the ends of sheets and in the joins, against
// the share of the surface those places are.
//   seam_study          the accounting
//   seam_study joins    the joins' accuracy against their number of rows
#include "../src/raster.hpp"
#include "../src/teapot.hpp"
#include <cstdio>
#include <cstring>
#include <string>
using namespace tconv;

namespace {

double triangle_area(const Mesh& mesh, std::size_t t) {
    const Vec3 &a = mesh.vertices[mesh.triangles[3 * t]].p, &b = mesh.vertices[mesh.triangles[3 * t + 1]].p, &c = mesh.vertices[mesh.triangles[3 * t + 2]].p;
    return 0.5 * length(cross(b - a, c - a));
}

Mesh bake(const Teapot& teapot, const MeshOptions& options) {
    std::vector<const Surface*> surfaces;
    std::vector<std::vector<const Trim*>> trims(teapot.parts.size());
    for (const auto& part : teapot.parts) { surfaces.push_back(&part->conv); }
    for (const TeapotJoin& join : teapot.joins) {
        trims[join.body_part].push_back(&join.body_trim);
        trims[join.tube_part].push_back(&join.tube_trim);
    }
    return build_mesh(surfaces, trims, options);
}

// Arc length along column 0 at every node row.
std::vector<double> row_arc(const Lattice& lattice) {
    std::vector<double> arc(lattice.nv, 0.0);
    for (std::size_t v = 1; v < lattice.nv; ++v) { arc[v] = arc[v - 1] + length(lattice.at(0, v) - lattice.at(0, v - 1)); }
    return arc;
}

void accounting() {
    TeapotOptions o;
    o.version = Version::v2026_unified;
    const Teapot teapot = make_teapot(o);
    MeshOptions mesh_options;
    mesh_options.layout = Layout::tree;
    mesh_options.tolerance = 1e-2;
    mesh_options.angle = 20.0;
    const Mesh mesh = bake(teapot, mesh_options);
    const double near = 0.1; // model units; the teapot is about 2.4 high

    std::printf("Nodes and vertices by part (tree layout, tolerance 1e-2, 20 degrees)\n");
    std::printf("%-12s %7s %6s %6s %8s %8s %9s %9s %10s %10s\n", "part", "lattice", "nodes", "hidden", "area", "nodes/A", "vertices", "triangles", "vertices/A", "near trim");
    std::size_t nodes_total = 0, hidden_total = 0;
    double area_total = 0.0;
    for (std::size_t index = 0; index < teapot.parts.size(); ++index) {
        const TeapotPart& part = *teapot.parts[index];
        const Lattice& lattice = part.conv.lattice;
        // Nodes under a trim: they shape nothing that is shown, except through their neighbours' stencils.
        std::size_t hidden = 0;
        for (std::size_t v = 0; v < lattice.nv; ++v) {
            for (std::size_t u = 0; u < lattice.nu; ++u) {
                bool gone = false;
                for (const TeapotJoin& join : teapot.joins) {
                    if (join.body_part == index) { gone = gone || join.body_trim.hides(double(u), double(v)); }
                    if (join.tube_part == index) { gone = gone || join.tube_trim.hides(double(u), double(v)); }
                }
                hidden += gone ? 1 : 0;
            }
        }
        const MeshPart& baked = mesh.parts[index];
        double area = 0.0;
        std::size_t trimmed = 0;
        for (std::size_t t = baked.first_triangle; t < baked.first_triangle + baked.triangle_count; ++t) {
            area += triangle_area(mesh, t);
            trimmed += mesh.triangle_trimmed[t];
        }
        char size[32];
        std::snprintf(size, sizeof size, "%zux%zu", lattice.nu, lattice.nv);
        std::printf("%-12s %7s %6zu %6zu %8.2f %8.1f %9zu %9zu %10.1f %10zu\n", part.name.c_str(), size, part.conv.node_count(), hidden, area,
                    double(part.conv.node_count()) / area, baked.vertex_count, baked.triangle_count, double(baked.vertex_count) / area, trimmed);
        nodes_total += part.conv.node_count();
        hidden_total += hidden;
        area_total += area;
    }
    std::printf("%-12s %7s %6zu %6zu %8.2f %8.1f %9zu %9zu %10.1f\n\n", "all", "", nodes_total, hidden_total, area_total, double(nodes_total) / area_total,
                mesh.vertices.size(), mesh.triangle_count(), double(mesh.vertices.size()) / area_total);

    std::printf("Rows and vertices within %.2f of a crease or an end of a sheet\n", near);
    std::printf("%-8s %8s %7s %7s %12s %12s %14s %14s\n", "sheet", "creases", "rows", "near", "length near", "rows near", "vertices near", "smallest h/median");
    for (std::size_t index = 0; index < 4 && index < teapot.parts.size(); ++index) {
        const TeapotPart& part = *teapot.parts[index];
        const Lattice& lattice = part.conv.lattice;
        const std::vector<double> arc = row_arc(lattice);
        std::vector<double> marks{arc.front(), arc.back()};
        std::size_t creases = 0;
        for (std::size_t v = 1; v + 1 < lattice.nv; ++v) {
            if (!lattice.seams_v.empty() && lattice.seams_v[v] != 0) {
                marks.push_back(arc[v]);
                ++creases;
            }
        }
        const auto distance = [&](double a) {
            double d = 1e30;
            for (double m : marks) { d = std::min(d, std::fabs(a - m)); }
            return d;
        };
        std::size_t rows_near = 0;
        for (std::size_t v = 0; v < lattice.nv; ++v) { rows_near += distance(arc[v]) < near ? 1 : 0; }
        double length_near = 0.0;
        const int fine = 4000;
        for (int k = 0; k < fine; ++k) { length_near += distance(arc.back() * (k + 0.5) / fine) < near ? arc.back() / fine : 0.0; }
        std::vector<double> h;
        for (std::size_t v = 1; v < lattice.nv; ++v) { h.push_back(arc[v] - arc[v - 1]); }
        std::vector<double> sorted = h;
        std::sort(sorted.begin(), sorted.end());
        const MeshPart& baked = mesh.parts[index];
        std::size_t vertices_near = 0;
        for (std::size_t k = baked.first_vertex; k < baked.first_vertex + baked.vertex_count; ++k) {
            const double v = std::min(double(lattice.nv - 1), std::max(0.0, double(mesh.vertices[k].v)));
            const std::size_t row = std::min(lattice.nv - 2, std::size_t(v));
            vertices_near += distance(arc[row] + (arc[row + 1] - arc[row]) * (v - double(row))) < near ? 1 : 0;
        }
        std::printf("%-8s %8zu %7zu %7zu %11.0f%% %11.0f%% %13.0f%% %14.3f\n", part.name.c_str(), creases, lattice.nv, rows_near, 100.0 * length_near / arc.back(),
                    100.0 * double(rows_near) / double(lattice.nv), 100.0 * double(vertices_near) / double(baked.vertex_count), sorted.front() / sorted[sorted.size() / 2]);
    }
}

// The joins against how they are stored: the blended band sampled as rows of
// nodes, or its two sides stored apart and blended in closed form. Measured
// against the two-sided band with 65 rows a side.
void joins() {
    TeapotOptions reference_options;
    reference_options.version = Version::v2026_unified;
    reference_options.join_rows = 65;
    const Teapot reference = make_teapot(reference_options);
    std::printf("Joins against how they are stored (reference: two sides of 65 rows)\n");
    std::printf("%-10s %5s %6s | %-14s %10s %10s %10s\n", "stored", "rows", "nodes", "join", "distance", "normal", "at edges");
    struct Case { bool two_sided; int rows, interior; };
    for (const Case& c : {Case{false, 7, 7}, Case{false, 9, 17}, Case{false, 17, 17}, Case{true, 2, 2}, Case{true, 3, 3}, Case{true, 4, 4}, Case{true, 5, 5}}) {
        TeapotOptions o = reference_options;
        o.join_two_sided = c.two_sided;
        o.join_rows = o.join_rows_sampled = c.rows;
        o.join_rows_interior = c.interior;
        const Teapot teapot = make_teapot(o);
        std::size_t total = 0;
        for (std::size_t index = 4; index < teapot.parts.size(); ++index) {
            const ConvSurface &a = teapot.parts[index]->conv, &b = reference.parts[index]->conv;
            double worst = 0.0, worst_angle = 0.0, edge = 0.0;
            for (int j = 0; j <= 64; ++j) {
                for (int k = 0; k < 64; ++k) {
                    const double u = double(a.cells_u()) * k / 64.0, tau = std::min(1.0 - 1e-6, std::max(1e-6, j / 64.0));
                    const SurfacePoint pa = a.eval(u, tau * double(a.cells_v())), pb = b.eval(u, tau * double(b.cells_v()));
                    worst = std::max(worst, length(pa.p - pb.p));
                    const double angle = std::acos(std::min(1.0, std::fabs(dot(normalize(cross(pa.du, pa.dv)), normalize(cross(pb.du, pb.dv)))))) * 180.0 / M_PI;
                    worst_angle = std::max(worst_angle, angle);
                    if (j == 0 || j == 64) { edge = std::max(edge, angle); }
                }
            }
            total += a.node_count();
            std::printf("%-10s %5d %6zu | %-14s %10.2e %9.2f° %9.2f°\n", c.two_sided ? "two sides" : "sampled", int(a.lattice.nv), a.node_count(),
                        (teapot.parts[index]->name + (index % 2 == 0 ? " a" : " b")).c_str(), worst, worst_angle, edge);
        }
        std::printf("%-10s       %6zu   in the four joins; %zu in the teapot\n", "", total, teapot.node_count());
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "joins") == 0) {
        joins();
    } else {
        accounting();
    }
    return 0;
}
