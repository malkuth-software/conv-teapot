// Vertex layouts for the baked mesh: how many vertices each needs for a
// given closeness to the surface and a given smoothness of shading.
//
// Layouts: the three of src/raster.hpp (uniform, bounds, tree) and one more
// tried only here:
//   clusters  k-means over dense surface samples in position and normal
//             (so clusters shrink where the normal turns), run with the
//             mini-batch moving-average updates of e-means (Bessa et al.,
//             arXiv 2611.09501); each cluster gives the sample nearest its
//             centre as a vertex, and the triangles are the dual of the
//             clusters on the sample grid.
// Measures, over points spread inside every triangle:
//   distance  from the flat triangle to the surface at the same (u, v)
//   normal    angle between the interpolated vertex normal and the surface's
// and for the whole picture, against the ray-traced frame: pixels whose
// coverage differs (silhouette) and the mean colour difference.
#include "../src/image.hpp"
#include "../src/teapot.hpp"
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
using namespace tconv;

namespace {

struct Quality {
    double dev_max = 0, dev_rms = 0, normal_max = 0, normal_rms = 0;
};

Quality measure(const Mesh& mesh, const std::vector<const Surface*>& surfaces) {
    Quality q;
    double dev2 = 0, normal2 = 0;
    std::size_t count = 0;
    static const double points[7][3] = {{1 / 3.0, 1 / 3.0, 1 / 3.0}, {0.6, 0.2, 0.2}, {0.2, 0.6, 0.2}, {0.2, 0.2, 0.6}, {0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}};
    for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
        const Surface& surface = *surfaces[mesh.triangle_part[t]];
        const MeshVertex& a = mesh.vertices[mesh.triangles[t * 3]];
        const MeshVertex& b = mesh.vertices[mesh.triangles[t * 3 + 1]];
        const MeshVertex& c = mesh.vertices[mesh.triangles[t * 3 + 2]];
        // u of the three corners on one turn.
        const double cells = double(surface.cells_u());
        double ub = b.u, uc = c.u;
        if (surface.periodic_u()) {
            ub += std::round((double(a.u) - ub) / cells) * cells;
            uc += std::round((double(a.u) - uc) / cells) * cells;
        }
        if (length(cross(b.p - a.p, c.p - a.p)) < 1e-12) { continue; } // collapsed at a pole
        for (const auto& w : points) {
            const Vec3 flat = a.p * w[0] + b.p * w[1] + c.p * w[2];
            const double u = double(a.u) * w[0] + ub * w[1] + uc * w[2], v = double(a.v) * w[0] + double(b.v) * w[1] + double(c.v) * w[2];
            // On an edge that runs along a crease, the surface on the triangle's own side.
            const double middle_v = (double(a.v) + double(b.v) + double(c.v)) / 3.0;
            const SurfacePoint truth = detail::mesh_point(surface, u, v + (middle_v - v) * 1e-3);
            const double d = length(flat - truth.p);
            const Vec3 n = normalize(a.n * w[0] + b.n * w[1] + c.n * w[2]);
            const double angle = std::acos(std::max(-1.0, std::min(1.0, dot(n, truth.du)))) * 180.0 / M_PI;
            q.dev_max = std::max(q.dev_max, d);
            dev2 += d * d;
            q.normal_max = std::max(q.normal_max, angle);
            normal2 += angle * angle;
            ++count;
        }
    }
    q.dev_rms = std::sqrt(dev2 / double(std::max<std::size_t>(1, count)));
    q.normal_rms = std::sqrt(normal2 / double(std::max<std::size_t>(1, count)));
    return q;
}

// k-means over dense samples, e-means style; `share` of the samples become vertices.
Mesh cluster_mesh(const std::vector<const Surface*>& surfaces, std::size_t wanted, double normal_weight, int epochs) {
    Mesh mesh;
    // Vertices are shared out among the parts by their sample counts weighted by how much the normal turns.
    struct Sample {
        float f[6];
        float u, v;
    };
    const int per = 6;
    std::vector<std::vector<Sample>> samples(surfaces.size());
    std::vector<double> weight(surfaces.size(), 0.0);
    std::vector<std::vector<Vec3>> normals(surfaces.size());
    for (std::size_t k = 0; k < surfaces.size(); ++k) {
        const Surface& s = *surfaces[k];
        const long nu = long(s.cells_u()) * per + (s.periodic_u() ? 0 : 1), nv = long(s.cells_v()) * per + 1;
        for (long j = 0; j < nv; ++j) {
            for (long i = 0; i < nu; ++i) {
                const SurfacePoint point = detail::mesh_point(s, double(i) / per, double(j) / per);
                Sample sample;
                sample.f[0] = float(point.p.x);
                sample.f[1] = float(point.p.y);
                sample.f[2] = float(point.p.z);
                sample.f[3] = float(point.du.x * normal_weight);
                sample.f[4] = float(point.du.y * normal_weight);
                sample.f[5] = float(point.du.z * normal_weight);
                sample.u = float(double(i) / per);
                sample.v = float(double(j) / per);
                samples[k].push_back(sample);
                normals[k].push_back(point.du);
            }
        }
        // Spread in feature space along the grid: the part's share of vertices.
        for (long j = 0; j + 1 < nv; ++j) {
            for (long i = 0; i + 1 < nu; ++i) {
                const Sample &a = samples[k][std::size_t(j * nu + i)], &b = samples[k][std::size_t(j * nu + i + 1)], &c = samples[k][std::size_t((j + 1) * nu + i)];
                double du = 0, dv = 0;
                for (int d = 0; d < 6; ++d) {
                    du += double(b.f[d] - a.f[d]) * double(b.f[d] - a.f[d]);
                    dv += double(c.f[d] - a.f[d]) * double(c.f[d] - a.f[d]);
                }
                weight[k] += std::sqrt(du * dv);
            }
        }
    }
    double total = 0;
    for (double w : weight) { total += w; }
    std::uint32_t state = 2463534242u;
    const auto random = [&] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (std::size_t part_index = 0; part_index < surfaces.size(); ++part_index) {
        const Surface& s = *surfaces[part_index];
        const std::vector<Sample>& data = samples[part_index];
        const long nu = long(s.cells_u()) * per + (s.periodic_u() ? 0 : 1), nv = long(s.cells_v()) * per + 1;
        const std::size_t n = data.size();
        const std::size_t k = std::max<std::size_t>(8, std::min(n / 3, std::size_t(double(wanted) * weight[part_index] / total)));
        std::vector<float> centre(k * 6), sum(k * 6, 0.0f), mass(k, 0.0f);
        for (std::size_t c = 0; c < k; ++c) {
            // Start on a jittered regular pick of the samples.
            const std::size_t pick = std::min(n - 1, c * n / k + random() % std::max<std::size_t>(1, n / k));
            std::memcpy(&centre[c * 6], data[pick].f, sizeof(float) * 6);
        }
        std::vector<std::uint32_t> label(n, 0);
        const auto nearest = [&](const float* f) {
            std::size_t best = 0;
            float best_d = 1e30f;
            for (std::size_t c = 0; c < k; ++c) {
                const float* q = &centre[c * 6];
                float d = 0;
                for (int e = 0; e < 6; ++e) { d += (f[e] - q[e]) * (f[e] - q[e]); }
                if (d < best_d) {
                    best_d = d;
                    best = c;
                }
            }
            return best;
        };
        // Mini-batches with moving-average sums and counts; the rate follows a cosine from alpha to a tenth of it.
        const std::size_t batch = std::max<std::size_t>(256, std::min(n, 4 * k));
        const std::size_t steps = std::size_t(epochs) * n / batch;
        std::vector<float> batch_sum(k * 6), batch_mass(k);
        for (std::size_t step = 0; step < steps; ++step) {
            const double alpha0 = 0.5, progress = double(step) / double(std::max<std::size_t>(1, steps - 1));
            const float alpha = float(alpha0 * (0.1 + 0.9 * 0.5 * (1.0 + std::cos(M_PI * progress))));
            std::fill(batch_sum.begin(), batch_sum.end(), 0.0f);
            std::fill(batch_mass.begin(), batch_mass.end(), 0.0f);
            for (std::size_t b = 0; b < batch; ++b) {
                const Sample& sample = data[random() % n];
                const std::size_t c = nearest(sample.f);
                for (int e = 0; e < 6; ++e) { batch_sum[c * 6 + std::size_t(e)] += sample.f[e]; }
                batch_mass[c] += 1.0f;
            }
            for (std::size_t c = 0; c < k; ++c) {
                for (int e = 0; e < 6; ++e) { sum[c * 6 + std::size_t(e)] = (1 - alpha) * sum[c * 6 + std::size_t(e)] + alpha * batch_sum[c * 6 + std::size_t(e)]; }
                mass[c] = (1 - alpha) * mass[c] + alpha * batch_mass[c];
                if (mass[c] > 1e-6f) {
                    for (int e = 0; e < 6; ++e) { centre[c * 6 + std::size_t(e)] = sum[c * 6 + std::size_t(e)] / mass[c]; }
                }
            }
        }
        // Finish on exact means, then one more assignment.
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<double> exact(k * 6, 0.0), members(k, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                label[i] = std::uint32_t(nearest(data[i].f));
                for (int e = 0; e < 6; ++e) { exact[label[i] * 6 + std::size_t(e)] += double(data[i].f[e]); }
                members[label[i]] += 1.0;
            }
            if (pass == 1) { break; }
            for (std::size_t c = 0; c < k; ++c) {
                if (members[c] > 0) {
                    for (int e = 0; e < 6; ++e) { centre[c * 6 + std::size_t(e)] = float(exact[c * 6 + std::size_t(e)] / members[c]); }
                }
            }
        }
        // A cluster's vertex: its member nearest the centre.
        std::vector<long> member(k, -1);
        std::vector<float> member_d(k, 1e30f);
        for (std::size_t i = 0; i < n; ++i) {
            const float* q = &centre[label[i] * 6];
            float d = 0;
            for (int e = 0; e < 6; ++e) { d += (data[i].f[e] - q[e]) * (data[i].f[e] - q[e]); }
            if (d < member_d[label[i]]) {
                member_d[label[i]] = d;
                member[label[i]] = long(i);
            }
        }
        MeshPart part;
        part.first_vertex = mesh.vertices.size();
        part.first_triangle = mesh.triangle_count();
        std::vector<long> vertex_of(k, -1);
        for (std::size_t c = 0; c < k; ++c) {
            if (member[c] < 0) { continue; }
            MeshVertex vertex;
            const Sample& sample = data[std::size_t(member[c])];
            vertex.p = {double(sample.f[0]), double(sample.f[1]), double(sample.f[2])};
            vertex.n = normals[part_index][std::size_t(member[c])];
            vertex.u = sample.u;
            vertex.v = sample.v;
            vertex.s = vertex.t = 0.0f;
            vertex_of[c] = long(mesh.vertices.size());
            mesh.vertices.push_back(vertex);
        }
        // The dual: a grid square whose corners lie in three clusters is a
        // triangle, in four two triangles.
        std::set<std::array<std::uint32_t, 3>> seen;
        const auto emit = [&](long a, long b, long c) {
            if (a == b || b == c || a == c) { return; }
            std::array<std::uint32_t, 3> sorted{std::uint32_t(a), std::uint32_t(b), std::uint32_t(c)};
            std::sort(sorted.begin(), sorted.end());
            if (!seen.insert(sorted).second) { return; }
            mesh.triangles.push_back(std::uint32_t(a));
            mesh.triangles.push_back(std::uint32_t(b));
            mesh.triangles.push_back(std::uint32_t(c));
            mesh.triangle_part.push_back(std::uint16_t(part_index));
            mesh.triangle_trimmed.push_back(0);
        };
        const long columns = s.periodic_u() ? nu : nu - 1;
        for (long j = 0; j + 1 < nv; ++j) {
            for (long i = 0; i < columns; ++i) {
                const long i1 = (i + 1) % nu;
                const long a = vertex_of[label[std::size_t(j * nu + i)]], b = vertex_of[label[std::size_t(j * nu + i1)]];
                const long c = vertex_of[label[std::size_t((j + 1) * nu + i1)]], d = vertex_of[label[std::size_t((j + 1) * nu + i)]];
                std::set<long> distinct{a, b, c, d};
                if (distinct.size() == 4) {
                    emit(a, b, c);
                    emit(a, c, d);
                } else if (distinct.size() == 3) {
                    if (a == b) { emit(a, c, d); }
                    else if (b == c) { emit(a, b, d); }
                    else if (c == d) { emit(a, b, c); }
                    else { emit(a, b, c); }
                }
            }
        }
        part.vertex_count = mesh.vertices.size() - part.first_vertex;
        part.triangle_count = mesh.triangle_count() - part.first_triangle;
        part.arc = {0.0f, 1.0f};
        mesh.parts.push_back(part);
    }
    mesh.version = 1;
    return mesh;
}

struct Picture {
    std::size_t coverage = 0; // pixels covered in one frame and not the other
    double colour = 0;        // mean absolute difference, 0..255
    double ms = 0;
};

} // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "quick";
    TeapotOptions o;
    o.version = Version::v2026_unified;
    Teapot teapot = make_teapot(o);
    Scene scene;
    std::vector<long> scene_index(teapot.parts.size(), -1);
    for (std::size_t k = 0; k < teapot.parts.size(); ++k) {
        RenderPart part;
        part.surface = &teapot.parts[k]->conv;
        part.orientation = teapot.parts[k]->orientation;
        part.per_u = part.per_v = 0;
        part.two_sided = true;
        scene.parts.push_back(part);
    }
    for (const TeapotJoin& join : teapot.joins) {
        scene.parts[join.body_part].trims.push_back(&join.body_trim);
        scene.parts[join.tube_part].trims.push_back(&join.tube_trim);
    }
    std::vector<const Surface*> surfaces;
    std::vector<std::vector<const Trim*>> trims, no_trims;
    for (const RenderPart& part : scene.parts) {
        surfaces.push_back(part.surface);
        trims.push_back(part.trims);
        no_trims.push_back({});
    }
    const int width = 960, height = 600;
    Camera camera;
    const double azimuth = 68.0 * M_PI / 180.0, elevation = 22.0 * M_PI / 180.0;
    camera.target = {0.15, 1.35, 0.0};
    camera.eye = camera.target + Vec3{std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth)} * 11.0;
    camera.fov_y = 34.0 * M_PI / 180.0;
    RenderOptions options;
    Frame reference, frame;
    reference.resize(width, height);
    frame.resize(width, height);
    Renderer renderer;
    std::vector<double> times;
    for (int k = 0; k < 5; ++k) {
        renderer.render(scene, camera, options, reference);
        times.push_back(renderer.stats.ms_total);
    }
    std::sort(times.begin(), times.end());
    const double reference_ms = times[2];
    RenderOptions one = options;
    one.threads = 1;
    renderer.render(scene, camera, one, frame);
    const double reference_ms_one = renderer.stats.ms_total;

    const auto picture = [&](Mesh& mesh, bool lit, int threads) {
        Picture p;
        scene.mesh = &mesh;
        RenderOptions raster = options;
        raster.method = Method::raster;
        raster.threads = threads;
        raster.baked_light = lit;
        std::vector<double> ms;
        for (int k = 0; k < 5; ++k) {
            renderer.render(scene, camera, raster, frame);
            ms.push_back(renderer.stats.ms_total);
        }
        std::sort(ms.begin(), ms.end());
        p.ms = ms[2];
        double sum = 0;
        for (std::size_t k = 0; k < frame.hits.size(); ++k) {
            if ((frame.hits[k].t > 0.0f) != (reference.hits[k].t > 0.0f)) { ++p.coverage; }
            for (int c = 0; c < 3; ++c) { sum += std::fabs(double(frame.rgba[k * 4 + std::size_t(c)]) - double(reference.rgba[k * 4 + std::size_t(c)])); }
        }
        p.colour = sum / double(frame.hits.size() * 3);
        scene.mesh = nullptr;
        return p;
    };

    std::printf("Unified teapot, %zu nodes. Ray-traced reference at %dx%d: %.1f ms (all threads), %.1f ms (one thread).\n", teapot.node_count(), width, height,
                reference_ms, reference_ms_one);
    std::printf("A pixel at the teapot is about 0.011 model units.\n\n");
    std::printf("%-22s %8s %9s %8s %8s | %9s %9s | %8s %8s | %9s %7s | %7s %7s\n", "layout", "vertices", "triangles", "evals", "build ms", "dev max", "dev rms", "nrm max",
                "nrm rms", "coverage", "colour", "ms all", "ms one");
    const auto row = [&](const std::string& name, Mesh& mesh) {
        const Quality q = measure(mesh, surfaces);
        const Picture all = picture(mesh, false, 0), single = picture(mesh, false, 1);
        std::printf("%-22s %8zu %9zu %8zu %8.1f | %9.2e %9.2e | %8.2f %8.3f | %9zu %7.3f | %7.2f %7.2f\n", name.c_str(), mesh.vertices.size(), mesh.triangle_count(),
                    mesh.evaluations, mesh.ms_build, q.dev_max, q.dev_rms, q.normal_max, q.normal_rms, all.coverage, all.colour, all.ms, single.ms);
        std::fflush(stdout);
    };
    const auto built = [&](MeshOptions m) {
        const double start = detail::now_ms();
        Mesh mesh = build_mesh(surfaces, trims, m);
        mesh.ms_build = detail::now_ms() - start;
        return mesh;
    };
    for (int cuts : {1, 2, 4, 8}) {
        MeshOptions m;
        m.layout = Layout::uniform;
        m.uniform = cuts;
        Mesh mesh = built(m);
        row("uniform " + std::to_string(cuts) + "x" + std::to_string(cuts), mesh);
    }
    for (double tolerance : {2e-2, 1e-2, 5e-3, 2e-3, 1e-3}) {
        char name[64];
        MeshOptions m;
        m.tolerance = tolerance;
        m.layout = Layout::bounds;
        Mesh a = built(m);
        std::snprintf(name, sizeof name, "bounds %.0e", tolerance);
        row(name, a);
    }
    for (double tolerance : {2e-2, 1e-2, 5e-3, 2e-3, 1e-3}) {
        char name[64];
        MeshOptions m;
        m.tolerance = tolerance;
        m.layout = Layout::tree;
        Mesh b = built(m);
        std::snprintf(name, sizeof name, "tree %.0e", tolerance);
        row(name, b);
    }
    // Distance alone lets a tight bend be one flat piece; a limit on the turn of the normal along an edge does not.
    for (const Layout layout : {Layout::bounds, Layout::tree}) {
        for (const auto& [tolerance, angle] : std::vector<std::pair<double, double>>{{5e-2, 30.0}, {2e-2, 30.0}, {2e-2, 20.0}, {1e-2, 20.0}, {1e-2, 15.0}, {5e-3, 15.0}, {5e-3, 10.0}, {2e-3, 10.0}, {2e-3, 5.0}}) {
            char name[64];
            MeshOptions m;
            m.tolerance = tolerance;
            m.layout = layout;
            m.angle = angle;
            Mesh b = built(m);
            std::snprintf(name, sizeof name, "%s %.0e + %.0f deg", layout == Layout::tree ? "tree" : "bounds", tolerance, angle);
            row(name, b);
        }
    }
    if (!quick) {
        for (std::size_t wanted : {3000u, 6000u, 12000u, 24000u}) {
            for (double normal_weight : {0.3, 1.0}) {
                const double start = detail::now_ms();
                Mesh mesh = cluster_mesh(surfaces, wanted, normal_weight, 8);
                mesh.ms_build = detail::now_ms() - start;
                char name[64];
                std::snprintf(name, sizeof name, "clusters %zu w%.1f", wanted, normal_weight);
                row(name, mesh);
            }
        }
    }

    // Light maps: the lit picture against the ray-traced one, and what a frame costs.
    std::printf("\nLight maps on tree 5e-03 (texel size in model units): texels, bake ms, colour difference, frame ms (all threads / one)\n");
    for (double texel : {0.032, 0.016, 0.008}) {
        MeshOptions m;
        m.tolerance = 5e-3;
        Mesh mesh = built(m);
        scene.mesh = &mesh;
        renderer.bake_light(scene, camera, options, width, height, texel);
        scene.mesh = nullptr;
        const Picture all = picture(mesh, true, 0), single = picture(mesh, true, 1);
        std::printf("  texel %.3f: %8zu texels, bake %7.1f ms, colour %.3f, frame %.2f / %.2f ms\n", texel, mesh.texels, mesh.ms_light, all.colour, all.ms, single.ms);
    }
    return 0;
}
