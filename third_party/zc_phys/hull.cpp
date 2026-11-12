// Shape cooking: convex hull by farthest-point insertion, coplanar-face
// merging, exact polyhedral mass properties, and the adjacency collision needs.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "math.hpp"

namespace zc::phys {

Vec3 operator+(Vec3 a, Vec3 b) { Vec3 out{a.x + b.x, a.y + b.y, a.z + b.z}; return out; }
Vec3 operator-(Vec3 a, Vec3 b) { Vec3 out{a.x - b.x, a.y - b.y, a.z - b.z}; return out; }
Vec3 operator*(Vec3 a, double s) { Vec3 out{a.x * s, a.y * s, a.z * s}; return out; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    Vec3 out{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    return out;
}
double length(Vec3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
Vec3 normalized(Vec3 a) {
    const double size = length(a);
    if (size == 0) {
        Vec3 zero;
        return zero;
    }
    return a * (1.0 / size);
}
Quat multiply(Quat a, Quat b) {
    Quat out;
    out.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    out.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    out.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    out.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return out;
}
Quat conjugate(Quat q) { Quat out; out.w = q.w; out.x = -q.x; out.y = -q.y; out.z = -q.z; return out; }
Vec3 rotate(Quat q, Vec3 v) {
    const Vec3 u{q.x, q.y, q.z};
    const Vec3 c = cross(u, v);
    const Vec3 d = cross(u, c);
    Vec3 out{v.x + 2 * (q.w * c.x + d.x), v.y + 2 * (q.w * c.y + d.y), v.z + 2 * (q.w * c.z + d.z)};
    return out;
}
Quat from_axis_angle(Vec3 axis, double angle) {
    const Vec3 unit = normalized(axis);
    const double s = std::sin(0.5 * angle);
    Quat out;
    out.w = std::cos(0.5 * angle);
    out.x = unit.x * s; out.y = unit.y * s; out.z = unit.z * s;
    return out;
}

namespace {

constexpr double kWeldDistance = 1.0e-4;        // metres
constexpr double kPlaneEpsilon = 1.0e-9;        // metres; closer points count as on the hull
constexpr double kMergeCosine = 0.9999619230641713;   // cos(0.5 degrees)
constexpr double kMinPrincipalMoment = 1.0e-9;

struct Triangle {
    int a = 0, b = 0, c = 0;
    Vec3 normal;
    double offset = 0;
    bool alive = true;
};

Triangle make_triangle(const std::vector<Vec3>& points, int a, int b, int c) {
    Triangle t;
    t.a = a; t.b = b; t.c = c;
    t.normal = normalized(cross(points[b] - points[a], points[c] - points[a]));
    t.offset = dot(t.normal, points[a]);
    return t;
}

int find_root(std::vector<int>& parent, int index) {
    int root = index;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[index] != root) {
        const int next = parent[index];
        parent[index] = root;
        index = next;
    }
    return root;
}

// Outward triangles over the welded point list, or empty if degenerate.
std::vector<Triangle> hull_triangles(const std::vector<Vec3>& points) {
    std::vector<Triangle> triangles;
    const int count = static_cast<int>(points.size());
    int lowest = 0;
    int highest = 0;
    for (int i = 1; i < count; i += 1) {
        if (points[i].x < points[lowest].x) { lowest = i; }
        if (points[i].x > points[highest].x) { highest = i; }
    }
    if (lowest == highest) {
        return triangles;
    }
    const Vec3 axis = normalized(points[highest] - points[lowest]);
    int third = -1;
    double third_distance = 0;
    for (int i = 0; i < count; i += 1) {
        const Vec3 rel = points[i] - points[lowest];
        const double distance = length(rel - axis * dot(rel, axis));
        if (distance > third_distance) { third_distance = distance; third = i; }
    }
    if (third < 0 || third_distance < kWeldDistance) {
        return triangles;
    }
    const Vec3 plane_normal = normalized(cross(points[highest] - points[lowest], points[third] - points[lowest]));
    int fourth = -1;
    double fourth_distance = 0;
    for (int i = 0; i < count; i += 1) {
        const double distance = std::abs(dot(plane_normal, points[i] - points[lowest]));
        if (distance > fourth_distance) { fourth_distance = distance; fourth = i; }
    }
    if (fourth < 0 || fourth_distance < kWeldDistance) {
        return triangles;
    }
    const int seed[4] = {lowest, highest, third, fourth};
    const Vec3 inside = (points[seed[0]] + points[seed[1]] + points[seed[2]] + points[seed[3]]) * 0.25;
    const int seed_faces[4][3] = {
        {seed[0], seed[1], seed[2]}, {seed[0], seed[1], seed[3]}, {seed[0], seed[2], seed[3]}, {seed[1], seed[2], seed[3]}};
    for (int f = 0; f < 4; f += 1) {
        Triangle t = make_triangle(points, seed_faces[f][0], seed_faces[f][1], seed_faces[f][2]);
        if (dot(t.normal, inside) - t.offset > 0) {
            t = make_triangle(points, seed_faces[f][0], seed_faces[f][2], seed_faces[f][1]);
        }
        triangles.push_back(t);
    }
    std::vector<char> used(points.size(), 0);
    for (int k = 0; k < 4; k += 1) {
        used[seed[k]] = 1;
    }
    std::vector<char> visible_edge(points.size() * points.size(), 0);
    std::vector<int> visible;
    std::vector<std::pair<int, int>> horizon;
    for (;;) {
        int best = -1;
        double best_distance = kPlaneEpsilon;
        for (int i = 0; i < count; i += 1) {
            if (used[i]) { continue; }
            double outside = 0;
            for (std::size_t t = 0; t < triangles.size(); t += 1) {
                if (!triangles[t].alive) { continue; }
                outside = std::max(outside, dot(triangles[t].normal, points[i]) - triangles[t].offset);
            }
            if (outside > best_distance) { best_distance = outside; best = i; }
        }
        if (best < 0) {
            break;
        }
        used[best] = 1;
        visible.clear();
        horizon.clear();
        for (std::size_t t = 0; t < triangles.size(); t += 1) {
            const Triangle& tri = triangles[t];
            if (tri.alive && dot(tri.normal, points[best]) - tri.offset > kPlaneEpsilon) {
                visible.push_back(static_cast<int>(t));
                visible_edge[tri.a * count + tri.b] = 1;
                visible_edge[tri.b * count + tri.c] = 1;
                visible_edge[tri.c * count + tri.a] = 1;
            }
        }
        for (std::size_t k = 0; k < visible.size(); k += 1) {
            Triangle& tri = triangles[visible[k]];
            const int corners[3] = {tri.a, tri.b, tri.c};
            for (int e = 0; e < 3; e += 1) {
                const int from = corners[e];
                const int to = corners[(e + 1) % 3];
                if (!visible_edge[to * count + from]) {
                    horizon.push_back(std::pair<int, int>(from, to));
                }
            }
            tri.alive = false;
        }
        for (std::size_t k = 0; k < visible.size(); k += 1) {
            const Triangle& tri = triangles[visible[k]];
            visible_edge[tri.a * count + tri.b] = 0;
            visible_edge[tri.b * count + tri.c] = 0;
            visible_edge[tri.c * count + tri.a] = 0;
        }
        for (std::size_t k = 0; k < horizon.size(); k += 1) {
            triangles.push_back(make_triangle(points, horizon[k].first, horizon[k].second, best));
        }
    }
    std::vector<Triangle> alive;
    for (std::size_t t = 0; t < triangles.size(); t += 1) {
        if (triangles[t].alive) { alive.push_back(triangles[t]); }
    }
    return alive;
}

HullData build_hull(const std::vector<Vec3>& input) {
    std::vector<Vec3> points;
    for (std::size_t i = 0; i < input.size(); i += 1) {
        bool duplicate = false;
        for (std::size_t k = 0; k < points.size(); k += 1) {
            if (length(input[i] - points[k]) < kWeldDistance) { duplicate = true; break; }
        }
        if (!duplicate) { points.push_back(input[i]); }
    }
    if (points.size() < 4) {
        throw std::runtime_error("hull needs at least four distinct points");
    }
    const std::vector<Triangle> triangles = hull_triangles(points);
    if (triangles.empty()) {
        throw std::runtime_error("hull points are degenerate");
    }
    const int count = static_cast<int>(points.size());
    const int triangle_count = static_cast<int>(triangles.size());
    // Owner of each directed edge, then union of adjacent near-coplanar triangles.
    std::vector<int> edge_owner(points.size() * points.size(), -1);
    for (int t = 0; t < triangle_count; t += 1) {
        edge_owner[triangles[t].a * count + triangles[t].b] = t;
        edge_owner[triangles[t].b * count + triangles[t].c] = t;
        edge_owner[triangles[t].c * count + triangles[t].a] = t;
    }
    std::vector<int> parent(triangles.size());
    for (int t = 0; t < triangle_count; t += 1) { parent[t] = t; }
    for (int t = 0; t < triangle_count; t += 1) {
        const int corners[3] = {triangles[t].a, triangles[t].b, triangles[t].c};
        for (int e = 0; e < 3; e += 1) {
            const int other = edge_owner[corners[(e + 1) % 3] * count + corners[e]];
            if (other >= 0 && dot(triangles[t].normal, triangles[other].normal) > kMergeCosine) {
                const int root_a = find_root(parent, t);
                const int root_b = find_root(parent, other);
                if (root_a != root_b) { parent[std::max(root_a, root_b)] = std::min(root_a, root_b); }
            }
        }
    }
    // Boundary loop of each group.
    std::vector<std::vector<int>> loops;
    std::vector<int> next(points.size(), -1);
    for (int root = 0; root < triangle_count; root += 1) {
        if (find_root(parent, root) != root) { continue; }
        int start = -1;
        int boundary_edges = 0;
        for (int t = 0; t < triangle_count; t += 1) {
            if (find_root(parent, t) != root) { continue; }
            const int corners[3] = {triangles[t].a, triangles[t].b, triangles[t].c};
            for (int e = 0; e < 3; e += 1) {
                const int from = corners[e];
                const int to = corners[(e + 1) % 3];
                const int twin = edge_owner[to * count + from];
                const bool interior = twin >= 0 && find_root(parent, twin) == root;
                if (!interior) {
                    next[from] = to;
                    boundary_edges += 1;
                    if (start < 0 || from < start) { start = from; }
                }
            }
        }
        std::vector<int> loop;
        int cursor = start;
        for (int guard = 0; guard <= boundary_edges; guard += 1) {
            loop.push_back(cursor);
            cursor = next[cursor];
            if (cursor == start) { break; }
        }
        loops.push_back(loop);
    }
    // A vertex shared by only two merged faces lies inside a straight edge.
    std::vector<int> incidence(points.size(), 0);
    for (std::size_t f = 0; f < loops.size(); f += 1) {
        for (std::size_t k = 0; k < loops[f].size(); k += 1) { incidence[loops[f][k]] += 1; }
    }
    HullData hull;
    std::vector<int> remap(points.size(), -1);
    hull.face_first.push_back(0);
    for (std::size_t f = 0; f < loops.size(); f += 1) {
        std::vector<int> corners;
        for (std::size_t k = 0; k < loops[f].size(); k += 1) {
            if (incidence[loops[f][k]] > 2) { corners.push_back(loops[f][k]); }
        }
        if (corners.size() < 3) { continue; }
        Vec3 normal;
        for (std::size_t k = 0; k < corners.size(); k += 1) {
            normal = normal + cross(points[corners[k]], points[corners[(k + 1) % corners.size()]]);
        }
        normal = normalized(normal);
        double offset = 0;
        for (std::size_t k = 0; k < corners.size(); k += 1) { offset += dot(normal, points[corners[k]]); }
        offset /= static_cast<double>(corners.size());
        for (std::size_t k = 0; k < corners.size(); k += 1) {
            if (remap[corners[k]] < 0) {
                remap[corners[k]] = static_cast<int>(hull.vertices.size());
                hull.vertices.push_back(points[corners[k]]);
            }
            hull.face_loop.push_back(remap[corners[k]]);
        }
        hull.face_first.push_back(static_cast<std::int32_t>(hull.face_loop.size()));
        hull.face_normals.push_back(normal);
        hull.face_offsets.push_back(offset);
    }
    return hull;
}

// Edges with their two faces, and each vertex's faces, from the face loops.
void finish_adjacency(HullData& hull) {
    const int vertex_count = static_cast<int>(hull.vertices.size());
    const int face_count = static_cast<int>(hull.face_normals.size());
    std::vector<int> edge_index(static_cast<std::size_t>(vertex_count) * vertex_count, -1);
    std::vector<std::vector<int>> vertex_faces(hull.vertices.size());
    hull.edge_v0.clear(); hull.edge_v1.clear(); hull.edge_face_a.clear(); hull.edge_face_b.clear();
    for (int f = 0; f < face_count; f += 1) {
        const int first = hull.face_first[f];
        const int size = hull.face_first[f + 1] - first;
        for (int k = 0; k < size; k += 1) {
            const int from = hull.face_loop[first + k];
            const int to = hull.face_loop[first + (k + 1) % size];
            vertex_faces[from].push_back(f);
            const int low = std::min(from, to);
            const int high = std::max(from, to);
            if (edge_index[low * vertex_count + high] >= 0) {
                hull.edge_face_b[edge_index[low * vertex_count + high]] = f;
            } else {
                edge_index[low * vertex_count + high] = static_cast<int>(hull.edge_v0.size());
                hull.edge_v0.push_back(from);
                hull.edge_v1.push_back(to);
                hull.edge_face_a.push_back(f);
                hull.edge_face_b.push_back(-1);
            }
        }
    }
    for (std::size_t e = 0; e < hull.edge_face_b.size(); e += 1) {
        if (hull.edge_face_b[e] < 0) {
            throw std::runtime_error("hull is not closed");
        }
    }
    hull.vertex_face_first.clear();
    hull.vertex_faces.clear();
    hull.vertex_face_first.push_back(0);
    for (int v = 0; v < vertex_count; v += 1) {
        for (std::size_t k = 0; k < vertex_faces[v].size(); k += 1) { hull.vertex_faces.push_back(vertex_faces[v][k]); }
        hull.vertex_face_first.push_back(static_cast<std::int32_t>(hull.vertex_faces.size()));
    }
}

}  // namespace

Shape cook(const ShapeDesc& desc) {
    Shape shape;
    double volume = 0;
    Vec3 first;
    double second[6] = {0, 0, 0, 0, 0, 0};   // xx xy xz yy yz zz
    for (std::size_t h = 0; h < desc.hulls.size(); h += 1) {
        HullData hull = build_hull(desc.hulls[h].points);
        const int face_count = static_cast<int>(hull.face_normals.size());
        // Signed tetrahedra (origin, a, b, c) over each face fan.
        for (int f = 0; f < face_count; f += 1) {
            const int begin = hull.face_first[f];
            const int size = hull.face_first[f + 1] - begin;
            const Vec3 a = hull.vertices[hull.face_loop[begin]];
            for (int k = 1; k + 1 < size; k += 1) {
                const Vec3 b = hull.vertices[hull.face_loop[begin + k]];
                const Vec3 c = hull.vertices[hull.face_loop[begin + k + 1]];
                const double det = dot(a, cross(b, c));
                volume += det / 6;
                const Vec3 sum = a + b + c;
                first = first + sum * (det / 24);
                const Vec3 corners[4] = {a, b, c, sum};
                const double weight = det / 120;
                for (int n = 0; n < 4; n += 1) {
                    const Vec3 p = corners[n];
                    second[0] += weight * p.x * p.x; second[1] += weight * p.x * p.y; second[2] += weight * p.x * p.z;
                    second[3] += weight * p.y * p.y; second[4] += weight * p.y * p.z; second[5] += weight * p.z * p.z;
                }
            }
        }
        shape.hulls.push_back(hull);
    }
    const Vec3 com = first * (1.0 / volume);
    const double density = desc.density;
    const double cxx = second[0] - volume * com.x * com.x;
    const double cxy = second[1] - volume * com.x * com.y;
    const double cxz = second[2] - volume * com.x * com.z;
    const double cyy = second[3] - volume * com.y * com.y;
    const double cyz = second[4] - volume * com.y * com.z;
    const double czz = second[5] - volume * com.z * com.z;
    shape.inertia = {
        std::max(density * (cyy + czz), kMinPrincipalMoment), -density * cxy, -density * cxz,
        -density * cxy, std::max(density * (cxx + czz), kMinPrincipalMoment), -density * cyz,
        -density * cxz, -density * cyz, std::max(density * (cxx + cyy), kMinPrincipalMoment)};
    shape.com_offset = com;
    shape.volume = volume;
    shape.mass = density * volume;
    shape.friction = desc.friction;
    shape.restitution = desc.restitution;
    shape.rolling_resistance = desc.rolling_resistance;
    for (std::size_t h = 0; h < shape.hulls.size(); h += 1) {
        HullData& hull = shape.hulls[h];
        Vec3 centroid;
        for (std::size_t v = 0; v < hull.vertices.size(); v += 1) {
            hull.vertices[v] = hull.vertices[v] - com;
            shape.radius = std::max(shape.radius, length(hull.vertices[v]));
            centroid = centroid + hull.vertices[v];
        }
        hull.centroid = centroid * (1.0 / static_cast<double>(hull.vertices.size()));
        for (std::size_t f = 0; f < hull.face_offsets.size(); f += 1) {
            hull.face_offsets[f] -= dot(hull.face_normals[f], com);
        }
        finish_adjacency(hull);
        shape.hull_vertices.push_back(hull.vertices);
    }
    return shape;
}

}  // namespace zc::phys
