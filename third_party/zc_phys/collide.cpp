#include "collide.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace zc::phys {

namespace {

constexpr double kEdgeParallelTolerance = 1.0e-6;
// An edge pair counts as a separating-axis candidate only if its edge-to-edge
// distance matches the hulls' true separation along that axis, in metres.
constexpr double kEdgeSupportTolerance = 1.0e-7;
// Preference for face axes (and for A's face), in metres. It keeps resting
// face contacts from flickering to single-point edge contacts. A biased face
// manifold is accepted only if it holds the depth its axis measured.
constexpr double kFaceBias = 1.0e-3;
constexpr double kWitnessTolerance = 2.0e-4;
constexpr double kWitnessReach = 5.0e-3;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct FaceQuery { double separation = -kInfinity; int face = -1; };
struct EdgeQuery { double separation = -kInfinity; int edge_a = -1, edge_b = -1; Vec3 axis; };

// Largest separation of `other` along the face normals of `reference`.
FaceQuery query_faces(const WorldHull& reference, const WorldHull& other) {
    FaceQuery best;
    const int face_count = static_cast<int>(reference.normals.size());
    const int vertex_count = static_cast<int>(other.vertices.size());
    for (int f = 0; f < face_count; f += 1) {
        const Vec3 normal = reference.normals[f];
        double lowest = kInfinity;
        for (int v = 0; v < vertex_count; v += 1) {
            lowest = std::min(lowest, dot(normal, other.vertices[v]));
        }
        const double separation = lowest - reference.offsets[f];
        if (separation > best.separation) {
            best.separation = separation;
            best.face = f;
        }
    }
    return best;
}

// Largest separation over edge pairs that form a face of the Minkowski
// difference (Gregorius, GDC 2013), each improving candidate verified against
// the hulls' true separation along its axis.
EdgeQuery query_edges(const WorldHull& hull_a, const WorldHull& hull_b) {
    EdgeQuery best;
    const HullData& data_a = *hull_a.hull;
    const HullData& data_b = *hull_b.hull;
    const int edges_a = static_cast<int>(data_a.edge_v0.size());
    const int edges_b = static_cast<int>(data_b.edge_v0.size());
    for (int ea = 0; ea < edges_a; ea += 1) {
        const Vec3 a = hull_a.normals[data_a.edge_face_a[ea]];
        const Vec3 b = hull_a.normals[data_a.edge_face_b[ea]];
        const Vec3 bxa = hull_a.edge_crosses[ea];
        const Vec3 head_a = hull_a.vertices[data_a.edge_v0[ea]];
        const Vec3 direction_a = hull_a.edge_directions[ea];
        for (int eb = 0; eb < edges_b; eb += 1) {
            const Vec3 nc = hull_b.normals[data_b.edge_face_a[eb]];
            const Vec3 nd = hull_b.normals[data_b.edge_face_b[eb]];
            const double cba = -dot(nc, bxa);
            const double dba = -dot(nd, bxa);
            if (!(cba * dba < 0)) {
                continue;
            }
            const Vec3 dxc = hull_b.edge_crosses[eb];
            const double adc = dot(a, dxc);
            const double bdc = dot(b, dxc);
            if (!(adc * bdc < 0 && cba * bdc > 0)) {
                continue;
            }
            const Vec3 direction_b = hull_b.edge_directions[eb];
            Vec3 axis = cross(direction_a, direction_b);
            const double axis_length = length(axis);
            if (axis_length < kEdgeParallelTolerance * length(direction_a) * length(direction_b)) {
                continue;
            }
            axis = axis * (1.0 / axis_length);
            if (dot(axis, head_a - hull_a.centroid) < 0) {
                axis = axis * -1.0;
            }
            const Vec3 head_b = hull_b.vertices[data_b.edge_v0[eb]];
            const double separation = dot(axis, head_b - head_a);
            if (separation > best.separation) {
                double lowest_b = kInfinity;
                for (std::size_t v = 0; v < hull_b.vertices.size(); v += 1) {
                    lowest_b = std::min(lowest_b, dot(axis, hull_b.vertices[v]));
                }
                double highest_a = -kInfinity;
                for (std::size_t v = 0; v < hull_a.vertices.size(); v += 1) {
                    highest_a = std::max(highest_a, dot(axis, hull_a.vertices[v]));
                }
                if (std::abs((lowest_b - highest_a) - separation) > kEdgeSupportTolerance) {
                    continue;
                }
                best.separation = separation;
                best.edge_a = ea;
                best.edge_b = eb;
                best.axis = axis;
            }
        }
    }
    return best;
}

// Sutherland-Hodgman: keeps the part of `polygon` with dot(normal, p) <= offset.
void clip_polygon(const std::vector<Vec3>& polygon, Vec3 normal, double offset, std::vector<Vec3>& out) {
    out.clear();
    const std::size_t count = polygon.size();
    for (std::size_t k = 0; k < count; k += 1) {
        const Vec3 current = polygon[k];
        const Vec3 following = polygon[(k + 1) % count];
        const double current_distance = dot(normal, current) - offset;
        const double following_distance = dot(normal, following) - offset;
        if (current_distance <= 0) {
            out.push_back(current);
        }
        if ((current_distance < 0 && following_distance > 0) || (current_distance > 0 && following_distance < 0)) {
            const double t = current_distance / (current_distance - following_distance);
            out.push_back(add_scaled(current, following - current, t));
        }
    }
}

ManifoldPoint face_point(Vec3 on_incident, Vec3 normal, double separation, bool reference_is_a) {
    ManifoldPoint point;
    const Vec3 on_reference = add_scaled(on_incident, normal, -separation);
    point.separation = separation;
    if (reference_is_a) {
        point.point_a = on_reference;
        point.point_b = on_incident;
        point.normal = normal * -1.0;
    } else {
        point.point_a = on_incident;
        point.point_b = on_reference;
        point.normal = normal;
    }
    return point;
}

struct FaceManifold {
    double deepest = kInfinity;
    double witness_separation = 0;
    double witness_outside = -kInfinity;
    ManifoldPoint witness_point;
};

// Clips the incident face of `inc` against reference face `face` of `ref`.
FaceManifold face_manifold(const WorldHull& ref, const WorldHull& inc, int face, double margin, bool reference_is_a,
                           CollideScratch& scratch, std::vector<ManifoldPoint>& points) {
    points.clear();
    const Vec3 normal = ref.normals[face];
    int support = 0;
    double support_distance = kInfinity;
    for (std::size_t v = 0; v < inc.vertices.size(); v += 1) {
        const double distance = dot(normal, inc.vertices[v]);
        if (distance < support_distance) {
            support_distance = distance;
            support = static_cast<int>(v);
        }
    }
    // The most anti-parallel incident face among those sharing the witness.
    const HullData& inc_data = *inc.hull;
    int incident = inc_data.vertex_faces[inc_data.vertex_face_first[support]];
    double lowest_dot = kInfinity;
    for (int k = inc_data.vertex_face_first[support]; k < inc_data.vertex_face_first[support + 1]; k += 1) {
        const double alignment = dot(inc.normals[inc_data.vertex_faces[k]], normal);
        if (alignment < lowest_dot) {
            lowest_dot = alignment;
            incident = inc_data.vertex_faces[k];
        }
    }
    scratch.polygon.clear();
    for (int k = inc_data.face_first[incident]; k < inc_data.face_first[incident + 1]; k += 1) {
        scratch.polygon.push_back(inc.vertices[inc_data.face_loop[k]]);
    }
    FaceManifold result;
    const HullData& ref_data = *ref.hull;
    const int begin = ref_data.face_first[face];
    const int size = ref_data.face_first[face + 1] - begin;
    for (int k = 0; k < size; k += 1) {
        const Vec3 from = ref.vertices[ref_data.face_loop[begin + k]];
        const Vec3 to = ref.vertices[ref_data.face_loop[begin + (k + 1) % size]];
        const Vec3 side = normalized(cross(to - from, normal));
        const double side_offset = dot(side, from);
        result.witness_outside = std::max(result.witness_outside, dot(side, inc.vertices[support]) - side_offset);
        if (!scratch.polygon.empty()) {
            clip_polygon(scratch.polygon, side, side_offset, scratch.clipped);
            scratch.polygon.swap(scratch.clipped);
        }
    }
    result.witness_separation = support_distance - ref.offsets[face];
    for (std::size_t k = 0; k < scratch.polygon.size(); k += 1) {
        const double separation = dot(normal, scratch.polygon[k]) - ref.offsets[face];
        if (separation > margin) {
            continue;
        }
        result.deepest = std::min(result.deepest, separation);
        points.push_back(face_point(scratch.polygon[k], normal, separation, reference_is_a));
    }
    result.witness_point = face_point(inc.vertices[support], normal, result.witness_separation, reference_is_a);
    return result;
}

ManifoldPoint edge_manifold(const WorldHull& hull_a, const WorldHull& hull_b, const EdgeQuery& query) {
    const Vec3 p1 = hull_a.vertices[(*hull_a.hull).edge_v0[query.edge_a]];
    const Vec3 d1 = hull_a.edge_directions[query.edge_a];
    const Vec3 p2 = hull_b.vertices[(*hull_b.hull).edge_v0[query.edge_b]];
    const Vec3 d2 = hull_b.edge_directions[query.edge_b];
    const Vec3 r = p1 - p2;
    const double a = dot(d1, d1);
    const double e = dot(d2, d2);
    const double f = dot(d2, r);
    const double c = dot(d1, r);
    const double b = dot(d1, d2);
    const double denominator = a * e - b * b;
    double s = 0;
    if (denominator > 0) {
        s = std::clamp((b * f - c * e) / denominator, 0.0, 1.0);
    }
    double t = (b * s + f) / e;
    if (t < 0) {
        t = 0;
        s = std::clamp(-c / a, 0.0, 1.0);
    } else if (t > 1) {
        t = 1;
        s = std::clamp((b - c) / a, 0.0, 1.0);
    }
    ManifoldPoint point;
    point.point_a = add_scaled(p1, d1, s);
    point.point_b = add_scaled(p2, d2, t);
    point.normal = query.axis * -1.0;
    point.separation = query.separation;
    return point;
}

void append(std::vector<ManifoldPoint>& out, const std::vector<ManifoldPoint>& points) {
    for (std::size_t k = 0; k < points.size(); k += 1) {
        out.push_back(points[k]);
    }
}

}  // namespace

void transform_hull(const HullData& hull, Vec3 position, Quat orientation, WorldHull& out) {
    const Mat3 rotation = to_matrix(orientation);
    out.hull = &hull;
    const std::size_t vertex_count = hull.vertices.size();
    out.vertices.resize(vertex_count);
    Vec3 low{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
    Vec3 high{-low.x, -low.y, -low.z};
    for (std::size_t v = 0; v < vertex_count; v += 1) {
        const Vec3 p = position + mul(rotation, hull.vertices[v]);
        out.vertices[v] = p;
        low.x = std::min(low.x, p.x); low.y = std::min(low.y, p.y); low.z = std::min(low.z, p.z);
        high.x = std::max(high.x, p.x); high.y = std::max(high.y, p.y); high.z = std::max(high.z, p.z);
    }
    out.low = low;
    out.high = high;
    const std::size_t face_count = hull.face_normals.size();
    out.normals.resize(face_count);
    out.offsets.resize(face_count);
    for (std::size_t f = 0; f < face_count; f += 1) {
        const Vec3 normal = mul(rotation, hull.face_normals[f]);
        out.normals[f] = normal;
        out.offsets[f] = hull.face_offsets[f] + dot(normal, position);
    }
    const std::size_t edge_count = hull.edge_v0.size();
    out.edge_directions.resize(edge_count);
    out.edge_crosses.resize(edge_count);
    for (std::size_t e = 0; e < edge_count; e += 1) {
        out.edge_directions[e] = out.vertices[hull.edge_v1[e]] - out.vertices[hull.edge_v0[e]];
        out.edge_crosses[e] = cross(out.normals[hull.edge_face_b[e]], out.normals[hull.edge_face_a[e]]);
    }
    out.centroid = position + mul(rotation, hull.centroid);
}

bool bounds_overlap(const WorldHull& a, const WorldHull& b, double margin) {
    if (a.low.x - margin > b.high.x || b.low.x - margin > a.high.x) { return false; }
    if (a.low.y - margin > b.high.y || b.low.y - margin > a.high.y) { return false; }
    if (a.low.z - margin > b.high.z || b.low.z - margin > a.high.z) { return false; }
    return true;
}

void collide_hulls(const WorldHull& hull_a, const WorldHull& hull_b, double margin, CollideScratch& scratch,
                   std::vector<ManifoldPoint>& out, double& separated_by) {
    separated_by = -kInfinity;
    const FaceQuery face_a = query_faces(hull_a, hull_b);
    if (face_a.separation > margin) {
        separated_by = face_a.separation;
        return;
    }
    const FaceQuery face_b = query_faces(hull_b, hull_a);
    if (face_b.separation > margin) {
        separated_by = face_b.separation;
        return;
    }
    const EdgeQuery edge = query_edges(hull_a, hull_b);
    if (edge.separation > margin) {
        separated_by = edge.separation;
        return;
    }
    bool use_a = true;
    double face_separation = face_a.separation;
    double other_separation = face_b.separation;
    if (face_b.separation > face_a.separation + kFaceBias + 0.05 * std::abs(face_a.separation)) {
        use_a = false;
        face_separation = face_b.separation;
        other_separation = face_a.separation;
    }
    const double bias = kFaceBias + 0.05 * std::abs(face_separation);
    if (edge.edge_a >= 0 && edge.separation > face_separation + bias) {
        out.push_back(edge_manifold(hull_a, hull_b, edge));
        return;
    }
    FaceManifold manifold;
    if (use_a) {
        manifold = face_manifold(hull_a, hull_b, face_a.face, margin, true, scratch, scratch.first);
    } else {
        manifold = face_manifold(hull_b, hull_a, face_b.face, margin, false, scratch, scratch.first);
    }
    // A face manifold stands only if it kept the depth the axis test measured.
    if (manifold.deepest <= manifold.witness_separation + kWitnessTolerance) {
        append(out, scratch.first);
        return;
    }
    if (other_separation >= face_separation - bias) {
        FaceManifold other;
        if (use_a) {
            other = face_manifold(hull_b, hull_a, face_b.face, margin, false, scratch, scratch.second);
        } else {
            other = face_manifold(hull_a, hull_b, face_a.face, margin, true, scratch, scratch.second);
        }
        if (other.deepest <= other.witness_separation + kWitnessTolerance) {
            append(out, scratch.second);
            return;
        }
    }
    if (edge.edge_a >= 0 && edge.separation >= face_separation - bias) {
        out.push_back(edge_manifold(hull_a, hull_b, edge));
        return;
    }
    append(out, scratch.first);
    if (manifold.witness_outside <= kWitnessReach + std::abs(manifold.witness_separation)) {
        out.push_back(manifold.witness_point);
    }
}

void collide_ground(const WorldHull& hull, double ground_z, const GroundSurface* surface, double margin, std::vector<ManifoldPoint>& out) {
    for (std::size_t v = 0; v < hull.vertices.size(); v += 1) {
        const Vec3 vertex = hull.vertices[v];
        const GroundSample ground = surface ? (*surface).sample(vertex.x, vertex.y) : GroundSample{ground_z, Vec3{0, 0, 1}};
        const double separation = (vertex.z - ground.height) * ground.normal.z;
        if (separation < margin) {
            ManifoldPoint point;
            point.point_a = vertex;
            point.point_b = vertex - ground.normal * separation;
            point.normal = ground.normal;
            point.separation = separation;
            out.push_back(point);
        }
    }
}

}  // namespace zc::phys
