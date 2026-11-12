#pragma once
// Narrow phase: separating-axis test with Gauss-map edge pruning, face
// clipping checked against the axis witness, and hull-ground vertices.
#include <vector>

#include "math.hpp"

namespace zc::phys {

// A cooked hull placed in the world. Storage is reused between frames.
struct WorldHull {
    const HullData* hull = nullptr;
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<double> offsets;
    std::vector<Vec3> edge_directions;   // vertex v1 - vertex v0 per edge
    std::vector<Vec3> edge_crosses;      // cross(normal of face b, normal of face a) per edge
    Vec3 centroid, low, high;
};

// A contact point: where it lies on each surface, the direction body B pushes
// body A, and the signed gap along that direction.
struct ManifoldPoint {
    Vec3 point_a, point_b, normal;
    double separation = 0;
};

struct CollideScratch {
    std::vector<Vec3> polygon, clipped;
    std::vector<ManifoldPoint> first, second;
};

void transform_hull(const HullData& hull, Vec3 position, Quat orientation, WorldHull& out);
bool bounds_overlap(const WorldHull& a, const WorldHull& b, double margin);
// Appends the contact points of two hulls to `out`; nothing beyond `margin`.
// `separated_by` receives the separation along the axis that proved the hulls
// apart (a lower bound on their true distance), or -infinity when they touch.
void collide_hulls(const WorldHull& a, const WorldHull& b, double margin, CollideScratch& scratch,
                   std::vector<ManifoldPoint>& out, double& separated_by);
// Appends the contact points of a hull against the plane z = ground_z; the hull is side A.
void collide_ground(const WorldHull& hull, double ground_z, const GroundSurface* surface, double margin, std::vector<ManifoldPoint>& out);

}  // namespace zc::phys
