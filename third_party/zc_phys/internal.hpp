#pragma once
// Records shared by the world and the contact solve.
#include <cstdint>
#include <memory>
#include <vector>

#include "block_ldl.hpp"
#include "collide.hpp"

namespace zc::phys {

struct Body {
    bool alive = true;
    bool is_static = false;
    bool asleep = false;
    bool hulls_fresh = false;        // a resting body's world hulls are current
    std::shared_ptr<const Shape> shape;
    Vec3 position, velocity, angular;
    Quat orientation;
    Pose previous_pose;              // last good pose, for the non-finite guard
    double mass = 0, inv_mass = 0;
    Mat3 inertia_local, inv_inertia_local, inertia_world, inv_inertia_world;
    std::int32_t island = 0;
    double quiet_time = 0;
    std::int32_t solver_index = -1;
    std::vector<WorldHull> world_hulls;
};

constexpr int kModeOpen = 0;
constexpr int kModeStick = 1;
constexpr int kModeSlide = 2;

struct ContactPoint {
    std::int32_t a = 0, b = 0;       // a is free; b is another body or kGround
    std::int32_t hull_a = 0, hull_b = 0;
    bool fixed_b = false;            // b is the ground or a static body
    bool matched = false;            // inherited last frame's impulse
    Vec3 point_a, point_b, normal, local_a;
    double separation = 0, friction = 0, restitution = 0;
    double normal_impulse = 0;       // this frame's result, also next frame's retained normal impulse
    Vec3 tangent_impulse;            // world vector; retained
    double warm_normal = 0;          // inherited
    Vec3 warm_tangent;               // inherited
    double bounce = 0;               // rebound speed owed on the next frame
    double pending_bounce = 0;       // rebound speed inherited from the last frame
    int mode = kModeOpen;
};

constexpr int kKindContact = 0;      // friction cone, retained
constexpr int kKindBall = 1;         // 3-vector bounded in length
constexpr int kKindHold = 2;         // crane linear spring

struct Row {
    int kind = kKindContact;
    bool retained = true;
    std::int32_t index_a = -1, index_b = -1;
    double ja[18], jb[18];           // three rows of six: tangent 1, tangent 2, normal
    Vec3 t1, t2;
    double friction = 0, local_inverse_mass = 0, r = 0;
    double vhat0 = 0, vhat1 = 0, vhat_n = 0, vhat_n_base = 0, shift = 0;
    double lambda0 = 0, lambda1 = 0, lambda_n = 0;
    double gamma0 = 0, gamma1 = 0, gamma_n = 0;
    double limit = 0, limit_lift = 0;
    int mode = kModeOpen;
    double g[9];
    std::int32_t contact = -1;       // index into the frame's contacts, for contact rows
    std::int32_t member_first = 0, member_count = 0;   // contact rows of a rolling row's pair
    double resistance = 0;
    std::uint64_t pair_key = 0;
};

struct RollingMemory { std::uint64_t pair_key = 0; Vec3 impulse; };

// One touching pair seen as a support: `upper` rests on `lower` through the
// contact rows [first, first + count).
struct Support {
    std::int32_t upper = 0, lower = 0, rank = 0, first = 0, count = 0;
    double weight = 0;
};

struct HoldRecord {
    bool active = false;
    BodyId id = 0;
    Pose target;
    HoldParams params;
    HoldState state;
};

// Storage the solve reuses between frames.
struct SolverWork {
    BlockFactor factor;
    std::vector<std::int32_t> graph, previous_graph, pairs;
    std::vector<Body*> bodies;
    std::vector<Row> rows;
    std::vector<double> twist, free_twist, gradient, momentum, impulse, direction, difference, mass_direction;
    std::vector<double> contact_velocity, contact_step, load;
    std::vector<std::int32_t> order, rank;
    std::vector<Support> supports;
    std::vector<RollingMemory> rolling, next_rolling;
    bool factor_valid = false;
    std::uint32_t factor_signature = 0;
};

// Advances every awake free body by one frame. Contacts carry inherited
// impulses in and this frame's impulses out.
void solve_frame(const WorldParams& params, const SolverParams& solver, std::vector<Body>& bodies,
                 std::vector<ContactPoint>& contacts, HoldRecord& hold, SolverWork& work, SolveStats& stats);

}  // namespace zc::phys
