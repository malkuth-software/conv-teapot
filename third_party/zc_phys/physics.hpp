#pragma once
// Zen Construction rigid-body engine. The declarations down to WorldParams and
// the first block of World follow the specification's section 4 exactly; the
// rest (static bodies, solver parameters, the rest report, the stable-set
// query and solver statistics) are additions.
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace zc::phys {

struct Vec3 { double x = 0, y = 0, z = 0; };
Vec3 operator+(Vec3 a, Vec3 b);
Vec3 operator-(Vec3 a, Vec3 b);
Vec3 operator*(Vec3 a, double s);
double dot(Vec3 a, Vec3 b);
Vec3 cross(Vec3 a, Vec3 b);
double length(Vec3 a);
Vec3 normalized(Vec3 a);

struct Quat { double w = 1, x = 0, y = 0, z = 0; };
Quat multiply(Quat a, Quat b);
Quat conjugate(Quat q);
Vec3 rotate(Quat q, Vec3 v);
Quat from_axis_angle(Vec3 axis, double angle);   // not for use inside step()

struct Pose { Vec3 p; Quat q; };                   // p = the body's centre of mass in world space

using BodyId = std::int32_t;     // assigned 0,1,2,... and never reused within a World
constexpr BodyId kGround = -1;   // the ground plane, in contacts and raycasts

struct HullDesc { std::vector<Vec3> points; };     // any points; the engine takes their convex hull
struct ShapeDesc {
    std::vector<HullDesc> hulls;                   // hulls in one shared local frame
    double density = 2600;                         // kg/m^3 (granite-ish)
    double friction = 0.75;                        // Coulomb coefficient, rock against rock
    double restitution = 0.05;
    double rolling_resistance = 0.002;             // metres (lever arm)
};

// One cooked convex hull in the body frame: polygonal faces with outward unit
// normals, unique edges with their two faces, and each vertex's faces.
struct HullData {
    std::vector<Vec3> vertices;
    std::vector<Vec3> face_normals;
    std::vector<double> face_offsets;              // dot(normal, x) = offset on the face
    std::vector<std::int32_t> face_first;          // face f owns loop entries [face_first[f], face_first[f + 1])
    std::vector<std::int32_t> face_loop;           // vertex indices, counter-clockwise seen from outside
    std::vector<std::int32_t> edge_v0, edge_v1, edge_face_a, edge_face_b;
    std::vector<std::int32_t> vertex_face_first;   // vertex v touches faces [first[v], first[v + 1]) of vertex_faces
    std::vector<std::int32_t> vertex_faces;
    Vec3 centroid;
};

struct Shape {                                     // "cooked": produced by cook(), shareable by many bodies
    std::vector<std::vector<Vec3>> hull_vertices;  // per hull, in the body frame (origin at the centre of mass)
    std::vector<HullData> hulls;                   // collision data for the same hulls
    Vec3 com_offset;     // the centre of mass in the ShapeDesc's input frame; body frame = input frame - com_offset
    double mass = 0, volume = 0, radius = 0;       // radius: max distance of any vertex from the centre of mass
    std::array<double, 9> inertia{};               // body-frame inertia tensor about the centre of mass, row-major
    double friction = 0.75, restitution = 0.05, rolling_resistance = 0.002;
};
Shape cook(const ShapeDesc& desc);                 // convex hulls + mass properties; may throw on degenerate input

struct BodyState { Pose pose; Vec3 v; Vec3 w; bool asleep = false; };

struct Contact {                                   // one contact point from the last step
    BodyId a, b;                                   // a is a free body; b is another body or kGround
    Vec3 point;                                    // world, midway between the surfaces
    Vec3 normal;                                   // unit; the direction b pushes a (from b toward a)
    double separation;                             // metres; negative when overlapping
    double normal_impulse;                         // N s over the last frame
};

struct HoldParams {
    double lin_hertz = 3.0, lin_zeta = 1.0;        // spring of the hook toward its target position
    double ang_hertz = 2.0, ang_zeta = 1.0;        // and toward its target orientation
    double max_lift = 1.5;                         // wire tension cap, in units of the body's weight m g
    double max_lateral = 0.35;                     // horizontal force cap, in units of m g
    double sling_length = 0.0;                     // optional hook-to-COM length; sideways travel takes up slack
    double max_torque = 0.35;                      // torque cap, in units of m g radius
};
struct HoldState {
    double tension = 0;                            // wire force / (m g) over the last frame: 1 hanging free, 0 slack
    Vec3 position_error;                           // target - actual centre of mass
    double lateral_force = 0;                      // horizontal hook force / (m g) over the last frame
};

// Immutable scene terrain. Sampling is allocation-free during the solve.
struct GroundSample { double height; Vec3 normal; };
class GroundSurface {
public:
    virtual ~GroundSurface() = default;
    virtual GroundSample sample(double x, double y) const = 0;
    virtual double raycast(Vec3 origin, Vec3 direction, double max_t) const = 0;
};

struct WorldParams {
    Vec3 gravity{0, 0, -9.81};
    double frame_dt = 1.0 / 60.0;                  // one step() advances this much
    int substeps = 8;                              // unused: the solver takes one solve per frame
    double contact_hertz = 30.0;                   // unused
    double contact_damping = 10.0;                 // unused
    double push_max_velocity = 0.5;                // m/s; the fastest overlap is ever pushed apart
    double linear_slop = 0.0005;                   // m; a gap wider than this delays a rebound by one frame
    double speculative_distance = 0.002;           // m; base contact margin (plus motion)
    double linear_damping = 0.02, angular_damping = 0.05;   // 1/s
    double sleep_linear = 0.004, sleep_angular = 0.02;      // m/s, rad/s
    double sleep_time = 0.5;                       // s below both thresholds before an island sleeps
    double ground_z = 0.0, ground_friction = 0.9;
    std::shared_ptr<const GroundSurface> ground_surface; // absent: the original infinite plane
    double restitution_threshold = 1.0;            // m/s; slower impacts never bounce
};

// Parameters of the contact solve (additions to the specification).
struct SolverParams {
    double regularization = 0.03;          // per-pass contact compliance, as a fraction of the inverse carried mass
    double relaxation_time = 0.1;          // s; overlap decay time
    double tight_tolerance = 1.0e-9;       // Newton gradient tolerance against the momentum scale
    double loose_tolerance = 1.0e-3;       // tolerance while the retained impulse field is still changing
    double pass_tolerance = 1.0e-3;        // relative impulse change that ends the passes
    double shift_tolerance = 1.0e-3;       // m/s change of the sliding shift that ends the passes
    int max_newton_iterations = 30;        // per pass
    int max_passes = 6;
    double ground_rolling_resistance = 0.004;   // metres; the sand bed absorbs rocking and spin
    bool sleeping = true;
};

// What the game reads each frame to know whether the pile has quieted.
struct RestReport {
    bool quiet = false;            // nothing awake, or the fastest body has been slow for sleep_time
    double quiet_seconds = 0;
    double kinetic_energy = 0;
    double max_speed = 0;          // fastest awake body, linear plus angular times radius
    int awake = 0;
};

struct SolveStats {
    std::int64_t frames = 0, passes = 0, newton_iterations = 0, factorizations = 0, factor_reuses = 0;
    std::int64_t line_search_evaluations = 0, analyses = 0, not_converged = 0, pass_limited = 0;
    std::int64_t pair_tests = 0, pair_cache_hits = 0;   // hull pairs reaching the narrow phase; those answered from the pair cache
    int last_passes = 0, last_newton_iterations = 0, last_contacts = 0;
    double last_residual = 0;
};

class World {
public:
    explicit World(const WorldParams& params = {});
    ~World();
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    BodyId add_body(const Shape& shape, const Pose& pose, Vec3 v = {}, Vec3 w = {}, bool start_asleep = false);
    void remove_body(BodyId id);                   // also releases the hold if it held this body; wakes its contacts
    bool has(BodyId id) const;
    std::vector<BodyId> bodies() const;            // ascending ids
    BodyState state(BodyId id) const;
    void set_pose(BodyId id, const Pose& pose);    // teleport; zeroes velocity; wakes the body and its contacts
    void wake(BodyId id);                          // wakes the whole touching group

    // the crane: at most one held body at a time
    void hold(BodyId id, const Pose& target, const HoldParams& params = {});
    void set_hold_target(const Pose& target);
    void release();
    std::optional<BodyId> held() const;
    HoldState hold_state() const;

    void step();                                   // advance by frame_dt

    std::vector<Contact> contacts() const;         // last step's contacts, deterministic order
    bool resting(BodyId id) const;                 // asleep, or below the sleep thresholds for >= 0.25 s
    std::optional<std::pair<BodyId, double>> raycast(Vec3 origin, Vec3 dir, double max_t) const;
    bool all_asleep() const;                       // ignoring a held body
    double kinetic_energy() const;
    std::uint64_t checksum() const;                // hash of the bit patterns of every pose and velocity
    bool out_of_bounds(BodyId id) const;           // centre below z = -0.5 or more than 5 m from the origin horizontally

    const WorldParams& params() const;

    // Additions.
    BodyId add_static_body(const Shape& shape, const Pose& pose);   // never moves; collides like the ground
    void set_solver_params(const SolverParams& params);
    const SolverParams& solver_params() const;
    RestReport rest_report() const;                // constant time
    std::vector<BodyId> stable_set() const;        // settled bodies joined to the ground through loaded contacts
    // Every loaded contact the world remembers: last frame's, and those of
    // sleeping bodies (which contacts() leaves out once they sleep). Points come
    // from body a's frame; separation is not tracked here (0).
    std::vector<Contact> retained_contacts() const;
    SolveStats solve_stats() const;
    int guard_count() const;                       // times a non-finite state was rolled back (tests expect 0)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace zc::phys
