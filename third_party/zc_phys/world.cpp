// World: bodies, once-per-frame collision, the retained impulse field,
// islands and sleeping, the crane, and the game-facing queries.
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <cstring>

#include "internal.hpp"

namespace zc::phys {

namespace {

constexpr std::int32_t kLoadedIsland = 1;     // bodies added asleep share one island until woken

// Last frame's impulse at one contact point, keyed by its manifold.
struct Persisted {
    std::uint64_t key = 0;
    std::int32_t a = 0, b = 0;
    bool fixed_b = false;
    bool used = false;
    Vec3 local_a, normal, tangent_impulse;
    double normal_impulse = 0, bounce = 0;
};

struct PersistedByKey {
    bool operator()(const Persisted& x, const Persisted& y) const { return x.key < y.key; }
};

std::uint64_t manifold_key(std::int32_t a, std::int32_t b, std::int32_t hull_a, std::int32_t hull_b) {
    return (((static_cast<std::uint64_t>(a) << 24) | static_cast<std::uint64_t>(b + 1)) << 8) |
           (static_cast<std::uint64_t>(hull_a) << 4) | static_cast<std::uint64_t>(hull_b);
}

bool finite(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

int find_root(std::vector<std::int32_t>& parent, int index) {
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

}  // namespace

// What one hull pair looked like when the narrow phase last ran on it: the two
// bodies' poses then, the margin used, and either the manifold (in each body's
// own frame) or a lower bound on the gap. A pair whose bodies have since moved
// less than the gap allows is still apart; a touching pair whose bodies have
// moved less than kReuseMotion keeps its manifold.
struct PairCache {
    Vec3 position_a, position_b;
    Quat orientation_a, orientation_b;
    double margin = 0;
    double separated_by = 0;          // gap bound when apart
    bool touching = false;
    std::vector<ManifoldPoint> local_points;   // point_a and normal in A's frame, point_b in B's frame
    std::int64_t stamp = 0;           // the frame it was last read or written
};

constexpr double kReuseMotion = 1.0e-7;   // m: below this a touching pair's manifold is reused (the slop is 5e-4)
constexpr double kReuseMargin = 1.0e-7;   // m
constexpr double kPi = 3.14159265358979323846;

struct World::Impl {
    WorldParams params;
    SolverParams solver;
    std::vector<Body> bodies;
    std::vector<ContactPoint> contacts;
    std::vector<Persisted> persisted, next_persisted;
    std::vector<ManifoldPoint> points;
    CollideScratch scratch;
    SolverWork work;
    SolveStats stats;
    HoldRecord hold;
    RestReport rest;
    std::vector<std::int32_t> parent;
    std::vector<char> island_quiet;
    std::vector<std::int32_t> island_id;
    std::int32_t next_island = kLoadedIsland + 1;
    int guards = 0;
    std::unordered_map<std::uint64_t, PairCache> pair_cache;   // looked up by key only; never iterated for results
    std::int64_t frame = 0;

    void wake_island(std::int32_t id);
    void wake_touching(std::int32_t id);
    void refresh_hulls(Body& body);
    double motion_bound(const Body& body) const;
    void warm_start(std::uint64_t key, const Body& body_a, std::size_t first);
    void add_points(std::int32_t a, std::int32_t b, std::int32_t hull_a, std::int32_t hull_b, bool fixed_b,
                    double friction, double restitution);
    void collide();
    void collide_pair(std::uint64_t key, const Body& first_body, const Body& second_body, const WorldHull& hull_a,
                      const WorldHull& hull_b, double margin);
    void prune_pair_cache();
    void persist();
    void guard();
    void update_sleep();
};

void World::Impl::wake_island(std::int32_t id) {
    Body& body = bodies[id];
    if (!body.asleep) {
        body.quiet_time = 0;
        return;
    }
    const std::int32_t island = body.island;
    for (std::size_t i = 0; i < bodies.size(); i += 1) {
        Body& other = bodies[i];
        if (other.alive && other.asleep && other.island == island) {
            other.asleep = false;
            other.quiet_time = 0;
        }
    }
}

// Wakes every body that shares a retained manifold with `id`.
void World::Impl::wake_touching(std::int32_t id) {
    for (std::size_t k = 0; k < persisted.size(); k += 1) {
        const Persisted& entry = persisted[k];
        if (entry.a == id && !entry.fixed_b) {
            wake_island(entry.b);
        } else if (entry.b == id && !entry.fixed_b) {
            wake_island(entry.a);
        }
    }
}

void World::Impl::refresh_hulls(Body& body) {
    // A resting body's world hulls stay valid once rebuilt at its final pose.
    if ((body.asleep || body.is_static) && body.hulls_fresh) {
        return;
    }
    const Shape& shape = *body.shape;
    body.world_hulls.resize(shape.hulls.size());
    for (std::size_t h = 0; h < shape.hulls.size(); h += 1) {
        transform_hull(shape.hulls[h], body.position, body.orientation, body.world_hulls[h]);
    }
    body.hulls_fresh = body.asleep || body.is_static;
}

double World::Impl::motion_bound(const Body& body) const {
    return (length(body.velocity) + length(body.angular) * (*body.shape).radius) * params.frame_dt;
}

// Inherits last frame's impulses for the contacts from index `first` on, for
// points that stayed within 2 mm on body A with a normal within about 25 degrees.
void World::Impl::warm_start(std::uint64_t key, const Body& body_a, std::size_t first) {
    Persisted probe;
    probe.key = key;
    const std::vector<Persisted>::iterator begin =
        std::lower_bound(persisted.begin(), persisted.end(), probe, PersistedByKey());
    for (std::size_t k = first; k < contacts.size(); k += 1) {
        ContactPoint& contact = contacts[k];
        std::vector<Persisted>::iterator best = persisted.end();
        double best_distance = 0.002;
        for (std::vector<Persisted>::iterator it = begin; it != persisted.end() && (*it).key == key; ++it) {
            if ((*it).used) {
                continue;
            }
            const double distance = length(contact.local_a - (*it).local_a);
            if (distance < best_distance && dot(contact.normal, (*it).normal) > 0.9) {
                best_distance = distance;
                best = it;
            }
        }
        if (best != persisted.end()) {
            (*best).used = true;
            contact.warm_normal = (*best).normal_impulse;
            contact.warm_tangent = (*best).tangent_impulse;
            contact.pending_bounce = (*best).bounce;
            contact.matched = true;
        }
    }
    (void)body_a;
}

// Turns the manifold points in `points` into contacts between a (free) and b.
void World::Impl::add_points(std::int32_t a, std::int32_t b, std::int32_t hull_a, std::int32_t hull_b, bool fixed_b,
                             double friction, double restitution) {
    const Body& body_a = bodies[a];
    const std::size_t first = contacts.size();
    const Quat inverse = conjugate(body_a.orientation);
    for (std::size_t k = 0; k < points.size(); k += 1) {
        ContactPoint contact;
        contact.a = a;
        contact.b = b;
        contact.hull_a = hull_a;
        contact.hull_b = hull_b;
        contact.fixed_b = fixed_b;
        contact.point_a = points[k].point_a;
        contact.point_b = points[k].point_b;
        contact.normal = points[k].normal;
        contact.separation = points[k].separation;
        contact.friction = friction;
        contact.restitution = restitution;
        contact.local_a = rotate(inverse, points[k].point_a - body_a.position);
        contacts.push_back(contact);
    }
    warm_start(manifold_key(a, b, hull_a, hull_b), body_a, first);
}

// An upper bound on how far any point of a body has moved since `position`,
// `orientation`: the centre's displacement plus the turn times the radius. The
// turn is bounded by pi times the length of the relative quaternion's vector
// part (the angle is 2 asin of it, and asin x <= pi x / 2 on [0, 1]).
double moved_since(const Body& body, Vec3 position, Quat orientation) {
    const double shift = length(body.position - position);
    const Quat relative = multiply(body.orientation, conjugate(orientation));
    const double vector_part = std::sqrt(relative.x * relative.x + relative.y * relative.y + relative.z * relative.z);
    const double turn = kPi * vector_part;
    return shift + turn * (*body.shape).radius;
}

// Runs the narrow phase on one hull pair into `points`, or answers from the
// pair's cache when the bodies have not moved enough to change the answer.
void World::Impl::collide_pair(std::uint64_t key, const Body& first_body, const Body& second_body, const WorldHull& hull_a,
                               const WorldHull& hull_b, double margin) {
    PairCache& cache = pair_cache[key];
    const bool known = cache.stamp > 0;
    cache.stamp = frame;
    stats.pair_tests += 1;
    if (known) {
        const double moved = moved_since(first_body, cache.position_a, cache.orientation_a) +
                             moved_since(second_body, cache.position_b, cache.orientation_b);
        if (!cache.touching && moved < cache.separated_by - margin) {
            stats.pair_cache_hits += 1;
            return;   // still apart: no point of either body can have closed the gap
        }
        // the margin carries each body's motion, which jitters at 1e-14 m between quiet
        // frames; a band of kReuseMargin can't hold a contact worth finding
        if (cache.touching && moved < kReuseMotion && margin <= cache.margin + kReuseMargin) {
            for (std::size_t k = 0; k < cache.local_points.size(); k += 1) {
                const ManifoldPoint& local = cache.local_points[k];
                if (local.separation > margin) {
                    continue;
                }
                ManifoldPoint point;
                point.point_a = rotate(first_body.orientation, local.point_a) + first_body.position;
                point.point_b = rotate(second_body.orientation, local.point_b) + second_body.position;
                point.normal = rotate(first_body.orientation, local.normal);
                point.separation = local.separation;
                points.push_back(point);
            }
            stats.pair_cache_hits += 1;
            return;
        }
    }
    double separated_by = 0;
    collide_hulls(hull_a, hull_b, margin, scratch, points, separated_by);
    cache.position_a = first_body.position;
    cache.orientation_a = first_body.orientation;
    cache.position_b = second_body.position;
    cache.orientation_b = second_body.orientation;
    cache.margin = margin;
    cache.touching = !points.empty();
    cache.separated_by = separated_by;
    cache.local_points.clear();
    if (cache.touching) {
        const Quat inverse_a = conjugate(first_body.orientation);
        const Quat inverse_b = conjugate(second_body.orientation);
        for (std::size_t k = 0; k < points.size(); k += 1) {
            ManifoldPoint local;
            local.point_a = rotate(inverse_a, points[k].point_a - first_body.position);
            local.point_b = rotate(inverse_b, points[k].point_b - second_body.position);
            local.normal = rotate(inverse_a, points[k].normal);
            local.separation = points[k].separation;
            cache.local_points.push_back(local);
        }
    }
}

// Forgets pairs not seen for a second (they drifted apart or a body left).
void World::Impl::prune_pair_cache() {
    std::unordered_map<std::uint64_t, PairCache>::iterator it = pair_cache.begin();
    while (it != pair_cache.end()) {
        if ((*it).second.stamp < frame - 60) {
            it = pair_cache.erase(it);
        } else {
            ++it;
        }
    }
}

void World::Impl::collide() {
    for (int restart = 0; restart < 64; restart += 1) {
        bool woke = false;
        contacts.clear();
        for (std::size_t k = 0; k < persisted.size(); k += 1) {
            persisted[k].used = false;
        }
        for (std::size_t i = 0; i < bodies.size(); i += 1) {
            if (bodies[i].alive) {
                refresh_hulls(bodies[i]);
            }
        }
        const std::int32_t count = static_cast<std::int32_t>(bodies.size());
        for (std::int32_t i = 0; i < count && !woke; i += 1) {
            Body& body_a = bodies[i];
            if (!body_a.alive) {
                continue;
            }
            const double motion_a = motion_bound(body_a);
            if (!body_a.asleep && !body_a.is_static) {
                const double margin = std::min(params.speculative_distance + motion_a, 0.05);
                const double friction = std::sqrt((*body_a.shape).friction * params.ground_friction);
                for (std::size_t h = 0; h < body_a.world_hulls.size(); h += 1) {
                    points.clear();
                    collide_ground(body_a.world_hulls[h], params.ground_z, params.ground_surface.get(), margin, points);
                    add_points(i, kGround, static_cast<std::int32_t>(h), 0, true, friction, (*body_a.shape).restitution);
                }
            }
            for (std::int32_t j = i + 1; j < count && !woke; j += 1) {
                Body& body_b = bodies[j];
                if (!body_b.alive) {
                    continue;
                }
                // A pair matters only if one of its bodies is awake and free.
                const bool resting_a = body_a.asleep || body_a.is_static;
                const bool resting_b = body_b.asleep || body_b.is_static;
                if (resting_a && resting_b) {
                    continue;
                }
                const double margin = std::min(params.speculative_distance + motion_a + motion_bound(body_b), 0.05);
                const double reach = (*body_a.shape).radius + (*body_b.shape).radius + margin;
                if (length(body_a.position - body_b.position) > reach) {
                    continue;
                }
                // The contact's first body is always a free one.
                const std::int32_t first = body_a.is_static ? j : i;
                const std::int32_t second = body_a.is_static ? i : j;
                const Body& body_first = bodies[first];
                const Body& body_second = bodies[second];
                const double friction = std::sqrt((*body_a.shape).friction * (*body_b.shape).friction);
                const double restitution = std::max((*body_a.shape).restitution, (*body_b.shape).restitution);
                for (std::size_t ha = 0; ha < body_first.world_hulls.size() && !woke; ha += 1) {
                    for (std::size_t hb = 0; hb < body_second.world_hulls.size(); hb += 1) {
                        const WorldHull& hull_a = body_first.world_hulls[ha];
                        const WorldHull& hull_b = body_second.world_hulls[hb];
                        if (!bounds_overlap(hull_a, hull_b, margin)) {
                            continue;
                        }
                        points.clear();
                        const std::uint64_t pair_key = manifold_key(first, second, static_cast<std::int32_t>(ha), static_cast<std::int32_t>(hb));
                        collide_pair(pair_key, body_first, body_second, hull_a, hull_b, margin);
                        if (points.empty()) {
                            continue;
                        }
                        if (body_a.asleep || body_b.asleep) {
                            // An awake body touching a sleeping one wakes its island.
                            wake_island(body_a.asleep ? i : j);
                            woke = true;
                            break;
                        }
                        add_points(first, second, static_cast<std::int32_t>(ha), static_cast<std::int32_t>(hb),
                                   body_second.is_static, friction, restitution);
                    }
                }
            }
        }
        if (!woke) {
            return;
        }
    }
}

// Keeps this frame's impulses, and those of sleeping bodies, for the next frame.
void World::Impl::persist() {
    next_persisted.clear();
    for (std::size_t k = 0; k < persisted.size(); k += 1) {
        const Persisted& entry = persisted[k];
        if (bodies[entry.a].alive && bodies[entry.a].asleep) {
            next_persisted.push_back(entry);
        }
    }
    for (std::size_t k = 0; k < contacts.size(); k += 1) {
        const ContactPoint& contact = contacts[k];
        Persisted entry;
        entry.key = manifold_key(contact.a, contact.b, contact.hull_a, contact.hull_b);
        entry.a = contact.a;
        entry.b = contact.b;
        entry.fixed_b = contact.fixed_b;
        entry.local_a = contact.local_a;
        entry.normal = contact.normal;
        entry.normal_impulse = contact.normal_impulse;
        entry.tangent_impulse = contact.tangent_impulse;
        entry.bounce = contact.bounce;
        next_persisted.push_back(entry);
    }
    std::stable_sort(next_persisted.begin(), next_persisted.end(), PersistedByKey());
    persisted.swap(next_persisted);
}

// Safety nets: a non-finite state is rolled back; speeds are capped.
void World::Impl::guard() {
    for (std::size_t i = 0; i < bodies.size(); i += 1) {
        Body& body = bodies[i];
        if (!body.alive || body.asleep || body.is_static) {
            continue;
        }
        const bool good = finite(body.position) && finite(body.velocity) && finite(body.angular) &&
                          std::isfinite(body.orientation.w) && std::isfinite(body.orientation.x) &&
                          std::isfinite(body.orientation.y) && std::isfinite(body.orientation.z);
        if (!good) {
            body.position = body.previous_pose.p;
            body.orientation = body.previous_pose.q;
            body.velocity = Vec3();
            body.angular = Vec3();
            guards += 1;
            continue;
        }
        const double speed = length(body.velocity);
        if (speed > 20.0) {
            body.velocity = body.velocity * (20.0 / speed);
        }
        const double spin = length(body.angular);
        if (spin > 50.0) {
            body.angular = body.angular * (50.0 / spin);
        }
    }
}

void World::Impl::update_sleep() {
    const std::size_t count = bodies.size();
    parent.resize(count);
    island_quiet.assign(count, 1);
    island_id.assign(count, 0);
    for (std::size_t i = 0; i < count; i += 1) {
        parent[i] = static_cast<std::int32_t>(i);
    }
    for (std::size_t k = 0; k < contacts.size(); k += 1) {
        const ContactPoint& contact = contacts[k];
        // Every reported contact joins islands, loaded or not: waking is triggered by any
        // contact inside the margin, so two touching groups must sleep together or they
        // wake each other in turn forever (seen with toppled compound rocks lying side by side).
        if (!contact.fixed_b) {
            const int root_a = find_root(parent, contact.a);
            const int root_b = find_root(parent, contact.b);
            if (root_a != root_b) {
                parent[std::max(root_a, root_b)] = std::min(root_a, root_b);
            }
        }
    }
    double max_speed = 0;
    double energy = 0;
    int awake = 0;
    for (std::size_t i = 0; i < count; i += 1) {
        Body& body = bodies[i];
        if (!body.alive || body.asleep || body.is_static) {
            continue;
        }
        awake += 1;
        const double linear = length(body.velocity);
        const double angular = length(body.angular);
        max_speed = std::max(max_speed, linear + angular * (*body.shape).radius);
        energy += 0.5 * body.mass * linear * linear +
                  0.5 * dot(body.angular, mul(body.inertia_world, body.angular));
        if (linear < params.sleep_linear && angular < params.sleep_angular) {
            body.quiet_time += params.frame_dt;
        } else {
            body.quiet_time = 0;
        }
        // The held body never sleeps, and keeps its island awake.
        const bool is_held = hold.active && hold.id == static_cast<BodyId>(i);
        if (body.quiet_time < params.sleep_time || is_held) {
            island_quiet[find_root(parent, static_cast<int>(i))] = 0;
        }
    }
    rest.max_speed = max_speed;
    rest.awake = awake;
    rest.kinetic_energy = energy;
    if (max_speed < params.sleep_linear) {
        rest.quiet_seconds += params.frame_dt;
    } else {
        rest.quiet_seconds = 0;
    }
    rest.quiet = awake == 0 || rest.quiet_seconds >= params.sleep_time;
    if (!solver.sleeping) {
        return;
    }
    for (std::size_t i = 0; i < count; i += 1) {
        Body& body = bodies[i];
        if (!body.alive || body.asleep || body.is_static) {
            continue;
        }
        const int root = find_root(parent, static_cast<int>(i));
        if (island_quiet[root]) {
            if (island_id[root] == 0) {
                island_id[root] = next_island;
                next_island += 1;
            }
            body.asleep = true;
            body.hulls_fresh = false;
            body.island = island_id[root];
            body.velocity = Vec3();
            body.angular = Vec3();
        }
    }
}

World::World(const WorldParams& params) : impl_(new Impl()) {
    (*impl_).params = params;
}

World::~World() = default;

namespace {

BodyId add_body_record(std::vector<Body>& bodies, const Shape& shape, const Pose& pose, Vec3 v, Vec3 w, bool is_static) {
    Body body;
    body.shape = std::make_shared<const Shape>(shape);
    body.is_static = is_static;
    body.position = pose.p;
    body.orientation = pose.q;
    body.previous_pose = pose;
    body.velocity = v;
    body.angular = w;
    body.mass = shape.mass;
    for (int k = 0; k < 9; k += 1) {
        body.inertia_local.m[k] = shape.inertia[k];
    }
    if (!is_static) {
        body.inv_mass = 1.0 / shape.mass;
        body.inv_inertia_local = inverse(body.inertia_local);
    }
    const Mat3 rotation = to_matrix(pose.q);
    body.inertia_world = rotated(rotation, body.inertia_local);
    body.inv_inertia_world = rotated(rotation, body.inv_inertia_local);
    bodies.push_back(body);
    return static_cast<BodyId>(bodies.size() - 1);
}

}  // namespace

BodyId World::add_body(const Shape& shape, const Pose& pose, Vec3 v, Vec3 w, bool start_asleep) {
    Impl& impl = *impl_;
    const BodyId id = add_body_record(impl.bodies, shape, pose, v, w, false);
    if (start_asleep) {
        impl.bodies[id].asleep = true;
        impl.bodies[id].island = kLoadedIsland;
        impl.bodies[id].velocity = Vec3();
        impl.bodies[id].angular = Vec3();
    }
    return id;
}

BodyId World::add_static_body(const Shape& shape, const Pose& pose) {
    return add_body_record((*impl_).bodies, shape, pose, Vec3(), Vec3(), true);
}

void World::remove_body(BodyId id) {
    Impl& impl = *impl_;
    if (!has(id)) {
        return;
    }
    if (impl.hold.active && impl.hold.id == id) {
        release();
    }
    impl.wake_island(id);
    impl.wake_touching(id);
    impl.bodies[id].alive = false;
}

bool World::has(BodyId id) const {
    const Impl& impl = *impl_;
    return id >= 0 && static_cast<std::size_t>(id) < impl.bodies.size() && impl.bodies[id].alive;
}

std::vector<BodyId> World::bodies() const {
    std::vector<BodyId> ids;
    for (std::size_t i = 0; i < (*impl_).bodies.size(); i += 1) {
        if ((*impl_).bodies[i].alive) {
            ids.push_back(static_cast<BodyId>(i));
        }
    }
    return ids;
}

BodyState World::state(BodyId id) const {
    const Body& body = (*impl_).bodies[id];
    BodyState out;
    out.pose.p = body.position;
    out.pose.q = body.orientation;
    out.v = body.velocity;
    out.w = body.angular;
    out.asleep = body.asleep;
    return out;
}

void World::set_pose(BodyId id, const Pose& pose) {
    Impl& impl = *impl_;
    impl.wake_island(id);
    impl.wake_touching(id);
    Body& body = impl.bodies[id];
    body.position = pose.p;
    body.orientation = pose.q;
    body.velocity = Vec3();
    body.angular = Vec3();
    body.hulls_fresh = false;
}

void World::wake(BodyId id) {
    (*impl_).wake_island(id);
}

void World::hold(BodyId id, const Pose& target, const HoldParams& params) {
    Impl& impl = *impl_;
    impl.wake_island(id);
    impl.hold = HoldRecord();
    impl.hold.active = true;
    impl.hold.id = id;
    impl.hold.target = target;
    impl.hold.params = params;
}

void World::set_hold_target(const Pose& target) {
    (*impl_).hold.target = target;
}

void World::release() {
    Impl& impl = *impl_;
    if (impl.hold.active) {
        impl.wake_island(impl.hold.id);
    }
    impl.hold = HoldRecord();
}

std::optional<BodyId> World::held() const {
    if ((*impl_).hold.active) {
        return (*impl_).hold.id;
    }
    return std::nullopt;
}

HoldState World::hold_state() const {
    return (*impl_).hold.state;
}

void World::step() {
    Impl& impl = *impl_;
    for (std::size_t i = 0; i < impl.bodies.size(); i += 1) {
        Body& body = impl.bodies[i];
        body.previous_pose.p = body.position;
        body.previous_pose.q = body.orientation;
    }
    impl.frame += 1;
    impl.collide();
    if (impl.frame % 60 == 0) {
        impl.prune_pair_cache();
    }
    solve_frame(impl.params, impl.solver, impl.bodies, impl.contacts, impl.hold, impl.work, impl.stats);
    impl.persist();
    impl.guard();
    impl.update_sleep();
}

std::vector<Contact> World::retained_contacts() const {
    std::vector<Contact> out;
    const Impl& impl = *impl_;
    for (std::size_t k = 0; k < impl.persisted.size(); k += 1) {
        const Persisted& entry = impl.persisted[k];
        if (entry.normal_impulse <= 0 || !impl.bodies[entry.a].alive) {
            continue;
        }
        if (entry.b >= 0 && !impl.bodies[entry.b].alive) {
            continue;
        }
        const Body& body = impl.bodies[entry.a];
        Contact contact;
        contact.a = entry.a;
        contact.b = entry.b;
        contact.point = rotate(body.orientation, entry.local_a) + body.position;
        contact.normal = entry.normal;
        contact.separation = 0;
        contact.normal_impulse = entry.normal_impulse;
        out.push_back(contact);
    }
    return out;
}

std::vector<Contact> World::contacts() const {
    std::vector<Contact> out;
    const Impl& impl = *impl_;
    for (std::size_t k = 0; k < impl.contacts.size(); k += 1) {
        const ContactPoint& point = impl.contacts[k];
        Contact contact;
        contact.a = point.a;
        contact.b = point.b;
        contact.point = (point.point_a + point.point_b) * 0.5;
        contact.normal = point.normal;
        contact.separation = point.separation;
        contact.normal_impulse = point.normal_impulse;
        out.push_back(contact);
    }
    return out;
}

bool World::resting(BodyId id) const {
    const Body& body = (*impl_).bodies[id];
    return body.asleep || body.quiet_time >= 0.25;
}

std::optional<std::pair<BodyId, double>> World::raycast(Vec3 origin, Vec3 dir, double max_t) const {
    const Impl& impl = *impl_;
    double best = max_t;
    BodyId hit = -2;
    if (impl.params.ground_surface || dir.z < 0) {
        const double t = impl.params.ground_surface ? (*impl.params.ground_surface).raycast(origin, dir, best)
                                                  : (impl.params.ground_z - origin.z) / dir.z;
        if (t >= 0 && t < best) {
            best = t;
            hit = kGround;
        }
    }
    for (std::size_t i = 0; i < impl.bodies.size(); i += 1) {
        const Body& body = impl.bodies[i];
        if (!body.alive) {
            continue;
        }
        const Quat inverse = conjugate(body.orientation);
        const Vec3 local_origin = rotate(inverse, origin - body.position);
        const Vec3 local_dir = rotate(inverse, dir);
        const Shape& shape = *body.shape;
        for (std::size_t h = 0; h < shape.hulls.size(); h += 1) {
            const HullData& hull = shape.hulls[h];
            double enter = 0;
            double leave = best;
            bool miss = false;
            for (std::size_t f = 0; f < hull.face_normals.size() && !miss; f += 1) {
                const double distance = dot(hull.face_normals[f], local_origin) - hull.face_offsets[f];
                const double rate = dot(hull.face_normals[f], local_dir);
                if (rate == 0) {
                    miss = distance > 0;
                } else if (rate < 0) {
                    enter = std::max(enter, -distance / rate);
                } else {
                    leave = std::min(leave, -distance / rate);
                }
                if (enter > leave) {
                    miss = true;
                }
            }
            if (!miss && enter < best) {
                best = enter;
                hit = static_cast<BodyId>(i);
            }
        }
    }
    if (hit == -2) {
        return std::nullopt;
    }
    return std::pair<BodyId, double>(hit, best);
}

bool World::all_asleep() const {
    const Impl& impl = *impl_;
    for (std::size_t i = 0; i < impl.bodies.size(); i += 1) {
        const Body& body = impl.bodies[i];
        const bool is_held = impl.hold.active && impl.hold.id == static_cast<BodyId>(i);
        if (body.alive && !body.asleep && !body.is_static && !is_held) {
            return false;
        }
    }
    return true;
}

double World::kinetic_energy() const {
    const Impl& impl = *impl_;
    double energy = 0;
    for (std::size_t i = 0; i < impl.bodies.size(); i += 1) {
        const Body& body = impl.bodies[i];
        if (!body.alive || body.asleep || body.is_static) {
            continue;
        }
        energy += 0.5 * body.mass * dot(body.velocity, body.velocity) +
                  0.5 * dot(body.angular, mul(body.inertia_world, body.angular));
    }
    return energy;
}

std::uint64_t World::checksum() const {
    const Impl& impl = *impl_;
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < impl.bodies.size(); i += 1) {
        const Body& body = impl.bodies[i];
        if (!body.alive) {
            continue;
        }
        const double values[13] = {
            body.position.x, body.position.y, body.position.z,
            body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z,
            body.velocity.x, body.velocity.y, body.velocity.z, body.angular.x, body.angular.y, body.angular.z};
        for (int k = 0; k < 13; k += 1) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &values[k], sizeof(bits));
            hash = (hash ^ bits) * 1099511628211ull;
        }
    }
    return hash;
}

bool World::out_of_bounds(BodyId id) const {
    const Body& body = (*impl_).bodies[id];
    return body.position.z < -0.5 || body.position.x * body.position.x + body.position.y * body.position.y > 25.0;
}

const WorldParams& World::params() const {
    return (*impl_).params;
}

void World::set_solver_params(const SolverParams& params) {
    (*impl_).solver = params;
}

const SolverParams& World::solver_params() const {
    return (*impl_).solver;
}

RestReport World::rest_report() const {
    return (*impl_).rest;
}

// Every body that is asleep or has been quiet for a quarter second and is
// joined to the ground or a static body through loaded contacts between such
// bodies: one breadth-first walk over the retained manifolds.
std::vector<BodyId> World::stable_set() const {
    const Impl& impl = *impl_;
    const std::size_t count = impl.bodies.size();
    std::vector<char> settled(count, 0);
    for (std::size_t i = 0; i < count; i += 1) {
        const Body& body = impl.bodies[i];
        if (body.alive && !body.is_static && (body.asleep || body.quiet_time >= 0.25)) {
            settled[i] = 1;
        }
    }
    std::vector<char> reached(count, 0);
    std::vector<BodyId> queue;
    std::vector<std::pair<std::int32_t, std::int32_t>> links;
    for (std::size_t k = 0; k < impl.persisted.size(); k += 1) {
        const Persisted& entry = impl.persisted[k];
        if (entry.normal_impulse <= 0 || !settled[entry.a]) {
            continue;
        }
        if (entry.fixed_b) {
            if (!reached[entry.a]) {
                reached[entry.a] = 1;
                queue.push_back(entry.a);
            }
        } else if (settled[entry.b]) {
            links.push_back(std::pair<std::int32_t, std::int32_t>(entry.a, entry.b));
        }
    }
    for (std::size_t head = 0; head < queue.size(); head += 1) {
        const BodyId current = queue[head];
        for (std::size_t k = 0; k < links.size(); k += 1) {
            std::int32_t other = -1;
            if (links[k].first == current) {
                other = links[k].second;
            } else if (links[k].second == current) {
                other = links[k].first;
            }
            if (other >= 0 && !reached[other]) {
                reached[other] = 1;
                queue.push_back(other);
            }
        }
    }
    std::vector<BodyId> stable;
    for (std::size_t i = 0; i < count; i += 1) {
        if (reached[i]) {
            stable.push_back(static_cast<BodyId>(i));
        }
    }
    return stable;
}

SolveStats World::solve_stats() const {
    return (*impl_).stats;
}

int World::guard_count() const {
    return (*impl_).guards;
}

}  // namespace zc::phys
