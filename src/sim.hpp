// The teapot in motion: one rigid body on the floor.
//
// The body is simulated by zc::phys, the rigid-body engine of PlaySuite's
// rock-stacking game, whose wrench-transport solver resolves contact,
// friction and rolling resistance in one solve per frame (third_party/
// zc_phys). The collision shape is a set of convex hulls taken straight from
// the CONV lattice: every node lies on the surface, so the hull of a group
// of nodes is a tight convex piece of the teapot, with no separate collision
// model to author. After each step the body's pose is applied to the rest
// nodes and the CONV surface is recompiled through them.
//
// The engine works in metres with z up; the model is in teapot units with
// y up. One model unit is taken as 7 cm, which makes the body 28 cm across.
#pragma once
#include "teapot.hpp"
#include "../third_party/zc_phys/physics.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

namespace tconv {

class TeapotSim {
public:
    static constexpr double metres = 0.07; // one model unit

    void build(Teapot& teapot) {
        teapot_ = &teapot;
        rest_.clear();
        for (const auto& part : teapot.parts) {
            rest_.insert(rest_.end(), part->conv.lattice.nodes.begin(), part->conv.lattice.nodes.end());
        }
        zc::phys::ShapeDesc desc;
        desc.density = 2400.0;      // porcelain
        desc.friction = 0.5;
        desc.restitution = 0.25;
        desc.rolling_resistance = 0.001;
        std::vector<zc::phys::HullDesc> groups;
        for (const auto& part : teapot.parts) {
            const Lattice& lattice = part->conv.lattice;
            if (part->hull == -1) { continue; }
            if (part->hull >= 0) {
                // Sampled patches: the nodes of every part of a group form one hull.
                if (groups.size() <= std::size_t(part->hull)) { groups.resize(std::size_t(part->hull) + 1); }
                for (const Vec3& node : lattice.nodes) { groups[std::size_t(part->hull)].points.push_back(to_physics(node)); }
                continue;
            }
            // The body is one hull; the other parts are cut along their length
            // into short convex pieces.
            const std::size_t stride = part->name == "body" ? lattice.nv - 1 : part->name == "lid" ? part->per_v : std::max<std::size_t>(1, part->per_v / 2);
            for (std::size_t first = 0; first + 1 < lattice.nv; first += stride) {
                zc::phys::HullDesc hull;
                const std::size_t last = std::min(lattice.nv - 1, first + stride);
                for (std::size_t v = first; v <= last; ++v) {
                    for (std::size_t u = 0; u < lattice.nu; ++u) { hull.points.push_back(to_physics(lattice.at(u, v))); }
                }
                desc.hulls.push_back(std::move(hull));
            }
        }
        for (zc::phys::HullDesc& hull : groups) {
            if (!hull.points.empty()) { desc.hulls.push_back(std::move(hull)); }
        }
        // Coincident nodes (a pole is one point many times over) are given to
        // the hull builder once.
        for (zc::phys::HullDesc& hull : desc.hulls) {
            std::vector<zc::phys::Vec3> distinct;
            for (const zc::phys::Vec3& p : hull.points) {
                bool seen = false;
                for (const zc::phys::Vec3& q : distinct) {
                    if (std::fabs(p.x - q.x) + std::fabs(p.y - q.y) + std::fabs(p.z - q.z) < 1e-7) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) { distinct.push_back(p); }
            }
            // A hull needs only a few hundred points of a sheet; the hull
            // builder's time grows steeply with the count.
            const std::size_t limit = 320;
            if (distinct.size() > limit) {
                std::vector<zc::phys::Vec3> fewer;
                for (std::size_t k = 0; k < limit; ++k) { fewer.push_back(distinct[k * distinct.size() / limit]); }
                distinct.swap(fewer);
            }
            // Rings of a revolved sheet are exactly coplanar, which the hull
            // builder does not handle; move each collision point by about a
            // micron, deterministically. The rendered surface is untouched.
            std::uint32_t state = 12345u;
            for (zc::phys::Vec3& p : distinct) {
                const auto next = [&] {
                    state = state * 1664525u + 1013904223u;
                    return (double(state >> 8) / 16777216.0 - 0.5) * 2e-6;
                };
                p.x += next();
                p.y += next();
                p.z += next();
            }
            hull.points = std::move(distinct);
        }
        // A group of nodes that does not span a volume cannot be a hull: keep
        // only those the engine accepts on their own.
        std::vector<zc::phys::HullDesc> accepted;
        for (zc::phys::HullDesc& hull : desc.hulls) {
            zc::phys::ShapeDesc single = desc;
            single.hulls.assign(1, hull);
            try {
                (void)zc::phys::cook(single);
                accepted.push_back(std::move(hull));
            } catch (const std::exception&) {
            }
        }
        desc.hulls = std::move(accepted);
        shape_ = zc::phys::cook(desc);
        hulls_ = desc.hulls.size();
        {
            // The inverse of the body-frame inertia, for the grab.
            const std::array<double, 9>& a = shape_.inertia;
            const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
            inverse_inertia_ = {(a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det, (a[1] * a[5] - a[2] * a[4]) / det,
                                (a[5] * a[6] - a[3] * a[8]) / det, (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
                                (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det, (a[0] * a[4] - a[1] * a[3]) / det};
        }
        zc::phys::WorldParams params;
        params.frame_dt = 1.0 / 120.0;
        world_ = std::make_unique<zc::phys::World>(params);
        zc::phys::Pose pose;
        pose.p = shape_.com_offset;
        body_ = world_->add_body(shape_, pose, {}, {}, true);
        moved_ = false;
        accumulator_ = 0.0;
        holding_ = false;
    }

    std::size_t hulls() const { return hulls_; }
    double mass() const { return shape_.mass; }
    bool moved() const { return moved_; }
    bool asleep() const { return world_->all_asleep(); }
    double kinetic() const { return world_->kinetic_energy(); }
    std::size_t contacts() const { return world_->contacts().size(); }
    const zc::phys::SolveStats stats() const { return world_->solve_stats(); }

    // Back to the rest pose, optionally lifted (model units) and tilted.
    void reset(double lift = 0.0, double tilt = 0.0) {
        if (holding_) { release(); }
        zc::phys::Pose pose;
        pose.q = zc::phys::from_axis_angle({0.0, 1.0, 0.0}, tilt);
        pose.p = shape_.com_offset + zc::phys::Vec3{0.0, 0.0, lift * metres};
        world_->set_pose(body_, pose);
        moved_ = lift != 0.0 || tilt != 0.0;
        write_back();
    }

    // Take hold of the teapot at a point of its surface. The grab is a soft
    // pin: the picked point of the body is pulled to the pointer by a
    // critically damped spring acting at that point, so the body swings and
    // turns about where it is held instead of being carried by its centre.
    // drag() moves the pointer end.
    void grab(Vec3 point) {
        const zc::phys::BodyState state = world_->state(body_);
        grab_local_ = zc::phys::rotate(zc::phys::conjugate(state.pose.q), to_physics(point) - state.pose.p);
        grab_target_ = to_physics(point);
        holding_ = true;
        world_->wake(body_);
    }
    void drag(Vec3 target) {
        if (!holding_) { return; }
        grab_target_ = to_physics(target);
        world_->wake(body_);
    }
    void release() { holding_ = false; }
    bool grabbing() const { return holding_; }

    // A push: a change of velocity at a surface point along a direction.
    void poke(Vec3 point, Vec3 direction, double speed = 1.2) {
        if (holding_) { release(); }
        const zc::phys::BodyState state = world_->state(body_);
        const zc::phys::Vec3 push = to_physics_direction(direction) * speed;
        const zc::phys::Vec3 arm = to_physics(point) - state.pose.p;
        const zc::phys::Vec3 spin = zc::phys::cross(arm, push) * (1.0 / (shape_.radius * shape_.radius));
        world_->remove_body(body_);
        body_ = world_->add_body(shape_, state.pose, state.v + push, state.w + spin);
        moved_ = true;
    }

    // Advance by dt seconds in fixed engine frames. Returns the steps taken.
    int step(double dt) {
        accumulator_ += dt;
        const double frame = world_->params().frame_dt;
        int steps = 0;
        while (accumulator_ >= frame && steps < 8) {
            if (holding_) { pull(frame); }
            world_->step();
            accumulator_ -= frame;
            ++steps;
        }
        if (steps == 8) { accumulator_ = 0.0; }
        if (steps > 0) {
            moved_ = true;
            write_back();
        }
        return steps;
    }

private:
    static zc::phys::Vec3 to_physics(Vec3 p) { return {p.x * metres, -p.z * metres, p.y * metres}; }
    static zc::phys::Vec3 to_physics_direction(Vec3 d) { return {d.x, -d.z, d.y}; }
    static Vec3 to_model(zc::phys::Vec3 p) { return {p.x / metres, p.z / metres, -p.y / metres}; }

    // One frame of the grab: the impulse at the held point that a spring and
    // damper between that point and the pointer would give over the frame,
    // found implicitly (a soft constraint), so it is stable at any stiffness.
    void pull(double h) {
        using zc::phys::Vec3;
        const zc::phys::BodyState state = world_->state(body_);
        const Vec3 r = zc::phys::rotate(state.pose.q, grab_local_);
        const Vec3 error = state.pose.p + r - grab_target_;
        const Vec3 speed = state.v + zc::phys::cross(state.w, r);
        const double omega = 2.0 * 3.14159265358979 * 8.0; // 8 Hz, critically damped: the weight sags it by 4 mm
        const double stiffness = shape_.mass * omega * omega, damping = 2.0 * shape_.mass * omega;
        const double gamma = 1.0 / (h * (damping + h * stiffness)), beta = h * stiffness * gamma;
        // World-frame inverse inertia applied to a vector.
        const auto spin = [&](Vec3 torque) {
            const Vec3 local = zc::phys::rotate(zc::phys::conjugate(state.pose.q), torque);
            const Vec3 turned{inverse_inertia_[0] * local.x + inverse_inertia_[1] * local.y + inverse_inertia_[2] * local.z,
                              inverse_inertia_[3] * local.x + inverse_inertia_[4] * local.y + inverse_inertia_[5] * local.z,
                              inverse_inertia_[6] * local.x + inverse_inertia_[7] * local.y + inverse_inertia_[8] * local.z};
            return zc::phys::rotate(state.pose.q, turned);
        };
        // How the point's velocity answers an impulse there, plus the softness.
        double k[3][3];
        for (int column = 0; column < 3; ++column) {
            const Vec3 unit{column == 0 ? 1.0 : 0.0, column == 1 ? 1.0 : 0.0, column == 2 ? 1.0 : 0.0};
            const Vec3 answer = unit * (1.0 / shape_.mass) + zc::phys::cross(spin(zc::phys::cross(r, unit)), r);
            k[0][column] = answer.x;
            k[1][column] = answer.y;
            k[2][column] = answer.z;
            k[column][column] += gamma;
        }
        const Vec3 wanted = (speed + error * beta) * -1.0;
        const double det = k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1]) - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0]) +
                           k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]);
        if (!(std::fabs(det) > 1e-300)) { return; }
        const auto solve_column = [&](int column) {
            double m[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) { m[i][j] = j == column ? (i == 0 ? wanted.x : i == 1 ? wanted.y : wanted.z) : k[i][j]; }
            }
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det;
        };
        Vec3 impulse{solve_column(0), solve_column(1), solve_column(2)};
        // A hand pulls with at most a few times the teapot's weight.
        const double limit = 6.0 * shape_.mass * 9.81 * h;
        const double size = std::sqrt(impulse.x * impulse.x + impulse.y * impulse.y + impulse.z * impulse.z);
        if (size > limit) { impulse = impulse * (limit / size); }
        world_->add_velocity(body_, impulse * (1.0 / shape_.mass), spin(zc::phys::cross(r, impulse)));
    }

    void write_back() {
        const zc::phys::Pose pose = world_->state(body_).pose;
        std::size_t index = 0;
        for (auto& part : teapot_->parts) {
            Lattice& lattice = part->conv.lattice;
            for (Vec3& node : lattice.nodes) {
                const zc::phys::Vec3 local = to_physics(rest_[index++]) - shape_.com_offset;
                node = to_model(pose.p + zc::phys::rotate(pose.q, local));
            }
            part->conv.compile();
        }
    }

    Teapot* teapot_ = nullptr;
    std::vector<Vec3> rest_;
    zc::phys::Shape shape_;
    std::unique_ptr<zc::phys::World> world_;
    zc::phys::BodyId body_ = 0;
    std::size_t hulls_ = 0;
    zc::phys::Vec3 grab_local_{}, grab_target_{}; // the held point in the body frame; where the pointer is
    std::array<double, 9> inverse_inertia_{};
    bool holding_ = false, moved_ = false;
    double accumulator_ = 0.0;
};

} // namespace tconv
