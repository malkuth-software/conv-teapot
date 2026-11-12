// Wrench-transport contact solve.
//
// Each contact point keeps the impulse it carried last (lambda). A frame runs
// a few passes; one pass minimises, over the twists V of the awake bodies,
//
//   l(V) = sum_b 1/2 (V_b - V*_b)^T M_b (V_b - V*_b)
//        + sum_c R_c/2 | P_K( lambda_c - (J_c V - vhat_c) / R_c ) |^2,
//
// with V* the free twist, P_K the projection onto the row's admissible set
// (friction cone, ball or box) and R_c a compliance set against the mass the
// contact carries. The projected vector is the pass's impulse and becomes
// lambda. Newton's method with an exact line search minimises l; each Newton
// system is solved exactly by BlockFactor. A group at rest is a fixed point:
// its first pass starts with a zero gradient and nothing moves.
#include <algorithm>
#include <cmath>

#include "internal.hpp"

namespace zc::phys {

namespace {

constexpr double kTwoPi = 6.283185307179586;

void tangent_basis(Vec3 normal, Vec3& t1, Vec3& t2) {
    Vec3 first;
    if (std::abs(normal.x) < 0.57735) {
        first = cross(Vec3{1, 0, 0}, normal);
    } else {
        first = cross(Vec3{0, 1, 0}, normal);
    }
    t1 = first * (1.0 / length(first));
    t2 = cross(normal, t1);
}

void set_identity_g(double* g, double value) {
    g[0] = value; g[1] = 0; g[2] = 0;
    g[3] = 0; g[4] = value; g[5] = 0;
    g[6] = 0; g[7] = 0; g[8] = value;
}

void clear_g(double* g) {
    for (int k = 0; k < 9; k += 1) {
        g[k] = 0;
    }
}

// Friction cone |g_t| <= mu g_n: impulse, mode and Hessian of one contact for
// contact-frame velocity (c0, c1, c2) = (tangent 1, tangent 2, normal).
void evaluate_contact(Row& row, double c0, double c1, double c2) {
    const double inverse_r = 1.0 / row.r;
    const double y0 = row.lambda0 - (c0 - row.vhat0) * inverse_r;
    const double y1 = row.lambda1 - (c1 - row.vhat1) * inverse_r;
    const double yn = row.lambda_n - (c2 - row.vhat_n) * inverse_r;
    const double yr = std::sqrt(y0 * y0 + y1 * y1);
    const double mu = row.friction;
    if (yr <= mu * yn) {
        row.mode = kModeStick;
        row.gamma0 = y0; row.gamma1 = y1; row.gamma_n = yn;
        set_identity_g(row.g, inverse_r);
        return;
    }
    if (yn <= -mu * yr) {
        row.mode = kModeOpen;
        row.gamma0 = 0; row.gamma1 = 0; row.gamma_n = 0;
        clear_g(row.g);
        return;
    }
    // Projection onto the cone's surface.
    row.mode = kModeSlide;
    const double factor = 1.0 / (1.0 + mu * mu);
    const double gamma_n = (yn + mu * yr) * factor;
    const double t0 = y0 / yr;
    const double t1 = y1 / yr;
    row.gamma_n = gamma_n;
    row.gamma0 = mu * gamma_n * t0;
    row.gamma1 = mu * gamma_n * t1;
    const double a = factor * inverse_r;
    const double p = mu * gamma_n * inverse_r / yr;
    double* g = row.g;
    g[0] = a * mu * mu * t0 * t0 + p * (1 - t0 * t0);
    g[1] = a * mu * mu * t0 * t1 - p * t0 * t1;
    g[2] = a * mu * t0;
    g[3] = g[1];
    g[4] = a * mu * mu * t1 * t1 + p * (1 - t1 * t1);
    g[5] = a * mu * t1;
    g[6] = g[2];
    g[7] = g[5];
    g[8] = a;
}

// A 3-vector bounded in length by row.limit: rolling resistance and the
// crane's angular spring.
void evaluate_ball(Row& row, double c0, double c1, double c2) {
    const double inverse_r = 1.0 / row.r;
    const double y0 = row.lambda0 - (c0 - row.vhat0) * inverse_r;
    const double y1 = row.lambda1 - (c1 - row.vhat1) * inverse_r;
    const double y2 = row.lambda_n - (c2 - row.vhat_n) * inverse_r;
    const double size = std::sqrt(y0 * y0 + y1 * y1 + y2 * y2);
    if (size <= row.limit) {
        row.mode = row.limit > 0 ? kModeStick : kModeOpen;
        row.gamma0 = y0; row.gamma1 = y1; row.gamma_n = y2;
        set_identity_g(row.g, inverse_r);
        return;
    }
    const double shrink = row.limit / size;
    row.mode = row.limit > 0 ? kModeSlide : kModeOpen;
    row.gamma0 = y0 * shrink; row.gamma1 = y1 * shrink; row.gamma_n = y2 * shrink;
    const double p = shrink * inverse_r;
    const double u0 = y0 / size;
    const double u1 = y1 / size;
    const double u2 = y2 / size;
    double* g = row.g;
    g[0] = p * (1 - u0 * u0); g[1] = -p * u0 * u1; g[2] = -p * u0 * u2;
    g[3] = g[1]; g[4] = p * (1 - u1 * u1); g[5] = -p * u1 * u2;
    g[6] = g[2]; g[7] = g[5]; g[8] = p * (1 - u2 * u2);
}

// The crane's linear spring: the two horizontal components are bounded
// together by row.limit; the vertical one (the wires) lies in [0, limit_lift].
void evaluate_hold(Row& row, double c0, double c1, double c2) {
    const double inverse_r = 1.0 / row.r;
    const double y0 = row.lambda0 - (c0 - row.vhat0) * inverse_r;
    const double y1 = row.lambda1 - (c1 - row.vhat1) * inverse_r;
    const double y2 = row.lambda_n - (c2 - row.vhat_n) * inverse_r;
    clear_g(row.g);
    row.mode = kModeStick;
    const double size = std::sqrt(y0 * y0 + y1 * y1);
    if (size <= row.limit) {
        row.gamma0 = y0; row.gamma1 = y1;
        row.g[0] = inverse_r;
        row.g[4] = inverse_r;
    } else {
        const double shrink = row.limit / size;
        row.mode = kModeSlide;
        row.gamma0 = y0 * shrink; row.gamma1 = y1 * shrink;
        const double p = shrink * inverse_r;
        const double u0 = y0 / size;
        const double u1 = y1 / size;
        row.g[0] = p * (1 - u0 * u0); row.g[1] = -p * u0 * u1;
        row.g[3] = row.g[1]; row.g[4] = p * (1 - u1 * u1);
    }
    if (y2 <= 0) {
        row.gamma_n = 0;
        row.mode = kModeSlide;
    } else if (y2 >= row.limit_lift) {
        row.gamma_n = row.limit_lift;
        row.mode = kModeSlide;
    } else {
        row.gamma_n = y2;
        row.g[8] = inverse_r;
    }
}

void evaluate_row(Row& row, double c0, double c1, double c2) {
    if (row.kind == kKindContact) {
        evaluate_contact(row, c0, c1, c2);
    } else if (row.kind == kKindBall) {
        evaluate_ball(row, c0, c1, c2);
    } else {
        evaluate_hold(row, c0, c1, c2);
    }
}

// Contact-frame velocity of every row for twist vector `twist`.
void row_velocities(const std::vector<Row>& rows, const std::vector<double>& twist, std::vector<double>& out) {
    const std::size_t count = rows.size();
    for (std::size_t k = 0; k < count; k += 1) {
        const Row& row = rows[k];
        const double* va = twist.data() + row.index_a * 6;
        double c0 = 0, c1 = 0, c2 = 0;
        for (int j = 0; j < 6; j += 1) {
            c0 += row.ja[j] * va[j];
            c1 += row.ja[6 + j] * va[j];
            c2 += row.ja[12 + j] * va[j];
        }
        if (row.index_b >= 0) {
            const double* vb = twist.data() + row.index_b * 6;
            for (int j = 0; j < 6; j += 1) {
                c0 += row.jb[j] * vb[j];
                c1 += row.jb[6 + j] * vb[j];
                c2 += row.jb[12 + j] * vb[j];
            }
        }
        out[k * 3] = c0; out[k * 3 + 1] = c1; out[k * 3 + 2] = c2;
    }
}

// y = M x for the block-diagonal mass matrix.
void apply_mass(const std::vector<Body*>& bodies, const std::vector<double>& x, std::vector<double>& y) {
    for (std::size_t i = 0; i < bodies.size(); i += 1) {
        const Body& body = *bodies[i];
        const double* in = x.data() + i * 6;
        double* out = y.data() + i * 6;
        const double* inertia = body.inertia_world.m;
        out[0] = body.mass * in[0]; out[1] = body.mass * in[1]; out[2] = body.mass * in[2];
        out[3] = inertia[0] * in[3] + inertia[1] * in[4] + inertia[2] * in[5];
        out[4] = inertia[3] * in[3] + inertia[4] * in[4] + inertia[5] * in[5];
        out[5] = inertia[6] * in[3] + inertia[7] * in[4] + inertia[8] * in[5];
    }
}

// Energy norm sqrt(x^T M^-1 x) of a generalised momentum vector.
double inverse_mass_norm(const std::vector<Body*>& bodies, const std::vector<double>& x) {
    double sum = 0;
    for (std::size_t i = 0; i < bodies.size(); i += 1) {
        const Body& body = *bodies[i];
        const double* in = x.data() + i * 6;
        const double* inv = body.inv_inertia_world.m;
        sum += body.inv_mass * (in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
        sum += in[3] * (inv[0] * in[3] + inv[1] * in[4] + inv[2] * in[5]) +
               in[4] * (inv[3] * in[3] + inv[4] * in[4] + inv[5] * in[5]) +
               in[5] * (inv[6] * in[3] + inv[7] * in[4] + inv[8] * in[5]);
    }
    return std::sqrt(sum);
}

// Adds J_x^T G J_y into a 6x6 block (or into its transpose).
void add_relation(double* block, const double* jx, const double* jy, const double* g, bool transposed) {
    for (int i = 0; i < 6; i += 1) {
        const double g0 = jx[i] * g[0] + jx[6 + i] * g[3] + jx[12 + i] * g[6];
        const double g1 = jx[i] * g[1] + jx[6 + i] * g[4] + jx[12 + i] * g[7];
        const double g2 = jx[i] * g[2] + jx[6 + i] * g[5] + jx[12 + i] * g[8];
        for (int j = 0; j < 6; j += 1) {
            const double value = g0 * jy[j] + g1 * jy[6 + j] + g2 * jy[12 + j];
            if (transposed) {
                block[j * 6 + i] += value;
            } else {
                block[i * 6 + j] += value;
            }
        }
    }
}

void fill_jacobian(double* j, Vec3 r, Vec3 t1, Vec3 t2, Vec3 normal, double sign, const Body& body, double& trace) {
    const Vec3 directions[3] = {t1, t2, normal};
    for (int d = 0; d < 3; d += 1) {
        const Vec3 arm = cross(r, directions[d]);
        j[d * 6] = sign * directions[d].x; j[d * 6 + 1] = sign * directions[d].y; j[d * 6 + 2] = sign * directions[d].z;
        j[d * 6 + 3] = sign * arm.x; j[d * 6 + 4] = sign * arm.y; j[d * 6 + 5] = sign * arm.z;
        trace += body.inv_mass + dot(arm, mul(body.inv_inertia_world, arm));
    }
}

void clear_row(Row& row) {
    row = Row();
    for (int k = 0; k < 18; k += 1) {
        row.ja[k] = 0;
        row.jb[k] = 0;
    }
    clear_g(row.g);
}

struct LineState {
    double slope = 0;
    double curvature = 0;
};

// First and second derivative of the cost along the step at parameter alpha.
void line_derivatives(std::vector<Row>& rows, const std::vector<double>& velocity, const std::vector<double>& step,
                      double slope_linear, double curvature_mass, double alpha, LineState& out) {
    double slope = slope_linear + alpha * curvature_mass;
    double curvature = curvature_mass;
    const std::size_t count = rows.size();
    for (std::size_t k = 0; k < count; k += 1) {
        Row& row = rows[k];
        const double d0 = step[k * 3];
        const double d1 = step[k * 3 + 1];
        const double d2 = step[k * 3 + 2];
        evaluate_row(row, velocity[k * 3] + alpha * d0, velocity[k * 3 + 1] + alpha * d1, velocity[k * 3 + 2] + alpha * d2);
        slope -= d0 * row.gamma0 + d1 * row.gamma1 + d2 * row.gamma_n;
        const double* g = row.g;
        curvature += d0 * (g[0] * d0 + g[1] * d1 + g[2] * d2) + d1 * (g[3] * d0 + g[4] * d1 + g[5] * d2) +
                     d2 * (g[6] * d0 + g[7] * d1 + g[8] * d2);
    }
    out.slope = slope;
    out.curvature = curvature;
}

// The step length that minimises the convex cost along the Newton direction:
// a bracketed Newton iteration on its derivative, bisecting only when the
// one-dimensional Newton step leaves the bracket.
double exact_line_search(std::vector<Row>& rows, const std::vector<double>& velocity, const std::vector<double>& step,
                         double slope_linear, double curvature_mass, double slope0, SolveStats& stats) {
    LineState state;
    const double tolerance = 1.0e-10 * std::abs(slope0);
    line_derivatives(rows, velocity, step, slope_linear, curvature_mass, 1.0, state);
    stats.line_search_evaluations += 1;
    if (std::abs(state.slope) <= tolerance) {
        return 1.0;
    }
    double low = 0;
    double high = 1;
    if (state.slope < 0) {
        low = 1;
        high = 2;
        for (int grow = 0; grow < 20; grow += 1) {
            line_derivatives(rows, velocity, step, slope_linear, curvature_mass, high, state);
            stats.line_search_evaluations += 1;
            if (state.slope >= 0) {
                break;
            }
            low = high;
            high *= 2;
        }
    }
    double alpha = 0.5 * (low + high);
    for (int refine = 0; refine < 60; refine += 1) {
        line_derivatives(rows, velocity, step, slope_linear, curvature_mass, alpha, state);
        stats.line_search_evaluations += 1;
        if (std::abs(state.slope) <= tolerance) {
            break;
        }
        if (state.slope > 0) {
            high = alpha;
        } else {
            low = alpha;
        }
        double next = alpha - state.slope / state.curvature;
        if (!(next > low && next < high)) {
            next = 0.5 * (low + high);
        }
        if (std::abs(next - alpha) <= 1.0e-15 * std::max(1.0, alpha)) {
            alpha = next;
            break;
        }
        alpha = next;
    }
    return alpha;
}

// Evaluates every row at the current twist and fills the gradient
// M (V - V*) - J^T gamma.
void evaluate_state(SolverWork& work) {
    const std::size_t size = work.bodies.size() * 6;
    row_velocities(work.rows, work.twist, work.contact_velocity);
    for (std::size_t i = 0; i < size; i += 1) {
        work.difference[i] = work.twist[i] - work.free_twist[i];
        work.impulse[i] = 0;
    }
    apply_mass(work.bodies, work.difference, work.gradient);
    const std::size_t count = work.rows.size();
    for (std::size_t k = 0; k < count; k += 1) {
        Row& row = work.rows[k];
        evaluate_row(row, work.contact_velocity[k * 3], work.contact_velocity[k * 3 + 1], work.contact_velocity[k * 3 + 2]);
        double* ia = work.impulse.data() + row.index_a * 6;
        for (int j = 0; j < 6; j += 1) {
            ia[j] += row.ja[j] * row.gamma0 + row.ja[6 + j] * row.gamma1 + row.ja[12 + j] * row.gamma_n;
        }
        if (row.index_b >= 0) {
            double* ib = work.impulse.data() + row.index_b * 6;
            for (int j = 0; j < 6; j += 1) {
                ib[j] += row.jb[j] * row.gamma0 + row.jb[6 + j] * row.gamma1 + row.jb[12 + j] * row.gamma_n;
            }
        }
    }
    for (std::size_t i = 0; i < size; i += 1) {
        work.gradient[i] -= work.impulse[i];
    }
}

// Newton's method with an exact line search from work.twist. Stops at
// `loose`, but a start that misses `tight` always takes at least one step.
bool newton_solve(SolverWork& work, double loose, double tight, int max_iterations, SolveStats& stats,
                  int& iterations, double& residual, double& scale) {
    const std::size_t count = work.bodies.size();
    const std::size_t size = count * 6;
    double previous_residual = 1.0e300;
    int stalls = 0;
    LineState line;
    for (int iteration = 0; iteration <= max_iterations; iteration += 1) {
        evaluate_state(work);
        apply_mass(work.bodies, work.twist, work.momentum);
        residual = inverse_mass_norm(work.bodies, work.gradient);
        scale = std::max(inverse_mass_norm(work.bodies, work.momentum), inverse_mass_norm(work.bodies, work.impulse));
        if (residual <= 1.0e-14 + tight * scale) {
            return true;
        }
        if (iteration > 0 && residual <= loose * scale) {
            return true;
        }
        if (iteration == max_iterations) {
            return false;
        }
        // Round-off floor: accept once the residual stops falling near zero.
        if (residual > 0.5 * previous_residual && residual <= 1.0e-6 * scale) {
            stalls += 1;
            if (stalls >= 3) {
                return true;
            }
        } else {
            stalls = 0;
        }
        previous_residual = residual;

        // Hessian: M plus one relation per touching pair. While no row
        // slides, it depends only on which rows press, so the factor is kept.
        std::uint32_t signature = 2166136261u;
        bool sliding = false;
        for (std::size_t k = 0; k < work.rows.size(); k += 1) {
            const int mode = work.rows[k].mode;
            if (mode == kModeSlide) {
                sliding = true;
                break;
            }
            signature = (signature ^ static_cast<std::uint32_t>(mode + 1)) * 16777619u;
        }
        if (sliding || !work.factor_valid || signature != work.factor_signature) {
            work.factor.clear_blocks();
            for (std::size_t i = 0; i < count; i += 1) {
                const Body& body = *work.bodies[i];
                double* block = work.factor.diagonal_block(static_cast<int>(i));
                block[0] = body.mass; block[7] = body.mass; block[14] = body.mass;
                for (int r = 0; r < 3; r += 1) {
                    for (int c = 0; c < 3; c += 1) {
                        block[(3 + r) * 6 + 3 + c] = body.inertia_world.m[r * 3 + c];
                    }
                }
            }
            for (std::size_t k = 0; k < work.rows.size(); k += 1) {
                const Row& row = work.rows[k];
                if (row.mode == kModeOpen) {
                    continue;
                }
                add_relation(work.factor.diagonal_block(row.index_a), row.ja, row.ja, row.g, false);
                if (row.index_b >= 0) {
                    add_relation(work.factor.diagonal_block(row.index_b), row.jb, row.jb, row.g, false);
                    bool transposed = false;
                    double* coupling = work.factor.coupling_block(row.index_a, row.index_b, transposed);
                    add_relation(coupling, row.ja, row.jb, row.g, transposed);
                }
            }
            if (!work.factor.factorize()) {
                return false;
            }
            stats.factorizations += 1;
            work.factor_valid = !sliding;
            work.factor_signature = signature;
        } else {
            stats.factor_reuses += 1;
        }
        for (std::size_t i = 0; i < size; i += 1) {
            work.gradient[i] = -work.gradient[i];
        }
        work.factor.solve(work.gradient, work.direction);
        iterations += 1;

        row_velocities(work.rows, work.direction, work.contact_step);
        apply_mass(work.bodies, work.direction, work.mass_direction);
        double slope_linear = 0;
        double curvature_mass = 0;
        for (std::size_t i = 0; i < size; i += 1) {
            slope_linear += work.mass_direction[i] * work.difference[i];
            curvature_mass += work.mass_direction[i] * work.direction[i];
        }
        line_derivatives(work.rows, work.contact_velocity, work.contact_step, slope_linear, curvature_mass, 0.0, line);
        stats.line_search_evaluations += 1;
        if (!(line.slope < 0)) {
            return residual <= 1.0e-6 * scale;
        }
        const double alpha = exact_line_search(work.rows, work.contact_velocity, work.contact_step, slope_linear,
                                               curvature_mass, line.slope, stats);
        for (std::size_t i = 0; i < size; i += 1) {
            work.twist[i] += alpha * work.direction[i];
        }
    }
    return false;
}

struct HeightOrder {
    const std::vector<Body*>* bodies;
    Vec3 up;
    bool operator()(std::int32_t i, std::int32_t j) const {
        const double hi = dot((*(*bodies)[i]).position, up);
        const double hj = dot((*(*bodies)[j]).position, up);
        if (hi != hj) {
            return hi > hj;
        }
        return i < j;
    }
};

struct SupportByRank {
    bool operator()(const Support& a, const Support& b) const {
        if (a.rank != b.rank) {
            return a.rank < b.rank;
        }
        return a.first < b.first;
    }
};

// Sets each contact row's compliance against the mass it carries: the body
// resting on a pair brings its own mass and the mass resting on it. Bodies
// are visited from the highest centre of mass down.
void assign_carried_mass(SolverWork& work, const std::vector<ContactPoint>& contacts, std::size_t contact_rows,
                         Vec3 gravity, double regularization) {
    const double gravity_length = length(gravity);
    if (gravity_length == 0) {
        return;
    }
    const Vec3 up = gravity * (-1.0 / gravity_length);
    const std::size_t count = work.bodies.size();
    work.order.resize(count);
    work.load.resize(count * 2);
    for (std::size_t i = 0; i < count; i += 1) {
        work.order[i] = static_cast<std::int32_t>(i);
        work.load[i] = (*work.bodies[i]).mass;
        work.load[count + i] = 0;                 // total support weight of body i
    }
    HeightOrder by_height;
    by_height.bodies = &work.bodies;
    by_height.up = up;
    std::sort(work.order.begin(), work.order.end(), by_height);
    std::vector<std::int32_t>& rank = work.rank;
    std::vector<Support>& supports = work.supports;
    rank.resize(count);
    for (std::size_t n = 0; n < count; n += 1) {
        rank[work.order[n]] = static_cast<std::int32_t>(n);
    }
    supports.clear();
    // Contact rows of one body pair are contiguous.
    std::size_t k = 0;
    while (k < contact_rows) {
        std::size_t end = k;
        double lift = 0;
        while (end < contact_rows && work.rows[end].index_a == work.rows[k].index_a &&
               work.rows[end].index_b == work.rows[k].index_b &&
               contacts[work.rows[end].contact].b == contacts[work.rows[k].contact].b) {
            lift += dot(contacts[work.rows[end].contact].normal, up);   // normal points from B to A
            end += 1;
        }
        const double mean = lift / static_cast<double>(end - k);
        Support support;
        support.first = static_cast<std::int32_t>(k);
        support.count = static_cast<std::int32_t>(end - k);
        support.weight = std::abs(lift);
        bool supported = true;
        if (mean > 0.1) {
            support.upper = work.rows[k].index_a;
            support.lower = work.rows[k].index_b;
        } else if (mean < -0.1 && work.rows[k].index_b >= 0) {
            support.upper = work.rows[k].index_b;
            support.lower = work.rows[k].index_a;
        } else {
            supported = false;
        }
        if (supported) {
            support.rank = rank[support.upper];
            work.load[count + support.upper] += support.weight;
            supports.push_back(support);
        }
        k = end;
    }
    std::sort(supports.begin(), supports.end(), SupportByRank());
    for (std::size_t s = 0; s < supports.size(); s += 1) {
        const Support& support = supports[s];
        const double share = work.load[support.upper] * support.weight / work.load[count + support.upper];
        for (std::int32_t r = support.first; r < support.first + support.count; r += 1) {
            Row& row = work.rows[r];
            row.r = regularization / std::max(1.0 / row.local_inverse_mass, share);
        }
        if (support.lower >= 0 && rank[support.lower] > support.rank) {
            work.load[support.lower] += share;
        }
    }
}

// Soft-constraint coefficients of an implicit spring over one step h, for
// unit effective mass.
void spring_coefficients(double hertz, double zeta, double h, double& bias_rate, double& stiffness) {
    const double omega = kTwoPi * hertz;
    const double a1 = 2 * zeta + h * omega;
    bias_rate = omega / a1;
    stiffness = h * omega * a1;
}

struct RollingByKey {
    bool operator()(const RollingMemory& a, const RollingMemory& b) const { return a.pair_key < b.pair_key; }
};

}  // namespace

void solve_frame(const WorldParams& params, const SolverParams& solver, std::vector<Body>& bodies,
                 std::vector<ContactPoint>& contacts, HoldRecord& hold, SolverWork& work, SolveStats& stats) {
    const double h = params.frame_dt;
    work.bodies.clear();
    for (std::size_t i = 0; i < bodies.size(); i += 1) {
        Body& body = bodies[i];
        body.solver_index = -1;
        if (body.alive && !body.asleep && !body.is_static) {
            body.solver_index = static_cast<std::int32_t>(work.bodies.size());
            work.bodies.push_back(&body);
            const Mat3 rotation = to_matrix(body.orientation);
            body.inertia_world = rotated(rotation, body.inertia_local);
            body.inv_inertia_world = rotated(rotation, body.inv_inertia_local);
        }
    }
    const std::size_t count = work.bodies.size();
    stats.frames += 1;
    stats.last_passes = 0;
    stats.last_newton_iterations = 0;
    stats.last_residual = 0;
    stats.last_contacts = static_cast<int>(contacts.size());
    if (count == 0) {
        return;
    }
    const std::size_t size = count * 6;
    work.twist.resize(size); work.free_twist.resize(size); work.gradient.resize(size); work.momentum.resize(size);
    work.impulse.resize(size); work.direction.resize(size); work.difference.resize(size); work.mass_direction.resize(size);
    const double linear_decay = 1.0 / (1.0 + h * params.linear_damping);
    const double angular_decay = 1.0 / (1.0 + h * params.angular_damping);
    for (std::size_t i = 0; i < count; i += 1) {
        const Body& body = *work.bodies[i];
        double* v = work.twist.data() + i * 6;
        double* f = work.free_twist.data() + i * 6;
        v[0] = body.velocity.x; v[1] = body.velocity.y; v[2] = body.velocity.z;
        v[3] = body.angular.x; v[4] = body.angular.y; v[5] = body.angular.z;
        f[0] = (body.velocity.x + h * params.gravity.x) * linear_decay;
        f[1] = (body.velocity.y + h * params.gravity.y) * linear_decay;
        f[2] = (body.velocity.z + h * params.gravity.z) * linear_decay;
        f[3] = body.angular.x * angular_decay; f[4] = body.angular.y * angular_decay; f[5] = body.angular.z * angular_decay;
    }

    // Contact rows.
    work.rows.clear();
    work.pairs.clear();
    for (std::size_t k = 0; k < contacts.size(); k += 1) {
        ContactPoint& contact = contacts[k];
        const Body& body_a = bodies[contact.a];
        work.rows.emplace_back();
        Row& row = work.rows.back();
        clear_row(row);
        row.kind = kKindContact;
        row.retained = true;
        row.contact = static_cast<std::int32_t>(k);
        row.index_a = body_a.solver_index;
        tangent_basis(contact.normal, row.t1, row.t2);
        double trace = 0;
        fill_jacobian(row.ja, contact.point_a - body_a.position, row.t1, row.t2, contact.normal, 1.0, body_a, trace);
        if (!contact.fixed_b) {
            const Body& body_b = bodies[contact.b];
            row.index_b = body_b.solver_index;
            fill_jacobian(row.jb, contact.point_b - body_b.position, row.t1, row.t2, contact.normal, -1.0, body_b, trace);
            work.pairs.push_back(row.index_a);
            work.pairs.push_back(row.index_b);
        }
        row.friction = contact.friction;
        row.local_inverse_mass = trace / 3;
        row.r = solver.regularization * trace / 3;
        // Normal target: a gap may close exactly; an overlap decays, never pops.
        if (contact.separation > 0) {
            row.vhat_n_base = -contact.separation / h;
        } else {
            row.vhat_n_base = std::min(-contact.separation / (h + solver.relaxation_time), params.push_max_velocity);
        }
        row.vhat_n = row.vhat_n_base;
        row.lambda0 = dot(contact.warm_tangent, row.t1);
        row.lambda1 = dot(contact.warm_tangent, row.t2);
        row.lambda_n = contact.warm_normal;
    }
    const std::size_t contact_rows = work.rows.size();

    // Rolling resistance: one row per touching pair, a torque on the relative
    // angular velocity bounded by a lever arm times the pair's normal impulse.
    {
        std::size_t k = 0;
        while (k < contact_rows) {
            const ContactPoint& lead = contacts[work.rows[k].contact];
            std::size_t end = k;
            double load = 0;
            while (end < contact_rows && contacts[work.rows[end].contact].a == lead.a &&
                   contacts[work.rows[end].contact].b == lead.b) {
                load += work.rows[end].lambda_n;
                end += 1;
            }
            const Body& body_a = bodies[lead.a];
            double resistance = (*body_a.shape).rolling_resistance;
            if (lead.b >= 0) {
                resistance = std::max(resistance, (*bodies[lead.b].shape).rolling_resistance);
            } else {
                resistance = std::max(resistance, solver.ground_rolling_resistance);
            }
            if (resistance > 0) {
                double trace = body_a.inv_inertia_world.m[0] + body_a.inv_inertia_world.m[4] + body_a.inv_inertia_world.m[8];
                std::int32_t index_b = -1;
                if (!lead.fixed_b) {
                    const Body& body_b = bodies[lead.b];
                    index_b = body_b.solver_index;
                    trace += body_b.inv_inertia_world.m[0] + body_b.inv_inertia_world.m[4] + body_b.inv_inertia_world.m[8];
                }
                work.rows.emplace_back();
                Row& row = work.rows.back();
                clear_row(row);
                row.kind = kKindBall;
                row.retained = true;
                row.index_a = body_a.solver_index;
                row.index_b = index_b;
                row.r = solver.regularization * trace / 3;
                for (int d = 0; d < 3; d += 1) {
                    row.ja[d * 6 + 3 + d] = 1;
                    row.jb[d * 6 + 3 + d] = index_b >= 0 ? -1.0 : 0.0;
                }
                row.member_first = static_cast<std::int32_t>(k);
                row.member_count = static_cast<std::int32_t>(end - k);
                row.resistance = resistance;
                row.pair_key = (static_cast<std::uint64_t>(lead.a) << 32) | static_cast<std::uint32_t>(lead.b + 1);
                row.limit = resistance * load;
                RollingMemory probe;
                probe.pair_key = row.pair_key;
                const std::vector<RollingMemory>::const_iterator found =
                    std::lower_bound(work.rolling.begin(), work.rolling.end(), probe, RollingByKey());
                if (found != work.rolling.end() && (*found).pair_key == row.pair_key) {
                    row.lambda0 = (*found).impulse.x; row.lambda1 = (*found).impulse.y; row.lambda_n = (*found).impulse.z;
                }
            }
            k = end;
        }
    }

    // The crane: a linear spring with gravity feed-forward, wires that only
    // pull, capped side force; and an angular spring with capped torque.
    std::int32_t lift_row = -1;
    if (hold.active && bodies[hold.id].solver_index >= 0) {
        const Body& body = bodies[hold.id];
        const double weight_impulse = body.mass * length(params.gravity) * h;
        double bias_rate = 0;
        double stiffness = 0;
        spring_coefficients(hold.params.lin_hertz, hold.params.lin_zeta, h, bias_rate, stiffness);
        work.rows.emplace_back();
        lift_row = static_cast<std::int32_t>(work.rows.size() - 1);
        Row& lift = work.rows.back();
        clear_row(lift);
        lift.kind = kKindHold;
        lift.retained = false;
        lift.index_a = body.solver_index;
        lift.r = 1.0 / (body.mass * stiffness);
        for (int d = 0; d < 3; d += 1) {
            lift.ja[d * 6 + d] = 1;
        }
        Vec3 error = hold.target.p - body.position;
        if (hold.params.sling_length > 0) {
            // A tilted sling takes up slack before pulling a grounded load.
            // Retain unilateral lift and the existing force caps: contact and
            // friction still decide whether it slides, tips, or lifts clear.
            const double length = hold.params.sling_length;
            const double sideways2 = error.x * error.x + error.y * error.y;
            error.z += length - std::sqrt(std::max(0.0, length * length - sideways2));
        }
        lift.vhat0 = bias_rate * error.x; lift.vhat1 = bias_rate * error.y; lift.vhat_n = bias_rate * error.z;
        lift.lambda_n = weight_impulse;       // feed-forward: a free-hanging rock does not droop
        lift.limit = hold.params.max_lateral * weight_impulse;
        lift.limit_lift = hold.params.max_lift * weight_impulse;

        spring_coefficients(hold.params.ang_hertz, hold.params.ang_zeta, h, bias_rate, stiffness);
        const double mean_inertia = (body.inertia_world.m[0] + body.inertia_world.m[4] + body.inertia_world.m[8]) / 3;
        work.rows.emplace_back();
        Row& turn = work.rows.back();
        clear_row(turn);
        turn.kind = kKindBall;
        turn.retained = false;
        turn.index_a = body.solver_index;
        turn.r = 1.0 / (mean_inertia * stiffness);
        for (int d = 0; d < 3; d += 1) {
            turn.ja[d * 6 + 3 + d] = 1;
        }
        // Twice the vector part of target * conj(current), on the short side.
        Quat difference = multiply(hold.target.q, conjugate(body.orientation));
        if (difference.w < 0) {
            difference.x = -difference.x; difference.y = -difference.y; difference.z = -difference.z;
        }
        turn.vhat0 = bias_rate * 2 * difference.x; turn.vhat1 = bias_rate * 2 * difference.y;
        turn.vhat_n = bias_rate * 2 * difference.z;
        turn.limit = hold.params.max_torque * weight_impulse * (*body.shape).radius;
    }

    // The elimination order is kept while the contact graph is unchanged.
    work.graph.clear();
    work.graph.push_back(static_cast<std::int32_t>(count));
    for (std::size_t k = 0; k + 1 < work.pairs.size(); k += 2) {
        const std::int32_t low = std::min(work.pairs[k], work.pairs[k + 1]);
        const std::int32_t high = std::max(work.pairs[k], work.pairs[k + 1]);
        work.graph.push_back(low * static_cast<std::int32_t>(count) + high);
    }
    std::sort(work.graph.begin() + 1, work.graph.end());
    work.graph.erase(std::unique(work.graph.begin() + 1, work.graph.end()), work.graph.end());
    if (work.graph != work.previous_graph) {
        work.factor.analyze(static_cast<int>(count), work.pairs);
        work.previous_graph = work.graph;
        stats.analyses += 1;
    }
    work.factor_valid = false;

    work.contact_velocity.resize(work.rows.size() * 3);
    work.contact_step.resize(work.rows.size() * 3);
    assign_carried_mass(work, contacts, contact_rows, params.gravity, solver.regularization);
    // A point that carried normal impulse g last frame was holding up a mass
    // of g / (|gravity| h); its compliance is set against that too.
    const double gravity_impulse = length(params.gravity) * h;
    if (gravity_impulse > 0) {
        for (std::size_t k = 0; k < contact_rows; k += 1) {
            Row& row = work.rows[k];
            const double held = row.lambda_n / gravity_impulse;
            if (held * row.r > solver.regularization) {
                row.r = solver.regularization / held;
            }
        }
    }
    // Sliding shift and restitution from the incoming velocities.
    row_velocities(work.rows, work.twist, work.contact_velocity);
    for (std::size_t k = 0; k < contact_rows; k += 1) {
        Row& row = work.rows[k];
        ContactPoint& contact = contacts[row.contact];
        const double s0 = work.contact_velocity[k * 3];
        const double s1 = work.contact_velocity[k * 3 + 1];
        const double closing = work.contact_velocity[k * 3 + 2];
        // A point closing faster than the threshold and due to touch within
        // this frame first closes its gap; the rebound is asked for on the
        // next frame, at the surface.
        contact.bounce = 0;
        if (contact.pending_bounce > 0) {
            row.vhat_n_base = std::max(row.vhat_n_base, contact.pending_bounce);
        } else if (closing < -params.restitution_threshold && contact.separation + h * closing < 0) {
            if (contact.separation > params.linear_slop) {
                contact.bounce = -contact.restitution * closing;
            } else {
                row.vhat_n_base = std::max(row.vhat_n_base, -contact.restitution * closing);
            }
        }
        row.shift = row.friction * std::sqrt(s0 * s0 + s1 * s1);
        row.vhat_n = row.vhat_n_base - row.shift;
    }

    int iterations = 0;
    int passes = 0;
    double residual = 0;
    double scale = 0;
    double pass_accuracy = solver.loose_tolerance;
    for (int pass = 0; pass < solver.max_passes; pass += 1) {
        passes += 1;
        if (!newton_solve(work, pass_accuracy, solver.tight_tolerance, solver.max_newton_iterations, stats, iterations,
                          residual, scale)) {
            stats.not_converged += 1;
        }
        // The pass's impulses become the retained field. The passes end when
        // the field and the sliding shifts have stopped changing.
        double change = 0;
        double magnitude = 0;
        double shift_change = 0;
        for (std::size_t k = 0; k < work.rows.size(); k += 1) {
            Row& row = work.rows[k];
            if (!row.retained) {
                continue;
            }
            const double d0 = row.gamma0 - row.lambda0;
            const double d1 = row.gamma1 - row.lambda1;
            const double dn = row.gamma_n - row.lambda_n;
            change += row.r * (d0 * d0 + d1 * d1 + dn * dn);
            magnitude += row.r * (row.gamma0 * row.gamma0 + row.gamma1 * row.gamma1 + row.gamma_n * row.gamma_n);
            row.lambda0 = row.gamma0; row.lambda1 = row.gamma1; row.lambda_n = row.gamma_n;
            if (row.kind != kKindContact) {
                continue;
            }
            const double s0 = work.contact_velocity[k * 3];
            const double s1 = work.contact_velocity[k * 3 + 1];
            const double shift = row.friction * std::sqrt(s0 * s0 + s1 * s1);
            shift_change = std::max(shift_change, std::abs(shift - row.shift));
            row.shift = shift;
            row.vhat_n = row.vhat_n_base - shift;
        }
        for (std::size_t k = contact_rows; k < work.rows.size(); k += 1) {
            Row& row = work.rows[k];
            if (row.member_count == 0) {
                continue;
            }
            double load = 0;
            for (std::int32_t m = row.member_first; m < row.member_first + row.member_count; m += 1) {
                load += work.rows[m].gamma_n;
            }
            row.limit = row.resistance * load;
        }
        const double relative_change = std::sqrt(change) / std::max(std::sqrt(magnitude), 1.0e-300);
        if (relative_change <= solver.pass_tolerance && shift_change <= solver.shift_tolerance) {
            break;
        }
        pass_accuracy = std::min(solver.loose_tolerance, std::max(solver.tight_tolerance, 0.1 * relative_change));
        if (pass == solver.max_passes - 1) {
            stats.pass_limited += 1;
        }
    }
    stats.passes += passes;
    stats.newton_iterations += iterations;
    stats.last_passes = passes;
    stats.last_newton_iterations = iterations;
    stats.last_residual = residual;

    for (std::size_t k = 0; k < contact_rows; k += 1) {
        const Row& row = work.rows[k];
        ContactPoint& contact = contacts[row.contact];
        contact.normal_impulse = row.gamma_n;
        contact.tangent_impulse = row.t1 * row.gamma0 + row.t2 * row.gamma1;
        contact.mode = row.mode;
    }
    work.next_rolling.clear();
    for (std::size_t k = contact_rows; k < work.rows.size(); k += 1) {
        const Row& row = work.rows[k];
        if (row.member_count > 0) {
            RollingMemory memory;
            memory.pair_key = row.pair_key;
            memory.impulse = Vec3{row.gamma0, row.gamma1, row.gamma_n};
            work.next_rolling.push_back(memory);
        }
    }
    std::sort(work.next_rolling.begin(), work.next_rolling.end(), RollingByKey());
    work.rolling.swap(work.next_rolling);
    if (lift_row >= 0) {
        const Row& lift = work.rows[lift_row];
        const Body& body = bodies[hold.id];
        const double weight_impulse = body.mass * length(params.gravity) * h;
        hold.state.tension = lift.gamma_n / weight_impulse;
        hold.state.lateral_force = std::sqrt(lift.gamma0 * lift.gamma0 + lift.gamma1 * lift.gamma1) / weight_impulse;
    }
    for (std::size_t i = 0; i < count; i += 1) {
        Body& body = *work.bodies[i];
        const double* v = work.twist.data() + i * 6;
        body.velocity = Vec3{v[0], v[1], v[2]};
        body.angular = Vec3{v[3], v[4], v[5]};
        body.position = add_scaled(body.position, body.velocity, h);
        body.orientation = integrate(body.orientation, body.angular, h);
    }
    if (hold.active) {
        hold.state.position_error = hold.target.p - bodies[hold.id].position;
    }
}

}  // namespace zc::phys
