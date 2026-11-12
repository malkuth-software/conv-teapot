// Parametric surfaces over a node lattice.
//
// Every surface here is addressed in node units: u in [0, cells_u],
// v in [0, cells_v], with integer coordinates at the nodes. ConvSurface is
// the CONV geometry: it passes through every node, and its value at (u, v)
// depends on the 6 by 6 block of nodes around the cell (8 by 8 next to a
// seam) and on nothing else.
#pragma once
#include "conv_line.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

namespace tconv {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
inline Vec3 operator*(double k, Vec3 a) { return {a.x * k, a.y * k, a.z * k}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) {
    const double n = length(a);
    return n > 0.0 ? a * (1.0 / n) : Vec3{0.0, 0.0, 0.0};
}

// A point with its two parametric derivatives (per node unit).
struct SurfacePoint {
    Vec3 p, du, dv;
};

class Surface {
public:
    virtual ~Surface() = default;
    virtual std::size_t cells_u() const = 0;
    virtual std::size_t cells_v() const = 0;
    virtual bool periodic_u() const = 0;
    virtual SurfacePoint eval(double u, double v) const = 0;
    virtual Vec3 position(double u, double v) const { return eval(u, v).p; }
};

// A rigid motion: p -> A p + b, A a rotation.
struct Rigid {
    double a[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Vec3 b;
    Vec3 direction(Vec3 d) const { return {a[0] * d.x + a[1] * d.y + a[2] * d.z, a[3] * d.x + a[4] * d.y + a[5] * d.z, a[6] * d.x + a[7] * d.y + a[8] * d.z}; }
    Vec3 point(Vec3 p) const { return direction(p) + b; }
    bool identity() const {
        return a[0] == 1 && a[4] == 1 && a[8] == 1 && a[1] == 0 && a[2] == 0 && a[3] == 0 && a[5] == 0 && a[6] == 0 && a[7] == 0 && b.x == 0 && b.y == 0 && b.z == 0;
    }
};

// A surface carried by a rigid motion. A body that moves rigidly is drawn
// this way, from its lattice at rest. Moving the nodes instead and building
// CONV through them is not the same surface: admission and the bubble act
// on each coordinate separately, so they do not commute with a rotation, and
// a turned lattice shows small creases that the lattice at rest does not.
class PosedSurface final : public Surface {
public:
    const Surface* base = nullptr;
    const Rigid* pose = nullptr;
    std::size_t cells_u() const override { return base->cells_u(); }
    std::size_t cells_v() const override { return base->cells_v(); }
    bool periodic_u() const override { return base->periodic_u(); }
    SurfacePoint eval(double u, double v) const override {
        SurfacePoint point = base->eval(u, v);
        point.p = pose->point(point.p);
        point.du = pose->direction(point.du);
        point.dv = pose->direction(point.dv);
        return point;
    }
    Vec3 position(double u, double v) const override { return pose->point(base->position(u, v)); }
};

// Splits a coordinate into a cell and a local parameter in [0,1].
inline void locate(double x, std::size_t cells, bool periodic, std::size_t& cell, double& t) {
    if (periodic) {
        x = x - std::floor(x / double(cells)) * double(cells);
        if (x >= double(cells)) { x = 0.0; }
    } else {
        x = x < 0.0 ? 0.0 : x > double(cells) ? double(cells) : x;
    }
    double whole = std::floor(x);
    if (whole >= double(cells)) { whole = double(cells) - 1.0; }
    cell = std::size_t(whole);
    t = x - whole;
}

// A lattice of nodes: positions or colours. Periodic in u when the surface
// closes on itself. Seams cut the difference stencils along a whole node
// row or column, which keeps a crease or a patch joint crisp.
struct Lattice {
    std::size_t nu = 0, nv = 0; // node counts
    bool wrap_u = false;
    Closure closure = Closure::cubic;
    std::vector<Vec3> nodes;             // [v * nu + u]
    std::vector<unsigned char> seams_u;  // [nu] or empty
    std::vector<unsigned char> seams_v;  // [nv] or empty
    std::size_t cells_u() const { return wrap_u ? nu : nu - 1; }
    std::size_t cells_v() const { return nv - 1; }
    Vec3& at(std::size_t u, std::size_t v) { return nodes[v * nu + u]; }
    const Vec3& at(std::size_t u, std::size_t v) const { return nodes[v * nu + u]; }
    LineLayout layout_u() const {
        return {nu, wrap_u, closure, seams_u.empty() ? nullptr : seams_u.data()};
    }
    LineLayout layout_v() const {
        return {nv, false, closure, seams_v.empty() ? nullptr : seams_v.data()};
    }
};

// CONV geometry: u-lines first, then one v-line through the u-line values
// at the query. compile() caches the admitted currents of every u-cell of
// every row; the v-factor is admitted at each query, because its inputs
// depend on u.
class ConvSurface final : public Surface {
public:
    Lattice lattice;
    Bubble bubble = Bubble::continuous;

    void compile() {
        const std::size_t cu = lattice.cells_u();
        const LineLayout layout_u = lattice.layout_u();
        const LineLayout layout_v = lattice.layout_v();
        shapes_u_.resize(cu);
        for (std::size_t cell = 0; cell < cu; ++cell) { shapes_u_[cell] = cell_shape(layout_u, cell); }
        shapes_v_.resize(lattice.cells_v());
        outer_low_.assign(lattice.cells_v(), 0);
        outer_high_.assign(lattice.cells_v(), 0);
        for (std::size_t cell = 0; cell < lattice.cells_v(); ++cell) {
            const CellShape shape = cell_shape(layout_v, cell);
            shapes_v_[cell] = shape;
            // Rows i-3 and i+4 are read only by the four-sample end stencils.
            const auto reaches_low = [](Jet jet) { return jet == Jet::last4 || jet == Jet::paper_last; };
            const auto reaches_high = [](Jet jet) { return jet == Jet::first4 || jet == Jet::paper_first; };
            outer_low_[cell] = reaches_low(shape.left) || reaches_low(shape.prev) || reaches_low(shape.right);
            outer_high_[cell] = reaches_high(shape.right) || reaches_high(shape.next) || reaches_high(shape.left);
        }
        rows_.assign(lattice.nv * cu * 3, Cell<double>{});
        for (std::size_t v = 0; v < lattice.nv; ++v) {
            for (std::size_t cell = 0; cell < cu; ++cell) {
                double w[3][8] = {};
                for (int k = 0; k < 8; ++k) {
                    long index = long(cell) - 3 + k;
                    if (lattice.wrap_u) {
                        index = ((index % long(lattice.nu)) + long(lattice.nu)) % long(lattice.nu);
                    } else if (index < 0 || index >= long(lattice.nu)) {
                        continue;
                    }
                    const Vec3& node = lattice.at(std::size_t(index), v);
                    w[0][k] = node.x;
                    w[1][k] = node.y;
                    w[2][k] = node.z;
                }
                for (int channel = 0; channel < 3; ++channel) {
                    rows_[(v * cu + cell) * 3 + std::size_t(channel)] = compile_cell(w[channel], shapes_u_[cell], bubble);
                }
            }
        }
    }

    std::size_t cells_u() const override { return lattice.cells_u(); }
    std::size_t cells_v() const override { return lattice.cells_v(); }
    bool periodic_u() const override { return lattice.wrap_u; }

    SurfacePoint eval(double u, double v) const override {
        const Dual ds_seed(0.0, 1.0, 0.0), dt_seed(0.0, 0.0, 1.0);
        Dual out[3];
        evaluate<Dual>(u, v, ds_seed, dt_seed, out);
        SurfacePoint point;
        point.p = {out[0].v, out[1].v, out[2].v};
        point.du = {out[0].ds, out[1].ds, out[2].ds};
        point.dv = {out[0].dt, out[1].dt, out[2].dt};
        return point;
    }

    Vec3 position(double u, double v) const override {
        double out[3];
        evaluate<double>(u, v, 0.0, 0.0, out);
        return {out[0], out[1], out[2]};
    }

private:
    template<class T> void evaluate(double u, double v, const T& s_seed, const T& t_seed, T* out) const {
        std::size_t cell_u = 0, cell_v = 0;
        double s = 0.0, t = 0.0;
        locate(u, cells_u(), lattice.wrap_u, cell_u, s);
        locate(v, cells_v(), false, cell_v, t);
        const Basis<T> basis_s = make_basis(T(s) + s_seed);
        const Basis<T> basis_t = make_basis(T(t) + t_seed);
        T w[3][8];
        const std::size_t cu = cells_u();
        for (int k = 0; k < 8; ++k) {
            const long row = long(cell_v) - 3 + k;
            if (row < 0 || row >= long(lattice.nv) || (k == 0 && !outer_low_[cell_v]) || (k == 7 && !outer_high_[cell_v])) {
                for (int channel = 0; channel < 3; ++channel) { w[channel][k] = T(0.0); }
                continue;
            }
            const Cell<double>* cells = &rows_[(std::size_t(row) * cu + cell_u) * 3];
            for (int channel = 0; channel < 3; ++channel) { w[channel][k] = eval_cell(cells[channel], basis_s); }
        }
        for (int channel = 0; channel < 3; ++channel) {
            out[channel] = eval_cell(compile_cell(w[channel], shapes_v_[cell_v], bubble), basis_t);
        }
    }

    std::vector<Cell<double>> rows_;
    std::vector<CellShape> shapes_u_, shapes_v_;
    std::vector<unsigned char> outer_low_, outer_high_;
};

// The same nodes joined bilinearly: the control mesh, for comparison.
class LinearSurface final : public Surface {
public:
    const Lattice* lattice = nullptr;
    std::size_t cells_u() const override { return lattice->cells_u(); }
    std::size_t cells_v() const override { return lattice->cells_v(); }
    bool periodic_u() const override { return lattice->wrap_u; }
    SurfacePoint eval(double u, double v) const override {
        std::size_t cu = 0, cv = 0;
        double s = 0.0, t = 0.0;
        locate(u, cells_u(), lattice->wrap_u, cu, s);
        locate(v, cells_v(), false, cv, t);
        const std::size_t u1 = lattice->wrap_u ? (cu + 1) % lattice->nu : cu + 1;
        const Vec3 a = lattice->at(cu, cv), b = lattice->at(u1, cv);
        const Vec3 c = lattice->at(cu, cv + 1), d = lattice->at(u1, cv + 1);
        SurfacePoint point;
        point.p = (a * (1 - s) + b * s) * (1 - t) + (c * (1 - s) + d * s) * t;
        point.du = (b - a) * (1 - t) + (d - c) * t;
        point.dv = (c - a) * (1 - s) + (d - b) * s;
        return point;
    }
};

// A grid of bicubic Bezier patches addressed in the node units of a
// lattice that samples each patch `per_u` by `per_v` times.
class BezierSurface final : public Surface {
public:
    std::size_t patches_u = 0, patches_v = 0; // patch counts
    std::size_t per_u = 1, per_v = 1;         // node intervals per patch
    bool wrap_u = false;
    std::vector<std::array<Vec3, 16>> patches; // [pv * patches_u + pu], control [i * 4 + j], i along u

    std::size_t cells_u() const override { return patches_u * per_u; }
    std::size_t cells_v() const override { return patches_v * per_v; }
    bool periodic_u() const override { return wrap_u; }

    SurfacePoint eval(double u, double v) const override {
        std::size_t pu = 0, pv = 0;
        double s = 0.0, t = 0.0;
        locate(u / double(per_u), patches_u, wrap_u, pu, s);
        locate(v / double(per_v), patches_v, false, pv, t);
        const std::array<Vec3, 16>& control = patches[pv * patches_u + pu];
        double bs[4], bt[4], ds[4], dt[4];
        basis(s, bs, ds);
        basis(t, bt, dt);
        SurfacePoint point;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const Vec3& c = control[std::size_t(i * 4 + j)];
                point.p = point.p + c * (bs[i] * bt[j]);
                point.du = point.du + c * (ds[i] * bt[j]);
                point.dv = point.dv + c * (bs[i] * dt[j]);
            }
        }
        point.du = point.du * (1.0 / double(per_u));
        point.dv = point.dv * (1.0 / double(per_v));
        return point;
    }

private:
    static void basis(double t, double* b, double* d) {
        const double s = 1.0 - t;
        b[0] = s * s * s;
        b[1] = 3.0 * s * s * t;
        b[2] = 3.0 * s * t * t;
        b[3] = t * t * t;
        d[0] = -3.0 * s * s;
        d[1] = 3.0 * s * s - 6.0 * s * t;
        d[2] = 6.0 * s * t - 3.0 * t * t;
        d[3] = 3.0 * t * t;
    }
};

} // namespace tconv
