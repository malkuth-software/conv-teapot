// CONV line cell: two-jet proposal, ledger-free box admission, bounded bubble.
//
// One cell of a CONV line is a function of the six samples y[i-2..i+3]
// alone (of y[i-3..i+4] next to a seam). This file evaluates that cell
// for any scalar type, so the same code serves plain values and the
// forward-mode duals that carry surface derivatives.
//
// The formulas follow the CONV paper (malkuth-software/papers), Sections V
// and XIII, and agree with the stx_conv stream reference model. The
// continuous bubble limiter (Bubble::continuous) is new here; see the paper
// in ../paper for why a surface needs it.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>

namespace tconv {

// Value plus two partial derivatives (surface parameters s and t).
struct Dual {
    double v = 0.0;
    double ds = 0.0;
    double dt = 0.0;
    Dual() = default;
    Dual(double value) : v(value) {}
    Dual(double value, double s, double t) : v(value), ds(s), dt(t) {}
};
inline Dual operator+(const Dual& a, const Dual& b) { return {a.v + b.v, a.ds + b.ds, a.dt + b.dt}; }
inline Dual operator-(const Dual& a, const Dual& b) { return {a.v - b.v, a.ds - b.ds, a.dt - b.dt}; }
inline Dual operator-(const Dual& a) { return {-a.v, -a.ds, -a.dt}; }
inline Dual operator*(const Dual& a, const Dual& b) {
    return {a.v * b.v, a.ds * b.v + a.v * b.ds, a.dt * b.v + a.v * b.dt};
}
inline Dual operator+(double k, const Dual& a) { return {k + a.v, a.ds, a.dt}; }
inline Dual operator*(double k, const Dual& a) { return {k * a.v, k * a.ds, k * a.dt}; }
inline Dual operator*(const Dual& a, double k) { return {k * a.v, k * a.ds, k * a.dt}; }
inline Dual operator/(const Dual& a, double k) { return {a.v / k, a.ds / k, a.dt / k}; }

inline double value_of(double x) { return x; }
inline double value_of(const Dual& x) { return x.v; }

// min and max select one argument whole, so a dual keeps the derivative of
// the selected branch: the map is piecewise affine and this is its
// derivative wherever one exists.
template<class T> inline T tmin(const T& a, const T& b) { return value_of(b) < value_of(a) ? b : a; }
template<class T> inline T tmax(const T& a, const T& b) { return value_of(b) > value_of(a) ? b : a; }
template<class T> inline T pos(const T& x) { return value_of(x) > 0.0 ? x : T(0.0); }       // [x]+
template<class T> inline T neg(const T& x) { return value_of(x) < 0.0 ? T(0.0) - x : T(0.0); } // [x]-

// Which difference stencil gives a node its scaled jet (m, q).
enum class Jet : unsigned char {
    interior,      // fourth-order centred, five samples (exact for quartics)
    paper_first,   // CONV paper closure, first node: m from 3 samples, q from 4
    paper_central, // CONV paper closure, second and second-to-last node
    paper_last,    // CONV paper closure, last node
    first4,        // cubic-exact, four samples to the right
    second4,       // cubic-exact, one sample to the left and two to the right
    penult4,       // cubic-exact, two to the left and one to the right
    last4,         // cubic-exact, four samples to the left
    first3,        // three-node segment: quadratic through the segment
    central3,
    last3,
    first2,        // two-node segment: the secant
    last2
};

enum class Bubble : unsigned char {
    none,
    standard,  // stx_conv's one-sided sign bound (discontinuous in the data)
    continuous // symmetric bound of this project (continuous in the data)
};

enum class Closure : unsigned char {
    paper, // second-order one-sided ends, as in the CONV paper and stx_conv
    cubic  // cubic-exact ends: a cubic segment is reproduced exactly
};

// The structure of a cell: decided by its place in the line, never by data.
// A node on a seam has two jets, one from each side; `left` and `right` are
// the jets of this cell's own segment, `prev` and `next` the jets with which
// the neighbouring cells see the shared nodes.
struct CellShape {
    Jet left = Jet::interior;
    Jet right = Jet::interior;
    Jet prev = Jet::interior; // node i as the previous cell sees it
    Jet next = Jet::interior; // node i+1 as the next cell sees it
    bool has_prev = true;     // a cell exists to the left
    bool has_next = true;     // a cell exists to the right
    bool six = true;          // the six-sample quintic term of the bubble applies
};

// A line of nodes with optional seams. A seam at a node cuts every
// difference stencil there; values stay continuous across it. The two ends
// of an open line are seams.
struct LineLayout {
    std::size_t nodes = 0;
    bool periodic = false;
    Closure closure = Closure::cubic;
    const unsigned char* seams = nullptr; // [nodes], nonzero at a seam; may be null

    std::size_t cells() const { return periodic ? nodes : nodes - 1; }
    bool seam(long node) const {
        if (!periodic && (node <= 0 || node >= long(nodes) - 1)) { return true; }
        if (seams == nullptr) { return false; }
        const long wrapped = ((node % long(nodes)) + long(nodes)) % long(nodes);
        return seams[wrapped] != 0;
    }
    bool any_seam() const {
        if (!periodic) { return true; }
        if (seams == nullptr) { return false; }
        for (std::size_t k = 0; k < nodes; ++k) {
            if (seams[k] != 0) { return true; }
        }
        return false;
    }
};

namespace detail {

// Jet of a node `dl` nodes after the start and `dr` nodes before the end of
// its segment.
inline Jet pick_jet(long dl, long dr, Closure closure) {
    if (dl >= 2 && dr >= 2) { return Jet::interior; }
    const long span = dl + dr; // segment nodes - 1
    if (closure == Closure::paper && span >= 4) {
        return dl == 0 ? Jet::paper_first : dr == 0 ? Jet::paper_last : Jet::paper_central;
    }
    if (span >= 3) { return dl == 0 ? Jet::first4 : dr == 0 ? Jet::last4 : dl == 1 ? Jet::second4 : Jet::penult4; }
    if (span == 2) { return dl == 0 ? Jet::first3 : dr == 0 ? Jet::last3 : Jet::central3; }
    return dl == 0 ? Jet::first2 : Jet::last2;
}

// Jet of `node` as seen from the cell [cell, cell+1] of `layout`.
inline Jet jet_in_cell(const LineLayout& layout, long cell, long node) {
    if (!layout.any_seam()) { return Jet::interior; }
    long start = cell;
    while (!layout.seam(start)) { --start; }
    long end = cell + 1;
    while (!layout.seam(end)) { ++end; }
    return pick_jet(node - start, end - node, layout.closure);
}

} // namespace detail

// Shape of cell `cell` of a line.
inline CellShape cell_shape(const LineLayout& layout, std::size_t cell) {
    CellShape shape;
    const long i = long(cell);
    shape.has_prev = layout.periodic || cell > 0;
    shape.has_next = layout.periodic || cell + 2 < layout.nodes;
    shape.left = detail::jet_in_cell(layout, i, i);
    shape.right = detail::jet_in_cell(layout, i, i + 1);
    shape.prev = shape.has_prev ? detail::jet_in_cell(layout, i - 1, i) : shape.left;
    shape.next = shape.has_next ? detail::jet_in_cell(layout, i + 1, i + 1) : shape.right;
    // Six samples y[i-2..i+3] inside one segment.
    shape.six = !layout.seam(i) && !layout.seam(i - 1) && !layout.seam(i + 1) && !layout.seam(i + 2);
    if (!layout.periodic && (cell < 2 || cell + 3 >= layout.nodes)) { shape.six = false; }
    return shape;
}

// Admitted currents and bubble amplitude of one cell.
template<class T> struct Cell {
    T anchor{};          // y[i]
    std::array<T, 5> c{}; // admitted currents, sum = y[i+1] - y[i]
    T bubble{};          // amplitude of psi(t) = [t(1-t)]^3 (2t-1)
};

namespace detail {

// Scaled jet (m, q) of the node at c[0] from its neighbours c[-3..3].
template<class T> inline void jet(const T* c, Jet kind, T& m, T& q) {
    switch (kind) {
    case Jet::interior:
        m = (c[-2] - 8.0 * c[-1] + 8.0 * c[1] - c[2]) / 12.0;
        q = (16.0 * c[1] - c[2] - 30.0 * c[0] + 16.0 * c[-1] - c[-2]) / 12.0;
        return;
    case Jet::paper_first:
        m = (4.0 * c[1] - 3.0 * c[0] - c[2]) / 2.0;
        q = 2.0 * c[0] - 5.0 * c[1] + 4.0 * c[2] - c[3];
        return;
    case Jet::paper_central:
    case Jet::central3:
        m = (c[1] - c[-1]) / 2.0;
        q = c[-1] - 2.0 * c[0] + c[1];
        return;
    case Jet::paper_last:
        m = (3.0 * c[0] - 4.0 * c[-1] + c[-2]) / 2.0;
        q = 2.0 * c[0] - 5.0 * c[-1] + 4.0 * c[-2] - c[-3];
        return;
    case Jet::first4:
        m = (18.0 * c[1] - 11.0 * c[0] - 9.0 * c[2] + 2.0 * c[3]) / 6.0;
        q = 2.0 * c[0] - 5.0 * c[1] + 4.0 * c[2] - c[3];
        return;
    case Jet::second4:
        m = (6.0 * c[1] - 2.0 * c[-1] - 3.0 * c[0] - c[2]) / 6.0;
        q = c[-1] - 2.0 * c[0] + c[1];
        return;
    case Jet::penult4:
        m = (2.0 * c[1] + 3.0 * c[0] - 6.0 * c[-1] + c[-2]) / 6.0;
        q = c[-1] - 2.0 * c[0] + c[1];
        return;
    case Jet::last4:
        m = (11.0 * c[0] - 18.0 * c[-1] + 9.0 * c[-2] - 2.0 * c[-3]) / 6.0;
        q = 2.0 * c[0] - 5.0 * c[-1] + 4.0 * c[-2] - c[-3];
        return;
    case Jet::first3:
        m = (4.0 * c[1] - 3.0 * c[0] - c[2]) / 2.0;
        q = c[0] - 2.0 * c[1] + c[2];
        return;
    case Jet::last3:
        m = (3.0 * c[0] - 4.0 * c[-1] + c[-2]) / 2.0;
        q = c[0] - 2.0 * c[-1] + c[-2];
        return;
    case Jet::first2:
        m = c[1] - c[0];
        q = T(0.0);
        return;
    case Jet::last2:
        m = c[0] - c[-1];
        q = T(0.0);
        return;
    }
}

template<class T> inline void order(T& a, T& b) {
    const T low = tmin(a, b);
    const T high = tmax(a, b);
    a = low;
    b = high;
}

// Capacity threshold: the largest of five prefix forms after a
// nine-comparator sort (CONV paper, Corollary "Five prefix forms").
template<class T> inline T capacity_threshold(std::array<T, 5> w, const T& mass) {
    order(w[0], w[1]);
    order(w[3], w[4]);
    order(w[2], w[4]);
    order(w[2], w[3]);
    order(w[1], w[4]);
    order(w[0], w[3]);
    order(w[0], w[2]);
    order(w[1], w[3]);
    order(w[1], w[2]);
    T result = mass / 5.0;
    T sum = w[0];
    result = tmax(result, (mass - sum) / 4.0);
    sum = sum + w[1];
    result = tmax(result, (mass - sum) / 3.0);
    sum = sum + w[2];
    result = tmax(result, (mass - sum) / 2.0);
    sum = sum + w[3];
    result = tmax(result, mass - sum);
    return result;
}

} // namespace detail

// One cell from its window w[0..7] = y[i-3..i+4]; the cell lies between
// w[3] and w[4]. Entries outside an open line are never read.
template<class T> Cell<T> compile_cell(const T* w, const CellShape& shape, Bubble bubble) {
    Cell<T> cell;
    cell.anchor = w[3];
    const T zero(0.0);
    T ml, ql, mr, qr;
    detail::jet(w + 3, shape.left, ml, ql);
    detail::jet(w + 4, shape.right, mr, qr);
    const T delta = w[4] - w[3];
    // Raw currents: differences of the six quintic controls.
    std::array<T, 5> a;
    a[0] = ml / 5.0;
    a[1] = ml / 5.0 + ql / 20.0;
    a[3] = mr / 5.0 - qr / 20.0;
    a[4] = mr / 5.0;
    a[2] = delta - a[0] - a[1] - a[3] - a[4];

    // Unsigned neighbour box.
    const T prev_delta = shape.has_prev ? w[3] - w[2] : zero;
    const T next_delta = shape.has_next ? w[5] - w[4] : zero;
    // Terminal raw currents of the neighbouring cells, a[i-1][4] and
    // a[i+1][0]. Away from a seam they equal a[0] and a[4].
    T prev_last = zero, next_first = zero;
    if (shape.has_prev) {
        T m = ml, q;
        if (shape.prev != shape.left) { detail::jet(w + 3, shape.prev, m, q); }
        prev_last = m / 5.0;
    }
    if (shape.has_next) {
        T m = mr, q;
        if (shape.next != shape.right) { detail::jet(w + 4, shape.next, m, q); }
        next_first = m / 5.0;
    }
    std::array<T, 5> room_pos, room_neg;
    T head_pos = tmin(pos(prev_delta), pos(pos(a[0]) - neg(prev_last)));
    T head_neg = tmin(neg(prev_delta), pos(neg(a[0]) - pos(prev_last)));
    for (int j = 0; j < 5; ++j) {
        head_pos = tmin(head_pos, pos(a[j]));
        head_neg = tmin(head_neg, neg(a[j]));
        room_pos[j] = head_pos;
        room_neg[j] = head_neg;
    }
    T tail_pos = tmin(pos(next_delta), pos(pos(a[4]) - neg(next_first)));
    T tail_neg = tmin(neg(next_delta), pos(neg(a[4]) - pos(next_first)));
    for (int j = 4; j >= 0; --j) {
        tail_pos = tmin(tail_pos, pos(a[j]));
        tail_neg = tmin(tail_neg, neg(a[j]));
        room_pos[j] = room_pos[j] + tail_pos;
        room_neg[j] = room_neg[j] + tail_neg;
    }
    T total_pos = zero, total_neg = zero;
    for (int j = 0; j < 5; ++j) {
        total_pos = total_pos + room_pos[j];
        total_neg = total_neg + room_neg[j];
    }
    const T plus = pos(delta);
    const T minus = neg(delta);
    const T upper_base = plus + tmin(total_neg, 4.0 * plus);
    const T lower_base = minus + tmin(total_pos, 4.0 * minus);

    // Preclip, then restore the mass by one capped common shift.
    std::array<T, 5> q, down, up;
    T sum = zero;
    for (int j = 0; j < 5; ++j) {
        const T high = room_pos[j] + upper_base;
        const T low = zero - (room_neg[j] + lower_base);
        q[j] = tmin(high, tmax(low, a[j]));
        down[j] = q[j] - low;
        up[j] = high - q[j];
        sum = sum + q[j];
    }
    const T residual = sum - delta;
    const bool lower = value_of(residual) > 0.0;
    const std::array<T, 5>& capacity = lower ? down : up;
    const T mass = lower ? residual : zero - residual;
    const T threshold = detail::capacity_threshold(capacity, mass);
    for (int j = 0; j < 5; ++j) {
        const T shift = tmin(threshold, capacity[j]);
        cell.c[j] = lower ? q[j] - shift : q[j] + shift;
    }

    if (bubble == Bubble::none) { return cell; }
    // Fitted amplitude: L2 projection onto psi of (six-sample quintic minus
    // admitted quintic).
    T fit = zero;
    if (shape.six) {
        const T fifth = (w[6] - w[1]) + 5.0 * (w[2] - w[5]) + 10.0 * (w[4] - w[3]);
        fit = (-41.0 / 30.0) * fifth;
    }
    static constexpr double kappa[5] = {35.0, 87.5, 112.5, 87.5, 35.0};
    for (int j = 0; j < 5; ++j) { fit = fit + kappa[j] * (a[j] - cell.c[j]); }
    // Degree-six coefficients of the admitted derivative at the three
    // positions the bubble touches.
    const std::array<T, 5>& c = cell.c;
    const T d2 = (c[0] + 8.0 * c[1] + 6.0 * c[2]) / 3.0;
    const T d3 = c[1] + 3.0 * c[2] + c[3];
    const T d4 = (6.0 * c[2] + 8.0 * c[3] + c[4]) / 3.0;
    if (bubble == Bubble::continuous) {
        // |amplitude| <= min(5|d2|, 5|d3|/2, 5|d4|): no coefficient changes
        // sign, and the bound is continuous in the data.
        const T bound = tmin(tmin(5.0 * tmax(d2, zero - d2), 2.5 * tmax(d3, zero - d3)), 5.0 * tmax(d4, zero - d4));
        cell.bubble = tmin(bound, tmax(zero - bound, fit));
        return cell;
    }
    // Standard one-sided bound (stx_conv): each coefficient keeps its sign.
    const double infinity = INFINITY;
    T low(-infinity), high(infinity);
    bool pinned = false;
    const auto bound = [&](const T& d, double scale, bool amplitude_subtracts) {
        // coefficient = d - amplitude/5 (subtracts) or d + 2 amplitude/5.
        const T edge = scale * d;
        const double dv = value_of(d);
        if (dv == 0.0) { pinned = true; return; }
        if (amplitude_subtracts) {
            if (dv > 0.0) { high = tmin(high, edge); } else { low = tmax(low, edge); }
        } else {
            if (dv > 0.0) { low = tmax(low, edge); } else { high = tmin(high, edge); }
        }
    };
    bound(d2, 5.0, true);
    bound(d3, -2.5, false);
    bound(d4, 5.0, true);
    cell.bubble = pinned ? zero : tmin(high, tmax(low, fit));
    return cell;
}

// The six functions a cell is a combination of, at local parameter t:
// tau[j] = sum of the Bernstein polynomials B_r^5 for r > j, and the bubble
// psi = [t(1-t)]^3 (2t-1).
template<class U> struct Basis {
    U tau[5];
    U psi;
};

template<class U> inline Basis<U> make_basis(const U& t) {
    Basis<U> basis;
    const U s = U(1.0) - t;
    const U t2 = t * t, t3 = t2 * t, t4 = t2 * t2;
    const U s2 = s * s, s3 = s2 * s, s4 = s2 * s2;
    basis.tau[4] = t4 * t;
    basis.tau[3] = basis.tau[4] + 5.0 * t4 * s;
    basis.tau[2] = basis.tau[3] + 10.0 * t3 * s2;
    basis.tau[1] = basis.tau[2] + 10.0 * t2 * s3;
    basis.tau[0] = basis.tau[1] + 5.0 * t * s4;
    basis.psi = t3 * s3 * (2.0 * t - U(1.0));
    return basis;
}

// Value of a compiled cell for a prepared basis.
template<class T, class U> inline auto eval_cell(const Cell<T>& cell, const Basis<U>& basis) {
    auto value = cell.anchor + cell.c[0] * basis.tau[0];
    value = value + cell.c[1] * basis.tau[1];
    value = value + cell.c[2] * basis.tau[2];
    value = value + cell.c[3] * basis.tau[3];
    value = value + cell.c[4] * basis.tau[4];
    value = value + cell.bubble * basis.psi;
    return value;
}

// Value of a compiled cell at local parameter t in [0,1].
template<class T, class U> inline auto eval_cell(const Cell<T>& cell, const U& t) {
    return eval_cell(cell, make_basis(t));
}

} // namespace tconv
