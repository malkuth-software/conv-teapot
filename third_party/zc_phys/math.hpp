#pragma once
// Internal small-matrix helpers shared by the engine's sources.
#include <cmath>

#include "physics.hpp"

namespace zc::phys {

struct Mat3 { double m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; };   // row-major

inline Vec3 add_scaled(Vec3 a, Vec3 b, double s) {
    Vec3 out{a.x + b.x * s, a.y + b.y * s, a.z + b.z * s};
    return out;
}

inline Vec3 mul(const Mat3& a, Vec3 v) {
    Vec3 out{
        a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z,
        a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
        a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z};
    return out;
}

inline Mat3 mul(const Mat3& a, const Mat3& b) {
    Mat3 out;
    for (int row = 0; row < 3; row += 1) {
        for (int col = 0; col < 3; col += 1) {
            out.m[row * 3 + col] =
                a.m[row * 3] * b.m[col] + a.m[row * 3 + 1] * b.m[3 + col] + a.m[row * 3 + 2] * b.m[6 + col];
        }
    }
    return out;
}

inline Mat3 transposed(const Mat3& a) {
    Mat3 out;
    out.m[0] = a.m[0]; out.m[1] = a.m[3]; out.m[2] = a.m[6];
    out.m[3] = a.m[1]; out.m[4] = a.m[4]; out.m[5] = a.m[7];
    out.m[6] = a.m[2]; out.m[7] = a.m[5]; out.m[8] = a.m[8];
    return out;
}

inline Mat3 inverse(const Mat3& a) {
    const double c00 = a.m[4] * a.m[8] - a.m[5] * a.m[7];
    const double c01 = a.m[5] * a.m[6] - a.m[3] * a.m[8];
    const double c02 = a.m[3] * a.m[7] - a.m[4] * a.m[6];
    const double inv = 1.0 / (a.m[0] * c00 + a.m[1] * c01 + a.m[2] * c02);
    Mat3 out;
    out.m[0] = c00 * inv;
    out.m[1] = (a.m[2] * a.m[7] - a.m[1] * a.m[8]) * inv;
    out.m[2] = (a.m[1] * a.m[5] - a.m[2] * a.m[4]) * inv;
    out.m[3] = c01 * inv;
    out.m[4] = (a.m[0] * a.m[8] - a.m[2] * a.m[6]) * inv;
    out.m[5] = (a.m[2] * a.m[3] - a.m[0] * a.m[5]) * inv;
    out.m[6] = c02 * inv;
    out.m[7] = (a.m[1] * a.m[6] - a.m[0] * a.m[7]) * inv;
    out.m[8] = (a.m[0] * a.m[4] - a.m[1] * a.m[3]) * inv;
    return out;
}

inline Mat3 to_matrix(Quat q) {
    const double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    Mat3 out;
    out.m[0] = 1 - 2 * (yy + zz); out.m[1] = 2 * (xy - wz); out.m[2] = 2 * (xz + wy);
    out.m[3] = 2 * (xy + wz); out.m[4] = 1 - 2 * (xx + zz); out.m[5] = 2 * (yz - wx);
    out.m[6] = 2 * (xz - wy); out.m[7] = 2 * (yz + wx); out.m[8] = 1 - 2 * (xx + yy);
    return out;
}

// rotation * a * rotation^T: a body-frame tensor in the world frame.
inline Mat3 rotated(const Mat3& rotation, const Mat3& a) {
    return mul(mul(rotation, a), transposed(rotation));
}

// q + (h/2) (0, w) q, renormalised: the first-order orientation update.
inline Quat integrate(Quat q, Vec3 w, double h) {
    const double half = 0.5 * h;
    Quat out;
    out.w = q.w + half * (-w.x * q.x - w.y * q.y - w.z * q.z);
    out.x = q.x + half * (w.x * q.w + w.y * q.z - w.z * q.y);
    out.y = q.y + half * (w.y * q.w + w.z * q.x - w.x * q.z);
    out.z = q.z + half * (w.z * q.w + w.x * q.y - w.y * q.x);
    const double size = std::sqrt(out.w * out.w + out.x * out.x + out.y * out.y + out.z * out.z);
    out.w /= size; out.x /= size; out.y /= size; out.z /= size;
    return out;
}

}  // namespace zc::phys
