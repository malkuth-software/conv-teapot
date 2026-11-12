// The Utah Teapot as four node lattices.
//
// Source: the original 28 bicubic Bezier patches of Martin Newell with Jim
// Blinn's scaling (data/teapot_bezier.txt, Utah Model Repository). Each part
// is closed around (u, periodic) and open along (v):
//   body   - rim, upper body, lower body (4 patches around, 3 along), and
//            optionally a flat bottom cap, which is not one of the 28 patches
//   lid    - knob, lid (4 around, 2 along)
//   handle - two half tubes around, 2 along
//   spout  - two half tubes around, 2 along
// The CONV teapot keeps only points ON the Bezier surface: per patch,
// `per_u` by `per_v` intervals of nodes.
#pragma once
#include "surface.hpp"
#include "blend.hpp"
#include "unified.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace tconv {

inline const std::array<std::array<Vec3, 16>, 28>& teapot_patches() {
    static const double raw[28][16][3] = {
#include "teapot_patches.inc"
    };
    static const std::array<std::array<Vec3, 16>, 28> patches = [] {
        std::array<std::array<Vec3, 16>, 28> out{};
        for (std::size_t p = 0; p < 28; ++p) {
            for (std::size_t k = 0; k < 16; ++k) { out[p][k] = {raw[p][k][0], raw[p][k][1], raw[p][k][2]}; }
        }
        return out;
    }();
    return patches;
}

// The versions of the Utah Teapot offered here. 1975 to 1987 are built from
// the original Bezier data; 2026 is built from points sampled on the surface
// of the official 2026 model (without its trimming and chamfers).
// v2026_unified: the 2026 shape as whole sheets (body with its interior, lid
// with its underside) instead of one lattice per source patch.
enum class Version : int { v1975 = 0, v1976 = 1, v1987 = 2, v2026 = 3, v2026_unified = 4 };

namespace utah {
#include "utah_samples.inc"
#include "utah_dense.inc"
}

struct TeapotOptions {
    Version version = Version::v1976;
    // Blinn's vertical scale of 3/4. Off gives Newell's taller proportions.
    bool blinn_scale = true;
    std::size_t per_u = 4;  // node intervals per patch, around
    std::size_t per_v = 4;  // node intervals per patch, along
    bool bottom_cap = true; // flat cap closing the body (Blinn's 1987 suggestion)
    bool seams = true;      // cut the stencils at the Bezier patch boundaries
    Closure closure = Closure::cubic;
    Bubble bubble = Bubble::continuous;
    // Round: every ring of the body and lid lies on a true circle and is
    // closed without seams, so the parts are round to the accuracy of a CONV
    // ring. The profile along v is unchanged.
    bool circular = false;
    // Smooth (experimental): seams only at true creases along v, with graded
    // row spacing between them. It trades fidelity at tight features for
    // continuity across the patch joints; see tools/qualities.cpp.
    bool smooth = false;
    // Unified version: blend the handle and spout into the body.
    bool joins = true;
    // Unified sheets: rows along the profile and nodes around.
    // The lid takes three quarters of the body's rows, the spout half, the
    // handle a third; the tubes have their own count around. The defaults
    // hold every sheet within about 1e-3 of its source (tools/unified_study).
    std::size_t sheet_rows = 72, sheet_around = 12, tube_around = 16;
    // Unified sheets: place rows by CONV's own error (plan_rows_by_error)
    // instead of by the profile's radius of curvature.
    bool rows_by_error = true;
    // Unified sheets: node rows across a join's band, where it meets the outer
    // wall and where the spout's channel meets the inner wall.
    // join_two_sided: the band's two sides are stored as separate lattices of
    // join_rows rows each and blended in closed form at evaluation. Otherwise
    // the blended band itself is sampled, with join_rows_sampled rows where
    // it meets the outer wall and join_rows_interior at the inner wall.
    bool join_two_sided = true;
    int join_rows = 3, join_rows_sampled = 9, join_rows_interior = 17;
};

struct TeapotPart {
    std::string name;
    BezierSurface bezier;  // the reference surface
    ConvSurface conv;      // CONV geometry through nodes of the reference
    LinearSurface linear;  // the same nodes, joined bilinearly
    Lattice control;       // a two-sided band has no one mesh of nodes: samples of it, for `linear`
    double orientation = 1.0; // sign making (du x dv) point outward
    std::size_t per_u = 1, per_v = 1;
    std::string group = "body"; // body, lid, handle or spout: what the part toggles act on
    bool interior = false;      // an inner surface (2026 version)
    // Unified sheets: rows from this one on face the cavity (the body's inside,
    // the lid's underside, the spout's channel). Beyond the last row: none.
    double interior_from = 1e30;
    bool sampled = false;       // built from surface samples; `bezier` is empty
    int hull = -2;              // collision: -2 cut along v (see sim.hpp), -1 none, >= 0 a hull group
    // The reference parameter (Bezier node units) of every lattice row, and
    // the arc-length table that carries a fractional row to it. In the
    // exact style a row's parameter is its index.
    std::vector<double> row_param;
    std::vector<double> arc_param, arc_length; // fine samples along v
    bool revolved = false;                     // rings are true circles

    // Reference parameter of the lattice coordinate v.
    double reference_v(double v) const {
        if (arc_param.empty()) { return v; }
        const std::size_t rows = row_param.size();
        const double clamped = v < 0.0 ? 0.0 : v > double(rows - 1) ? double(rows - 1) : v;
        std::size_t row = std::size_t(clamped);
        if (row + 1 >= rows) { row = rows - 2; }
        const double t = clamped - double(row);
        const double arc = arc_at(row_param[row]) * (1.0 - t) + arc_at(row_param[row + 1]) * t;
        return param_at(arc);
    }
    // The reference surface point that the lattice coordinate (u, v) stands for.
    Vec3 reference(double u, double v) const {
        const double rv = reference_v(v);
        if (!revolved) { return bezier.position(u, rv); }
        const Vec3 profile = bezier.position(0.0, rv);
        const double angle = 6.283185307179586 * u / double(bezier.cells_u());
        return {profile.x * std::cos(angle), profile.y, profile.x * std::sin(angle)};
    }
    double arc_at(double param) const {
        const double x = param / arc_step_;
        std::size_t k = std::size_t(x);
        if (k + 1 >= arc_length.size()) { return arc_length.back(); }
        return arc_length[k] + (arc_length[k + 1] - arc_length[k]) * (x - double(k));
    }
    double param_at(double arc) const {
        std::size_t low = 0, high = arc_length.size() - 1;
        while (high - low > 1) {
            const std::size_t mid = (low + high) / 2;
            if (arc_length[mid] <= arc) { low = mid; } else { high = mid; }
        }
        const double span = arc_length[high] - arc_length[low];
        return (double(low) + (span > 0.0 ? (arc - arc_length[low]) / span : 0.0)) * arc_step_;
    }
    double arc_step_ = 1.0;
};

// A blended join: which parts it trims, and the group it belongs to.
struct TeapotJoin {
    std::string group;            // handle or spout
    std::size_t body_part = 0, tube_part = 0;
    Trim body_trim, tube_trim;
};

struct Teapot {
    std::vector<std::unique_ptr<TeapotPart>> parts;
    std::vector<TeapotJoin> joins;
    std::size_t node_count() const {
        std::size_t total = 0;
        for (const auto& part : parts) { total += part->conv.node_count(); }
        return total;
    }
};

namespace detail {

// One patch of surface samples (9 by 9, row-major) as an open lattice.
inline std::unique_ptr<TeapotPart> make_sampled_part(const std::string& tag, const float (*points)[3], double y_scale,
                                                     const TeapotOptions& options, int hull) {
    auto part = std::make_unique<TeapotPart>();
    const std::size_t full = std::size_t(utah::utah_res);
    const std::size_t per = options.per_u >= full ? full : full / 2; // 8 or 4 intervals
    const std::size_t stride = full / per;
    part->name = tag;
    part->interior = tag.find("_interior") != std::string::npos;
    part->group = tag.substr(0, tag.find('_'));
    if (part->group == "bottom") { part->group = "body"; }
    part->sampled = true;
    part->hull = part->interior ? -1 : hull;
    part->per_u = part->per_v = per;
    Lattice& lattice = part->conv.lattice;
    lattice.nu = lattice.nv = per + 1;
    lattice.wrap_u = false;
    lattice.closure = Closure::cubic;
    lattice.nodes.resize(lattice.nu * lattice.nv);
    for (std::size_t v = 0; v <= per; ++v) {
        for (std::size_t u = 0; u <= per; ++u) {
            const float* p = points[(v * stride) * (full + 1) + u * stride];
            lattice.at(u, v) = {double(p[0]), double(p[1]) * y_scale, double(p[2])};
        }
    }
    part->row_param.resize(lattice.nv);
    for (std::size_t v = 0; v < lattice.nv; ++v) { part->row_param[v] = double(v); }
    part->conv.bubble = options.bubble;
    part->conv.compile();
    part->linear.lattice = &part->conv.lattice;
    return part;
}

inline std::unique_ptr<TeapotPart> make_part(const char* name, std::size_t first, std::size_t around, std::size_t along,
                                             const TeapotOptions& options, bool cap, double y_scale = 1.0, double y_lift = 0.0) {
    auto part = std::make_unique<TeapotPart>();
    part->name = name;
    part->group = name;
    part->per_u = options.per_u;
    part->per_v = options.per_v;
    BezierSurface& bezier = part->bezier;
    bezier.patches_u = around;
    bezier.patches_v = along + (cap ? 1 : 0);
    bezier.per_u = options.per_u;
    bezier.per_v = options.per_v;
    bezier.wrap_u = true;
    for (std::size_t b = 0; b < along; ++b) {
        for (std::size_t a = 0; a < around; ++a) {
            std::array<Vec3, 16> patch = teapot_patches()[first + b * around + a];
            for (Vec3& control : patch) { control.y = (control.y + y_lift) * y_scale; }
            bezier.patches.push_back(patch);
        }
    }
    if (cap) {
        // A patch linear along v from the last edge of the body to the
        // centre of that edge's ring: a flat disc, with a crease at its rim.
        Vec3 centre{};
        for (std::size_t a = 0; a < around; ++a) {
            const auto& edge = bezier.patches[(along - 1) * around + a];
            for (int i = 0; i < 4; ++i) { centre = centre + edge[std::size_t(i * 4 + 3)]; }
        }
        centre = centre * (1.0 / double(4 * around));
        centre.x = 0.0;
        centre.z = 0.0;
        for (std::size_t a = 0; a < around; ++a) {
            const auto& edge = bezier.patches[(along - 1) * around + a];
            std::array<Vec3, 16> patch;
            for (int i = 0; i < 4; ++i) {
                const Vec3 rim = edge[std::size_t(i * 4 + 3)];
                for (int j = 0; j < 4; ++j) { patch[std::size_t(i * 4 + j)] = rim + (centre - rim) * (double(j) / 3.0); }
            }
            bezier.patches.push_back(patch);
        }
    }
    Lattice& lattice = part->conv.lattice;
    lattice.wrap_u = true;
    lattice.nu = bezier.cells_u();
    lattice.nv = bezier.cells_v() + 1;
    lattice.nodes.resize(lattice.nu * lattice.nv);
    lattice.closure = options.closure;
    part->row_param.resize(lattice.nv);
    for (std::size_t v = 0; v < lattice.nv; ++v) { part->row_param[v] = double(v); }
    if (!options.smooth) {
        part->revolved = options.circular && around == 4;
        for (std::size_t v = 0; v < lattice.nv; ++v) {
            for (std::size_t u = 0; u < lattice.nu; ++u) { lattice.at(u, v) = part->reference(double(u), double(v)); }
        }
        if (options.seams) {
            lattice.seams_v.assign(lattice.nv, 0);
            for (std::size_t v = 0; v < lattice.nv; v += options.per_v) { lattice.seams_v[v] = 1; }
            if (!part->revolved) {
                lattice.seams_u.assign(lattice.nu, 0);
                for (std::size_t u = 0; u < lattice.nu; u += options.per_u) { lattice.seams_u[u] = 1; }
            }
        }
    } else {
        // Arc length along v, averaged over samples around the part.
        const int fine = 64, around_samples = 16;
        part->arc_step_ = double(options.per_v) / fine;
        const std::size_t count = bezier.patches_v * std::size_t(fine) + 1;
        part->arc_length.assign(count, 0.0);
        for (std::size_t k = 1; k < count; ++k) {
            double sum = 0.0;
            for (int a = 0; a < around_samples; ++a) {
                const double u = double(bezier.cells_u()) * a / around_samples;
                sum += length(bezier.position(u, double(k) * part->arc_step_) - bezier.position(u, double(k - 1) * part->arc_step_));
            }
            part->arc_length[k] = part->arc_length[k - 1] + sum / around_samples;
        }
        part->arc_param.assign(1, 0.0); // marks the table as present
        // A patch joint is a crease when the tangent along v turns there.
        lattice.seams_v.assign(lattice.nv, 0);
        std::vector<std::size_t> breaks{0};
        for (std::size_t joint = 1; joint < bezier.patches_v; ++joint) {
            const double v = double(joint * options.per_v);
            double worst = 1.0;
            for (int a = 0; a < around_samples; ++a) {
                const double u = double(bezier.cells_u()) * (a + 0.37) / around_samples;
                const Vec3 before = normalize(bezier.position(u, v) - bezier.position(u, v - 1e-4));
                const Vec3 after = normalize(bezier.position(u, v + 1e-4) - bezier.position(u, v));
                worst = std::fmin(worst, dot(before, after));
            }
            if (worst < std::cos(4.0 * 3.14159265358979 / 180.0)) { breaks.push_back(joint); }
        }
        breaks.push_back(bezier.patches_v);
        // Between creases: start from the Bezier layout (each patch cut into
        // `per_v` equal arcs, so small features keep their share of rows),
        // then grade the interval lengths so that they change gradually
        // across the patch joints. CONV differentiates in the row index; a
        // sudden change of spacing would read as a kink.
        for (std::size_t span = 0; span + 1 < breaks.size(); ++span) {
            const std::size_t first_row = breaks[span] * options.per_v, last_row = breaks[span + 1] * options.per_v;
            lattice.seams_v[first_row] = 1;
            lattice.seams_v[last_row] = 1;
            const std::size_t intervals = last_row - first_row;
            std::vector<double> log_length(intervals);
            for (std::size_t k = 0; k < intervals; ++k) {
                const std::size_t patch = (first_row + k) / options.per_v;
                const double arc = part->arc_at(double((patch + 1) * options.per_v)) - part->arc_at(double(patch * options.per_v));
                log_length[k] = std::log(arc / double(options.per_v));
            }
            for (int pass = 0; pass < int(options.per_v) * 2; ++pass) {
                std::vector<double> next(log_length);
                for (std::size_t k = 0; k < intervals; ++k) {
                    const double left = log_length[k == 0 ? 0 : k - 1], right = log_length[k + 1 == intervals ? k : k + 1];
                    next[k] = 0.25 * left + 0.5 * log_length[k] + 0.25 * right;
                }
                log_length.swap(next);
            }
            const double arc_a = part->arc_at(double(first_row)), arc_b = part->arc_at(double(last_row));
            double total = 0.0;
            for (double value : log_length) { total += std::exp(value); }
            double arc = arc_a;
            for (std::size_t k = 0; k <= intervals; ++k) {
                const std::size_t row = first_row + k;
                part->row_param[row] = k == 0 ? double(first_row) : k == intervals ? double(last_row) : part->param_at(arc);
                if (k < intervals) { arc += std::exp(log_length[k]) * (arc_b - arc_a) / total; }
            }
        }
        part->revolved = around == 4;
        for (std::size_t v = 0; v < lattice.nv; ++v) {
            for (std::size_t u = 0; u < lattice.nu; ++u) { lattice.at(u, v) = part->reference(double(u), double(v)); }
        }
    }
    part->conv.bubble = options.bubble;
    part->conv.compile();
    part->linear.lattice = &part->conv.lattice;
    // Outward orientation: compare the normal with the direction away from
    // the centroid of the ring through a mid-surface point.
    const double v_mid = 0.37 * double(bezier.cells_v());
    Vec3 ring{};
    const int samples = 32;
    for (int k = 0; k < samples; ++k) { ring = ring + bezier.eval(double(bezier.cells_u()) * k / samples, v_mid).p; }
    ring = ring * (1.0 / samples);
    const SurfacePoint probe = bezier.eval(0.3 * double(bezier.cells_u()), v_mid);
    part->orientation = dot(cross(probe.du, probe.dv), probe.p - ring) >= 0.0 ? 1.0 : -1.0;
    return part;
}

} // namespace detail

inline Teapot make_teapot(const TeapotOptions& options) {
    Teapot teapot;
    const double y_scale = options.blinn_scale ? 1.0 : 4.0 / 3.0;
    if (options.version == Version::v2026_unified) {
        // Body and lid: one sheet each, from the meridian of the sampled patches.
        for (const char* group : {"body", "lid"}) {
            const bool is_body = std::string(group) == "body";
            const int run_count = is_body ? utah::dense_body_runs : utah::dense_lid_runs;
            std::vector<ProfileCurve::Run> runs;
            for (int r = 0; r < run_count; ++r) {
                ProfileCurve::Run run;
                for (int k = 0; k < utah::dense_profile_points; ++k) {
                    const float* p = is_body ? utah::dense_body_profile[r][k] : utah::dense_lid_profile[r][k];
                    run.x.push_back(double(p[0]));
                    run.y.push_back(double(p[1]) * y_scale);
                }
                runs.push_back(run);
            }
            const ProfileCurve profile = chain_runs(runs);
            const std::size_t rows = std::string(group) == "body" ? options.sheet_rows : options.sheet_rows * 3 / 4;
            const SheetPlan plan = options.rows_by_error ? plan_rows_by_error(profile, rows, true)
                                                         : plan_rows(profile, rows, profile.length / 3000.0, profile.length / 30.0);
            auto part = std::make_unique<TeapotPart>();
            part->name = group;
            part->group = group;
            part->sampled = true;
            part->hull = std::string(group) == "body" ? 0 : 1;
            revolve_sheet(profile, plan, options.sheet_around, part->conv.lattice);
            part->per_u = part->per_v = 0; // no patch outlines inside a sheet
            part->row_param.assign(plan.row_arc.begin(), plan.row_arc.end());
            {
                // The body turns inward at the top of its rim, the lid at its widest row.
                const Lattice& l = part->conv.lattice;
                std::size_t turn = 0;
                for (std::size_t v = 0; v < l.nv; ++v) {
                    const double here = is_body ? l.at(0, v).y : std::hypot(l.at(0, v).x, l.at(0, v).z);
                    const double best = is_body ? l.at(0, turn).y : std::hypot(l.at(0, turn).x, l.at(0, turn).z);
                    if (here > best) { turn = v; }
                }
                part->interior_from = double(turn);
            }
            part->conv.bubble = options.bubble;
            part->conv.compile();
            part->linear.lattice = &part->conv.lattice;
            teapot.parts.push_back(std::move(part));
        }
        // Handle and spout: one tube sheet each, the spout with its interior.
        for (const char* group : {"handle", "spout"}) {
            const bool is_handle = std::string(group) == "handle";
            const int count = is_handle ? utah::dense_handle_patches : utah::dense_spout_patches;
            std::vector<const float (*)[3]> patches;
            for (int k = 0; k < count; ++k) { patches.push_back(is_handle ? utah::dense_handle_points[k] : utah::dense_spout_points[k]); }
            auto part = std::make_unique<TeapotPart>();
            part->name = group;
            part->group = group;
            part->sampled = true;
            part->hull = is_handle ? 2 : 3;
            tube_sheet(patches.data(), count, utah::dense_tube_points, is_handle ? options.sheet_rows / 3 : options.sheet_rows / 2,
                       std::max<std::size_t>(8, (options.tube_around + 1) / 2 * 2), part->conv.lattice, part->row_param, options.rows_by_error);
            for (Vec3& node : part->conv.lattice.nodes) { node.y *= y_scale; }
            if (!is_handle) {
                // The spout turns into its channel at the lip, its furthest row from the axis.
                const Lattice& l = part->conv.lattice;
                std::size_t lip = 0;
                double far = -1.0;
                for (std::size_t v = 0; v < l.nv; ++v) {
                    double reach = 0.0;
                    for (std::size_t u = 0; u < l.nu; ++u) { reach += std::hypot(l.at(u, v).x, l.at(u, v).z); }
                    if (reach > far) {
                        far = reach;
                        lip = v;
                    }
                }
                // The channel is the stretch of rows on the narrower side of the lip.
                const auto ring = [&](std::size_t v) { return length(l.at(0, v) - l.at(l.nu / 2, v)); };
                part->interior_from = ring(0) >= ring(l.nv - 1) ? double(lip) : -double(lip) - 1.0; // negative: rows before the lip
            }
            part->per_u = part->per_v = 0;
            part->conv.bubble = options.bubble;
            part->conv.compile();
            part->linear.lattice = &part->conv.lattice;
            teapot.parts.push_back(std::move(part));
        }
        if (options.joins) {
            const Surface& body = teapot.parts[0]->conv;
            const Lattice& body_lattice = teapot.parts[0]->conv.lattice;
            // The body's height rises along its rows to the top of the rim and falls after it.
            std::size_t top = 0;
            for (std::size_t v = 0; v < body_lattice.nv; ++v) {
                if (body_lattice.at(0, v).y > body_lattice.at(0, top).y) { top = v; }
            }
            BodyBranch exterior{&body, 0.0, double(top), true};
            BodyBranch interior{&body, double(top), double(body_lattice.nv - 1), false};
            const std::size_t part_count = teapot.parts.size();
            for (std::size_t index = 2; index < part_count; ++index) {
                const std::string group = teapot.parts[index]->group;
                const Surface& tube = teapot.parts[index]->conv;
                // The wider end of the spout is its exterior base; the other end is the channel into the cavity.
                const auto ring_size = [&](double w) { return length(tube.position(0.0, w) - tube.position(0.5 * double(tube.cells_u()), w)); };
                const bool first_is_wider = ring_size(0.0) >= ring_size(double(tube.cells_v()));
                for (int end = 0; end < 2; ++end) {
                    const bool at_first = end == 0;
                    const bool exterior_end = group == "handle" || (at_first == first_is_wider);
                    const double width = (group == "handle" ? 0.11 : exterior_end ? 0.14 : 0.07) * y_scale;
                    // At either end the part of the tube to remove is the part nearer the
                    // axis than the wall it meets: inside the body at the spout's base,
                    // inside the cavity at the end of its channel. The channel is cut where
                    // it crosses the inner wall and blended to that wall there.
                    JoinResult made = make_join(body, exterior_end ? exterior : interior, true, tube, at_first, width, body_lattice.nu, 
                                                options.join_two_sided ? options.join_rows : exterior_end ? options.join_rows_sampled : options.join_rows_interior, options.join_two_sided);
                    if (!made.ok) { continue; }
                    auto part = std::make_unique<TeapotPart>();
                    part->name = group + "_join";
                    if (!exterior_end) { part->interior_from = 0.0; }
                    part->group = group;
                    part->sampled = true;
                    part->hull = -1;
                    part->conv.lattice = made.blend;
                    if (options.join_two_sided) {
                        part->conv.toward = std::make_shared<ConvSurface>();
                        part->conv.toward->lattice = made.toward;
                        part->conv.toward->bubble = options.bubble;
                        part->conv.spread = 4;
                    }
                    part->per_u = part->per_v = 0;
                    part->row_param.assign(part->conv.lattice.nv, 0.0);
                    part->conv.bubble = options.bubble;
                    part->conv.compile();
                    part->linear.lattice = &part->conv.lattice;
                    if (options.join_two_sided) {
                        part->control = made.blend;
                        part->control.nv = 9;
                        part->control.nodes.resize(part->control.nu * 9);
                        for (std::size_t j = 0; j < 9; ++j) {
                            for (std::size_t k = 0; k < part->control.nu; ++k) {
                                part->control.at(k, j) = part->conv.position(double(k), double(part->conv.cells_v()) * double(j) / 8.0);
                            }
                        }
                        part->linear.lattice = &part->control;
                    }
                    teapot.parts.push_back(std::move(part));
                    TeapotJoin join;
                    join.group = group;
                    join.body_part = 0;
                    join.tube_part = index;
                    join.body_trim = made.body_trim;
                    join.tube_trim = made.tube_trim;
                    teapot.joins.push_back(join);
                }
            }
        }
        return teapot;
    }
    if (options.version == Version::v2026) {
        // Collision hulls: one each for the body, the lid, the handle and the
        // spout. (A hull per half-tube patch is too thin for the hull builder.)
        for (int k = 0; k < utah::utah2026_count; ++k) {
            const std::string tag = utah::utah2026_tags[k];
            const int hull = tag == "body" ? 0 : tag == "lid" ? 1 : tag == "handle" ? 2 : 3;
            teapot.parts.push_back(detail::make_sampled_part(tag, utah::utah2026_points[k], y_scale, options, hull));
        }
        return teapot;
    }
    const bool crow = options.version == Version::v1987;
    const double lift = crow ? double(utah::utah1987_lift) : 0.0;
    const bool cap = options.bottom_cap && !crow;
    teapot.parts.push_back(detail::make_part("body", 0, 4, 3, options, cap, y_scale, lift));
    teapot.parts.push_back(detail::make_part("lid", 20, 4, 2, options, false, y_scale, lift));
    teapot.parts.push_back(detail::make_part("handle", 12, 2, 2, options, false, y_scale, lift));
    teapot.parts.push_back(detail::make_part("spout", 16, 2, 2, options, false, y_scale, lift));
    if (crow) {
        // Crow's round bottom: four patches, taken as surface samples. They
        // join the hull of the body, so they need none of their own.
        for (int k = 0; k < utah::utah1987_bottom_count; ++k) {
            teapot.parts.push_back(detail::make_sampled_part("bottom", utah::utah1987_bottom_points[k], y_scale, options, -1));
        }
    }
    return teapot;
}

} // namespace tconv
