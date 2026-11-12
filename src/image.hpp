// Minimal PNG writer (stored deflate blocks) and the built-in textures.
#pragma once
#include "render.hpp"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace tconv {

inline bool write_png(const std::string& path, int width, int height, const std::uint8_t* rgba) {
    std::vector<std::uint8_t> raw;
    raw.reserve(std::size_t(height) * (std::size_t(width) * 3 + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        for (int x = 0; x < width; ++x) {
            const std::uint8_t* p = rgba + (std::size_t(y) * std::size_t(width) + std::size_t(x)) * 4;
            raw.push_back(p[0]);
            raw.push_back(p[1]);
            raw.push_back(p[2]);
        }
    }
    static std::uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) { c = (c & 1u) != 0 ? 0xedb88320u ^ (c >> 1) : c >> 1; }
            table[n] = c;
        }
        ready = true;
    }
    const auto crc = [&](const std::vector<std::uint8_t>& data) {
        std::uint32_t c = 0xffffffffu;
        for (std::uint8_t byte : data) { c = table[(c ^ byte) & 0xffu] ^ (c >> 8); }
        return c ^ 0xffffffffu;
    };
    std::vector<std::uint8_t> file = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    const auto put32 = [](std::vector<std::uint8_t>& out, std::uint32_t v) {
        out.push_back(std::uint8_t(v >> 24));
        out.push_back(std::uint8_t(v >> 16));
        out.push_back(std::uint8_t(v >> 8));
        out.push_back(std::uint8_t(v));
    };
    const auto chunk = [&](const char* type, const std::vector<std::uint8_t>& body) {
        put32(file, std::uint32_t(body.size()));
        std::vector<std::uint8_t> typed(type, type + 4);
        typed.insert(typed.end(), body.begin(), body.end());
        file.insert(file.end(), typed.begin(), typed.end());
        put32(file, crc(typed));
    };
    std::vector<std::uint8_t> header;
    put32(header, std::uint32_t(width));
    put32(header, std::uint32_t(height));
    header.insert(header.end(), {8, 2, 0, 0, 0});
    chunk("IHDR", header);
    std::vector<std::uint8_t> z = {0x78, 0x01};
    std::uint32_t a = 1, b = 0;
    for (std::size_t offset = 0; offset < raw.size();) {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - offset);
        z.push_back(offset + n == raw.size() ? 1 : 0);
        z.push_back(std::uint8_t(n));
        z.push_back(std::uint8_t(n >> 8));
        z.push_back(std::uint8_t(~n));
        z.push_back(std::uint8_t((~n) >> 8));
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint8_t byte = raw[offset + k];
            z.push_back(byte);
            a = (a + byte) % 65521u;
            b = (b + a) % 65521u;
        }
        offset += n;
    }
    put32(z, (b << 16) | a);
    chunk("IDAT", z);
    chunk("IEND", {});
    std::FILE* out = std::fopen(path.c_str(), "wb");
    if (out == nullptr) { return false; }
    const bool ok = std::fwrite(file.data(), 1, file.size(), out) == file.size();
    std::fclose(out);
    return ok;
}

inline std::unique_ptr<Texture> make_texture(std::size_t width, std::size_t height, const std::vector<Vec3>& texels) {
    auto texture = std::make_unique<Texture>();
    Lattice& lattice = texture->conv.lattice;
    lattice.nu = width;
    lattice.nv = height;
    lattice.wrap_u = true;
    lattice.closure = Closure::paper;
    lattice.nodes = texels;
    texture->conv.bubble = Bubble::continuous;
    texture->conv.compile();
    return texture;
}

// Built-in low-resolution textures; the renderer magnifies them.
inline std::vector<std::unique_ptr<Texture>> make_textures() {
    std::vector<std::unique_ptr<Texture>> textures;
    {   // 1: checker, two texels per square
        const std::size_t w = 16, h = 16;
        std::vector<Vec3> t(w * h);
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                t[y * w + x] = (((x / 2) + (y / 2)) & 1) != 0 ? Vec3{0.93, 0.91, 0.86} : Vec3{0.13, 0.27, 0.52};
            }
        }
        textures.push_back(make_texture(w, h, t));
    }
    {   // 2: "UTAH" in a 5 by 7 bitmap face, two texels per dot, between glaze bands
        static const char* rows[7] = {
            "#...#.#####..###..#...#",
            "#...#...#...#...#.#...#",
            "#...#...#...#...#.#...#",
            "#...#...#...#####.#####",
            "#...#...#...#...#.#...#",
            "#...#...#...#...#.#...#",
            ".###....#...#...#.#...#"};
        const std::size_t w = 64, h = 96;
        std::vector<Vec3> t(w * h, Vec3{0.94, 0.92, 0.87});
        for (std::size_t x = 0; x < w; ++x) {
            for (std::size_t y : {22u, 23u, 49u, 50u}) { t[y * w + x] = Vec3{0.72, 0.16, 0.12}; }
        }
        for (std::size_t y = 0; y < 14; ++y) {
            for (std::size_t x = 0; x < 46; ++x) {
                if (rows[y / 2][x / 2] == '#') { t[(y + 29) * w + x + 9] = Vec3{0.10, 0.22, 0.48}; }
            }
        }
        textures.push_back(make_texture(w, h, t));
    }
    {   // 3: glaze bands with a slow hue drift
        const std::size_t w = 24, h = 12;
        std::vector<Vec3> t(w * h);
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                const double a = 6.2831853 * double(x) / double(w);
                const double band = 0.5 + 0.5 * std::sin(1.7 * double(y) + 2.0 * std::sin(a));
                t[y * w + x] = Vec3{0.25 + 0.6 * band, 0.45 + 0.35 * std::cos(a + 0.4 * double(y)) * 0.5 + 0.1 * band, 0.55 + 0.35 * (1.0 - band)};
            }
        }
        textures.push_back(make_texture(w, h, t));
    }
    return textures;
}

} // namespace tconv
