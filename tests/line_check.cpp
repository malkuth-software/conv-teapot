// Checks conv_line.hpp against lines produced by the stx_conv stream model.
#include "../src/conv_line.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace tconv;

static std::vector<double> read_row(std::istream& in) {
    std::string line;
    std::getline(in, line);
    std::istringstream row(line);
    std::vector<double> values;
    std::string token;
    while (row >> token) { values.push_back(std::strtod(token.c_str(), nullptr)); }
    return values;
}

static std::vector<double> resample(const std::vector<double>& y, std::size_t m, Bubble bubble) {
    const std::size_t n = y.size();
    std::vector<double> out(m);
    for (std::size_t r = 0; r < m; ++r) {
        const std::size_t numerator = r * (n - 1);
        std::size_t cell = numerator / (m - 1);
        double t = double(numerator % (m - 1)) / double(m - 1);
        if (cell == n - 1) { cell = n - 2; t = 1.0; }
        double w[8] = {};
        for (int k = 0; k < 8; ++k) {
            const long index = long(cell) - 3 + k;
            if (index >= 0 && index < long(n)) { w[k] = y[std::size_t(index)]; }
        }
        LineLayout layout;
        layout.nodes = n;
        layout.closure = Closure::paper;
        const Cell<double> compiled = compile_cell(w, cell_shape(layout, cell), bubble);
        out[r] = eval_cell(compiled, t);
    }
    return out;
}

int main(int argc, char** argv) {
    std::ifstream in(argc > 1 ? argv[1] : "tests/line_reference.txt");
    if (!in) { std::puts("missing reference"); return 1; }
    double worst_base = 0.0, worst_bubble = 0.0;
    int lines = 0;
    std::string header;
    while (std::getline(in, header)) {
        std::size_t n = 0, m = 0;
        std::sscanf(header.c_str(), "%zu %zu", &n, &m);
        const std::vector<double> y = read_row(in), base = read_row(in), bub = read_row(in);
        const std::vector<double> mine_base = resample(y, m, Bubble::none);
        const std::vector<double> mine_bub = resample(y, m, Bubble::standard);
        double scale = 1.0;
        for (double v : y) { scale = std::fmax(scale, std::fabs(v)); }
        for (std::size_t r = 0; r < m; ++r) {
            worst_base = std::fmax(worst_base, std::fabs(mine_base[r] - base[r]) / scale);
            worst_bubble = std::fmax(worst_bubble, std::fabs(mine_bub[r] - bub[r]) / scale);
        }
        ++lines;
    }
    std::printf("lines %d  worst scaled difference: base %.3e  standard bubble %.3e\n", lines, worst_base, worst_bubble);
    return worst_base < 1e-12 && worst_bubble < 1e-12 ? 0 : 1;
}
