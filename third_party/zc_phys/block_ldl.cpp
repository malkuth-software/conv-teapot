#include "block_ldl.hpp"

#include <algorithm>
#include <cmath>

namespace zc::phys {

namespace {

constexpr int kB = 6;
constexpr int kBB = 36;

// In-place Cholesky of a 6x6 block: the lower triangle receives L.
bool cholesky_block(double* block) {
    for (int j = 0; j < kB; j += 1) {
        double pivot = block[j * kB + j];
        for (int k = 0; k < j; k += 1) {
            pivot -= block[j * kB + k] * block[j * kB + k];
        }
        if (!(pivot > 0)) {
            return false;
        }
        pivot = std::sqrt(pivot);
        block[j * kB + j] = pivot;
        const double inverse = 1.0 / pivot;
        for (int i = j + 1; i < kB; i += 1) {
            double value = block[i * kB + j];
            for (int k = 0; k < j; k += 1) {
                value -= block[i * kB + k] * block[j * kB + k];
            }
            block[i * kB + j] = value * inverse;
        }
    }
    return true;
}

// Solves (L L^T) x = b for six values at `values` with stride `stride`.
void cholesky_solve(const double* lower, double* values, int stride) {
    for (int i = 0; i < kB; i += 1) {
        double value = values[i * stride];
        for (int k = 0; k < i; k += 1) {
            value -= lower[i * kB + k] * values[k * stride];
        }
        values[i * stride] = value / lower[i * kB + i];
    }
    for (int i = kB - 1; i >= 0; i -= 1) {
        double value = values[i * stride];
        for (int k = i + 1; k < kB; k += 1) {
            value -= lower[k * kB + i] * values[k * stride];
        }
        values[i * stride] = value / lower[i * kB + i];
    }
}

// target -= left^T * right for 6x6 blocks.
void subtract_transpose_product(double* target, const double* left, const double* right) {
    for (int i = 0; i < kB; i += 1) {
        for (int j = 0; j < kB; j += 1) {
            double sum = 0;
            for (int k = 0; k < kB; k += 1) {
                sum += left[k * kB + i] * right[k * kB + j];
            }
            target[i * kB + j] -= sum;
        }
    }
}

}  // namespace

void BlockFactor::analyze(int node_count, const std::vector<std::int32_t>& pairs) {
    node_count_ = node_count;
    const std::size_t n = static_cast<std::size_t>(node_count);
    adjacency_.assign(n * n, 0);
    degree_.assign(n, 0);
    for (std::size_t k = 0; k + 1 < pairs.size(); k += 2) {
        const int i = pairs[k];
        const int j = pairs[k + 1];
        if (i != j && !adjacency_[i * n + j]) {
            adjacency_[i * n + j] = 1;
            adjacency_[j * n + i] = 1;
            degree_[i] += 1;
            degree_[j] += 1;
        }
    }
    order_.assign(n, 0);
    position_.assign(n, -1);
    row_first_.assign(n + 1, 0);
    structure_.clear();
    // Greedy minimum degree; degrees are kept current as fill appears. The
    // structure is first recorded as nodes, then mapped to positions.
    for (int step = 0; step < node_count; step += 1) {
        int best = -1;
        int best_degree = node_count + 1;
        for (int i = 0; i < node_count; i += 1) {
            if (position_[i] < 0 && degree_[i] < best_degree) {
                best_degree = degree_[i];
                best = i;
            }
        }
        position_[best] = step;
        order_[step] = best;
        neighbours_.clear();
        for (int j = 0; j < node_count; j += 1) {
            if (position_[j] < 0 && adjacency_[best * n + j]) {
                neighbours_.push_back(j);
            }
        }
        for (std::size_t a = 0; a < neighbours_.size(); a += 1) {
            degree_[neighbours_[a]] -= 1;
            for (std::size_t b = a + 1; b < neighbours_.size(); b += 1) {
                const int u = neighbours_[a];
                const int v = neighbours_[b];
                if (!adjacency_[u * n + v]) {
                    adjacency_[u * n + v] = 1;
                    adjacency_[v * n + u] = 1;
                    degree_[u] += 1;
                    degree_[v] += 1;
                }
            }
            structure_.push_back(neighbours_[a]);
        }
        row_first_[step + 1] = static_cast<std::int32_t>(structure_.size());
    }
    lookup_.assign(n * n, -1);
    for (int p = 0; p < node_count; p += 1) {
        for (int k = row_first_[p]; k < row_first_[p + 1]; k += 1) {
            structure_[k] = position_[structure_[k]];
        }
        std::sort(structure_.begin() + row_first_[p], structure_.begin() + row_first_[p + 1]);
        for (int k = row_first_[p]; k < row_first_[p + 1]; k += 1) {
            lookup_[static_cast<std::size_t>(p) * n + structure_[k]] = k;
        }
    }
    diagonal_.assign(n * kBB, 0);
    upper_.assign(structure_.size() * kBB, 0);
    transfer_.assign(structure_.size() * kBB, 0);
    work_.assign(n * kB, 0);
}

void BlockFactor::clear_blocks() {
    std::fill(diagonal_.begin(), diagonal_.end(), 0.0);
    std::fill(upper_.begin(), upper_.end(), 0.0);
}

double* BlockFactor::diagonal_block(int node) {
    return diagonal_.data() + static_cast<std::size_t>(position_[node]) * kBB;
}

double* BlockFactor::coupling_block(int node_i, int node_j, bool& transposed) {
    const int p = position_[node_i];
    const int q = position_[node_j];
    const std::size_t n = static_cast<std::size_t>(node_count_);
    if (p < q) {
        transposed = false;
        return upper_.data() + static_cast<std::size_t>(lookup_[p * n + q]) * kBB;
    }
    transposed = true;
    return upper_.data() + static_cast<std::size_t>(lookup_[q * n + p]) * kBB;
}

bool BlockFactor::factorize() {
    const std::size_t n = static_cast<std::size_t>(node_count_);
    for (int p = 0; p < node_count_; p += 1) {
        double* lower = diagonal_.data() + static_cast<std::size_t>(p) * kBB;
        if (!cholesky_block(lower)) {
            return false;
        }
        const int first = row_first_[p];
        const int last = row_first_[p + 1];
        for (int k = first; k < last; k += 1) {
            double* transfer = transfer_.data() + static_cast<std::size_t>(k) * kBB;
            const double* upper = upper_.data() + static_cast<std::size_t>(k) * kBB;
            for (int e = 0; e < kBB; e += 1) {
                transfer[e] = upper[e];
            }
            for (int column = 0; column < kB; column += 1) {
                cholesky_solve(lower, transfer + column, kB);
            }
        }
        for (int k1 = first; k1 < last; k1 += 1) {
            const int q = structure_[k1];
            const double* upper = upper_.data() + static_cast<std::size_t>(k1) * kBB;
            subtract_transpose_product(diagonal_.data() + static_cast<std::size_t>(q) * kBB, upper,
                                       transfer_.data() + static_cast<std::size_t>(k1) * kBB);
            for (int k2 = k1 + 1; k2 < last; k2 += 1) {
                const int entry = lookup_[static_cast<std::size_t>(q) * n + structure_[k2]];
                subtract_transpose_product(upper_.data() + static_cast<std::size_t>(entry) * kBB, upper,
                                           transfer_.data() + static_cast<std::size_t>(k2) * kBB);
            }
        }
    }
    return true;
}

void BlockFactor::solve(const std::vector<double>& rhs, std::vector<double>& x) {
    double* work = work_.data();
    for (int p = 0; p < node_count_; p += 1) {
        const int node = order_[p];
        for (int i = 0; i < kB; i += 1) {
            work[p * kB + i] = rhs[node * kB + i];
        }
    }
    // Up sweep: each eliminated body hands its load to the bodies it touches.
    for (int p = 0; p < node_count_; p += 1) {
        for (int k = row_first_[p]; k < row_first_[p + 1]; k += 1) {
            const double* transfer = transfer_.data() + static_cast<std::size_t>(k) * kBB;
            const int q = structure_[k];
            for (int j = 0; j < kB; j += 1) {
                double sum = 0;
                for (int i = 0; i < kB; i += 1) {
                    sum += transfer[i * kB + j] * work[p * kB + i];
                }
                work[q * kB + j] -= sum;
            }
        }
    }
    for (int p = 0; p < node_count_; p += 1) {
        cholesky_solve(diagonal_.data() + static_cast<std::size_t>(p) * kBB, work + p * kB, 1);
    }
    // Down sweep: each body receives the response of the bodies it handed to.
    for (int p = node_count_ - 1; p >= 0; p -= 1) {
        for (int k = row_first_[p]; k < row_first_[p + 1]; k += 1) {
            const double* transfer = transfer_.data() + static_cast<std::size_t>(k) * kBB;
            const int q = structure_[k];
            for (int i = 0; i < kB; i += 1) {
                double sum = 0;
                for (int j = 0; j < kB; j += 1) {
                    sum += transfer[i * kB + j] * work[q * kB + j];
                }
                work[p * kB + i] -= sum;
            }
        }
    }
    for (int p = 0; p < node_count_; p += 1) {
        const int node = order_[p];
        for (int i = 0; i < kB; i += 1) {
            x[node * kB + i] = work[p * kB + i];
        }
    }
}

}  // namespace zc::phys
