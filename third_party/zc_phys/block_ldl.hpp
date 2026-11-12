#pragma once
// Macro-frame elimination: a sparse block Cholesky factorisation over the
// contact graph, six degrees of freedom per body. Eliminating a body folds its
// inertia and load into the bodies it still touches; one up sweep and one down
// sweep then solve the system exactly.
#include <cstdint>
#include <vector>

namespace zc::phys {

class BlockFactor {
public:
    // Chooses a minimum-degree elimination order for `node_count` nodes joined
    // by `pairs` (flat node index pairs) and lays out every block.
    void analyze(int node_count, const std::vector<std::int32_t>& pairs);
    void clear_blocks();
    double* diagonal_block(int node);                       // 36 doubles, row-major
    // The stored block coupling two different nodes; `transposed` is set when
    // the stored block is (node_j, node_i) rather than (node_i, node_j).
    double* coupling_block(int node_i, int node_j, bool& transposed);
    bool factorize();                                       // false if a pivot is not positive
    void solve(const std::vector<double>& rhs, std::vector<double>& x);   // six values per node
    int node_count() const { return node_count_; }

private:
    int node_count_ = 0;
    std::vector<std::int32_t> order_;          // order_[position] = node
    std::vector<std::int32_t> position_;       // position_[node] = elimination position
    std::vector<std::int32_t> row_first_;      // position p owns structure entries [row_first_[p], row_first_[p + 1])
    std::vector<std::int32_t> structure_;      // later positions coupled to p, ascending
    std::vector<std::int32_t> lookup_;         // p * node_count + q -> entry index, or -1
    std::vector<double> diagonal_;             // 36 per position; overwritten by its Cholesky factor
    std::vector<double> upper_;                // 36 per structure entry: block (p, q)
    std::vector<double> transfer_;             // 36 per structure entry: inverse(diagonal p) * upper
    std::vector<double> work_;
    std::vector<std::uint8_t> adjacency_;
    std::vector<std::int32_t> degree_;
    std::vector<std::int32_t> neighbours_;
};

}  // namespace zc::phys
