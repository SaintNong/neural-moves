#include <cstdint>

#include <doctest.h>

#include "nmo/evaluation.hpp"
#include "nmo/positions.hpp"

TEST_CASE("Material evaluation reflects advantage from the side to move") {
    const auto white_to_move = nmo::parse_board("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    const auto black_to_move = nmo::parse_board("7k/8/8/8/8/8/R7/K7 b - - 0 1");
    CHECK(nmo::evaluate_material(white_to_move) > 0);
    CHECK(nmo::evaluate_material(black_to_move) == -nmo::evaluate_material(white_to_move));
    CHECK(nmo::evaluate_material(chess::Board{}) == 0);
}

TEST_CASE("Adding material changes the evaluation in the owning side's favor") {
    const auto base = nmo::parse_board("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    const auto friendly_pawn = nmo::parse_board("7k/8/8/8/8/8/RP6/K7 w - - 0 1");
    const auto enemy_pawn = nmo::parse_board("7k/6p1/8/8/8/8/R7/K7 w - - 0 1");
    CHECK(nmo::evaluate_material(friendly_pawn) > nmo::evaluate_material(base));
    CHECK(nmo::evaluate_material(enemy_pawn) < nmo::evaluate_material(base));
}

TEST_CASE("Effective branching factor reconstructs a uniform search tree") {
    for (const std::uint64_t branching : {1U, 2U, 4U}) {
        for (const int depth : {1, 3, 5}) {
            CAPTURE(branching);
            CAPTURE(depth);
            std::uint64_t nodes = 1;
            std::uint64_t level = 1;
            for (int ply = 0; ply < depth; ++ply) {
                level *= branching;
                nodes += level;
            }
            CHECK(nmo::estimate_ebf(nodes, depth) == doctest::Approx(static_cast<double>(branching)));
        }
    }
}
