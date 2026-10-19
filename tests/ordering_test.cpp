#include <array>
#include <string_view>

#include <doctest.h>

#include "support.hpp"

using namespace test_support;

TEST_CASE("Ordering preserves every legal move exactly once") {
    for (const auto fen : {std::string_view(chess::constants::STARTPOS), free_queen,
                           castling, en_passant, promotion}) {
        CAPTURE(fen);
        const auto board = nmo::parse_board(fen);
        const auto legal = legal_moves(board);
        const auto expected = move_set(legal);
        REQUIRE_FALSE(legal.empty());
        for (const auto strategy : nmo::heuristic_strategies) {
            INFO("strategy: ", nmo::strategy_name(strategy));
            nmo::MoveOrderer orderer(strategy);
            auto ordered = legal;
            orderer.order(board, ordered);
            CHECK(move_set(ordered) == expected);
        }
    }
}
