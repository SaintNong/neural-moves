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

TEST_CASE("Static exchange evaluation counts the recapture sequence") {
    const auto see = [](std::string_view fen, std::string_view uci) {
        const auto board = nmo::parse_board(fen);
        return nmo::static_exchange(board, chess::uci::uciToMove(board, std::string(uci)));
    };
    // Undefended pawn: the rook wins it outright.
    CHECK(see("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5") == 100);
    // Pawn takes a knight defended by a pawn: wins 320 and loses 100.
    CHECK(see("4k3/8/2p5/3n4/4P3/8/8/4K3 w - - 0 1", "e4d5") == 220);
    // Queen takes a pawn defended by a pawn: loses the queen for a pawn.
    CHECK(see("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", "d1d5") == -800);
    // Rook takes a pawn defended by a rook, backed up by a second rook: wins the pawn.
    CHECK(see("3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5") == 100);
}

TEST_CASE("Strong ordering searches losing captures after quiet moves") {
    // The queen can take a pawn defended by a pawn (losing) or move quietly.
    const auto board = nmo::parse_board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1");
    auto moves = legal_moves(board);
    nmo::MoveOrderer orderer(nmo::OrderingStrategy::strong);
    orderer.order(board, moves);
    CHECK(moves[moves.size() - 1] == chess::uci::uciToMove(board, "d1d5"));
    nmo::MoveOrderer simple(nmo::OrderingStrategy::mvvlva);
    auto by_mvv_lva = legal_moves(board);
    simple.order(board, by_mvv_lva);
    CHECK(by_mvv_lva[0] == chess::uci::uciToMove(board, "d1d5"));
}
