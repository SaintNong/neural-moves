#include <array>
#include <string>
#include <string_view>

#include <doctest.h>

#include "support.hpp"

using namespace test_support;

TEST_CASE("Search takes an undefended queen") {
    const auto board = nmo::parse_board(free_queen);
    const auto winning_capture = move_from_uci(board, "d2d5");
    for_each_searcher([&](auto& searcher, auto strategy) {
        const auto result = searcher.search(board, 3, strategy);
        CHECK(result.best_move == winning_capture);
    });
}

TEST_CASE("Search accounts for the opponent's reply before taking material") {
    const auto board = nmo::parse_board(poisoned_rook);
    // Qxd5 wins a rook immediately, but the e6 pawn can then take the queen.
    const auto losing_capture = move_from_uci(board, "d2d5");
    for_each_searcher([&](auto& searcher, auto strategy) {
        const auto result = searcher.search(board, 2, strategy);
        REQUIRE(board.isLegal(result.best_move));
        CHECK(result.best_move != losing_capture);
    });
}

TEST_CASE("Search delivers checkmate when a mate in one is available") {
    const auto board = nmo::parse_board(mate_in_one);
    for_each_searcher([&](auto& searcher, auto strategy) {
        const auto result = searcher.search(board, 2, strategy);
        REQUIRE(board.isLegal(result.best_move));
        auto after = board;
        after.makeMove(result.best_move);
        CHECK(after.inCheck());
        CHECK(legal_moves(after).empty());
    });
}

TEST_CASE("Search plays the sole legal reply to check") {
    const auto board = nmo::parse_board("7k/6Q1/8/4K3/8/8/8/8 b - - 0 1");
    const auto replies = legal_moves(board);
    REQUIRE(board.inCheck());
    REQUIRE(replies.size() == 1);
    for_each_searcher([&](auto& searcher, auto strategy) {
        CHECK(searcher.search(board, 3, strategy).best_move == replies.front());
    });
}

TEST_CASE("Search can win material through en passant and promotion") {
    for (const auto& [fen, winning_move] : std::array{
             std::pair{en_passant, std::string_view("e5d6")},
             std::pair{promotion, std::string_view("a7a8q")}}) {
        CAPTURE(fen);
        const auto board = nmo::parse_board(fen);
        const auto expected = move_from_uci(board, winning_move);
        for_each_searcher([&](auto& searcher, auto strategy) {
            CHECK(searcher.search(board, 2, strategy).best_move == expected);
        });
    }
}

TEST_CASE("Ordering and caching preserve the search value and caller's position") {
    // These fixtures include tied moves, tactical replies and special moves.
    // Compare values and legality, allowing each strategy to break ties freely.
    for (const auto fen : {std::string_view(chess::constants::STARTPOS), free_queen,
                           poisoned_rook, castling, en_passant, promotion}) {
        CAPTURE(fen);
        const auto board = nmo::parse_board(fen);
        const auto original_fen = board.getFen();
        const auto original_moves = move_set(legal_moves(board));
        nmo::AlphaBetaSearcher reference({0, false});
        const auto expected = reference.search(board, 4).score_cp;
        for_each_searcher([&](auto& searcher, auto strategy) {
            const auto result = searcher.search(board, 4, strategy);
            CHECK(result.score_cp == expected);
            CHECK(board.isLegal(result.best_move));
            CHECK(board.getFen() == original_fen);
            CHECK(move_set(legal_moves(board)) == original_moves);
        });
    }
}

TEST_CASE("Earlier searches cannot change a later position's result") {
    const auto target = nmo::parse_board(poisoned_rook);
    nmo::AlphaBetaSearcher fresh;
    const auto expected = fresh.search(target, 2).score_cp;
    for_each_searcher([&](auto& searcher, auto strategy) {
        (void)searcher.search(target, 4, strategy);
        (void)searcher.search(nmo::parse_board(free_queen), 3, strategy);
        const auto result = searcher.search(target, 2, strategy);
        CHECK(result.score_cp == expected);
        CHECK(target.isLegal(result.best_move));
    });
}

TEST_CASE("Terminal positions have no selected move") {
    SUBCASE("Checkmate is a loss for the side to move") {
        const auto board = nmo::parse_board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
        for_each_searcher([&](auto& searcher, auto strategy) {
            const auto result = searcher.search(board, 3, strategy);
            CHECK(result.best_move == chess::Move::NO_MOVE);
            CHECK(result.score_cp < 0);
        });
    }
    SUBCASE("Stalemate and insufficient material are draws") {
        for (const auto fen : {
                 "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  // Stalemate.
                 "7k/8/8/8/8/8/N7/K7 w - - 0 1",    // King and knight against king.
                 "7k/8/8/6B1/8/4B3/8/K1B5 w - - 0 1"}) { // Bishops on one color.
            CAPTURE(fen);
            const auto board = nmo::parse_board(fen);
            for_each_searcher([&](auto& searcher, auto strategy) {
                const auto result = searcher.search(board, 3, strategy);
                CHECK(result.best_move == chess::Move::NO_MOVE);
                CHECK(result.score_cp == 0);
            });
        }
    }
}

