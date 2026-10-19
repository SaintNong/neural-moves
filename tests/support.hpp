#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <doctest.h>

#include "nmo/positions.hpp"
#include "nmo/search.hpp"

namespace test_support {

inline constexpr std::string_view free_queen = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1";
inline constexpr std::string_view poisoned_rook = "7k/8/4p3/3r4/8/8/3Q4/K7 w - - 0 1";
inline constexpr std::string_view mate_in_one = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1";
inline constexpr std::string_view en_passant = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
inline constexpr std::string_view promotion = "7k/P7/8/8/8/8/8/4K3 w - - 0 1";
inline constexpr std::string_view castling = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";

inline chess::Movelist legal_moves(const chess::Board& board) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    return moves;
}

inline chess::Move move_from_uci(const chess::Board& board, std::string_view uci) {
    const auto moves = legal_moves(board);
    const auto found = std::find_if(moves.begin(), moves.end(), [uci](chess::Move move) {
        return chess::uci::moveToUci(move) == uci;
    });
    INFO("fixture FEN: ", board.getFen(), "; requested move: ", uci);
    REQUIRE(found != moves.end());
    return *found;
}

inline std::vector<std::string> move_set(const chess::Movelist& moves) {
    std::vector<std::string> result;
    for (const auto move : moves) result.push_back(chess::uci::moveToUci(move));
    std::sort(result.begin(), result.end());
    return result;
}

// Exercise supported search settings through the public API. Tests may accept
// any equally good move; they do not prescribe traversal order or node counts.
template <class Check>
void for_each_searcher(Check check) {
    constexpr std::array<nmo::SearchOptions, 4> options{{
        {0, false}, {0, true}, {1, false}, {1, true}
    }};
    for (const auto& option : options) {
        for (const auto strategy : nmo::heuristic_strategies) {
            INFO("strategy: ", nmo::strategy_name(strategy),
                 "; caching: ", option.table_megabytes != 0,
                 "; iterative deepening: ", option.iterative_deepening);
            nmo::AlphaBetaSearcher searcher(option);
            check(searcher, strategy);
        }
    }
}

}  // namespace test_support
