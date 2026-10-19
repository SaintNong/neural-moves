#pragma once

#include <cstddef>
#include <cstdint>

#include <chess.hpp>

#include "nmo/ordering.hpp"
#include "nmo/transposition_table.hpp"

namespace nmo {

struct SearchOptions {
    // TT storage budget in MiB; zero disables it. Actual allocation may be smaller.
    std::size_t table_megabytes = 16;
    // Search depths 1 through the requested depth, retaining TT/history between them.
    bool iterative_deepening = true;
};

struct SearchResult {
    // NO_MOVE for terminal roots, including positions adjudicated as draws.
    chess::Move best_move{chess::Move::NO_MOVE};
    // Root side-to-move perspective: positive favors that side, zero is equal/drawn.
    // Ordinary scores are material centipawns. A forced mate at distance p plies
    // scores +(mate_score - p) for a win, -(mate_score - p) for a loss.
    int score_cp = 0;
    // Requested depth in plies, even if a terminal root ends the search early.
    int depth = 0;
    // Counters accumulate over all iterative-deepening iterations in this call.
    std::uint64_t nodes = 0;  // All visits, including roots, leaves and TT hits.
    // Beta cutoffs while searching moves; excludes TT cutoffs.
    std::uint64_t cutoffs = 0;
    // Probes with a matching transposition-table entry.
    std::uint64_t tt_hits = 0;
    // TT hits that return a score without searching children.
    std::uint64_t tt_cutoffs = 0;

    // Solve nodes = 1 + b + ... + b^depth for b >= 0. This measures total
    // search work, including earlier iterations, not just the final search tree.
    // Throws std::invalid_argument if nodes or depth is less than one.
    [[nodiscard]] double ebf() const;
};

// Fixed-horizon alpha-beta search with material-only leaf evaluation; no
// quiescence search or extensions. Adjudicates stalemate, insufficient material
// (see is_insufficient_material), and the automatic 75-move rule. Checkmate
// takes precedence. Repetition and optional 50-move claims are not adjudicated.
// Instances hold mutable search state and must not be used concurrently.
class AlphaBetaSearcher {
public:
    explicit AlphaBetaSearcher(SearchOptions options = {});
    // A board copy isolates the caller from search mutation. TT and history are
    // reset per call, then retained across iterative-deepening iterations.
    // Requires a valid standard-chess position. Depth must be in [1, max_depth],
    // otherwise throws std::invalid_argument. Neural ordering throws
    // std::logic_error because inference is unimplemented. Equally scoring
    // moves may yield different best moves under different ordering strategies.
    [[nodiscard]] SearchResult search(chess::Board board, int depth,
                                      OrderingStrategy strategy = OrderingStrategy::captures);
    [[nodiscard]] std::size_t table_bytes() const noexcept { return table_.bytes(); }

private:
    int negamax(chess::Board& board, int depth, int alpha, int beta, int ply,
                chess::Move* root_best_move = nullptr);

    SearchOptions options_;
    TranspositionTable table_;
    MoveOrderer orderer_;
    SearchResult result_{};
};

}  // namespace nmo

