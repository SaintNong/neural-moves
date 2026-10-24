#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <chess.hpp>

#include "nmo/ordering.hpp"
#include "nmo/transposition_table.hpp"

namespace nmo {

struct SearchOptions {
    // TT storage budget in MiB; zero disables it. Actual allocation may be smaller.
    std::size_t table_megabytes = 16;
    // Search depths 1 through the requested depth, retaining TT/history between them.
    bool iterative_deepening = true;
    // Extend depth-zero nodes with capture and promotion searches.
    bool quiescence = true;
    // Prioritize a transposition-table move during move ordering.
    bool hash_move_ordering = true;
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

// Receives the move-ordering decisions of a search; used to sample training data.
class SearchObserver {
public:
    virtual ~SearchObserver() = default;
    // Called just before a node's moves are ordered. Interior nodes have
    // depth >= 1; quiescence nodes have depth 0 and list only captures and
    // promotions. The hash move (searched first) may be NO_MOVE. Alpha and
    // beta are the node's window; ply is from the root.
    virtual void on_order(const chess::Board& board, const chess::Movelist& moves, chess::Move hash_move,
                          int depth, int alpha, int beta, int ply) = 0;
};

struct ProbeResult {
    int score = 0;               // Fail-soft score within the requested window.
    std::uint64_t nodes = 0;     // Nodes visited, as counted by SearchResult.
};

// Alpha-beta search with material evaluation and optional quiescence search
// (enabled by default). Adjudicates stalemate, insufficient material (see
// is_insufficient_material), and the 100-halfmove draw rule. Checkmate takes
// precedence. Repetition is not adjudicated.
// Instances hold mutable search state and must not be used concurrently.
class AlphaBetaSearcher {
public:
    explicit AlphaBetaSearcher(SearchOptions options = {});
    // A board copy isolates the caller from search mutation. TT and history are
    // reset per call, then retained across iterative-deepening iterations.
    // Requires a valid standard-chess position. Depth must be in [1, max_depth],
    // otherwise throws std::invalid_argument. Neural ordering throws
    // std::logic_error unless a policy was set. Equally scoring
    // moves may yield different best moves under different ordering strategies.
    [[nodiscard]] SearchResult search(chess::Board board, int depth,
                                      OrderingStrategy strategy = OrderingStrategy::mvvlva);
    // Search a non-root position with an explicit window, starting from empty
    // tables and iteratively deepening to depth (zero runs the leaf search).
    // With quiescence_only, the position is searched as a quiescence node.
    // Scores use the given root ply for mate distances. Data generation probes
    // every move of a sampled node this way to label it.
    [[nodiscard]] ProbeResult probe(chess::Board board, int depth, int alpha, int beta, int ply,
                                    OrderingStrategy strategy, bool quiescence_only = false);
    [[nodiscard]] std::size_t table_bytes() const noexcept { return table_.bytes(); }
    // The policy network used by OrderingStrategy::neural.
    void set_policy(std::shared_ptr<const Policy> policy) noexcept { policy_ = std::move(policy); }
    // Not owned; null disables observation.
    void set_observer(SearchObserver* observer) noexcept { observer_ = observer; }

private:
    int negamax(chess::Board& board, int depth, int alpha, int beta, int ply,
                chess::Move* root_best_move = nullptr);
    int quiescence(chess::Board& board, int alpha, int beta, int ply, bool count_node = true);

    void begin(const chess::Board& board, OrderingStrategy strategy);

    SearchOptions options_;
    TranspositionTable table_;
    MoveOrderer orderer_;
    std::shared_ptr<const Policy> policy_;
    SearchObserver* observer_ = nullptr;
    SearchResult result_{};
};

}  // namespace nmo

