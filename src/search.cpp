#include "nmo/search.hpp"

#include <algorithm>
#include <stdexcept>

#include "nmo/evaluation.hpp"

namespace nmo {

double SearchResult::ebf() const {
    return estimate_ebf(nodes, depth);
}

AlphaBetaSearcher::AlphaBetaSearcher(SearchOptions options) : options_(options), table_(options.table_megabytes) {}

SearchResult AlphaBetaSearcher::search(chess::Board board, int depth, OrderingStrategy strategy) {
    if (depth < 1 || depth > max_depth) {
        throw std::invalid_argument("search depth must be between 1 and 64");
    }
    if (strategy == OrderingStrategy::neural) {
        throw std::logic_error("neural move ordering is a stub; inference is not implemented");
    }
    result_ = {};
    table_.clear();
    orderer_.reset(strategy);
    const int first_depth = options_.iterative_deepening ? 1 : depth;
    for (int iteration = first_depth; iteration <= depth; ++iteration) {
        result_.score_cp = negamax(board, iteration, -infinity, infinity, 0, result_.best_move);
        result_.depth = iteration;
        if (result_.best_move == chess::Move::NO_MOVE) break;
    }
    // Terminal roots still report the requested depth for benchmark consistency.
    result_.depth = depth;
    return result_;
}

int AlphaBetaSearcher::negamax(chess::Board& board, int depth, int alpha, int beta, int ply,
                              chess::Move& best_move) {
    ++result_.nodes;
    best_move = chess::Move::NO_MOVE;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    // Mate takes precedence over rule draws, including at the horizon.
    if (moves.empty()) return board.inCheck() ? -mate_score + ply : 0;
    // Match the Python experiment's automatic draw rules (no optional claims).
    if (board.halfMoveClock() >= 150 || is_insufficient_material(board)) return 0;
    if (depth == 0) return evaluate_material(board);

    const auto key = TranspositionTable::position_key(board);
    chess::Move hash_move{chess::Move::NO_MOVE};
    if (const auto* entry = table_.probe(key)) {
        // Only order a move actually present in this node's legal move list.
        if (std::find(moves.begin(), moves.end(), entry->best_move) != moves.end()) {
            ++result_.tt_hits;
            hash_move = entry->best_move;
            // Exact depth preserves fixed-horizon minimax semantics even when
            // the same board is reached at different plies after captures.
            if (entry->depth == depth) {
                const int score = score_from_table(entry->score, ply);
                if (entry->bound == Bound::exact || (entry->bound == Bound::lower && score >= beta) ||
                    (entry->bound == Bound::upper && score <= alpha)) {
                    ++result_.tt_cutoffs;
                    best_move = hash_move;
                    return score;
                }
            }
        }
    }
    orderer_.order(board, moves, hash_move);
    const int original_alpha = alpha;
    int best_score = -infinity;
    for (const auto move : moves) {
        board.makeMove(move);
        chess::Move child_best{chess::Move::NO_MOVE};
        const int score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, child_best);
        board.unmakeMove(move);
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            ++result_.cutoffs;
            orderer_.on_cutoff(board, move, depth);
            break;
        }
    }
    const auto bound = best_score <= original_alpha ? Bound::upper : best_score >= beta ? Bound::lower : Bound::exact;
    table_.store(key, depth, score_to_table(best_score, ply), bound, best_move);
    return best_score;
}

}  // namespace nmo
