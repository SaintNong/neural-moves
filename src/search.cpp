#include "nmo/search.hpp"

#include <algorithm>
#include <stdexcept>

#include "nmo/evaluation.hpp"

namespace nmo {

double SearchResult::ebf() const {
    return estimate_ebf(nodes, depth);
}

AlphaBetaSearcher::AlphaBetaSearcher(SearchOptions options) : options_(options), table_(options.table_megabytes) {}

/**
 * Returns whether these are valid search starting arguments
 */
void validate_search_args(int depth, OrderingStrategy strategy) {
    // depth must not underflow or overflow
    if (depth < 1 || depth > max_depth) {
        throw std::invalid_argument("search depth must be between 1 and 64");
    }

    // strategy actually exists
    if (strategy == OrderingStrategy::neural) {
        throw std::logic_error("neural move ordering not implemented yet");
    }
}

/**
 * Run iterative deepening on this board with a given depth, using a specific ordering strategy
 * @param board
 * @param depth
 * @param strategy
 */
SearchResult AlphaBetaSearcher::search(chess::Board board, int depth, OrderingStrategy strategy) {
    validate_search_args(depth, strategy);

    // Clear tables for search
    result_ = {};
    table_.clear();
    orderer_.reset(strategy);

    // Begin iterative deepening loop
    const int start_depth = options_.iterative_deepening ? 1 : depth;
    for (int iteration = start_depth; iteration <= depth; ++iteration) {

        result_.score_cp = negamax(board, iteration, -infinity, infinity, 0, &result_.best_move);
        result_.depth = iteration;

        // Edge case: We're in a terminal position
        if (result_.best_move == chess::Move::NO_MOVE)
            break;
    }

    result_.depth = depth;
    return result_;
}

int AlphaBetaSearcher::negamax(chess::Board& board, int depth, int alpha, int beta, int ply,
                              chess::Move* root_best_move) {
    // Non-stalemate draws
    const bool drawn = board.halfMoveClock() >= 100 || is_insufficient_material(board);
    ++result_.nodes;
    if (root_best_move != nullptr) *root_best_move = chess::Move::NO_MOVE;

    //  50-move rule
    if (drawn && !board.inCheck())
        return 0;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    // Stalemate or checkmate
    if (moves.empty())
        return board.inCheck() ? -mate_score + ply : 0;
    if (drawn)
        return 0;

    // Leaf node
    if (depth == 0)
        return evaluate_material(board);

    // TT lookup on this node
    // attempt to avoid searching node by looking it up in TT.
    const auto key = TranspositionTable::position_key(board);
    chess::Move hash_move = chess::Move::NO_MOVE;
    if (const auto* entry = table_.probe(key)) {
        // TT hit.
        // now we know the exact score and best move from last search in this position.
        ++result_.tt_hits;
        hash_move = entry->best_move;
        if (ply != 0 && entry->depth >= depth) {
            // Check if TT info is informative enough that we can directly return.
            const int score = score_from_table(entry->score, ply);
            if (entry->bound == Bound::exact || (entry->bound == Bound::lower && score >= beta) ||
                (entry->bound == Bound::upper && score <= alpha)) {
                ++result_.tt_cutoffs;
                return score;
            }
        }
    }

    // Initialise our custom move orderer on this node
    orderer_.order(board, moves, hash_move);

    // track if alpha was beaten for TT score boundary
    const int original_alpha = alpha;
    int best_score = -infinity;
    chess::Move best_move = chess::Move::NO_MOVE;
    for (const auto move : moves) {
        board.makeMove(move);
        const int score = -negamax(board, depth - 1, -beta, -alpha, ply + 1);
        board.unmakeMove(move);

        if (score > best_score) {
            best_score = score;
            best_move = move;
            if (root_best_move != nullptr) *root_best_move = move;
        }
        alpha = std::max(alpha, score);

        // Beta cutoff
        if (alpha >= beta) {
            ++result_.cutoffs;

            // If our orderer supports storing cutoffs, update the tables
            orderer_.on_cutoff(board, move, depth);
            break;
        }
    }

    // Store resultant score of searching this node in hash table, and the boundary associated with it.
    const auto bound = best_score <= original_alpha
        ? Bound::upper : best_score >= beta ? Bound::lower : Bound::exact;
    table_.store(key, depth, score_to_table(best_score, ply), bound, best_move);

    return best_score;
}

}  // namespace nmo
