#include "nmo/search.hpp"

#include <algorithm>
#include <stdexcept>

#include "nmo/evaluation.hpp"

namespace nmo {
namespace {
constexpr int fifty_move_rule_halfmoves = 100;
}  // namespace

double SearchResult::ebf() const {
    return estimate_ebf(nodes, depth);
}

AlphaBetaSearcher::AlphaBetaSearcher(SearchOptions options) : options_(options), table_(options.table_megabytes) {}

/**
 * Returns whether these are valid search starting arguments
 */
void validate_search_args(int depth) {
    // depth must not underflow or overflow
    if (depth < 1 || depth > max_depth) {
        throw std::invalid_argument("search depth must be between 1 and 64");
    }
}

/**
 * Clear tables and prepare the orderer to search from this root
 */
void AlphaBetaSearcher::begin(const chess::Board& board, OrderingStrategy strategy) {
    result_ = {};
    table_.clear();
    orderer_.reset(strategy, policy_);
    orderer_.set_root(board);
}

/**
 * Run iterative deepening on this board with a given depth, using a specific ordering strategy
 * @param board
 * @param depth
 * @param strategy
 */
SearchResult AlphaBetaSearcher::search(chess::Board board, int depth, OrderingStrategy strategy) {
    validate_search_args(depth);

    // Clear tables for search
    begin(board, strategy);

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

ProbeResult AlphaBetaSearcher::probe(chess::Board board, int depth, int alpha, int beta, int ply,
                                     OrderingStrategy strategy, bool quiescence_only) {
    if (depth < 0 || depth > max_depth) {
        throw std::invalid_argument("probe depth must be between 0 and 64");
    }
    begin(board, strategy);
    int score = 0;
    if (quiescence_only) {
        score = quiescence(board, alpha, beta, ply);
    } else {
        const int start_depth = options_.iterative_deepening ? std::min(depth, 1) : depth;
        for (int iteration = start_depth; iteration <= depth; ++iteration) {
            score = negamax(board, iteration, alpha, beta, ply);
        }
    }
    return {score, result_.nodes};
}

int AlphaBetaSearcher::negamax(chess::Board& board, int depth, int alpha, int beta, int ply,
                              chess::Move* root_best_move) {
    if (root_best_move != nullptr)
        *root_best_move = chess::Move::NO_MOVE;

    // Non-stalemate draws
    const bool drawn = board.halfMoveClock() >= fifty_move_rule_halfmoves || is_insufficient_material(board);
    ++result_.nodes;

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
    if (depth == 0 && options_.quiescence)
        return quiescence(board, alpha, beta, ply, false);
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
        if (options_.hash_move_ordering)
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

    // Initialise our custom move orderer on this node. Orderers that defer to
    // the hash move are only consulted once it fails to cut: it is searched
    // first either way, so this skips network inference without changing the order.
    if (observer_ != nullptr)
        observer_->on_order(board, moves, hash_move, depth, alpha, beta, ply);
    const bool deferred = orderer_.defers_to_hash_move() && hash_move != chess::Move::NO_MOVE &&
                          std::find(moves.begin(), moves.end(), hash_move) != moves.end();
    OrderedMoves ordered;
    if (!deferred)
        orderer_.rank(board, moves, ordered, hash_move, depth);

    // track if alpha was beaten for TT score boundary
    const int original_alpha = alpha;
    int best_score = -infinity;
    chess::Move best_move = chess::Move::NO_MOVE;
    for (int index = 0; index < moves.size(); ++index) {
        if (deferred && index == 1) {
            chess::Movelist rest;
            for (const auto other : moves)
                if (other != hash_move) rest.add(other);
            orderer_.rank(board, rest, ordered, chess::Move::NO_MOVE, depth);
        }
        const auto move = deferred && index == 0 ? hash_move : ordered.next();
        orderer_.push(board, move, depth - 1);
        board.makeMove(move);
        const int score = -negamax(board, depth - 1, -beta, -alpha, ply + 1);
        board.unmakeMove(move);
        orderer_.pop();

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

int AlphaBetaSearcher::quiescence(chess::Board& board, int alpha, int beta, int ply, bool count_node) {
    if (count_node) ++result_.nodes;
    if (board.halfMoveClock() >= fifty_move_rule_halfmoves || is_insufficient_material(board))
        return 0;

    chess::Movelist legal;
    chess::movegen::legalmoves(legal, board);

    // Use the static evaluation as the stand-pat score, then search captures
    // and promotions.
    int best_score = evaluate_material(board);
    if (best_score >= beta)
        return best_score;
    alpha = std::max(alpha, best_score);

    // Captures include en passant; promotions include quiet promotions.
    chess::Movelist moves;
    for (const auto move : legal) {
        if (board.isCapture(move) || move.typeOf() == chess::Move::PROMOTION)
            moves.add(move);
    }
    if (moves.empty())
        return best_score;

    if (observer_ != nullptr)
        observer_->on_order(board, moves, chess::Move::NO_MOVE, 0, alpha, beta, ply);
    OrderedMoves ordered;
    orderer_.rank(board, moves, ordered, chess::Move::NO_MOVE, 0);
    for (int index = 0; index < ordered.size(); ++index) {
        const auto move = ordered.next();
        orderer_.push(board, move, 0);
        board.makeMove(move);
        const int score = -quiescence(board, -beta, -alpha, ply + 1);
        board.unmakeMove(move);
        orderer_.pop();

        best_score = std::max(best_score, score);
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            ++result_.cutoffs;
            break;
        }
    }

    return best_score;
}

}  // namespace nmo
