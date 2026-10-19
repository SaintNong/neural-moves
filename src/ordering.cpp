#include "nmo/ordering.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "nmo/evaluation.hpp"
#include "nmo/neural.hpp"

namespace nmo {
namespace {
constexpr int history_limit = 7183;
constexpr int capture_history_limit = 10692;
constexpr int bonus_per_ply = 131;
constexpr int additive_history_limit = 1'000'000'000;

struct RankedMove {
    chess::Move move{chess::Move::NO_MOVE};
    bool is_hash_move = false;
    int stage = 0;
    int score = 0;
    int index = 0;
};
}  // namespace

std::string_view strategy_name(OrderingStrategy strategy) noexcept {
    switch (strategy) {
        case OrderingStrategy::captures: return "captures";
        case OrderingStrategy::history: return "history";
        case OrderingStrategy::history_gravity: return "history-gravity";
        case OrderingStrategy::capture_hist: return "capture-hist";
        case OrderingStrategy::mvv_lva_history: return "mvv-lva-history";
        case OrderingStrategy::neural: return "neural";
    }
    return "unknown";
}

OrderingStrategy parse_strategy(std::string_view name) {
    for (const auto strategy : heuristic_strategies) {
        if (name == strategy_name(strategy)) {
            return strategy;
        }
    }
    if (name == "neural") {
        return OrderingStrategy::neural;
    }
    throw std::invalid_argument("unknown strategy: " + std::string(name));
}

int tactical_priority(const chess::Board& board, chess::Move move, bool capture) noexcept {
    int score = 0;
    if (capture) {
        const auto victim = static_cast<std::size_t>(static_cast<int>(board.getCapturing<chess::PieceType>(move)));
        const auto attacker = static_cast<std::size_t>(static_cast<int>(board.at<chess::PieceType>(move.from())));
        score += 10'000 + 10 * piece_values[victim] - piece_values[attacker];
    }
    if (move.typeOf() == chess::Move::PROMOTION) {
        score += 20'000 + piece_values[static_cast<std::size_t>(static_cast<int>(move.promotionType()))];
    }
    return score;
}

int gravity_update(int value, int bonus, int limit) noexcept {
    bonus = std::clamp(bonus, -limit, limit);
    const auto adjustment = static_cast<std::int64_t>(value) * std::abs(bonus) / limit;
    return value + bonus - static_cast<int>(adjustment);
}

void MoveOrderer::reset(OrderingStrategy strategy) noexcept {
    strategy_ = strategy;
    history_.fill(0);
    capture_history_.fill(0);
}

std::size_t MoveOrderer::quiet_key(const chess::Board& board, chess::Move move) noexcept {
    return static_cast<std::size_t>(static_cast<int>(board.sideToMove()) * 4096 +
                                    move.from().index() * 64 + move.to().index());
}

std::size_t MoveOrderer::capture_key(const chess::Board& board, chess::Move move) noexcept {
    const int attacker = board.at<chess::PieceType>(move.from());
    const int victim = board.getCapturing<chess::PieceType>(move);
    return static_cast<std::size_t>(((static_cast<int>(board.sideToMove()) * 6 + attacker) * 64 +
                                     move.to().index()) * 6 + victim);
}

void MoveOrderer::order(const chess::Board& board, chess::Movelist& moves, chess::Move hash_move) const {
    if (strategy_ == OrderingStrategy::neural) {
        NeuralPolicy{}.rank(board, moves);
    }
    std::array<RankedMove, chess::constants::MAX_MOVES> ranked{};
    for (int index = 0; index < moves.size(); ++index) {
        const auto move = moves[index];
        const bool promotion = move.typeOf() == chess::Move::PROMOTION;
        auto& rank = ranked[static_cast<std::size_t>(index)];
        rank.move = move;
        rank.is_hash_move = move == hash_move;
        rank.index = index;
        switch (strategy_) {
            case OrderingStrategy::captures: {
                const bool capture = board.isCapture(move);
                rank.score = tactical_priority(board, move, capture);
                break;
            }
            case OrderingStrategy::history:
            case OrderingStrategy::history_gravity: {
                const bool capture = !promotion && board.isCapture(move);
                rank.stage = promotion || capture ? 1 : 0;
                rank.score = rank.stage ? 0 : history_[quiet_key(board, move)];
                break;
            }
            case OrderingStrategy::capture_hist: {
                if (promotion) {
                    rank.stage = 2;
                    rank.score = static_cast<int>(move.promotionType());
                } else if (board.isCapture(move)) {
                    rank.stage = 1;
                    rank.score = capture_history_[capture_key(board, move)];
                } else {
                    rank.stage = 0;
                    rank.score = history_[quiet_key(board, move)];
                }
                break;
            }
            case OrderingStrategy::mvv_lva_history: {
                const bool capture = board.isCapture(move);
                rank.stage = promotion || capture ? 1 : 0;
                rank.score = rank.stage ? tactical_priority(board, move, capture)
                                        : history_[quiet_key(board, move)];
                break;
            }
            case OrderingStrategy::neural: break;  // The stub above always throws.
        }
    }
    std::sort(ranked.begin(), ranked.begin() + moves.size(), [](const RankedMove& left, const RankedMove& right) {
        if (left.is_hash_move != right.is_hash_move) return left.is_hash_move;
        if (left.stage != right.stage) return left.stage > right.stage;
        if (left.score != right.score) return left.score > right.score;
        return left.index < right.index;
    });
    for (int index = 0; index < moves.size(); ++index) {
        moves[index] = ranked[static_cast<std::size_t>(index)].move;
    }
}

void MoveOrderer::on_cutoff(const chess::Board& board, chess::Move move, int depth) noexcept {
    if (strategy_ == OrderingStrategy::captures || strategy_ == OrderingStrategy::neural) return;
    if (board.isCapture(move)) {
        if (strategy_ == OrderingStrategy::capture_hist) {
            auto& value = capture_history_[capture_key(board, move)];
            value = gravity_update(value, bonus_per_ply * depth, capture_history_limit);
        }
        return;
    }
    if (move.typeOf() == chess::Move::PROMOTION) return;
    auto& value = history_[quiet_key(board, move)];
    if (strategy_ == OrderingStrategy::history_gravity || strategy_ == OrderingStrategy::capture_hist) {
        value = gravity_update(value, bonus_per_ply * depth, history_limit);
    } else {
        value = std::min(additive_history_limit, value + depth * depth);
    }
}

}  // namespace nmo

