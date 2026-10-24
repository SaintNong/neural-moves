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
// Fixed-point resolution used to rank network logits as integers.
constexpr float logit_resolution = 4096.0F;

// MVV-LVA scores indexed by victim then attacker, in PieceType order:
// pawn, knight, bishop, rook, queen, king, none.
constexpr std::array<std::array<int, 7>, 7> mvv_lva_scores{{
    {{15, 14, 13, 12, 11, 10, 0}},
    {{25, 24, 23, 22, 21, 20, 0}},
    {{35, 34, 33, 32, 31, 30, 0}},
    {{45, 44, 43, 42, 41, 40, 0}},
    {{55, 54, 53, 52, 51, 50, 0}},
    {{0, 0, 0, 0, 0, 0, 0}},
    {{0, 0, 0, 0, 0, 0, 0}},
}};

}  // namespace

std::string_view strategy_name(OrderingStrategy strategy) noexcept {
    switch (strategy) {
        case OrderingStrategy::mvvlva: return "mvv-lva";
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
        score += mvv_lva_scores[victim][attacker];
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

MoveOrderer::MoveOrderer(OrderingStrategy strategy) : strategy_(strategy) {}
MoveOrderer::~MoveOrderer() = default;
MoveOrderer::MoveOrderer(MoveOrderer&&) noexcept = default;
MoveOrderer& MoveOrderer::operator=(MoveOrderer&&) noexcept = default;

void MoveOrderer::reset(OrderingStrategy strategy, std::shared_ptr<const Policy> policy) {
    if (strategy == OrderingStrategy::neural && !policy) {
        throw std::logic_error("neural move ordering requires a policy network");
    }
    if (strategy != OrderingStrategy::neural) {
        policy.reset();
    }
    if (policy != policy_) {
        policy_state_ = policy ? policy->new_state() : nullptr;
        policy_ = std::move(policy);
    }
    strategy_ = strategy;
    history_.fill(0);
    capture_history_.fill(0);
}

void MoveOrderer::set_root(const chess::Board& board) {
    if (policy_state_) policy_state_->reset(board);
}

void MoveOrderer::push(const chess::Board& before, chess::Move move, int child_depth) {
    if (policy_state_) policy_state_->push(before, move, child_depth);
}

void MoveOrderer::pop() noexcept {
    if (policy_state_) policy_state_->pop();
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

chess::Move OrderedMoves::next() noexcept {
    const auto first = moves_.begin() + picked_;
    const auto last = moves_.begin() + count_;
    if (picked_ < lazy_picks) {
        std::iter_swap(first, std::min_element(first, last, RankedMove::before));
    } else if (picked_ == lazy_picks) {
        // A node that needed this many moves probably needs them all.
        std::sort(first, last, RankedMove::before);
    }
    return moves_[static_cast<std::size_t>(picked_++)].move;
}

void MoveOrderer::order(const chess::Board& board, chess::Movelist& moves, chess::Move hash_move, int depth) {
    OrderedMoves ranked;
    rank(board, moves, ranked, hash_move, depth);
    std::sort(ranked.moves_.begin(), ranked.moves_.begin() + ranked.count_, RankedMove::before);
    for (int index = 0; index < moves.size(); ++index) {
        moves[index] = ranked.moves_[static_cast<std::size_t>(index)].move;
    }
}

void MoveOrderer::rank(const chess::Board& board, const chess::Movelist& moves, OrderedMoves& out,
                       chess::Move hash_move, int depth) {
    const bool network = strategy_ == OrderingStrategy::neural && (depth > 0 || policy_->quiescence);
    std::array<float, chess::constants::MAX_MOVES> logits;  // Written only for network ordering.
    if (network) {
        policy_state_->evaluate(board, moves, depth, logits.data());
    }
    out.count_ = moves.size();
    out.picked_ = 0;
    for (int index = 0; index < moves.size(); ++index) {
        const auto move = moves[index];
        const bool promotion = move.typeOf() == chess::Move::PROMOTION;
        auto& rank = out.moves_[static_cast<std::size_t>(index)];
        rank = {move, move == hash_move, 0, 0, index};
        switch (strategy_) {
            case OrderingStrategy::mvvlva: {
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
            case OrderingStrategy::neural: {
                const bool capture = board.isCapture(move);
                if (!network || (policy_->staged && (capture || promotion))) {
                    rank.stage = network ? 1 : 0;
                    rank.score = tactical_priority(board, move, capture);
                    break;
                }
                float score = logits[static_cast<std::size_t>(index)];
                if (policy_->history_weight != 0.0 && !promotion && !capture) {
                    score += static_cast<float>(policy_->history_weight) * static_cast<float>(history_[quiet_key(board, move)]);
                }
                rank.score = static_cast<int>(score * logit_resolution);
                break;
            }
        }
    }
}

void MoveOrderer::on_cutoff(const chess::Board& board, chess::Move move, int depth) noexcept {
    if (strategy_ == OrderingStrategy::mvvlva) return;
    if (strategy_ == OrderingStrategy::neural && policy_->history_weight == 0.0) return;
    if (board.isCapture(move)) {
        if (strategy_ == OrderingStrategy::capture_hist) {
            auto& value = capture_history_[capture_key(board, move)];
            value = gravity_update(value, bonus_per_ply * depth, capture_history_limit);
        }
        return;
    }
    if (move.typeOf() == chess::Move::PROMOTION) return;
    auto& value = history_[quiet_key(board, move)];
    if (strategy_ == OrderingStrategy::history_gravity || strategy_ == OrderingStrategy::capture_hist ||
        strategy_ == OrderingStrategy::neural) {
        value = gravity_update(value, bonus_per_ply * depth, history_limit);
    } else {
        value = std::min(additive_history_limit, value + depth * depth);
    }
}

}  // namespace nmo

