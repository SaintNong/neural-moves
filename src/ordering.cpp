#include "nmo/ordering.hpp"

#include <algorithm>
#include <bit>
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
// Capture history is divided by this before it is added to 16 x victim value
// (tuned on quiet-100 over 2, 4, 8 and 16; 4 to 16 were within 0.2%).
constexpr int capture_history_divisor = 8;
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
        case OrderingStrategy::mvv_lva_history: return "mvv-lva-history";
        case OrderingStrategy::history_gravity: return "history-gravity";
        case OrderingStrategy::capture_hist: return "capture-hist";
        case OrderingStrategy::killers: return "killers";
        case OrderingStrategy::cont_hist: return "cont-hist";
        case OrderingStrategy::strong: return "strong";
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
    if (name == "history") {
        return OrderingStrategy::mvv_lva_history;
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

int static_exchange(const chess::Board& board, chess::Move move) noexcept {
    // Values for exchanges; the king is large so that capturing it ends the sequence.
    constexpr std::array<int, 7> value{100, 320, 330, 500, 900, 20'000, 0};
    const auto to = move.to();
    const auto from = move.from();
    const bool en_passant = move.typeOf() == chess::Move::ENPASSANT;
    int gain[32];
    int count = 0;
    gain[0] = en_passant ? value[0] : value[static_cast<int>(board.at<chess::PieceType>(to))];
    auto occupied = board.occ() ^ chess::Bitboard::fromSquare(from);
    if (en_passant) {
        occupied ^= chess::Bitboard::fromSquare(chess::Square(to.file(), from.rank()));
    }
    int on_square = value[static_cast<int>(board.at<chess::PieceType>(from))];
    auto side = ~board.sideToMove();
    const auto bishops = board.pieces(chess::PieceType::BISHOP) | board.pieces(chess::PieceType::QUEEN);
    const auto rooks = board.pieces(chess::PieceType::ROOK) | board.pieces(chess::PieceType::QUEEN);
    while (count < 31) {
        const auto attackers =
            ((chess::attacks::pawn(chess::Color::WHITE, to) & board.pieces(chess::PieceType::PAWN, chess::Color::BLACK)) |
             (chess::attacks::pawn(chess::Color::BLACK, to) & board.pieces(chess::PieceType::PAWN, chess::Color::WHITE)) |
             (chess::attacks::knight(to) & board.pieces(chess::PieceType::KNIGHT)) |
             (chess::attacks::bishop(to, occupied) & bishops) | (chess::attacks::rook(to, occupied) & rooks) |
             (chess::attacks::king(to) & board.pieces(chess::PieceType::KING))) &
            occupied;
        const auto mine = attackers & board.us(side);
        if (mine.empty()) break;
        int type = 0;
        chess::Bitboard least;
        for (; type < 6; ++type) {
            least = mine & board.pieces(static_cast<chess::PieceType::underlying>(type));
            if (!least.empty()) break;
        }
        // A king may not capture into a square the opponent still attacks.
        if (type == 5 && !(attackers & board.us(~side)).empty()) break;
        ++count;
        gain[count] = on_square - gain[count - 1];
        on_square = value[type];
        occupied ^= chess::Bitboard::fromSquare(std::countr_zero(least.getBits()));
        side = ~side;
    }
    while (count > 0) {
        gain[count - 1] = -std::max(-gain[count - 1], gain[count]);
        --count;
    }
    return gain[0];
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
    for (auto& pair : killers_) pair = {chess::Move::NO_MOVE, chess::Move::NO_MOVE};
    std::fill(continuation_history_.begin(), continuation_history_.end(), 0);
}

void MoveOrderer::set_root(const chess::Board& board) {
    path_.clear();
    if (policy_state_) policy_state_->reset(board);
}

void MoveOrderer::push(const chess::Board& before, chess::Move move, int child_depth) {
    path_.push_back({static_cast<int>(before.at(move.from())), move.to().index()});
    if (policy_state_) policy_state_->push(before, move, child_depth);
}

void MoveOrderer::pop() noexcept {
    path_.pop_back();
    if (policy_state_) policy_state_->pop();
}

bool MoveOrderer::uses_continuation() const noexcept {
    return strategy_ == OrderingStrategy::cont_hist || strategy_ == OrderingStrategy::strong ||
           (strategy_ == OrderingStrategy::neural && policy_->continuation);
}

int MoveOrderer::continuation(const chess::Board& board, chess::Move move) const noexcept {
    if (path_.empty()) return 0;
    const auto& previous = path_.back();
    const auto piece = static_cast<std::size_t>(static_cast<int>(board.at(move.from())));
    return continuation_history_[((static_cast<std::size_t>(previous.piece) * 64 + previous.to) * 12 + piece) * 64 +
                                 move.to().index()];
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
            case OrderingStrategy::mvv_lva_history:
            case OrderingStrategy::history_gravity: {
                const bool capture = board.isCapture(move);
                rank.stage = promotion || capture ? 1 : 0;
                rank.score = rank.stage ? tactical_priority(board, move, capture)
                                        : history_[quiet_key(board, move)];
                break;
            }
            case OrderingStrategy::capture_hist:
            case OrderingStrategy::killers:
            case OrderingStrategy::cont_hist:
            case OrderingStrategy::strong: {
                // Stages: 4 winning or even tactical moves, 3/2 killers, 1 quiet
                // moves, 0 losing captures (strong only).
                const bool capture = board.isCapture(move);
                const bool full = strategy_ != OrderingStrategy::capture_hist;
                if (promotion) {
                    rank.stage = 4;
                    rank.score = 1'000'000 + tactical_priority(board, move, capture);
                } else if (capture) {
                    const int victim = move.typeOf() == chess::Move::ENPASSANT
                        ? 0 : static_cast<int>(board.at<chess::PieceType>(move.to()));
                    const bool by_value = strategy_ == OrderingStrategy::capture_hist || strategy_ == OrderingStrategy::strong;
                    rank.stage = 4;
                    // Victim value plus capture history, with the cheaper attacker first on ties.
                    const int attacker = static_cast<int>(board.at<chess::PieceType>(move.from()));
                    rank.score = by_value ? 16 * piece_values[static_cast<std::size_t>(victim)] +
                                                capture_history_[capture_key(board, move)] / capture_history_divisor - attacker
                                          : tactical_priority(board, move, capture);
                    if (strategy_ == OrderingStrategy::strong && static_exchange(board, move) < 0) rank.stage = 0;
                } else {
                    const auto ply = std::min<std::size_t>(path_.size(), killers_.size() - 1);
                    if (full && depth > 0 && move == killers_[ply][0]) {
                        rank.stage = 3;
                    } else if (full && depth > 0 && move == killers_[ply][1]) {
                        rank.stage = 2;
                    } else {
                        rank.stage = 1;
                        rank.score = history_[quiet_key(board, move)];
                        if (uses_continuation()) rank.score += continuation(board, move);
                    }
                }
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
                    int history = history_[quiet_key(board, move)];
                    if (policy_->continuation) history += continuation(board, move);
                    score += static_cast<float>(policy_->history_weight) * static_cast<float>(history);
                }
                rank.score = static_cast<int>(score * logit_resolution);
                break;
            }
        }
    }
}

void MoveOrderer::on_cutoff(const chess::Board& board, chess::Move move, int depth,
                            const chess::Movelist& failed_quiets) noexcept {
    if (strategy_ == OrderingStrategy::mvvlva) return;
    if (strategy_ == OrderingStrategy::neural && policy_->history_weight == 0.0) return;
    const int bonus = bonus_per_ply * depth;
    if (board.isCapture(move)) {
        if (strategy_ == OrderingStrategy::capture_hist || strategy_ == OrderingStrategy::strong) {
            auto& value = capture_history_[capture_key(board, move)];
            value = gravity_update(value, bonus, capture_history_limit);
        }
        return;
    }
    if (move.typeOf() == chess::Move::PROMOTION) return;
    auto& value = history_[quiet_key(board, move)];
    if (strategy_ == OrderingStrategy::mvv_lva_history) {
        value = std::min(additive_history_limit, value + depth * depth);
        return;
    }
    value = gravity_update(value, bonus, history_limit);
    // The network blend was tuned without a malus, so neural ordering keeps none.
    const bool malus = strategy_ != OrderingStrategy::neural;
    if (malus) {
        for (const auto failed : failed_quiets) {
            auto& other = history_[quiet_key(board, failed)];
            other = gravity_update(other, -bonus, history_limit);
        }
    }
    if (strategy_ == OrderingStrategy::killers || strategy_ == OrderingStrategy::cont_hist ||
        strategy_ == OrderingStrategy::strong) {
        auto& pair = killers_[std::min<std::size_t>(path_.size(), killers_.size() - 1)];
        if (pair[0] != move) {
            pair[1] = pair[0];
            pair[0] = move;
        }
    }
    if (uses_continuation() && !path_.empty()) {
        const auto& previous = path_.back();
        const auto index = [&](chess::Move quiet) {
            const auto piece = static_cast<std::size_t>(static_cast<int>(board.at(quiet.from())));
            return ((static_cast<std::size_t>(previous.piece) * 64 + previous.to) * 12 + piece) * 64 + quiet.to().index();
        };
        auto& entry = continuation_history_[index(move)];
        entry = gravity_update(entry, bonus, history_limit);
        if (malus) {
            for (const auto failed : failed_quiets) {
                auto& other = continuation_history_[index(failed)];
                other = gravity_update(other, -bonus, history_limit);
            }
        }
    }
}

}  // namespace nmo

