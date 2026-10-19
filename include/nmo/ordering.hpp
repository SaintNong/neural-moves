#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include <chess.hpp>

namespace nmo {

enum class OrderingStrategy { captures, history, history_gravity, capture_hist, mvv_lva_history, neural };

inline constexpr std::array heuristic_strategies{
    OrderingStrategy::captures, OrderingStrategy::history, OrderingStrategy::history_gravity,
    OrderingStrategy::capture_hist, OrderingStrategy::mvv_lva_history};

[[nodiscard]] std::string_view strategy_name(OrderingStrategy strategy) noexcept;
[[nodiscard]] OrderingStrategy parse_strategy(std::string_view name);
[[nodiscard]] int tactical_priority(const chess::Board& board, chess::Move move) noexcept;
[[nodiscard]] int gravity_update(int value, int bonus, int limit) noexcept;

class MoveOrderer {
public:
    explicit MoveOrderer(OrderingStrategy strategy = OrderingStrategy::captures) : strategy_(strategy) {}

    // Select a strategy and discard all accumulated cutoff history.
    void reset(OrderingStrategy strategy) noexcept;
    // A legal TT move has first priority for every strategy. Ties preserve generator order.
    // Requires moves legal in board; reorders without adding/removing moves.
    // Neural ordering throws std::logic_error (inference is unimplemented).
    void order(const chess::Board& board, chess::Movelist& moves,
               chess::Move tt_move = chess::Move::NO_MOVE) const;
    // Reward a legal move that caused a beta cutoff, using the board before that
    // move and remaining search depth in [1, max_depth]. Updates only the history
    // relevant to the selected strategy; captures-only ordering has no history.
    void on_cutoff(const chess::Board& board, chess::Move move, int depth) noexcept;

private:
    [[nodiscard]] static std::size_t quiet_key(const chess::Board& board, chess::Move move) noexcept;
    [[nodiscard]] static std::size_t capture_key(const chess::Board& board, chess::Move move) noexcept;

    OrderingStrategy strategy_;
    std::array<int, 2 * 64 * 64> history_{};
    std::array<int, 2 * 6 * 64 * 6> capture_history_{};
};

}  // namespace nmo

