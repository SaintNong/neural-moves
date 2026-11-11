#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <chess.hpp>

namespace nmo {

class Policy;
class PolicyState;

// Every heuristic orders captures and promotions ahead of quiet moves, by MVV-LVA
// unless it has its own capture ordering, so the heuristics differ only in what
// they add on top. "history" is accepted as another name for mvv-lva-history.
//   mvv-lva          captures by MVV-LVA, quiet moves in generator order
//   mvv-lva-history  + quiet moves by the additive history heuristic (Schaeffer)
//   history-gravity  + gravity history with a penalty for quiet moves that failed to cut
//   capture-hist     + captures by victim value plus capture history
//   killers          history-gravity + two killer moves per ply
//   cont-hist        killers + one-ply continuation history added to quiet scores
//   strong           cont-hist + capture history, with losing captures (SEE < 0) last
enum class OrderingStrategy {
    mvvlva, mvv_lva_history, history_gravity, capture_hist, killers, cont_hist, strong, neural
};

inline constexpr std::array heuristic_strategies{
    OrderingStrategy::mvvlva, OrderingStrategy::mvv_lva_history, OrderingStrategy::history_gravity,
    OrderingStrategy::capture_hist, OrderingStrategy::killers, OrderingStrategy::cont_hist,
    OrderingStrategy::strong};

[[nodiscard]] std::string_view strategy_name(OrderingStrategy strategy) noexcept;
[[nodiscard]] OrderingStrategy parse_strategy(std::string_view name);
[[nodiscard]] int tactical_priority(const chess::Board& board, chess::Move move, bool capture) noexcept;
[[nodiscard]] int gravity_update(int value, int bonus, int limit) noexcept;
// Static exchange evaluation: the material the side to move gains by making this
// capture and then trading off on the destination square with least valuable
// attackers first. Ignores pins and checks.
[[nodiscard]] int static_exchange(const chess::Board& board, chess::Move move) noexcept;

// A move with its ordering keys. Moves are searched by: hash move first, then
// higher stage, then higher score, then generator order (index).
struct RankedMove {
    chess::Move move{chess::Move::NO_MOVE};
    bool is_hash_move = false;
    int stage = 0;
    int score = 0;
    int index = 0;

    [[nodiscard]] static bool before(const RankedMove& left, const RankedMove& right) noexcept {
        if (left.is_hash_move != right.is_hash_move) return left.is_hash_move;
        if (left.stage != right.stage) return left.stage > right.stage;
        if (left.score != right.score) return left.score > right.score;
        return left.index < right.index;
    }
};

// The ranked moves of one node, handed out best-first. Selection is lazy
// because most nodes cut off after one or two moves; the order is the same as
// a full sort since RankedMove::before is a total order.
class OrderedMoves {
public:
    [[nodiscard]] int size() const noexcept { return count_; }
    // The next move in order; call at most size() times.
    [[nodiscard]] chess::Move next() noexcept;

private:
    friend class MoveOrderer;
    static constexpr int lazy_picks = 3;

    std::array<RankedMove, chess::constants::MAX_MOVES> moves_;
    int count_ = 0;
    int picked_ = 0;
};

class MoveOrderer {
public:
    explicit MoveOrderer(OrderingStrategy strategy = OrderingStrategy::mvvlva);
    ~MoveOrderer();
    MoveOrderer(MoveOrderer&&) noexcept;
    MoveOrderer& operator=(MoveOrderer&&) noexcept;

    // Select a strategy and discard all accumulated cutoff history. Neural
    // ordering needs a policy network; other strategies ignore it.
    // Throws std::logic_error for neural ordering without a policy.
    void reset(OrderingStrategy strategy, std::shared_ptr<const Policy> policy = nullptr);
    // Start a new search path at this root. Each move searched from the current
    // position is pushed, with the remaining depth of the node it leads to,
    // before Board::makeMove and popped after unmakeMove, so incrementally
    // updated networks track the board.
    void set_root(const chess::Board& board);
    void push(const chess::Board& before, chess::Move move, int child_depth);
    void pop() noexcept;
    // The hash move is placed first regardless of strategy; remaining ties
    // preserve generator order. Reorders without adding or removing moves.
    // Depth is the node's remaining depth; zero marks a quiescence node, whose
    // list contains only captures and promotions.
    void order(const chess::Board& board, chess::Movelist& moves,
               chess::Move hash_move = chess::Move::NO_MOVE, int depth = 1);
    // Rank moves as order() does, for best-first iteration with OrderedMoves::next().
    void rank(const chess::Board& board, const chess::Movelist& moves, OrderedMoves& out,
              chess::Move hash_move = chess::Move::NO_MOVE, int depth = 1);
    // Whether the search should try the hash move before asking for an order,
    // because ordering is expensive (network inference).
    [[nodiscard]] bool defers_to_hash_move() const noexcept { return strategy_ == OrderingStrategy::neural; }
    // Reward a legal move that caused a beta cutoff, using the board before that
    // move and remaining search depth in [1, max_depth]. Updates only the history
    // relevant to the selected strategy; MVV-LVA ordering has no history.
    // Quiet moves searched at this node before the cutoff are penalised by the
    // strategies that use a malus.
    void on_cutoff(const chess::Board& board, chess::Move move, int depth,
                   const chess::Movelist& failed_quiets = {}) noexcept;

private:
    [[nodiscard]] static std::size_t quiet_key(const chess::Board& board, chess::Move move) noexcept;
    [[nodiscard]] static std::size_t capture_key(const chess::Board& board, chess::Move move) noexcept;

    OrderingStrategy strategy_;
    std::shared_ptr<const Policy> policy_;
    std::unique_ptr<PolicyState> policy_state_;
    std::array<int, 2 * 64 * 64> history_{};
    std::array<int, 2 * 6 * 64 * 6> capture_history_{};
    // Moved piece and destination of each move on the current path, for
    // continuation history; its size is the current ply.
    struct PathMove { int piece; int to; };
    std::vector<PathMove> path_;
    std::array<std::array<chess::Move, 2>, 128> killers_{};
    // Indexed by previous (piece, to) then current (piece, to); 12 pieces x 64 squares.
    std::vector<int> continuation_history_ = std::vector<int>(12 * 64 * 12 * 64, 0);
    [[nodiscard]] int continuation(const chess::Board& board, chess::Move move) const noexcept;
    [[nodiscard]] bool uses_continuation() const noexcept;
};

}  // namespace nmo
