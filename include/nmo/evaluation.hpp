#pragma once

#include <array>
#include <cstdint>

#include <chess.hpp>

namespace nmo {

inline constexpr int mate_score = 100'000;
inline constexpr int max_depth = 64;
inline constexpr int infinity = mate_score + 1;
// Indexed by chess::PieceType; NONE is included for non-captures.
inline constexpr std::array<int, 7> piece_values{100, 320, 330, 500, 900, 0, 0};

// Material balance in centipawns from the side-to-move perspective. Does not
// adjudicate mate or draws and includes no positional or mobility terms.
[[nodiscard]] int evaluate_material(const chess::Board& board) noexcept;
// Recognizes bare kings, king + one knight versus king, and bishops-only
// material when every bishop occupies the same square color (including
// promoted bishops). Not a general detector of all possible dead positions.
[[nodiscard]] bool is_insufficient_material(const chess::Board& board) noexcept;
// Nonnegative b satisfying total_nodes = sum(b^ply, ply=0..depth).
// Returns zero for one node; throws std::invalid_argument for zero nodes or
// nonpositive depth. The caller chooses whether counts include prior iterations.
[[nodiscard]] double estimate_ebf(std::uint64_t total_nodes, int depth);

}  // namespace nmo

