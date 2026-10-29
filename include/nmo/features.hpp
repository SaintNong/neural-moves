#pragma once

#include <array>
#include <bit>
#include <cstdint>

#include <chess.hpp>

// Board and move encodings shared by network inference and the training-data
// featurizer, so the two cannot drift apart. Everything is expressed from a
// perspective: ranks are mirrored for black so each side sees itself moving up.
namespace nmo::features {

inline constexpr int piece_square_count = 768;
inline constexpr int max_active = 32;
inline constexpr int no_victim = 6;
inline constexpr int threat_combinations = 256;

// Threat bits describing a move's squares before it is played. "Lower" means
// attacked by an enemy piece of lower value class than the piece that ends on
// the square (pawn < knight = bishop < rook < queen < king).
enum ThreatBit : std::uint8_t {
    to_attacked = 1 << 0,
    to_attacked_by_lower = 1 << 1,
    to_defended = 1 << 2,
    from_attacked = 1 << 3,
    from_attacked_by_lower = 1 << 4,
    from_defended = 1 << 5,
    gives_direct_check = 1 << 6,
    // Another own piece stays attacked, so the opponent keeps a capture.
    others_attacked = 1 << 7,
};

[[nodiscard]] constexpr int relative_square(chess::Color perspective, int square) noexcept {
    return perspective == chess::Color::WHITE ? square : square ^ 56;
}

// The perspective's own pieces occupy [0, 384) and the opponent's [384, 768),
// each indexed as piece_type * 64 + relative square.
[[nodiscard]] constexpr int piece_square(chess::Color perspective, chess::Piece piece, int square) noexcept {
    const int theirs = piece.color() == perspective ? 0 : 1;
    return (theirs * 6 + static_cast<int>(piece.type())) * 64 + relative_square(perspective, square);
}

struct ActiveFeatures {
    std::array<std::int16_t, max_active> index{};
    int count = 0;
};

// Piece-square features of every piece on the board, in square order.
[[nodiscard]] ActiveFeatures active_features(const chess::Board& board, chess::Color perspective) noexcept;

struct PieceChange {
    chess::Piece piece;
    int square = 0;
};

// Pieces removed from and added to the board by a move (absolute squares).
// Castling moves two pieces; a capturing promotion removes two.
struct MoveDelta {
    std::array<PieceChange, 2> removed{};
    std::array<PieceChange, 2> added{};
    int removed_count = 0;
    int added_count = 0;
};

[[nodiscard]] MoveDelta move_delta(const chess::Board& before, chess::Move move) noexcept;

// Squares attacked by each side, computed once per position and reused for all
// of its moves.
struct ThreatContext {
    // enemy_lower[v]: squares attacked by an enemy piece of value class below v.
    std::array<chess::Bitboard, 5> enemy_lower{};
    chess::Bitboard enemy_attacks;
    chess::Bitboard own_attacks;
    chess::Bitboard own_attacked;  // Own pieces attacked by the enemy.
    // Squares from which a piece of each type would attack the enemy king.
    std::array<chess::Bitboard, 6> check_squares{};
};

[[nodiscard]] ThreatContext threat_context(const chess::Board& board) noexcept;

// A move relative to the side to move. Castling uses the king's destination.
// Eight bytes so it can be assembled in a register.
struct MoveInfo {
    std::uint8_t from = 0;
    std::uint8_t to = 0;
    std::uint8_t piece = 0;      // Moving piece type, 0..5.
    std::uint8_t victim = 0;     // Captured piece type 0..4, or no_victim.
    std::uint8_t promotion = 0;  // 0 for none, otherwise 1..4 for knight..queen.
    std::uint8_t threats = 0;    // ThreatBit mask.
    std::uint8_t reserved[2]{};
};
static_assert(sizeof(MoveInfo) == 8);

namespace detail {
// Value classes used to decide whether an attacker is "lower" than a piece.
inline constexpr std::array<int, 6> value_class{0, 1, 1, 2, 3, 4};
// Piece type of each chess::Piece value, with no_victim for an empty square.
inline constexpr std::array<int, 13> piece_type_of{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, no_victim};

[[nodiscard]] inline std::uint64_t bit(chess::Bitboard bitboard, int square) noexcept {
    return (bitboard.getBits() >> square) & 1U;
}
}  // namespace detail

// Inline because it runs for every move of every ordered node; it works on raw
// indices and bitboards rather than the library's wrappers.
[[nodiscard]] inline MoveInfo describe(const chess::Board& board, const ThreatContext& context, chess::Move move) noexcept {
    using detail::bit;
    const auto us = board.sideToMove();
    const int raw = move.move();
    const int origin = (raw >> 6) & 63;
    int target = raw & 63;
    const int type = raw & (3 << 14);
    const int mover = detail::piece_type_of[static_cast<std::size_t>(static_cast<int>(board.at(chess::Square(origin))))];
    int victim = detail::piece_type_of[static_cast<std::size_t>(static_cast<int>(board.at(chess::Square(target))))];
    int landing = mover;
    int promotion = 0;
    if (type != chess::Move::NORMAL) [[unlikely]] {
        if (type == chess::Move::CASTLING) {
            target = chess::Square::castling_king_square(target > origin, us).index();
            victim = no_victim;
        } else if (type == chess::Move::ENPASSANT) {
            victim = 0;
        } else {
            promotion = landing = ((raw >> 12) & 3) + 1;  // Knight..queen map to 1..4.
        }
    }
    const auto lower = [&](int piece) {
        return context.enemy_lower[static_cast<std::size_t>(detail::value_class[static_cast<std::size_t>(piece)])];
    };
    const std::uint64_t threats =
        bit(context.enemy_attacks, target) | bit(lower(landing), target) << 1 | bit(context.own_attacks, target) << 2 |
        bit(context.enemy_attacks, origin) << 3 | bit(lower(mover), origin) << 4 | bit(context.own_attacks, origin) << 5 |
        bit(context.check_squares[static_cast<std::size_t>(landing)], target) << 6 |
        static_cast<std::uint64_t>((context.own_attacked.getBits() & ~(std::uint64_t{1} << origin)) != 0) << 7;
    const auto flip = static_cast<std::uint64_t>(us == chess::Color::WHITE ? 0 : 56);
    const std::uint64_t packed = (static_cast<std::uint64_t>(origin) ^ flip) | (static_cast<std::uint64_t>(target) ^ flip) << 8 |
                                 static_cast<std::uint64_t>(mover) << 16 | static_cast<std::uint64_t>(victim) << 24 |
                                 static_cast<std::uint64_t>(promotion) << 32 | threats << 40;
    static_assert(std::endian::native == std::endian::little);
    return std::bit_cast<MoveInfo>(packed);
}

}  // namespace nmo::features
