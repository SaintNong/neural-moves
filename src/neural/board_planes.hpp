#pragma once

#include <chess.hpp>

#include "common.hpp"

namespace nmo::neural {

inline constexpr int plane_count = 18;

// [18][64] input planes relative to the side to move, matching board_planes()
// in training/nmo_train/models.py: 12 piece planes (own pieces first), 4
// castling planes (own king/queen side, then the opponent's), an en passant
// plane and a constant plane.
inline void board_planes(const chess::Board& board, float* planes) noexcept {
    for (int index = 0; index < plane_count * 64; ++index) planes[index] = 0.0F;
    const auto us = board.sideToMove();
    const auto active = features::active_features(board, us);
    for (int feature = 0; feature < active.count; ++feature) planes[active.index[static_cast<std::size_t>(feature)]] = 1.0F;
    using Side = chess::Board::CastlingRights::Side;
    const auto rights = board.castlingRights();
    const bool castling[4] = {rights.has(us, Side::KING_SIDE), rights.has(us, Side::QUEEN_SIDE),
                              rights.has(~us, Side::KING_SIDE), rights.has(~us, Side::QUEEN_SIDE)};
    for (int bit = 0; bit < 4; ++bit) {
        if (!castling[bit]) continue;
        for (int square = 0; square < 64; ++square) planes[(12 + bit) * 64 + square] = 1.0F;
    }
    if (board.enpassantSq() != chess::Square::NO_SQ) {
        planes[16 * 64 + 40 + static_cast<int>(board.enpassantSq().file())] = 1.0F;
    }
    for (int square = 0; square < 64; ++square) planes[17 * 64 + square] = 1.0F;
}

}  // namespace nmo::neural
