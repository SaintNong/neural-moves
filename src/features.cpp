#include "nmo/features.hpp"

#include <bit>

namespace nmo::features {
namespace {


chess::Bitboard pawn_attacks(chess::Color color, chess::Bitboard pawns) noexcept {
    using chess::attacks;
    if (color == chess::Color::WHITE) {
        return attacks::pawnLeftAttacks<chess::Color::WHITE>(pawns) |
               attacks::pawnRightAttacks<chess::Color::WHITE>(pawns);
    }
    return attacks::pawnLeftAttacks<chess::Color::BLACK>(pawns) |
           attacks::pawnRightAttacks<chess::Color::BLACK>(pawns);
}

// Attacks of one side, indexed by attacking piece type.
std::array<chess::Bitboard, 6> attacks_by_type(const chess::Board& board, chess::Color color) noexcept {
    using chess::attacks;
    using chess::PieceType;
    const auto occupied = board.occ();
    std::array<chess::Bitboard, 6> result{};
    result[0] = pawn_attacks(color, board.pieces(PieceType::PAWN, color));
    const auto accumulate = [&](PieceType type, auto attack) {
        auto pieces = board.pieces(type, color).getBits();
        chess::Bitboard total;
        while (pieces != 0) {
            const auto square = chess::Square(std::countr_zero(pieces));
            pieces &= pieces - 1;
            total = total | attack(square);
        }
        result[static_cast<std::size_t>(static_cast<int>(type))] = total;
    };
    accumulate(PieceType::KNIGHT, [](chess::Square sq) { return attacks::knight(sq); });
    accumulate(PieceType::BISHOP, [&](chess::Square sq) { return attacks::bishop(sq, occupied); });
    accumulate(PieceType::ROOK, [&](chess::Square sq) { return attacks::rook(sq, occupied); });
    accumulate(PieceType::QUEEN, [&](chess::Square sq) { return attacks::queen(sq, occupied); });
    result[5] = attacks::king(board.kingSq(color));
    return result;
}
}  // namespace

ActiveFeatures active_features(const chess::Board& board, chess::Color perspective) noexcept {
    ActiveFeatures features;
    auto occupied = board.occ().getBits();
    while (occupied != 0 && features.count < max_active) {
        const int square = std::countr_zero(occupied);
        occupied &= occupied - 1;
        features.index[static_cast<std::size_t>(features.count++)] = static_cast<std::int16_t>(
            piece_square(perspective, board.at(chess::Square(square)), square));
    }
    return features;
}

MoveDelta move_delta(const chess::Board& before, chess::Move move) noexcept {
    MoveDelta delta;
    const auto us = before.sideToMove();
    const auto from = move.from();
    const auto to = move.to();
    const auto piece = before.at(from);
    if (move.typeOf() == chess::Move::CASTLING) {
        // The library encodes castling as the king capturing its own rook.
        const bool king_side = to > from;
        delta.removed = {{{piece, from.index()}, {before.at(to), to.index()}}};
        delta.added = {{{piece, chess::Square::castling_king_square(king_side, us).index()},
                        {before.at(to), chess::Square::castling_rook_square(king_side, us).index()}}};
        delta.removed_count = delta.added_count = 2;
        return delta;
    }
    delta.removed[static_cast<std::size_t>(delta.removed_count++)] = {piece, from.index()};
    if (move.typeOf() == chess::Move::ENPASSANT) {
        delta.removed[static_cast<std::size_t>(delta.removed_count++)] = {
            chess::Piece(chess::PieceType::PAWN, ~us), to.ep_square().index()};
    } else if (before.at(to) != chess::Piece::NONE) {
        delta.removed[static_cast<std::size_t>(delta.removed_count++)] = {before.at(to), to.index()};
    }
    const auto landing = move.typeOf() == chess::Move::PROMOTION ? chess::Piece(move.promotionType(), us) : piece;
    delta.added[static_cast<std::size_t>(delta.added_count++)] = {landing, to.index()};
    return delta;
}

ThreatContext threat_context(const chess::Board& board) noexcept {
    const auto us = board.sideToMove();
    const auto them = ~us;
    const auto enemy = attacks_by_type(board, them);
    const auto own = attacks_by_type(board, us);

    ThreatContext context;
    context.enemy_lower[1] = enemy[0];
    context.enemy_lower[2] = context.enemy_lower[1] | enemy[1] | enemy[2];
    context.enemy_lower[3] = context.enemy_lower[2] | enemy[3];
    context.enemy_lower[4] = context.enemy_lower[3] | enemy[4];
    context.enemy_attacks = context.enemy_lower[4] | enemy[5];
    context.own_attacks = own[0] | own[1] | own[2] | own[3] | own[4] | own[5];
    context.own_attacked = context.enemy_attacks & board.us(us);

    using chess::attacks;
    const auto king = board.kingSq(them);
    const auto occupied = board.occ();
    context.check_squares[0] = attacks::pawn(them, king);
    context.check_squares[1] = attacks::knight(king);
    context.check_squares[2] = attacks::bishop(king, occupied);
    context.check_squares[3] = attacks::rook(king, occupied);
    context.check_squares[4] = context.check_squares[2] | context.check_squares[3];
    return context;
}

}  // namespace nmo::features
