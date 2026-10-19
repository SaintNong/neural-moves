#include "nmo/evaluation.hpp"

#include <stdexcept>

namespace nmo {

int evaluate_material(const chess::Board& board) noexcept {
    int score = 0;
    for (int type = 0; type < 5; ++type) {
        const auto piece = chess::PieceType(static_cast<chess::PieceType::underlying>(type));
        score += piece_values[static_cast<std::size_t>(type)] *
                 (board.pieces(piece, chess::Color::WHITE).count() -
                  board.pieces(piece, chess::Color::BLACK).count());
    }
    return board.sideToMove() == chess::Color::WHITE ? score : -score;
}

bool is_insufficient_material(const chess::Board& board) noexcept {
    if (board.pieces(chess::PieceType::PAWN, chess::PieceType::ROOK, chess::PieceType::QUEEN)) return false;
    const auto bishops = board.pieces(chess::PieceType::BISHOP).getBits();
    const auto knights = board.pieces(chess::PieceType::KNIGHT).count();
    if (bishops == 0) return knights <= 1;
    if (knights != 0) return false;
    // Include arbitrarily many promoted bishops. The library's basic detector
    // only checks up to four total pieces; Python also draws this larger case.
    constexpr std::uint64_t dark_squares = 0xaa55aa55aa55aa55ULL;
    return (bishops & dark_squares) == 0 || (bishops & ~dark_squares) == 0;
}

double estimate_ebf(std::uint64_t total_nodes, int depth) {
    if (total_nodes < 1 || depth < 1) {
        throw std::invalid_argument("EBF requires at least one node and positive depth");
    }
    if (total_nodes == 1) {
        return 0.0;
    }
    const auto target = static_cast<double>(total_nodes);
    const auto tree_size = [depth, target](double branching) {
        double sum = 1.0;
        double level = 1.0;
        for (int ply = 0; ply < depth && sum < target; ++ply) {
            level *= branching;
            sum += level;
        }
        return sum;
    };
    double low = 0.0;
    double high = 1.0;
    while (tree_size(high) < target) {
        high *= 2.0;
    }
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double mid = (low + high) / 2.0;
        if (tree_size(mid) < target) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return (low + high) / 2.0;
}

}  // namespace nmo

