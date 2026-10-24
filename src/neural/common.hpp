#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <chess.hpp>

#include "nmo/features.hpp"
#include "nmo/neural.hpp"
#include "simd.hpp"

namespace nmo::neural {

[[nodiscard]] inline AlignedFloats load_tensor(const WeightFile& file, const std::string& name,
                                               const std::vector<int>& shape) {
    const auto& data = file.tensor(name, shape);
    AlignedFloats out(data.size());
    std::copy(data.begin(), data.end(), out.data());
    return out;
}

enum class Activation { relu, crelu, screlu };

inline void activate(const float* in, float* out, int size, Activation activation) noexcept {
    for (int index = 0; index < size; ++index) {
        const float value = in[index];
        switch (activation) {
            case Activation::relu: out[index] = std::max(value, 0.0F); break;
            case Activation::crelu: out[index] = std::clamp(value, 0.0F, 1.0F); break;
            case Activation::screlu: {
                const float clipped = std::clamp(value, 0.0F, 1.0F);
                out[index] = clipped * clipped;
                break;
            }
        }
    }
}

// Squares of a move relative to the side to move; castling uses the king's
// destination, matching features::describe.
struct MoveSquares {
    int from = 0;
    int to = 0;
    int promotion = 0;  // 0 for none, otherwise 1..4 for knight..queen.
};

[[nodiscard]] inline MoveSquares move_squares(chess::Color us, chess::Move move) noexcept {
    auto to = move.to();
    if (move.typeOf() == chess::Move::CASTLING) to = chess::Square::castling_king_square(to > move.from(), us);
    const int promotion = move.typeOf() == chess::Move::PROMOTION ? static_cast<int>(move.promotionType()) : 0;
    return {features::relative_square(us, move.from().index()), features::relative_square(us, to.index()), promotion};
}

// State for networks evaluated from scratch at every node.
class StatelessState : public PolicyState {
public:
    void reset(const chess::Board&) override {}
    void push(const chess::Board&, chess::Move, int) override {}
    void pop() override {}
};

}  // namespace nmo::neural
