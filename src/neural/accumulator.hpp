#pragma once

#include <array>
#include <cstdint>

#include <chess.hpp>

#include "nmo/features.hpp"
#include "simd.hpp"

namespace nmo::neural {

// NNUE-style feature transformer state along the current search path. Both
// perspectives (white and black) are kept per ply. Pushing a move only records
// its piece changes; the accumulator is computed on first use from the nearest
// computed ancestor, so nodes that never order moves cost nothing.
class AccumulatorStack {
public:
    static constexpr int capacity = 256;

    // weights: [768, hidden] rows indexed by features::piece_square; bias: [hidden].
    AccumulatorStack(const float* weights, const float* bias, int hidden);

    void reset(const chess::Board& root);
    // Untracked pushes record no change: their subtree must not need the
    // accumulator, or get() falls back to a full refresh.
    // Throws std::length_error beyond `capacity` plies.
    void push(const chess::Board& before, chess::Move move, bool tracked = true);
    void pop() noexcept { --ply_; }
    // Accumulator of a perspective for the current position, `board`.
    [[nodiscard]] const float* get(const chess::Board& board, chess::Color perspective);

private:
    float* slot(int ply, int perspective) noexcept {
        return values_.data() + (static_cast<std::size_t>(ply) * 2 + static_cast<std::size_t>(perspective)) *
                                    static_cast<std::size_t>(hidden_);
    }

    const float* weights_;
    const float* bias_;
    int hidden_;
    int ply_ = 0;
    AlignedFloats values_;
    void refresh(const chess::Board& board, int ply);

    std::array<features::MoveDelta, capacity> deltas_{};
    std::array<bool, capacity> computed_{};
    std::array<bool, capacity> tracked_{};
};

}  // namespace nmo::neural
