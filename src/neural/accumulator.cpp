#include "accumulator.hpp"

#include <stdexcept>

namespace nmo::neural {

AccumulatorStack::AccumulatorStack(const float* weights, const float* bias, int hidden)
    : weights_(weights), bias_(bias), hidden_(hidden),
      values_(static_cast<std::size_t>(capacity) * 2 * static_cast<std::size_t>(hidden)) {}

void AccumulatorStack::reset(const chess::Board& root) {
    ply_ = 0;
    refresh(root, 0);
}

void AccumulatorStack::refresh(const chess::Board& board, int ply) {
    for (int perspective = 0; perspective < 2; ++perspective) {
        const auto active = features::active_features(board, chess::Color(perspective));
        float* out = slot(ply, perspective);
        for (int index = 0; index < hidden_; ++index) out[index] = bias_[index];
        for (int feature = 0; feature < active.count; ++feature) {
            const float* row = weights_ + static_cast<std::ptrdiff_t>(active.index[static_cast<std::size_t>(feature)]) * hidden_;
            for (int index = 0; index < hidden_; ++index) out[index] += row[index];
        }
    }
    computed_[static_cast<std::size_t>(ply)] = true;
    tracked_[static_cast<std::size_t>(ply)] = true;
}

void AccumulatorStack::push(const chess::Board& before, chess::Move move, bool tracked) {
    if (ply_ + 1 >= capacity) throw std::length_error("search path exceeds the accumulator stack");
    ++ply_;
    const auto index = static_cast<std::size_t>(ply_);
    computed_[index] = false;
    tracked_[index] = tracked;
    if (tracked) deltas_[index] = features::move_delta(before, move);
}

const float* AccumulatorStack::get(const chess::Board& board, chess::Color perspective) {
    if (!computed_[static_cast<std::size_t>(ply_)]) {
        int base = ply_;
        while (!computed_[static_cast<std::size_t>(base)] && tracked_[static_cast<std::size_t>(base)]) --base;
        if (!computed_[static_cast<std::size_t>(base)]) {
            // An untracked move lies on the path: rebuild from the board.
            refresh(board, ply_);
            return slot(ply_, static_cast<int>(perspective));
        }
        for (int ply = base + 1; ply <= ply_; ++ply) {
            const auto& delta = deltas_[static_cast<std::size_t>(ply)];
            for (int side = 0; side < 2; ++side) {
                const chess::Color view(side);
                const float* added[2];
                const float* removed[2];
                for (int index = 0; index < delta.added_count; ++index) {
                    const auto& change = delta.added[static_cast<std::size_t>(index)];
                    added[index] = weights_ + static_cast<std::ptrdiff_t>(
                                                  features::piece_square(view, change.piece, change.square)) * hidden_;
                }
                for (int index = 0; index < delta.removed_count; ++index) {
                    const auto& change = delta.removed[static_cast<std::size_t>(index)];
                    removed[index] = weights_ + static_cast<std::ptrdiff_t>(
                                                    features::piece_square(view, change.piece, change.square)) * hidden_;
                }
                apply_rows(slot(ply, side), slot(ply - 1, side), added, delta.added_count, removed,
                           delta.removed_count, hidden_);
            }
            computed_[static_cast<std::size_t>(ply)] = true;
        }
    }
    return slot(ply_, static_cast<int>(perspective));
}

}  // namespace nmo::neural
