#include "nmo/transposition_table.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

#include "nmo/evaluation.hpp"

namespace nmo {

TranspositionTable::TranspositionTable(std::size_t megabytes) {
    constexpr std::size_t mib = 1024 * 1024;
    if (megabytes > std::numeric_limits<std::size_t>::max() / mib) {
        throw std::invalid_argument("transposition table size is too large");
    }
    const auto count = megabytes * mib / sizeof(Bucket);
    if (count > 0) {
        buckets_.resize(std::bit_floor(count));
    }
}

void TranspositionTable::clear() noexcept {
    // Advancing the generation invalidates old entries without touching each
    // slot. On wraparound, reset the slots before reusing generation one.
    ++generation_;
    if (generation_ == 0) {
        std::fill(buckets_.begin(), buckets_.end(), Bucket{});
        generation_ = 1;
    }
}

const TranspositionEntry* TranspositionTable::probe(std::uint64_t key) const noexcept {
    if (buckets_.empty())
        return nullptr;
    const auto& entry = buckets_[key & (buckets_.size() - 1)];
    if (entry.generation != generation_ || entry.depth < 0 || entry.key != key)
        return nullptr;
    return &entry;
}

void TranspositionTable::store(std::uint64_t key, int depth, int score, Bound bound,
                               chess::Move best_move) noexcept {
    if (buckets_.empty())
        return;
    buckets_[key & (buckets_.size() - 1)] = {
        key, score, static_cast<std::int16_t>(depth), bound, generation_, best_move};
}

std::uint64_t TranspositionTable::position_key(const chess::Board& board) noexcept {
    return board.hash();
}

int score_to_table(int score, int ply) noexcept {
    if (score >= mate_score - max_depth)
        return score + ply;
    if (score <= -mate_score + max_depth)
        return score - ply;
    return score;
}

int score_from_table(int score, int ply) noexcept {
    if (score >= mate_score - max_depth)
        return score - ply;
    if (score <= -mate_score + max_depth)
        return score + ply;
    return score;
}

}  // namespace nmo

