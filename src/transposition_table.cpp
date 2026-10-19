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
    // A fresh root search must not see earlier scores. An epoch avoids writing
    // the entire allocation for every short benchmark search.
    ++generation_;
    if (generation_ == 0) {
        std::fill(buckets_.begin(), buckets_.end(), Bucket{});
        generation_ = 1;
    }
}

const TranspositionEntry* TranspositionTable::probe(std::uint64_t key) const noexcept {
    if (buckets_.empty()) return nullptr;
    const auto& bucket = buckets_[key & (buckets_.size() - 1)];
    for (const auto& entry : bucket) {
        if (entry.generation == generation_ && entry.depth >= 0 && entry.key == key) return &entry;
    }
    return nullptr;
}

void TranspositionTable::store(std::uint64_t key, int depth, int score, Bound bound, chess::Move best_move) noexcept {
    if (buckets_.empty()) return;
    auto& bucket = buckets_[key & (buckets_.size() - 1)];
    auto* target = &bucket.front();
    const auto replacement_depth = [this](const TranspositionEntry& entry) {
        return entry.generation == generation_ ? entry.depth : -1;
    };
    for (auto& entry : bucket) {
        if (entry.generation == generation_ && entry.depth >= 0 && entry.key == key) {
            // Preserve a deeper entry, or an exact value over an equal-depth bound.
            if (entry.depth > depth || (entry.depth == depth && entry.bound == Bound::exact && bound != Bound::exact)) {
                return;
            }
            target = &entry;
            break;
        }
        if (replacement_depth(entry) < replacement_depth(*target)) target = &entry;
    }
    *target = {key, best_move, score, static_cast<std::int16_t>(depth), bound, generation_};
}

std::uint64_t TranspositionTable::position_key(const chess::Board& board) noexcept {
    // SplitMix64 finalizer gives each rule-clock value an independent salt.
    auto clock = static_cast<std::uint64_t>(board.halfMoveClock()) + 0x9e3779b97f4a7c15ULL;
    clock = (clock ^ (clock >> 30)) * 0xbf58476d1ce4e5b9ULL;
    clock = (clock ^ (clock >> 27)) * 0x94d049bb133111ebULL;
    return board.hash() ^ (clock ^ (clock >> 31));
}

int score_to_table(int score, int ply) noexcept {
    if (score >= mate_score - max_depth) return score + ply;
    if (score <= -mate_score + max_depth) return score - ply;
    return score;
}

int score_from_table(int score, int ply) noexcept {
    if (score >= mate_score - max_depth) return score - ply;
    if (score <= -mate_score + max_depth) return score + ply;
    return score;
}

}  // namespace nmo

