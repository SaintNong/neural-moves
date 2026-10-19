#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <chess.hpp>

namespace nmo {

enum class Bound : std::uint8_t { exact, lower, upper };

struct TranspositionEntry {
    std::uint64_t key = 0;
    int score = 0;
    std::int16_t depth = -1;  // An empty slot can also hold key zero.
    Bound bound = Bound::upper;
    std::uint32_t generation = 0;
};

class TranspositionTable {
public:
    // Budget in MiB, rounded down to a supported bucket count; zero disables
    // storage. Throws std::invalid_argument on size overflow; allocation may fail.
    explicit TranspositionTable(std::size_t megabytes = 16);

    // Invalidate all entries while retaining the allocated storage.
    void clear() noexcept;
    // Null on a miss. Returned entry contents may change after store/clear.
    [[nodiscard]] const TranspositionEntry* probe(std::uint64_t key) const noexcept;
    // Depth must be in [0, max_depth]; score must use position-relative mate
    // encoding (score_to_table). Replacement may retain a stronger existing entry.
    void store(std::uint64_t key, int depth, int score, Bound bound) noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return buckets_.size() * ways; }
    [[nodiscard]] std::size_t bytes() const noexcept { return buckets_.size() * sizeof(Bucket); }

    // The library's Zobrist hash omits the rule clock; cached scores must include it.
    [[nodiscard]] static std::uint64_t position_key(const chess::Board& board) noexcept;

private:
    static constexpr std::size_t ways = 4;
    using Bucket = std::array<TranspositionEntry, ways>;
    std::vector<Bucket> buckets_;
    std::uint32_t generation_ = 1;
};

// Mate scores stored in the TT are relative to the cached position, not the root.
[[nodiscard]] int score_to_table(int score, int ply) noexcept;
[[nodiscard]] int score_from_table(int score, int ply) noexcept;

}  // namespace nmo

