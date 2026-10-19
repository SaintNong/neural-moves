#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <chess.hpp>

namespace nmo {

// Describes whether a stored search score is exact or a bound on the value.
enum class Bound : std::uint8_t { exact, lower, upper };

struct TranspositionEntry {
    std::uint64_t key = 0;
    int score = 0;
    std::int16_t depth = -1;  // Negative depth marks an unused slot.
    Bound bound = Bound::upper;
    std::uint32_t generation = 0;
};

class TranspositionTable {
public:
    // Allocate the largest power-of-two entry count that fits the MiB budget.
    // Zero disables storage. Throws std::invalid_argument on size overflow;
    // allocation may fail.
    explicit TranspositionTable(std::size_t megabytes = 16);

    // Invalidate entries without clearing the table's storage.
    void clear() noexcept;
    // Return the indexed entry when its key and generation match and it is
    // initialized; nullptr indicates a miss. Contents may change after store()
    // or clear().
    [[nodiscard]] const TranspositionEntry* probe(std::uint64_t key) const noexcept;
    // Replace the indexed entry unconditionally. Depth must be in [0, max_depth]
    // and mate scores must use position-relative encoding (score_to_table).
    void store(std::uint64_t key, int depth, int score, Bound bound) noexcept;
    // Number of direct-mapped entries and their storage size, excluding vector overhead.
    [[nodiscard]] std::size_t capacity() const noexcept { return buckets_.size(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return buckets_.size() * sizeof(Bucket); }

    // Use the chess library's position hash directly as the table key.
    [[nodiscard]] static std::uint64_t position_key(const chess::Board& board) noexcept;

private:
    using Bucket = TranspositionEntry;
    std::vector<Bucket> buckets_;
    std::uint32_t generation_ = 1;
};

// Convert root-ply mate scores to position-relative scores before storing and
// convert them back when probing. Ordinary evaluation scores are unchanged.
[[nodiscard]] int score_to_table(int score, int ply) noexcept;
[[nodiscard]] int score_from_table(int score, int ply) noexcept;

}  // namespace nmo

