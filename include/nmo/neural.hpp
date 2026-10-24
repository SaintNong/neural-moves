#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <chess.hpp>

namespace nmo {

// Named float tensors and numeric settings read from a .nmo file written by
// training/nmo_train/export.py. Layout (little-endian): "NMOW", u32 version,
// architecture string, u32 setting count of (string, f64), u32 tensor count of
// (string, u32 rank, u32 dims[rank], f32 data[]). Strings are u32 length + bytes.
class WeightFile {
public:
    // Throws std::runtime_error if the file is missing or malformed.
    explicit WeightFile(const std::filesystem::path& path);

    [[nodiscard]] const std::string& architecture() const noexcept { return architecture_; }
    [[nodiscard]] bool has(const std::string& name) const { return tensors_.contains(name); }
    // Throws std::runtime_error if the tensor is missing or has another shape.
    [[nodiscard]] const std::vector<float>& tensor(const std::string& name, const std::vector<int>& shape) const;
    [[nodiscard]] const std::vector<int>& shape(const std::string& name) const;
    [[nodiscard]] double setting(const std::string& key, double fallback) const;

private:
    struct Tensor {
        std::vector<int> shape;
        std::vector<float> data;
    };
    std::string architecture_;
    std::map<std::string, Tensor, std::less<>> tensors_;
    std::map<std::string, double, std::less<>> settings_;
};

// Mutable per-search state of a policy network, such as incrementally updated
// accumulators. The orderer pushes each move before Board::makeMove and pops it
// after Board::unmakeMove, so the state always describes the current board.
// child_depth is the remaining depth of the node the move leads to (0 for leaf
// and quiescence nodes); it never increases along a path, so a state may skip
// work in subtrees where it will not be evaluated.
class PolicyState {
public:
    virtual ~PolicyState() = default;
    virtual void reset(const chess::Board& root) = 0;
    virtual void push(const chess::Board& before, chess::Move move, int child_depth) = 0;
    virtual void pop() = 0;
    // Write one logit per move of the current position; larger logits search
    // first. Depth is the node's remaining depth, zero for quiescence nodes.
    virtual void evaluate(const chess::Board& board, const chess::Movelist& moves, int depth, float* logits) = 0;
};

// Immutable network weights plus how the orderer should use them. Safe to share
// between searchers; each searcher creates its own PolicyState.
class Policy {
public:
    virtual ~Policy() = default;
    [[nodiscard]] virtual std::unique_ptr<PolicyState> new_state() const = 0;

    // Blend weight for the gravity history of quiet moves, in logits per history
    // unit; zero orders purely by the network.
    double history_weight = 0.0;
    // Order quiescence captures with the policy (its network, or its leaf head
    // when it has one) instead of MVV-LVA.
    bool quiescence = false;
    // Search captures and promotions first by MVV-LVA, as mvv-lva-history
    // does, and use the network only to order quiet moves.
    bool staged = false;

    // Construct the architecture named in a weight file. Settings in the file
    // initialise history_weight, quiescence and staged. Throws
    // std::runtime_error.
    [[nodiscard]] static std::shared_ptr<Policy> load(const std::filesystem::path& path);
};

}  // namespace nmo
