// Efficiently updatable policies: "nnue-policy" (Monty-style) and "upn" (ours).
// See training/nmo_train/models.py for the matching PyTorch definitions.

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "accumulator.hpp"
#include "architectures.hpp"
#include "common.hpp"

namespace nmo::neural {
namespace {

// Activations in [0, 1] are quantized to 0..127 for the int8 move table.
constexpr float activation_scale = 127.0F;

class NnuePolicy final : public Policy {
public:
    NnuePolicy(const WeightFile& file, bool upn) : upn_(upn) {
        hidden_ = static_cast<int>(file.setting("hidden", 0));
        width_ = 2 * hidden_;
        if (hidden_ <= 0 || hidden_ % 8 != 0) throw std::runtime_error("hidden size must be a positive multiple of 8");
        activation_ = static_cast<Activation>(static_cast<int>(file.setting("activation", 1)));
        input_weight_ = load_tensor(file, "input.weight", {768, hidden_});
        input_bias_ = load_tensor(file, "input.bias", {hidden_});
        move_weight_ = load_tensor(file, "move.weight", {4096, width_});
        move_bias_ = load_tensor(file, "move.bias", {4096});
        promotion_weight_ = load_tensor(file, "promotion.weight", {5, width_});
        promotion_bias_ = load_tensor(file, "promotion.bias", {5});
        // Every move adds promotion row 0 ("none"); fold it into the move rows
        // and make the other promotion rows relative to it.
        for (int slot = 0; slot < 4096; ++slot) {
            for (int index = 0; index < width_; ++index) move_weight_[at(slot, index)] += promotion_weight_[static_cast<std::size_t>(index)];
            move_bias_[static_cast<std::size_t>(slot)] += promotion_bias_[0];
        }
        for (int promotion = 4; promotion >= 0; --promotion) {
            for (int index = 0; index < width_; ++index) {
                promotion_weight_[at(promotion, index)] -= promotion_weight_[static_cast<std::size_t>(index)];
            }
            promotion_bias_[static_cast<std::size_t>(promotion)] -= promotion_bias_[0];
        }
        // Clipped activations allow an int8 move table with one scale per row.
        quantized_ = activation_ != Activation::relu && width_ % 32 == 0 && file.setting("quantized", 1.0) != 0.0;
        if (quantized_) quantize_moves();
        if (!upn_) return;
        context_ = static_cast<int>(file.setting("context", 0));
        buckets_ = static_cast<int>(file.setting("depth_buckets", 1));
        if (context_ < 0 || context_ % 8 != 0) throw std::runtime_error("context size must be a multiple of 8");
        constexpr int threats = features::threat_combinations;
        tactic_bias_ = load_tensor(file, "tactic.bias", {buckets_ * 6 * 7 * threats});
        if (context_ > 0) {
            project_weight_ = load_tensor(file, "project.weight", {context_, width_});
            project_bias_ = load_tensor(file, "project.bias", {context_});
            threat_ = load_tensor(file, "threat.weight", {buckets_ * 6 * threats, context_});
            capture_ = load_tensor(file, "capture.weight", {6 * 7 * 5, context_});
        }
        // Optional tactical leaf head, used instead of the network at nodes with
        // remaining depth <= leaf_depth (0: quiescence only). Those nodes then
        // never touch the accumulator.
        if (file.has("leaf.tactic.bias")) {
            leaf_depth_ = static_cast<int>(file.setting("leaf_depth", 0));
            leaf_tactic_ = load_tensor(file, "leaf.tactic.bias", {(leaf_depth_ + 1) * 6 * 7 * threats});
            leaf_move_ = load_tensor(file, "leaf.move.bias", {4096});
            leaf_promotion_ = load_tensor(file, "leaf.promotion.bias", {5});
        }
    }

    [[nodiscard]] std::unique_ptr<PolicyState> new_state() const override;

private:
    friend class NnueState;
    [[nodiscard]] std::size_t at(int row, int column) const noexcept {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(column);
    }

    void quantize_moves() {
        move_quantized_ = AlignedArray<std::int8_t>(static_cast<std::size_t>(4096) * static_cast<std::size_t>(width_));
        move_scale_ = AlignedFloats(4096);
        for (int slot = 0; slot < 4096; ++slot) {
            float largest = 1e-12F;
            for (int index = 0; index < width_; ++index) largest = std::max(largest, std::abs(move_weight_[at(slot, index)]));
            const float scale = 127.0F / largest;
            for (int index = 0; index < width_; ++index) {
                move_quantized_[at(slot, index)] = static_cast<std::int8_t>(std::lround(move_weight_[at(slot, index)] * scale));
            }
            move_scale_[static_cast<std::size_t>(slot)] = 1.0F / (scale * activation_scale);
        }
    }

    bool upn_;
    bool quantized_ = false;
    int hidden_ = 0;
    int width_ = 0;
    int context_ = 0;
    int buckets_ = 1;
    Activation activation_ = Activation::crelu;
    AlignedFloats input_weight_, input_bias_, move_weight_, move_bias_, promotion_weight_, promotion_bias_;
    AlignedArray<std::int8_t> move_quantized_;
    AlignedFloats move_scale_;
    AlignedFloats tactic_bias_, project_weight_, project_bias_, threat_, capture_;
    int leaf_depth_ = -1;
    AlignedFloats leaf_tactic_, leaf_move_, leaf_promotion_;
};

class NnueState final : public PolicyState {
public:
    explicit NnueState(const NnuePolicy& net)
        : net_(net), stack_(net.input_weight_.data(), net.input_bias_.data(), net.hidden_),
          hidden_(static_cast<std::size_t>(net.width_)), quantized_(static_cast<std::size_t>(net.width_)),
          context_(static_cast<std::size_t>(std::max(net.context_, 8))),
          scratch_(static_cast<std::size_t>(std::max(net.context_, 8))),
          untracked_depth_(net.leaf_depth_ >= 0 ? net.leaf_depth_ : net.quiescence ? -1 : 0) {}

    void reset(const chess::Board& root) override { stack_.reset(root); }
    // Subtrees whose nodes are all ordered by the leaf head (or by MVV-LVA in
    // quiescence) never read the accumulator, so their moves are not tracked.
    void push(const chess::Board& before, chess::Move move, int child_depth) override {
        stack_.push(before, move, child_depth > untracked_depth_);
    }
    void pop() override { stack_.pop(); }

    void evaluate(const chess::Board& board, const chess::Movelist& moves, int depth, float* logits) override {
        if (depth <= net_.leaf_depth_) {
            evaluate_leaf(board, moves, depth, logits);
            return;
        }
        const auto us = board.sideToMove();
        const int hidden = net_.hidden_;
        activate(stack_.get(board, us), hidden_.data(), hidden, net_.activation_);
        activate(stack_.get(board, ~us), hidden_.data() + hidden, hidden, net_.activation_);
        if (net_.quantized_) {
            for (int index = 0; index < net_.width_; ++index) {
                quantized_[static_cast<std::size_t>(index)] =
                    static_cast<std::uint8_t>(hidden_[static_cast<std::size_t>(index)] * activation_scale + 0.5F);
            }
        }

        if (!net_.upn_) {
            for (int index = 0; index < moves.size(); ++index) {
                const auto squares = move_squares(us, moves[index]);
                logits[index] = move_logit(squares.from * 64 + squares.to, squares.promotion);
            }
            return;
        }

        // Low-rank position context for the tactical interaction terms.
        const int context = net_.context_;
        for (int row = 0; row < context; ++row) {
            context_[static_cast<std::size_t>(row)] = dot(net_.project_weight_.data() + net_.at(row, 0), hidden_.data(),
                                                          net_.width_) +
                                                      net_.project_bias_[static_cast<std::size_t>(row)];
        }
        const int bucket = std::min(depth, net_.buckets_ - 1);
        const auto threats = features::threat_context(board);
        constexpr int threat_count = features::threat_combinations;
        for (int index = 0; index < moves.size(); ++index) {
            const auto info = features::describe(board, threats, moves[index]);
            const int piece = info.piece;
            float logit = move_logit(info.from * 64 + info.to, info.promotion);
            logit += net_.tactic_bias_[static_cast<std::size_t>(((bucket * 6 + piece) * 7 + info.victim) * threat_count + info.threats)];
            if (context == 0) {
                logits[index] = logit;
                continue;
            }
            const float* threat = net_.threat_.data() + static_cast<std::size_t>((bucket * 6 + piece) * threat_count + info.threats) *
                                                            static_cast<std::size_t>(context);
            const float* capture = net_.capture_.data() + static_cast<std::size_t>((piece * 7 + info.victim) * 5 + info.promotion) *
                                                              static_cast<std::size_t>(context);
            for (int row = 0; row < context; ++row) scratch_[static_cast<std::size_t>(row)] = threat[row] + capture[row];
            logits[index] = logit + dot(scratch_.data(), context_.data(), context);
        }
    }

private:
    void evaluate_leaf(const chess::Board& board, const chess::Movelist& moves, int depth, float* logits) const noexcept {
        const auto threats = features::threat_context(board);
        for (int index = 0; index < moves.size(); ++index) {
            const auto info = features::describe(board, threats, moves[index]);
            const auto tactic = ((depth * 6 + info.piece) * 7 + info.victim) * features::threat_combinations + info.threats;
            logits[index] = net_.leaf_tactic_[static_cast<std::size_t>(tactic)] +
                            net_.leaf_move_[static_cast<std::size_t>(info.from * 64 + info.to)] +
                            net_.leaf_promotion_[info.promotion];
        }
    }

    // Position-dependent score of a (from, to) slot plus its promotion terms.
    [[nodiscard]] float move_logit(int slot, int promotion) const noexcept {
        const int width = net_.width_;
        float logit = net_.move_bias_[static_cast<std::size_t>(slot)];
        if (net_.quantized_) {
            logit += static_cast<float>(dot_u8i8(quantized_.data(), net_.move_quantized_.data() + net_.at(slot, 0), width)) *
                     net_.move_scale_[static_cast<std::size_t>(slot)];
        } else {
            logit += dot(net_.move_weight_.data() + net_.at(slot, 0), hidden_.data(), width);
        }
        if (promotion != 0) {
            logit += dot(net_.promotion_weight_.data() + net_.at(promotion, 0), hidden_.data(), width) +
                     net_.promotion_bias_[static_cast<std::size_t>(promotion)];
        }
        return logit;
    }

    const NnuePolicy& net_;
    AccumulatorStack stack_;
    AlignedFloats hidden_;
    AlignedArray<std::uint8_t> quantized_;
    AlignedFloats context_;
    AlignedFloats scratch_;
    int untracked_depth_;  // Deepest remaining depth never evaluated by the network.
};

std::unique_ptr<PolicyState> NnuePolicy::new_state() const { return std::make_unique<NnueState>(*this); }

}  // namespace

std::shared_ptr<Policy> make_nnue_policy(const WeightFile& file) { return std::make_shared<NnuePolicy>(file, false); }
std::shared_ptr<Policy> make_upn(const WeightFile& file) { return std::make_shared<NnuePolicy>(file, true); }

}  // namespace nmo::neural
