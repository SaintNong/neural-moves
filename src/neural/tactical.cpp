// Ablation of UPN without a network: learned tactical and move biases only.

#include "architectures.hpp"
#include "common.hpp"

namespace nmo::neural {
namespace {

class TacticalPolicy final : public Policy {
public:
    explicit TacticalPolicy(const WeightFile& file) {
        buckets_ = static_cast<int>(file.setting("depth_buckets", 1));
        tactic_bias_ = load_tensor(file, "tactic.bias", {buckets_ * 6 * 7 * features::threat_combinations});
        move_bias_ = load_tensor(file, "move.bias", {4096});
        promotion_bias_ = load_tensor(file, "promotion.bias", {5});
    }

    [[nodiscard]] std::unique_ptr<PolicyState> new_state() const override;

    int buckets_ = 1;
    AlignedFloats tactic_bias_, move_bias_, promotion_bias_;
};

class TacticalState final : public StatelessState {
public:
    explicit TacticalState(const TacticalPolicy& net) : net_(net) {}

    void evaluate(const chess::Board& board, const chess::Movelist& moves, int depth, float* logits) override {
        const int bucket = std::min(depth, net_.buckets_ - 1);
        const auto threats = features::threat_context(board);
        for (int index = 0; index < moves.size(); ++index) {
            const auto info = features::describe(board, threats, moves[index]);
            const auto tactic = ((bucket * 6 + info.piece) * 7 + info.victim) * features::threat_combinations + info.threats;
            logits[index] = net_.tactic_bias_[static_cast<std::size_t>(tactic)] +
                            net_.move_bias_[static_cast<std::size_t>(info.from * 64 + info.to)] +
                            net_.promotion_bias_[info.promotion];
        }
    }

private:
    const TacticalPolicy& net_;
};

std::unique_ptr<PolicyState> TacticalPolicy::new_state() const { return std::make_unique<TacticalState>(*this); }

}  // namespace

std::shared_ptr<Policy> make_tactical(const WeightFile& file) { return std::make_shared<TacticalPolicy>(file); }

}  // namespace nmo::neural
