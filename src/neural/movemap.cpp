// Neural MoveMap (Kocsis et al., 2003): recomputed from scratch at every node.

#include <stdexcept>

#include "architectures.hpp"
#include "common.hpp"

namespace nmo::neural {
namespace {

class MoveMapPolicy final : public Policy {
public:
    explicit MoveMapPolicy(const WeightFile& file) {
        hidden_ = static_cast<int>(file.setting("hidden", 0));
        if (hidden_ <= 0 || hidden_ % 8 != 0) throw std::runtime_error("hidden size must be a positive multiple of 8");
        input_weight_ = load_tensor(file, "input.weight", {768, hidden_});
        input_bias_ = load_tensor(file, "input.bias", {hidden_});
        output_weight_ = load_tensor(file, "output.weight", {128, hidden_});
        output_bias_ = load_tensor(file, "output.bias", {128});
    }

    [[nodiscard]] std::unique_ptr<PolicyState> new_state() const override;

    int hidden_ = 0;
    AlignedFloats input_weight_, input_bias_, output_weight_, output_bias_;
};

class MoveMapState final : public StatelessState {
public:
    explicit MoveMapState(const MoveMapPolicy& net)
        : net_(net), hidden_(static_cast<std::size_t>(net.hidden_)), output_(128) {}

    void evaluate(const chess::Board& board, const chess::Movelist& moves, int, float* logits) override {
        const int hidden = net_.hidden_;
        const auto us = board.sideToMove();
        const auto active = features::active_features(board, us);
        float* h = hidden_.data();
        for (int index = 0; index < hidden; ++index) h[index] = net_.input_bias_[static_cast<std::size_t>(index)];
        for (int feature = 0; feature < active.count; ++feature) {
            const float* row = net_.input_weight_.data() +
                               static_cast<std::ptrdiff_t>(active.index[static_cast<std::size_t>(feature)]) * hidden;
            for (int index = 0; index < hidden; ++index) h[index] += row[index];
        }
        activate(h, h, hidden, Activation::relu);
        for (int unit = 0; unit < 128; ++unit) {
            output_[static_cast<std::size_t>(unit)] =
                dot(net_.output_weight_.data() + static_cast<std::ptrdiff_t>(unit) * hidden, h, hidden) +
                net_.output_bias_[static_cast<std::size_t>(unit)];
        }
        for (int index = 0; index < moves.size(); ++index) {
            const auto squares = move_squares(us, moves[index]);
            logits[index] = output_[static_cast<std::size_t>(squares.from)] + output_[static_cast<std::size_t>(64 + squares.to)];
        }
    }

private:
    const MoveMapPolicy& net_;
    AlignedFloats hidden_;
    AlignedFloats output_;
};

std::unique_ptr<PolicyState> MoveMapPolicy::new_state() const { return std::make_unique<MoveMapState>(*this); }

}  // namespace

std::shared_ptr<Policy> make_movemap(const WeightFile& file) { return std::make_shared<MoveMapPolicy>(file); }

}  // namespace nmo::neural
