// AlphaZero/Lc0-style residual CNN with a 73x64 convolutional policy head,
// recomputed from scratch at every node. Batch norm is folded into the
// convolutions by the exporter.

#include <array>
#include <cstdlib>
#include <stdexcept>

#include "architectures.hpp"
#include "board_planes.hpp"
#include "common.hpp"

namespace nmo::neural {
namespace {

// AlphaZero move planes, matching alphazero_plane_table() in models.py.
constexpr std::array<std::array<int, 2>, 8> queen_directions{{{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}}};
constexpr std::array<std::array<int, 2>, 8> knight_deltas{{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}};

int policy_plane(int from, int to, int promotion) noexcept {
    const int df = to % 8 - from % 8;
    const int dr = to / 8 - from / 8;
    if (promotion >= 1 && promotion <= 3 && std::abs(df) <= 1 && dr == 1) return 64 + (promotion - 1) * 3 + df + 1;
    for (int index = 0; index < 8; ++index) {
        if (knight_deltas[static_cast<std::size_t>(index)][0] == df && knight_deltas[static_cast<std::size_t>(index)][1] == dr)
            return 56 + index;
    }
    const int sx = (df > 0) - (df < 0);
    const int sy = (dr > 0) - (dr < 0);
    for (int index = 0; index < 8; ++index) {
        if (queen_directions[static_cast<std::size_t>(index)][0] == sx && queen_directions[static_cast<std::size_t>(index)][1] == sy)
            return index * 7 + std::max(std::abs(df), std::abs(dr)) - 1;
    }
    return 0;
}

struct Conv {
    int inputs = 0;
    int outputs = 0;
    AlignedFloats weight;  // [outputs][inputs][9]
    AlignedFloats bias;    // [outputs]
};

Conv load_conv(const WeightFile& file, const std::string& name, int inputs, int outputs) {
    return {inputs, outputs, load_tensor(file, name + ".weight", {outputs, inputs, 3, 3}),
            load_tensor(file, name + ".bias", {outputs})};
}

class CnnPolicy final : public Policy {
public:
    explicit CnnPolicy(const WeightFile& file) {
        filters_ = static_cast<int>(file.setting("filters", 0));
        const int blocks = static_cast<int>(file.setting("blocks", 0));
        if (filters_ <= 0 || blocks < 0) throw std::runtime_error("invalid CNN shape");
        stem_ = load_conv(file, "stem", plane_count, filters_);
        for (int block = 0; block < blocks; ++block) {
            const auto prefix = "block" + std::to_string(block);
            tower_.push_back({load_conv(file, prefix + ".conv1", filters_, filters_),
                              load_conv(file, prefix + ".conv2", filters_, filters_)});
        }
        head_ = load_conv(file, "head", filters_, filters_);
        policy_weight_ = load_tensor(file, "policy.weight", {73, filters_, 1, 1});
        policy_bias_ = load_tensor(file, "policy.bias", {73});
    }

    [[nodiscard]] std::unique_ptr<PolicyState> new_state() const override;

    int filters_ = 0;
    Conv stem_, head_;
    std::vector<std::array<Conv, 2>> tower_;
    AlignedFloats policy_weight_, policy_bias_;
};

class CnnState final : public StatelessState {
public:
    explicit CnnState(const CnnPolicy& net)
        : net_(net), planes_(plane_count * 64), shifted_(static_cast<std::size_t>(std::max(net.filters_, plane_count)) * 9 * 64),
          x_(static_cast<std::size_t>(net.filters_) * 64), y_(static_cast<std::size_t>(net.filters_) * 64),
          z_(static_cast<std::size_t>(net.filters_) * 64) {}

    void evaluate(const chess::Board& board, const chess::Movelist& moves, int, float* logits) override {
        board_planes(board, planes_.data());
        convolve(net_.stem_, planes_.data(), x_.data(), nullptr);
        for (const auto& block : net_.tower_) {
            convolve(block[0], x_.data(), y_.data(), nullptr);
            convolve(block[1], y_.data(), z_.data(), x_.data());
            std::swap(x_, z_);
        }
        convolve(net_.head_, x_.data(), y_.data(), nullptr);
        const auto us = board.sideToMove();
        const int filters = net_.filters_;
        for (int index = 0; index < moves.size(); ++index) {
            const auto squares = move_squares(us, moves[index]);
            const int plane = policy_plane(squares.from, squares.to, squares.promotion);
            float logit = net_.policy_bias_[static_cast<std::size_t>(plane)];
            const float* weight = net_.policy_weight_.data() + static_cast<std::ptrdiff_t>(plane) * filters;
            for (int channel = 0; channel < filters; ++channel) {
                logit += weight[channel] * y_[static_cast<std::size_t>(channel * 64 + squares.from)];
            }
            logits[index] = logit;
        }
    }

private:
    // out = relu(conv(in) + bias [+ residual]) with zero padding on the 8x8 board.
    void convolve(const Conv& conv, const float* in, float* out, const float* residual) {
        for (int channel = 0; channel < conv.inputs; ++channel) {
            for (int tap = 0; tap < 9; ++tap) {
                const int dy = tap / 3 - 1;
                const int dx = tap % 3 - 1;
                float* plane = shifted_.data() + (channel * 9 + tap) * 64;
                for (int square = 0; square < 64; ++square) {
                    const int row = square / 8 + dy;
                    const int column = square % 8 + dx;
                    plane[square] = row >= 0 && row < 8 && column >= 0 && column < 8 ? in[channel * 64 + row * 8 + column] : 0.0F;
                }
            }
        }
        for (int output = 0; output < conv.outputs; ++output) {
            float sum[64];
            for (float& value : sum) value = conv.bias[static_cast<std::size_t>(output)];
            const float* weights = conv.weight.data() + static_cast<std::ptrdiff_t>(output) * conv.inputs * 9;
            for (int tap = 0; tap < conv.inputs * 9; ++tap) {
                const float weight = weights[tap];
                const float* plane = shifted_.data() + tap * 64;
                for (int square = 0; square < 64; ++square) sum[square] += weight * plane[square];
            }
            float* target = out + output * 64;
            for (int square = 0; square < 64; ++square) {
                const float value = residual ? sum[square] + residual[output * 64 + square] : sum[square];
                target[square] = std::max(value, 0.0F);
            }
        }
    }

    const CnnPolicy& net_;
    AlignedFloats planes_, shifted_, x_, y_, z_;
};

std::unique_ptr<PolicyState> CnnPolicy::new_state() const { return std::make_unique<CnnState>(*this); }

}  // namespace

std::shared_ptr<Policy> make_cnn(const WeightFile& file) { return std::make_shared<CnnPolicy>(file); }

}  // namespace nmo::neural
