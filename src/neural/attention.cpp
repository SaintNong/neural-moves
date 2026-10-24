// Transformer over the 64 squares with an Lc0-style attention policy head,
// recomputed from scratch at every node.

#include <cmath>
#include <stdexcept>

#include "architectures.hpp"
#include "board_planes.hpp"
#include "common.hpp"

namespace nmo::neural {
namespace {

struct Linear {
    int inputs = 0;
    int outputs = 0;
    AlignedFloats weight;  // [outputs][inputs]
    AlignedFloats bias;

    // out[t][o] for each of `tokens` input rows of length `inputs`.
    void apply(const float* in, float* out, int tokens) const noexcept {
        for (int token = 0; token < tokens; ++token) {
            const float* row = in + static_cast<std::ptrdiff_t>(token) * inputs;
            for (int output = 0; output < outputs; ++output) {
                out[static_cast<std::ptrdiff_t>(token) * outputs + output] =
                    dot(weight.data() + static_cast<std::ptrdiff_t>(output) * inputs, row, inputs) +
                    bias[static_cast<std::size_t>(output)];
            }
        }
    }
};

Linear load_linear(const WeightFile& file, const std::string& name, int inputs, int outputs) {
    return {inputs, outputs, load_tensor(file, name + ".weight", {outputs, inputs}), load_tensor(file, name + ".bias", {outputs})};
}

struct Norm {
    AlignedFloats weight, bias;
};

Norm load_norm(const WeightFile& file, const std::string& name, int dim) {
    return {load_tensor(file, name + ".weight", {dim}), load_tensor(file, name + ".bias", {dim})};
}

struct Layer {
    Norm norm1, norm2;
    Linear qkv, attn_out, ffn1, ffn2;
};

class AttentionPolicy final : public Policy {
public:
    explicit AttentionPolicy(const WeightFile& file) {
        dim_ = static_cast<int>(file.setting("dim", 0));
        heads_ = static_cast<int>(file.setting("heads", 0));
        ffn_ = static_cast<int>(file.setting("ffn", 0));
        const int layers = static_cast<int>(file.setting("layers", 0));
        if (dim_ <= 0 || dim_ % 8 != 0 || heads_ <= 0 || dim_ % (heads_ * 8) != 0 || ffn_ % 8 != 0) {
            throw std::runtime_error("invalid attention network shape");
        }
        embed_weight_ = load_tensor(file, "embed.weight", {dim_, plane_count});
        embed_bias_ = load_tensor(file, "embed.bias", {dim_});
        position_ = load_tensor(file, "position", {64, dim_});
        for (int layer = 0; layer < layers; ++layer) {
            const auto prefix = "layer" + std::to_string(layer) + ".";
            layers_.push_back({load_norm(file, prefix + "norm1", dim_), load_norm(file, prefix + "norm2", dim_),
                               load_linear(file, prefix + "qkv", dim_, 3 * dim_),
                               load_linear(file, prefix + "attn_out", dim_, dim_),
                               load_linear(file, prefix + "ffn1", dim_, ffn_), load_linear(file, prefix + "ffn2", ffn_, dim_)});
        }
        final_norm_ = load_norm(file, "final_norm", dim_);
        query_ = load_linear(file, "query", dim_, dim_);
        key_ = load_linear(file, "key", dim_, dim_);
        promotion_ = load_tensor(file, "promotion.weight", {5, dim_});
    }

    [[nodiscard]] std::unique_ptr<PolicyState> new_state() const override;

    int dim_ = 0, heads_ = 0, ffn_ = 0;
    AlignedFloats embed_weight_, embed_bias_, position_, promotion_;
    std::vector<Layer> layers_;
    Norm final_norm_;
    Linear query_, key_;
};

class AttentionState final : public StatelessState {
public:
    explicit AttentionState(const AttentionPolicy& net)
        : net_(net), planes_(plane_count * 64), x_(64 * size(net.dim_)), y_(64 * size(net.dim_)),
          qkv_(64 * 3 * size(net.dim_)), q_(64 * size(net.dim_)), k_(64 * size(net.dim_)), v_(64 * size(net.dim_)),
          attended_(64 * size(net.dim_)), hidden_(64 * size(std::max(net.ffn_, net.dim_))) {}

    void evaluate(const chess::Board& board, const chess::Movelist& moves, int, float* logits) override {
        const int dim = net_.dim_;
        board_planes(board, planes_.data());
        for (int square = 0; square < 64; ++square) {
            for (int channel = 0; channel < dim; ++channel) {
                float value = net_.embed_bias_[size(channel)] + net_.position_[size(square * dim + channel)];
                for (int plane = 0; plane < plane_count; ++plane) {
                    value += net_.embed_weight_[size(channel * plane_count + plane)] * planes_[size(plane * 64 + square)];
                }
                x_[size(square * dim + channel)] = value;
            }
        }
        for (const auto& layer : net_.layers_) {
            normalize(layer.norm1, x_.data(), y_.data());
            attend(layer);
            layer.attn_out.apply(attended_.data(), y_.data(), 64);
            for (std::size_t index = 0; index < size(64 * dim); ++index) x_[index] += y_[index];
            normalize(layer.norm2, x_.data(), y_.data());
            layer.ffn1.apply(y_.data(), hidden_.data(), 64);
            for (std::size_t index = 0; index < size(64 * net_.ffn_); ++index) hidden_[index] = std::max(hidden_[index], 0.0F);
            layer.ffn2.apply(hidden_.data(), y_.data(), 64);
            for (std::size_t index = 0; index < size(64 * dim); ++index) x_[index] += y_[index];
        }
        normalize(net_.final_norm_, x_.data(), y_.data());
        net_.query_.apply(y_.data(), q_.data(), 64);
        net_.key_.apply(y_.data(), k_.data(), 64);
        const auto us = board.sideToMove();
        const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
        for (int index = 0; index < moves.size(); ++index) {
            const auto squares = move_squares(us, moves[index]);
            const float* query = q_.data() + squares.from * dim;
            const float* key = k_.data() + squares.to * dim;
            const float* promotion = net_.promotion_.data() + squares.promotion * dim;
            float logit = 0.0F;
            for (int channel = 0; channel < dim; ++channel) logit += (query[channel] + promotion[channel]) * key[channel];
            logits[index] = logit * scale;
        }
    }

private:
    static std::size_t size(int value) noexcept { return static_cast<std::size_t>(value); }

    void normalize(const Norm& norm, const float* in, float* out) const noexcept {
        const int dim = net_.dim_;
        for (int token = 0; token < 64; ++token) {
            const float* row = in + token * dim;
            float mean = 0.0F;
            for (int channel = 0; channel < dim; ++channel) mean += row[channel];
            mean /= static_cast<float>(dim);
            float variance = 0.0F;
            for (int channel = 0; channel < dim; ++channel) variance += (row[channel] - mean) * (row[channel] - mean);
            const float inverse = 1.0F / std::sqrt(variance / static_cast<float>(dim) + 1e-5F);
            for (int channel = 0; channel < dim; ++channel) {
                out[token * dim + channel] = (row[channel] - mean) * inverse * norm.weight[size(channel)] + norm.bias[size(channel)];
            }
        }
    }

    // Multi-head self-attention of y_ into attended_ (heads concatenated).
    void attend(const Layer& layer) noexcept {
        const int dim = net_.dim_;
        const int head_dim = dim / net_.heads_;
        layer.qkv.apply(y_.data(), qkv_.data(), 64);
        const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        for (int head = 0; head < net_.heads_; ++head) {
            // Gather contiguous per-head rows so dot products are SIMD-friendly.
            for (int token = 0; token < 64; ++token) {
                for (int channel = 0; channel < head_dim; ++channel) {
                    const int source = token * 3 * dim + head * head_dim + channel;
                    q_[size(token * head_dim + channel)] = qkv_[size(source)];
                    k_[size(token * head_dim + channel)] = qkv_[size(source + dim)];
                    v_[size(token * head_dim + channel)] = qkv_[size(source + 2 * dim)];
                }
            }
            for (int token = 0; token < 64; ++token) {
                float weights[64];
                float largest = -1e30F;
                for (int other = 0; other < 64; ++other) {
                    weights[other] = dot(q_.data() + token * head_dim, k_.data() + other * head_dim, head_dim) * scale;
                    largest = std::max(largest, weights[other]);
                }
                float total = 0.0F;
                for (float& weight : weights) {
                    weight = std::exp(weight - largest);
                    total += weight;
                }
                float* out = attended_.data() + token * dim + head * head_dim;
                for (int channel = 0; channel < head_dim; ++channel) out[channel] = 0.0F;
                for (int other = 0; other < 64; ++other) {
                    const float weight = weights[other] / total;
                    const float* value = v_.data() + other * head_dim;
                    for (int channel = 0; channel < head_dim; ++channel) out[channel] += weight * value[channel];
                }
            }
        }
    }

    const AttentionPolicy& net_;
    AlignedFloats planes_, x_, y_, qkv_, q_, k_, v_, attended_, hidden_;
};

std::unique_ptr<PolicyState> AttentionPolicy::new_state() const { return std::make_unique<AttentionState>(*this); }

}  // namespace

std::shared_ptr<Policy> make_attention(const WeightFile& file) { return std::make_shared<AttentionPolicy>(file); }

}  // namespace nmo::neural
