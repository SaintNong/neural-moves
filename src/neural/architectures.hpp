#pragma once

#include <memory>

#include "nmo/neural.hpp"

namespace nmo::neural {

// Construct the architecture a weight file names, or return null if unknown.
[[nodiscard]] std::shared_ptr<Policy> make_policy(const WeightFile& file);

[[nodiscard]] std::shared_ptr<Policy> make_movemap(const WeightFile& file);
[[nodiscard]] std::shared_ptr<Policy> make_nnue_policy(const WeightFile& file);
[[nodiscard]] std::shared_ptr<Policy> make_upn(const WeightFile& file);
[[nodiscard]] std::shared_ptr<Policy> make_tactical(const WeightFile& file);
[[nodiscard]] std::shared_ptr<Policy> make_cnn(const WeightFile& file);
[[nodiscard]] std::shared_ptr<Policy> make_attention(const WeightFile& file);

}  // namespace nmo::neural
