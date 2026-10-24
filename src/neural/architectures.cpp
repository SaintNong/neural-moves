#include "architectures.hpp"

namespace nmo::neural {

std::shared_ptr<Policy> make_policy(const WeightFile& file) {
    const auto& name = file.architecture();
    if (name == "movemap") return make_movemap(file);
    if (name == "nnue-policy") return make_nnue_policy(file);
    if (name == "upn") return make_upn(file);
    if (name == "tactical") return make_tactical(file);
    if (name == "cnn") return make_cnn(file);
    if (name == "attention") return make_attention(file);
    return nullptr;
}

}  // namespace nmo::neural
