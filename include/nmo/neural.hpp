#pragma once

#include <stdexcept>

#include <chess.hpp>

namespace nmo {

// Extension point only. No model, feature encoding, weights, or inference exists.
class NeuralPolicy {
public:
    [[noreturn]] void rank(const chess::Board&, chess::Movelist&) const {
        throw std::logic_error("neural move ordering is a stub; inference is not implemented");
    }

    [[noreturn]] void train() const {
        throw std::logic_error("neural training is a stub; training is not implemented");
    }
};

}  // namespace nmo

