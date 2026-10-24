#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "architectures.hpp"
#include "nmo/neural.hpp"

namespace nmo {
namespace {
constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t max_string = 1 << 16;
constexpr std::uint32_t max_rank = 8;

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : path_(path), input_(path, std::ios::binary) {
        if (!input_) fail("cannot open weight file");
    }

    template <class T>
    T read() {
        T value{};
        input_.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!input_) fail("truncated weight file");
        return value;
    }

    std::string string() {
        const auto length = read<std::uint32_t>();
        if (length > max_string) fail("oversized string in weight file");
        std::string value(length, '\0');
        input_.read(value.data(), length);
        if (!input_) fail("truncated weight file");
        return value;
    }

    void floats(std::vector<float>& values) {
        input_.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!input_) fail("truncated weight file");
    }

    bool at_end() { return input_.peek() == std::char_traits<char>::eof(); }

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(message + ": " + path_.string());
    }

private:
    std::filesystem::path path_;
    std::ifstream input_;
};

std::string shape_text(const std::vector<int>& shape) {
    std::string text = "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) text += ", ";
        text += std::to_string(shape[index]);
    }
    return text + "]";
}
}  // namespace

WeightFile::WeightFile(const std::filesystem::path& path) {
    Reader reader(path);
    char magic[4];
    for (auto& symbol : magic) symbol = reader.read<char>();
    if (std::memcmp(magic, "NMOW", 4) != 0) reader.fail("not a .nmo weight file");
    if (reader.read<std::uint32_t>() != format_version) reader.fail("unsupported weight file version");
    architecture_ = reader.string();
    const auto setting_count = reader.read<std::uint32_t>();
    for (std::uint32_t index = 0; index < setting_count; ++index) {
        auto key = reader.string();
        settings_[std::move(key)] = reader.read<double>();
    }
    const auto tensor_count = reader.read<std::uint32_t>();
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
        auto name = reader.string();
        const auto rank = reader.read<std::uint32_t>();
        if (rank > max_rank) reader.fail("tensor rank too large");
        Tensor tensor;
        std::size_t elements = 1;
        for (std::uint32_t axis = 0; axis < rank; ++axis) {
            const auto dimension = reader.read<std::uint32_t>();
            if (dimension == 0 || dimension > (1U << 24)) reader.fail("invalid tensor dimension");
            tensor.shape.push_back(static_cast<int>(dimension));
            elements *= dimension;
            if (elements > (std::size_t{1} << 28)) reader.fail("tensor too large");
        }
        tensor.data.resize(elements);
        reader.floats(tensor.data);
        tensors_[std::move(name)] = std::move(tensor);
    }
    if (!reader.at_end()) reader.fail("trailing data in weight file");
}

const std::vector<float>& WeightFile::tensor(const std::string& name, const std::vector<int>& shape) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) throw std::runtime_error("weight file lacks tensor " + name);
    if (found->second.shape != shape) {
        throw std::runtime_error("tensor " + name + " has shape " + shape_text(found->second.shape) +
                                 ", expected " + shape_text(shape));
    }
    return found->second.data;
}

const std::vector<int>& WeightFile::shape(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) throw std::runtime_error("weight file lacks tensor " + name);
    return found->second.shape;
}

double WeightFile::setting(const std::string& key, double fallback) const {
    const auto found = settings_.find(key);
    return found == settings_.end() ? fallback : found->second;
}

std::shared_ptr<Policy> Policy::load(const std::filesystem::path& path) {
    const WeightFile file(path);
    std::shared_ptr<Policy> policy = neural::make_policy(file);
    if (!policy) throw std::runtime_error("unknown network architecture '" + file.architecture() + "' in " + path.string());
    policy->history_weight = file.setting("history_weight", 0.0);
    policy->quiescence = file.setting("quiescence", 0.0) != 0.0;
    policy->staged = file.setting("staged", 0.0) != 0.0;
    return policy;
}

}  // namespace nmo
