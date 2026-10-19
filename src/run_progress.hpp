#pragma once

#include <cstdint>
#include <memory>

class RunProgress {
public:
    explicit RunProgress(std::uint64_t total);
    ~RunProgress();

    RunProgress(const RunProgress&) = delete;
    RunProgress& operator=(const RunProgress&) = delete;

    void update(std::uint64_t completed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
