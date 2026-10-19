#include "run_progress.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <indicators/color.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/terminal_size.hpp>

struct RunProgress::Impl {
    explicit Impl(std::uint64_t count)
        : total(count), started_at(std::chrono::steady_clock::now()) {
        if (!stderr_is_terminal()) return;

        auto columns = indicators::terminal_width();
        if (columns == 0) columns = 80;
        const auto count_width = position_text(total).size();
        const auto fixed_width = std::string_view("Search ").size() + 2 + 5 + count_width + 1;
        constexpr std::size_t time_width = 21;
        const bool show_times = columns >= fixed_width + time_width + 10;
        show_time_estimate = show_times;
        const auto reserved = fixed_width + (show_times ? time_width : 0) + 2;
        const auto bar_width = std::clamp(columns > reserved ? columns - reserved : std::size_t{8},
                                          std::size_t{8}, std::size_t{28});

        bar = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{bar_width},
            indicators::option::PrefixText{"Search "},
            indicators::option::Start{"["},
            indicators::option::Fill{"━"},
            indicators::option::Lead{"━"},
            indicators::option::Remainder{"─"},
            indicators::option::End{"]"},
            indicators::option::PostfixText{postfix_text(0)},
            indicators::option::ShowPercentage{true},
            indicators::option::FillColor{indicators::Color::cyan},
            indicators::option::LeadColor{indicators::Color::yellow},
            indicators::option::RemainderColor{indicators::Color::grey},
            indicators::option::MaxProgress{total},
            indicators::option::Stream{std::cerr});
        hide_cursor();
        bar->set_progress(0);
    }

    ~Impl() { restore_cursor(); }

    static bool stderr_is_terminal() noexcept {
#ifdef _WIN32
        return _isatty(_fileno(stderr)) != 0;
#else
        return isatty(STDERR_FILENO) != 0;
#endif
    }

    void hide_cursor() noexcept {
#ifdef _WIN32
        const auto console = GetStdHandle(STD_ERROR_HANDLE);
        CONSOLE_CURSOR_INFO info{};
        if (GetConsoleCursorInfo(console, &info)) {
            cursor_was_visible = info.bVisible != FALSE;
            cursor_state_known = true;
            if (cursor_was_visible) {
                info.bVisible = FALSE;
                SetConsoleCursorInfo(console, &info);
            }
        }
#else
        std::cerr << "\033[?25l" << std::flush;
        cursor_state_known = true;
#endif
    }

    void restore_cursor() noexcept {
        if (!cursor_state_known) return;
#ifdef _WIN32
        const auto console = GetStdHandle(STD_ERROR_HANDLE);
        CONSOLE_CURSOR_INFO info{};
        if (GetConsoleCursorInfo(console, &info)) {
            info.bVisible = cursor_was_visible ? TRUE : FALSE;
            SetConsoleCursorInfo(console, &info);
        }
#else
        std::cerr << "\033[?25h" << std::flush;
#endif
        cursor_state_known = false;
    }

    std::string position_text(std::uint64_t completed) const {
        return " " + std::to_string(completed) + "/" + std::to_string(total);
    }

    static std::string format_duration(std::chrono::seconds duration) {
        const auto seconds = (std::max)(duration.count(), std::chrono::seconds::rep{0});
        const auto hours = seconds / 3600;
        const auto minutes = (seconds / 60) % 60;
        const auto remaining_seconds = seconds % 60;

        std::ostringstream output;
        output << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2)
               << minutes << ':' << std::setw(2) << remaining_seconds;
        return output.str();
    }

    std::string postfix_text(std::uint64_t completed) const {
        auto text = position_text(completed);
        if (!show_time_estimate) return text;

        const auto elapsed = std::chrono::steady_clock::now() - started_at;
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(elapsed);
        text += " " + format_duration(elapsed_seconds) + " / ~";

        if (completed == 0 || total == 0) {
            text += "--:--:--";
            return text;
        }

        const auto elapsed_fraction = std::chrono::duration<long double>(elapsed).count();
        const auto estimated_seconds = std::ceil(
            elapsed_fraction * static_cast<long double>(total) /
            static_cast<long double>(completed));
        const auto max_seconds = static_cast<long double>(
            (std::numeric_limits<std::chrono::seconds::rep>::max)());
        const auto bounded_seconds = (std::min)(estimated_seconds, max_seconds);
        text += format_duration(std::chrono::seconds{
            static_cast<std::chrono::seconds::rep>(bounded_seconds)});
        return text;
    }

    std::uint64_t total;
    std::unique_ptr<indicators::ProgressBar> bar;
    std::chrono::steady_clock::time_point started_at;
    bool show_time_estimate = false;
    bool cursor_state_known = false;
    bool cursor_was_visible = true;
};

RunProgress::RunProgress(std::uint64_t total) : impl_(std::make_unique<Impl>(total)) {}

RunProgress::~RunProgress() = default;

void RunProgress::update(std::uint64_t completed) {
    if (!impl_->bar) return;
    const auto progress = std::min(completed, impl_->total);
    impl_->bar->set_option(indicators::option::PostfixText{impl_->postfix_text(progress)});
    impl_->bar->set_progress(static_cast<std::size_t>(progress));
    if (progress == impl_->total) impl_->restore_cursor();
}
