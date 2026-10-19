#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "nmo/evaluation.hpp"
#include "nmo/neural.hpp"
#include "nmo/ordering.hpp"
#include "nmo/positions.hpp"
#include "nmo/search.hpp"
#include "run_progress.hpp"

namespace {
struct Arguments {
    std::optional<std::filesystem::path> positions;
    std::optional<std::string> fen;
    std::optional<std::filesystem::path> results;
    int depth = 4;
    std::size_t table_megabytes = 16;
    std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    bool iterative = true;
    bool csv = false;
    bool help = false;
    bool version = false;
    std::vector<nmo::OrderingStrategy> strategies;
};

void print_help() {
    std::cout << R"(Usage: neural-move-ordering POSITIONS [options]
       neural-move-ordering --fen "FEN" [options]

Compare chess move ordering using material-only alpha-beta search.
POSITIONS contains one FEN or FEN | legal_uci_move per line.
Blank lines and lines beginning with # are ignored.

Options:
  --depth N           Search depth in plies, 1..64 (default: 4)
  --strategy NAME     Select a strategy; repeat to compare several (default: all)
                      mvv-lva, history, history-gravity, capture-hist,
                      mvv-lva-history; neural is an unimplemented stub
  --tt-mb N           Transposition table memory budget, 0..4096 MiB (default: 16)
                      0 disables the table; actual allocation rounds down
  --no-iterative      Search only the requested depth
  --limit N           Read at most N positions (positive)
  --csv               Print the summary as CSV
  --results FILE      Write per-position moves, scores, and statistics as CSV
  --fen FEN           Search a single FEN instead of a file
  --help, -h          Show this help
  --version           Show version

Neural inference and the 'train' subcommand are intentionally unimplemented.
)";
}

std::uint64_t integer(std::string_view value, std::string_view option, std::uint64_t low, std::uint64_t high) {
    std::uint64_t number = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end != value.data() + value.size() || number < low || number > high) {
        throw std::invalid_argument(std::string(option) + " must be an integer in [" + std::to_string(low) +
                                    ", " + std::to_string(high) + ']');
    }
    return number;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        const auto value = [&]() -> std::string_view {
            if (index + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(arg));
            return argv[++index];
        };
        if (arg == "--help" || arg == "-h") args.help = true;
        else if (arg == "--version") args.version = true;
        else if (arg == "--depth") args.depth = static_cast<int>(integer(value(), arg, 1, nmo::max_depth));
        else if (arg == "--tt-mb") args.table_megabytes = static_cast<std::size_t>(integer(value(), arg, 0, 4096));
        else if (arg == "--limit") args.limit = integer(value(), arg, 1, std::numeric_limits<std::uint64_t>::max());
        else if (arg == "--fen") {
            if (args.fen) throw std::invalid_argument("--fen can be specified only once");
            args.fen = value();
        } else if (arg == "--results") {
            if (args.results) throw std::invalid_argument("--results can be specified only once");
            args.results = value();
        } else if (arg == "--strategy") {
            const auto strategy = nmo::parse_strategy(value());
            if (std::find(args.strategies.begin(), args.strategies.end(), strategy) != args.strategies.end()) {
                throw std::invalid_argument("duplicate strategy");
            }
            args.strategies.push_back(strategy);
        } else if (arg == "--no-iterative") args.iterative = false;
        else if (arg == "--csv") args.csv = true;
        else if (arg == "train" && index == 1) nmo::NeuralPolicy{}.train();
        else if (arg.starts_with('-')) throw std::invalid_argument("unknown option: " + std::string(arg));
        else {
            if (args.positions) throw std::invalid_argument("provide only one positions file");
            args.positions = std::filesystem::path(arg);
        }
    }
    if (args.help || args.version) return args;
    if (args.positions.has_value() == args.fen.has_value()) {
        throw std::invalid_argument("provide a positions file or --fen");
    }
    if (args.strategies.empty()) {
        args.strategies.assign(nmo::heuristic_strategies.begin(), nmo::heuristic_strategies.end());
    }
    if (std::find(args.strategies.begin(), args.strategies.end(), nmo::OrderingStrategy::neural) != args.strategies.end()) {
        throw std::logic_error("neural move ordering is a stub; inference is not implemented");
    }
    if (args.results && args.positions) {
        std::error_code error;
        if (std::filesystem::equivalent(*args.results, *args.positions, error)) {
            throw std::invalid_argument("results file must differ from the positions file");
        }
    }
    return args;
}

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t cutoffs = 0;
    std::uint64_t tt_hits = 0;
    std::uint64_t tt_cutoffs = 0;
    double ebf = 0.0;
    double seconds = 0.0;
};

std::uint64_t count_positions(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open positions file: " + path.string());

    std::uint64_t count = 0;
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r\n\f\v");
        if (first == std::string::npos || line[first] == '#') continue;
        if (count == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("positions file contains too many records");
        }
        ++count;
    }
    if (input.bad()) throw std::runtime_error("failed to read positions file: " + path.string());
    return count;
}

void csv_field(std::ostream& out, std::string_view value) {
    out << '"';
    for (const char symbol : value) {
        if (symbol == '"') out << '"';
        out << symbol;
    }
    out << '"';
}

std::string move_text(chess::Move move) {
    return move == chess::Move::NO_MOVE ? "" : chess::uci::moveToUci(move);
}

void write_result(std::ostream& out, std::uint64_t index, const nmo::Position& position,
                  nmo::OrderingStrategy strategy, const nmo::SearchResult& result, double seconds) {
    out << index << ',' << position.line_number << ',';
    csv_field(out, position.fen);
    out << ',' << nmo::strategy_name(strategy) << ',' << result.depth << ',' << move_text(result.best_move) << ',';
    if (position.label) out << move_text(*position.label);
    out << ',' << result.score_cp << ',' << result.nodes << ',' << result.cutoffs << ',' << result.tt_hits
        << ',' << result.tt_cutoffs << ',' << std::fixed << std::setprecision(6) << result.ebf() << ','
        << std::setprecision(9) << seconds << '\n';
}

void print_summary(const Arguments& args, const std::vector<Totals>& totals, std::uint64_t count) {
    if (args.csv) {
        std::cout << "strategy,positions,depth,mean_ebf,mean_nodes,total_nodes,cutoffs,tt_hits,tt_cutoffs,seconds\n";
    } else {
        std::cout << "Positions: " << count << " | depth: " << args.depth << " | iterative: "
                  << (args.iterative ? "yes" : "no") << '\n';
        std::cout << std::left << std::setw(19) << "Strategy" << std::right
                  << std::setw(11) << "Mean EBF" << std::setw(14) << "Mean nodes"
                  << std::setw(15) << "Total nodes" << std::setw(13) << "Cutoffs"
                  << std::setw(13) << "TT hits" << std::setw(13) << "TT cutoffs" << std::setw(12) << "Time (s)" << '\n';
    }
    const auto divisor = static_cast<double>(count);
    for (std::size_t index = 0; index < totals.size(); ++index) {
        const auto& total = totals[index];
        if (args.csv) {
            std::cout << nmo::strategy_name(args.strategies[index]) << ',' << count << ',' << args.depth
                      << ',' << std::fixed << std::setprecision(6) << total.ebf / divisor << ','
                      << static_cast<double>(total.nodes) / divisor << ',' << total.nodes << ',' << total.cutoffs
                      << ',' << total.tt_hits << ',' << total.tt_cutoffs << ',' << total.seconds << '\n';
        } else {
            std::cout << std::left << std::setw(19) << nmo::strategy_name(args.strategies[index]) << std::right
                      << std::fixed << std::setprecision(3) << std::setw(11) << total.ebf / divisor
                      << std::setprecision(1) << std::setw(14) << static_cast<double>(total.nodes) / divisor
                      << std::setw(15) << total.nodes << std::setw(13) << total.cutoffs
                      << std::setw(13) << total.tt_hits << std::setw(13) << total.tt_cutoffs
                      << std::setprecision(3) << std::setw(12) << total.seconds << '\n';
        }
    }
}

int run(const Arguments& args) {
    std::unique_ptr<nmo::PositionReader> reader;
    std::optional<nmo::Position> single_position;
    std::uint64_t total_positions = 1;
    if (args.positions) {
        total_positions = std::min(count_positions(*args.positions), args.limit);
        if (total_positions == 0) throw std::invalid_argument("positions input contains no positions");
        reader = std::make_unique<nmo::PositionReader>(*args.positions);
    } else {
        single_position = nmo::parse_position(*args.fen);
    }

    std::ofstream detailed;
    if (args.results) {
        detailed.open(*args.results);
        if (!detailed) throw std::runtime_error("cannot open results file: " + args.results->string());
        detailed << "position,line,fen,strategy,depth,best_move,label,score_cp,nodes,cutoffs,tt_hits,tt_cutoffs,ebf,seconds\n";
    }

    nmo::AlphaBetaSearcher searcher({args.table_megabytes, args.iterative});
    std::vector<Totals> totals(args.strategies.size());
    std::uint64_t count = 0;
    std::cerr << "TT: " << searcher.table_bytes() / (1024 * 1024) << " MiB allocated; searching positions\n";
    RunProgress progress(total_positions);
    while (count < args.limit) {
        auto position = reader ? reader->next() : std::exchange(single_position, std::nullopt);
        if (!position) break;
        ++count;
        std::optional<int> expected_score;
        for (std::size_t index = 0; index < args.strategies.size(); ++index) {
            const auto strategy = args.strategies[index];
            const auto begin = std::chrono::steady_clock::now();
            const auto result = searcher.search(position->board, args.depth, strategy);
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            if (expected_score && *expected_score != result.score_cp) {
                throw std::logic_error("strategies disagree on the minimax score at position " + std::to_string(count));
            }
            expected_score = result.score_cp;
            auto& total = totals[index];
            total.nodes += result.nodes;
            total.cutoffs += result.cutoffs;
            total.tt_hits += result.tt_hits;
            total.tt_cutoffs += result.tt_cutoffs;
            total.ebf += result.ebf();
            total.seconds += seconds;
            if (args.results) write_result(detailed, count, *position, strategy, result, seconds);
        }
        progress.update(count);
    }
    if (count == 0) throw std::invalid_argument("positions input contains no positions");
    if (args.results) {
        detailed.close();
        if (!detailed) throw std::runtime_error("failed to write results file: " + args.results->string());
    }
    print_summary(args, totals, count);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        if (args.help) {
            print_help();
            return 0;
        }
        if (args.version) {
            std::cout << "neural-move-ordering 1.0.0 (C++20)\n";
            return 0;
        }
        return run(args);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
