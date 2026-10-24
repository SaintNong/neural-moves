// Generate move-ordering training data from the benchmark's own search trees.
//
//   nmo-datagen label ROOTS.fen --out RAW.bin [options]
//       Search each root to --depth with the benchmark searcher and sample the
//       nodes where moves are ordered. Every move of a sampled node is then
//       probed with the node's (alpha, beta) window, recording the move's score
//       (fail-soft, from the node's perspective), the probe's node count and
//       whether it was the search's hash move.
//   nmo-datagen featurize RAW.bin --out PREFIX
//       Convert raw records into flat little-endian arrays (PREFIX.*.bin plus
//       PREFIX.json) using the same feature encoder as network inference.
//   nmo-datagen stats ROOTS.fen [--depth D]
//       Count ordering decisions per remaining depth, to choose sampling rates.
//   nmo-datagen import LABELED.fen --out RAW.bin
//       Convert "FEN | best_move_uci" lines (e.g. Stockfish labels) into raw
//       records whose only refutation is the labelled move.
//   nmo-datagen evaluate RAW.bin --strategy NETWORK.nmo --out LOGITS.bin [--limit N]
//       Write C++ network logits for the first N records (inference checks).

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "nmo/evaluation.hpp"
#include "nmo/features.hpp"
#include "nmo/neural.hpp"
#include "nmo/positions.hpp"
#include "nmo/search.hpp"

namespace {

// Raw record layout. Squares hold chess::Piece values (12 = empty).
struct RawHeader {
    std::uint8_t squares[64];
    std::uint8_t side_to_move;
    std::uint8_t castling;  // Bits: white king, white queen, black king, black queen side.
    std::uint8_t en_passant;  // Square index, or 64 for none.
    std::uint8_t halfmove;
    std::int8_t depth;        // Remaining depth; zero for quiescence nodes.
    std::int8_t label_depth;  // Depth each move was probed to (children at depth - 1).
    std::uint8_t ply;
    std::uint8_t move_count;
    std::int32_t alpha;
    std::int32_t beta;
    std::uint32_t root;
};
static_assert(sizeof(RawHeader) == 84);

struct RawMove {
    std::uint16_t move;
    std::uint16_t flags;  // Bit 0: the search's hash move, which is searched first.
    std::int32_t score;
    std::uint32_t nodes;
};
static_assert(sizeof(RawMove) == 12);

struct Options {
    std::string mode;
    std::string input;
    std::string output;
    int depth = 5;
    int threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    std::uint64_t limit = UINT64_MAX;
    std::uint64_t seed = 1;
    // Probability of sampling a decision at remaining depth d is
    // min(1, rate * boost^(d - 1)); quiescence decisions use qrate.
    double rate = 0.001;
    double boost = 4.0;
    double qrate = 0.0;
    int max_label_depth = 4;
    int repeat = 0;  // evaluate: also time this many passes of inference.
    std::string strategy = "mvv-lva-history";
    std::string label_strategy = "mvv-lva-history";
};

[[noreturn]] void usage() {
    std::cerr << "usage: nmo-datagen label ROOTS.fen --out RAW.bin [--depth D] [--threads N] [--limit N]\n"
                 "                    [--seed S] [--rate R] [--boost B] [--qrate Q] [--max-label-depth L]\n"
                 "                    [--strategy NAME|neural:FILE] [--label-strategy NAME]\n"
                 "       nmo-datagen featurize RAW.bin --out PREFIX\n"
                 "       nmo-datagen stats ROOTS.fen [--depth D] [--limit N] [--strategy NAME]\n"
                 "       nmo-datagen import LABELED.fen --out RAW.bin\n"
                 "       nmo-datagen evaluate RAW.bin --strategy NETWORK.nmo --out LOGITS.bin [--limit N]\n";
    std::exit(2);
}

template <class T>
T number(std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("invalid number: " + std::string(text));
    }
    return value;
}

Options parse(int argc, char** argv) {
    if (argc < 3) usage();
    Options options;
    options.mode = argv[1];
    options.input = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string_view arg = argv[index];
        if (index + 1 >= argc) usage();
        const std::string_view value = argv[++index];
        if (arg == "--out") options.output = value;
        else if (arg == "--depth") options.depth = number<int>(value);
        else if (arg == "--threads") options.threads = number<int>(value);
        else if (arg == "--limit") options.limit = number<std::uint64_t>(value);
        else if (arg == "--seed") options.seed = number<std::uint64_t>(value);
        else if (arg == "--rate") options.rate = number<double>(value);
        else if (arg == "--boost") options.boost = number<double>(value);
        else if (arg == "--qrate") options.qrate = number<double>(value);
        else if (arg == "--max-label-depth") options.max_label_depth = number<int>(value);
        else if (arg == "--repeat") options.repeat = number<int>(value);
        else if (arg == "--strategy") options.strategy = value;
        else if (arg == "--label-strategy") options.label_strategy = value;
        else usage();
    }
    if (options.mode != "stats" && options.output.empty()) usage();
    return options;
}

std::vector<std::string> read_roots(const std::string& path, std::uint64_t limit) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> roots;
    std::string line;
    while (roots.size() < limit && std::getline(input, line)) {
        const auto bar = line.find('|');
        if (bar != std::string::npos) line.resize(bar);
        if (line.empty() || line.front() == '#') continue;
        roots.push_back(line);
    }
    return roots;
}

struct Strategy {
    nmo::OrderingStrategy kind = nmo::OrderingStrategy::mvv_lva_history;
    std::shared_ptr<const nmo::Policy> policy;
};

Strategy load_strategy(const std::string& name) {
    if (name.starts_with("neural:")) return {nmo::OrderingStrategy::neural, nmo::Policy::load(name.substr(7))};
    return {nmo::parse_strategy(name), nullptr};
}

struct Sample {
    std::string fen;
    std::vector<chess::Move> moves;
    chess::Move hash_move{chess::Move::NO_MOVE};
    int depth = 0;
    int alpha = 0;
    int beta = 0;
    int ply = 0;
};

class Sampler : public nmo::SearchObserver {
public:
    Sampler(const Options& options, std::uint64_t seed) : options_(options), rng_(seed) {
        for (int depth = 0; depth <= nmo::max_depth; ++depth) {
            rates_[static_cast<std::size_t>(depth)] =
                depth == 0 ? options.qrate : std::min(1.0, options.rate * std::pow(options.boost, depth - 1));
        }
    }

    void on_order(const chess::Board& board, const chess::Movelist& moves, chess::Move hash_move, int depth,
                  int alpha, int beta, int ply) override {
        ++decisions[static_cast<std::size_t>(std::min(depth, 15))];
        const double rate = rates_[static_cast<std::size_t>(depth)];
        if (rate <= 0.0 || moves.size() < 2 || uniform_(rng_) >= rate) return;
        Sample sample{board.getFen(), {moves.begin(), moves.end()}, hash_move, depth, alpha, beta, ply};
        samples.push_back(std::move(sample));
    }

    std::vector<Sample> samples;
    std::array<std::uint64_t, 16> decisions{};

private:
    const Options& options_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
    std::array<double, nmo::max_depth + 1> rates_{};
};

RawHeader header_of(const chess::Board& board, const Sample& sample, int label_depth, std::uint32_t root) {
    RawHeader header{};
    for (int square = 0; square < 64; ++square) {
        header.squares[square] = static_cast<std::uint8_t>(static_cast<int>(board.at(chess::Square(square))));
    }
    header.side_to_move = static_cast<std::uint8_t>(static_cast<int>(board.sideToMove()));
    const auto rights = board.castlingRights();
    using Side = chess::Board::CastlingRights::Side;
    header.castling = static_cast<std::uint8_t>(
        rights.has(chess::Color::WHITE, Side::KING_SIDE) | rights.has(chess::Color::WHITE, Side::QUEEN_SIDE) << 1 |
        rights.has(chess::Color::BLACK, Side::KING_SIDE) << 2 | rights.has(chess::Color::BLACK, Side::QUEEN_SIDE) << 3);
    header.en_passant = static_cast<std::uint8_t>(
        board.enpassantSq() == chess::Square::NO_SQ ? 64 : board.enpassantSq().index());
    header.halfmove = static_cast<std::uint8_t>(std::min<std::uint32_t>(board.halfMoveClock(), 255));
    header.depth = static_cast<std::int8_t>(sample.depth);
    header.label_depth = static_cast<std::int8_t>(label_depth);
    header.ply = static_cast<std::uint8_t>(std::min(sample.ply, 255));
    header.move_count = static_cast<std::uint8_t>(sample.moves.size());
    header.alpha = sample.alpha;
    header.beta = sample.beta;
    header.root = root;
    return header;
}

int label(const Options& options) {
    const auto roots = read_roots(options.input, options.limit);
    const auto strategy = load_strategy(options.strategy);
    const auto label_strategy = load_strategy(options.label_strategy);
    std::ofstream output(options.output, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open " + options.output);

    std::mutex mutex;
    std::atomic<std::size_t> next{0};
    std::atomic<std::uint64_t> records{0};
    std::atomic<std::uint64_t> probe_nodes{0};
    const auto worker = [&]() {
        nmo::AlphaBetaSearcher searcher;
        nmo::AlphaBetaSearcher prober({4});
        searcher.set_policy(strategy.policy);
        prober.set_policy(label_strategy.policy);
        std::vector<char> buffer;
        for (std::size_t index = next++; index < roots.size(); index = next++) {
            Sampler sampler(options, options.seed * 0x9E3779B97F4A7C15ULL + index);
            searcher.set_observer(&sampler);
            const auto root = nmo::parse_board(roots[index]);
            (void)searcher.search(root, options.depth, strategy.kind);

            buffer.clear();
            for (const auto& sample : sampler.samples) {
                auto board = nmo::parse_board(sample.fen);
                const bool quiescence = sample.depth == 0;
                const int label_depth = quiescence ? 0 : std::min(sample.depth, options.max_label_depth + 1);
                const auto header = header_of(board, sample, label_depth, static_cast<std::uint32_t>(index));
                buffer.insert(buffer.end(), reinterpret_cast<const char*>(&header),
                              reinterpret_cast<const char*>(&header) + sizeof(header));
                for (const auto move : sample.moves) {
                    board.makeMove(move);
                    const auto probe = prober.probe(board, std::max(label_depth - 1, 0), -sample.beta, -sample.alpha,
                                                    sample.ply + 1, label_strategy.kind, quiescence);
                    board.unmakeMove(move);
                    probe_nodes += probe.nodes;
                    const RawMove raw{move.move(), static_cast<std::uint16_t>(move == sample.hash_move), -probe.score,
                                      static_cast<std::uint32_t>(std::min<std::uint64_t>(probe.nodes, UINT32_MAX))};
                    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&raw),
                                  reinterpret_cast<const char*>(&raw) + sizeof(raw));
                }
                ++records;
            }
            const std::lock_guard lock(mutex);
            output.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            if ((index + 1) % 500 == 0) {
                std::cerr << "roots " << index + 1 << "/" << roots.size() << ", records " << records
                          << ", probe nodes " << probe_nodes << "\n";
            }
        }
    };
    std::vector<std::thread> threads;
    for (int thread = 0; thread < options.threads; ++thread) threads.emplace_back(worker);
    for (auto& thread : threads) thread.join();
    std::cerr << "done: " << records << " records from " << roots.size() << " roots\n";
    return output ? 0 : 1;
}

int stats(const Options& options) {
    const auto roots = read_roots(options.input, options.limit);
    const auto strategy = load_strategy(options.strategy);
    nmo::AlphaBetaSearcher searcher;
    searcher.set_policy(strategy.policy);
    Options counting = options;
    counting.rate = 0.0;
    Sampler sampler(counting, 0);
    searcher.set_observer(&sampler);
    std::uint64_t nodes = 0;
    for (const auto& fen : roots) nodes += searcher.search(nmo::parse_board(fen), options.depth, strategy.kind).nodes;
    std::cout << "roots " << roots.size() << ", nodes " << nodes << "\n";
    for (std::size_t depth = 0; depth < sampler.decisions.size(); ++depth) {
        if (sampler.decisions[depth] != 0) {
            std::cout << "depth " << depth << (depth == 0 ? " (quiescence)" : "") << ": " << sampler.decisions[depth]
                      << " decisions\n";
        }
    }
    return 0;
}

template <class T>
void write_array(const std::string& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) throw std::runtime_error("cannot write " + path);
}

chess::Board board_of(const RawHeader& header) {
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const chess::Piece piece(static_cast<chess::Piece::underlying>(header.squares[rank * 8 + file]));
            if (piece == chess::Piece::NONE) {
                ++empty;
                continue;
            }
            if (empty != 0) fen += static_cast<char>('0' + std::exchange(empty, 0));
            fen += static_cast<std::string>(piece);
        }
        if (empty != 0) fen += static_cast<char>('0' + empty);
        if (rank != 0) fen += '/';
    }
    fen += header.side_to_move == 0 ? " w " : " b ";
    std::string rights;
    if (header.castling & 1) rights += 'K';
    if (header.castling & 2) rights += 'Q';
    if (header.castling & 4) rights += 'k';
    if (header.castling & 8) rights += 'q';
    fen += rights.empty() ? "-" : rights;
    fen += ' ';
    if (header.en_passant == 64) {
        fen += '-';
    } else {
        fen += static_cast<char>('a' + header.en_passant % 8);
        fen += static_cast<char>('1' + header.en_passant / 8);
    }
    fen += ' ' + std::to_string(header.halfmove) + " 1";
    chess::Board board;
    if (!board.setFen(fen)) throw std::runtime_error("bad record position: " + fen);
    return board;
}

bool read_record(std::istream& input, RawHeader& header, std::vector<RawMove>& moves) {
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header))) return false;
    moves.resize(header.move_count);
    input.read(reinterpret_cast<char*>(moves.data()), static_cast<std::streamsize>(moves.size() * sizeof(RawMove)));
    if (!input) throw std::runtime_error("truncated record");
    return true;
}

// Records have depth 1 and window (0, 1): the labelled move scores 1 and every
// other move 0, each costing one node.
int import_labels(const Options& options) {
    nmo::PositionReader reader(options.input);
    std::ofstream output(options.output, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open " + options.output);
    std::uint64_t records = 0;
    std::uint32_t index = 0;
    while (auto position = reader.next()) {
        if (!position->label) throw std::runtime_error("unlabelled position at line " + std::to_string(position->line_number));
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, position->board);
        Sample sample{position->fen, {moves.begin(), moves.end()}, chess::Move::NO_MOVE, 1, 0, 1, 0};
        const auto header = header_of(position->board, sample, 1, index++);
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        for (const auto move : moves) {
            const RawMove raw{move.move(), 0, move == *position->label ? 1 : 0, 1};
            output.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
        }
        ++records;
    }
    std::cerr << "imported " << records << " records\n";
    return output ? 0 : 1;
}

// Write the network's logits for every move of the first --limit records, as
// float32 in record order, to check C++ inference against PyTorch.
int evaluate(const Options& options) {
    std::ifstream input(options.input, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + options.input);
    const auto policy = nmo::Policy::load(options.strategy);
    auto state = policy->new_state();
    std::vector<float> logits;
    RawHeader header{};
    std::vector<RawMove> raw;
    std::vector<std::tuple<chess::Board, chess::Movelist, int>> records;
    for (std::uint64_t count = 0; count < options.limit && read_record(input, header, raw); ++count) {
        const auto board = board_of(header);
        chess::Movelist moves;
        for (const auto& entry : raw) moves.add(chess::Move(entry.move));
        std::vector<float> out(raw.size());
        state->reset(board);
        state->evaluate(board, moves, header.depth, out.data());
        logits.insert(logits.end(), out.begin(), out.end());
        if (options.repeat > 0) records.emplace_back(board, moves, header.depth);
    }
    write_array(options.output, logits);
    std::cerr << "wrote " << logits.size() << " logits\n";
    if (options.repeat > 0) {
        // Time evaluate() alone; reset() is excluded because search updates incrementally.
        std::vector<float> out(chess::constants::MAX_MOVES);
        std::array<std::uint64_t, 3> moves{}, calls{};
        std::array<double, 3> seconds{};
        for (int pass = 0; pass < options.repeat; ++pass) {
            for (const auto& [board, list, depth] : records) {
                state->reset(board);
                const auto begin = std::chrono::steady_clock::now();
                state->evaluate(board, list, depth, out.data());
                const auto bucket = static_cast<std::size_t>(std::min(depth, 2));
                seconds[bucket] += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                moves[bucket] += static_cast<std::uint64_t>(list.size());
                ++calls[bucket];
            }
        }
        for (std::size_t bucket = 0; bucket < 3; ++bucket) {
            if (calls[bucket] == 0) continue;
            std::cout << "evaluate depth " << (bucket == 2 ? "2+" : std::to_string(bucket)) << ": "
                      << seconds[bucket] / static_cast<double>(calls[bucket]) * 1e9 << " ns per call, "
                      << seconds[bucket] / static_cast<double>(moves[bucket]) * 1e9 << " ns per move\n";
        }
    }
    return 0;
}


int featurize(const Options& options) {
    std::ifstream input(options.input, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + options.input);
    // Per position: 32 stm features, 32 opponent features (-1 padded).
    std::vector<std::int16_t> features;
    // Per position: depth, label_depth, ply, alpha, beta, move offset, move count, root.
    std::vector<std::int32_t> info;
    // Per position: relative castling bits (ours K, ours Q, theirs K, theirs Q),
    // en passant file + 1 (0 for none), halfmove clock, in check, piece count.
    std::vector<std::uint8_t> aux;
    // Per move: from, to, piece, victim, promotion, threats, castling, en passant,
    // hash move, then reserved bytes.
    std::vector<std::uint8_t> moves;
    // Per move: score, nodes.
    std::vector<std::int32_t> labels;

    RawHeader header{};
    std::vector<RawMove> raw;
    std::uint64_t positions = 0;
    while (read_record(input, header, raw)) {
        const auto board = board_of(header);

        const auto us = board.sideToMove();
        for (const auto perspective : {us, ~us}) {
            const auto active = nmo::features::active_features(board, perspective);
            for (int index = 0; index < nmo::features::max_active; ++index) {
                features.push_back(index < active.count ? active.index[static_cast<std::size_t>(index)] : -1);
            }
        }
        info.insert(info.end(), {header.depth, header.label_depth, header.ply, header.alpha, header.beta,
                                 static_cast<std::int32_t>(labels.size() / 2), header.move_count,
                                 static_cast<std::int32_t>(header.root)});
        using Side = chess::Board::CastlingRights::Side;
        const auto castling = board.castlingRights();
        const int own_rights = castling.has(us, Side::KING_SIDE) | castling.has(us, Side::QUEEN_SIDE) << 1;
        const int their_rights = castling.has(~us, Side::KING_SIDE) | castling.has(~us, Side::QUEEN_SIDE) << 1;
        const auto context = nmo::features::threat_context(board);
        aux.insert(aux.end(), {static_cast<std::uint8_t>(own_rights | their_rights << 2),
                               static_cast<std::uint8_t>(header.en_passant == 64 ? 0 : header.en_passant % 8 + 1),
                               header.halfmove, static_cast<std::uint8_t>(board.inCheck()),
                               static_cast<std::uint8_t>(board.occ().count()), 0, 0, 0});

        for (const auto& entry : raw) {
            const chess::Move move(entry.move);
            const auto described = nmo::features::describe(board, context, move);
            moves.insert(moves.end(), {described.from, described.to, described.piece, described.victim,
                                       described.promotion, described.threats,
                                       static_cast<std::uint8_t>(move.typeOf() == chess::Move::CASTLING),
                                       static_cast<std::uint8_t>(move.typeOf() == chess::Move::ENPASSANT),
                                       static_cast<std::uint8_t>(entry.flags & 1U), 0, 0, 0, 0, 0, 0, 0});
            labels.push_back(entry.score);
            labels.push_back(static_cast<std::int32_t>(std::min<std::uint32_t>(entry.nodes, INT32_MAX)));
        }
        ++positions;
    }
    write_array(options.output + ".features.bin", features);
    write_array(options.output + ".info.bin", info);
    write_array(options.output + ".aux.bin", aux);
    write_array(options.output + ".moves.bin", moves);
    write_array(options.output + ".labels.bin", labels);
    std::ofstream meta(options.output + ".json");
    meta << "{\"positions\": " << positions << ", \"moves\": " << labels.size() / 2 << ", \"version\": 1}\n";
    std::cerr << "featurized " << positions << " positions, " << labels.size() / 2 << " moves\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.mode == "label") return label(options);
        if (options.mode == "featurize") return featurize(options);
        if (options.mode == "stats") return stats(options);
        if (options.mode == "evaluate") return evaluate(options);
        if (options.mode == "import") return import_labels(options);
        usage();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
