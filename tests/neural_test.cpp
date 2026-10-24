#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest.h>

#include "nmo/features.hpp"
#include "nmo/neural.hpp"
#include "support.hpp"

using namespace test_support;

namespace {

struct TensorSpec {
    std::string name;
    std::vector<std::uint32_t> shape;
};

void put_string(std::ofstream& out, const std::string& value) {
    const auto length = static_cast<std::uint32_t>(value.size());
    out.write(reinterpret_cast<const char*>(&length), sizeof(length));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

template <class T>
void put(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Write a weight file with random tensors of the given shapes.
std::filesystem::path random_network(const std::string& name, const std::map<std::string, double>& settings,
                                     const std::vector<TensorSpec>& tensors, const std::string& architecture = "") {
    const auto path = std::filesystem::temp_directory_path() / ("nmo-test-" + name + ".nmo");
    std::ofstream out(path, std::ios::binary);
    out.write("NMOW", 4);
    put<std::uint32_t>(out, 1);
    put_string(out, architecture.empty() ? name : architecture);
    put(out, static_cast<std::uint32_t>(settings.size()));
    for (const auto& [key, value] : settings) {
        put_string(out, key);
        put(out, value);
    }
    put(out, static_cast<std::uint32_t>(tensors.size()));
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0F, 0.3F);
    for (const auto& tensor : tensors) {
        put_string(out, tensor.name);
        put(out, static_cast<std::uint32_t>(tensor.shape.size()));
        std::size_t count = 1;
        for (const auto dimension : tensor.shape) {
            put(out, dimension);
            count *= dimension;
        }
        for (std::size_t index = 0; index < count; ++index) put(out, normal(rng));
    }
    return path;
}

// A random UPN; the clipped activation exercises the int8 move table, and the
// optional leaf head replaces the network at remaining depths 0 and 1.
std::filesystem::path random_upn(bool leaf_head = false, std::map<std::string, double> extra = {}) {
    constexpr std::uint32_t hidden = 16;
    constexpr std::uint32_t context = 8;
    constexpr std::uint32_t width = 2 * hidden;
    std::map<std::string, double> settings{{"hidden", hidden}, {"activation", 1}, {"context", context}, {"depth_buckets", 4}};
    settings.insert(extra.begin(), extra.end());
    std::vector<TensorSpec> tensors{{"input.weight", {768, hidden}}, {"input.bias", {hidden}},
                                    {"move.weight", {4096, width}}, {"move.bias", {4096}},
                                    {"promotion.weight", {5, width}}, {"promotion.bias", {5}},
                                    {"tactic.bias", {4 * 6 * 7 * 256}}, {"project.weight", {context, width}},
                                    {"project.bias", {context}}, {"threat.weight", {4 * 6 * 256, context}},
                                    {"capture.weight", {6 * 7 * 5, context}}};
    if (leaf_head) {
        tensors.push_back({"leaf.tactic.bias", {2 * 6 * 7 * 256}});
        tensors.push_back({"leaf.move.bias", {4096}});
        tensors.push_back({"leaf.promotion.bias", {5}});
        settings["quiescence"] = 1;
        settings["leaf_depth"] = 1;
    }
    return random_network(leaf_head ? "upn-leaf" : "upn", settings, tensors, "upn");
}

std::vector<float> logits_of(nmo::PolicyState& state, const chess::Board& board, int depth) {
    const auto moves = legal_moves(board);
    std::vector<float> logits(static_cast<std::size_t>(moves.size()));
    state.evaluate(board, moves, depth, logits.data());
    return logits;
}

}  // namespace

TEST_CASE("Move deltas reproduce the board's piece-square features after every move kind") {
    for (const auto fen : {std::string_view(chess::constants::STARTPOS), castling, en_passant, promotion,
                           std::string_view("r3k2r/1P6/8/8/8/8/6p1/R3K2R w KQkq - 0 1")}) {
        const auto board = nmo::parse_board(fen);
        for (const auto move : legal_moves(board)) {
            CAPTURE(fen);
            CAPTURE(chess::uci::moveToUci(move));
            for (const auto perspective : {chess::Color::WHITE, chess::Color::BLACK}) {
                std::multiset<int> features;
                const auto before = nmo::features::active_features(board, perspective);
                for (int index = 0; index < before.count; ++index) features.insert(before.index[static_cast<std::size_t>(index)]);
                const auto delta = nmo::features::move_delta(board, move);
                for (int index = 0; index < delta.removed_count; ++index) {
                    const auto& change = delta.removed[static_cast<std::size_t>(index)];
                    const auto found = features.find(nmo::features::piece_square(perspective, change.piece, change.square));
                    REQUIRE(found != features.end());
                    features.erase(found);
                }
                for (int index = 0; index < delta.added_count; ++index) {
                    const auto& change = delta.added[static_cast<std::size_t>(index)];
                    features.insert(nmo::features::piece_square(perspective, change.piece, change.square));
                }
                auto after = board;
                after.makeMove(move);
                const auto expected = nmo::features::active_features(after, perspective);
                CHECK(features == std::multiset<int>(expected.index.begin(), expected.index.begin() + expected.count));
            }
        }
    }
}

TEST_CASE("Incrementally updated networks match a fresh evaluation along random lines") {
    const auto policy = nmo::Policy::load(random_upn(true));
    auto incremental = policy->new_state();
    auto fresh = policy->new_state();
    std::mt19937 rng(3);
    for (const auto fen : {std::string_view(chess::constants::STARTPOS), castling, en_passant,
                           std::string_view("r3k2r/1P6/8/8/8/8/6p1/R3K2R w KQkq - 0 1")}) {
        for (int line = 0; line < 20; ++line) {
            auto board = nmo::parse_board(fen);
            incremental->reset(board);
            std::vector<chess::Move> played;
            for (int ply = 0; ply < 24; ++ply) {
                fresh->reset(board);
                const auto expected = logits_of(*fresh, board, ply % 5);
                const auto actual = logits_of(*incremental, board, ply % 5);
                REQUIRE(actual.size() == expected.size());
                for (std::size_t index = 0; index < actual.size(); ++index) {
                    CHECK(actual[index] == doctest::Approx(expected[index]).epsilon(1e-4));
                }
                const auto moves = legal_moves(board);
                if (moves.empty()) break;
                const auto move = moves[static_cast<int>(rng() % static_cast<unsigned>(moves.size()))];
                incremental->push(board, move, static_cast<int>(rng() % 4));  // Some untracked.
                board.makeMove(move);
                played.push_back(move);
                // Occasionally back up a ply, as the search does.
                if (rng() % 4 == 0) {
                    board.unmakeMove(played.back());
                    played.pop_back();
                    incremental->pop();
                }
            }
        }
    }
}

TEST_CASE("Neural ordering preserves moves and the search value") {
    const std::vector<std::filesystem::path> networks{random_upn(), random_upn(true, {{"history_weight", 0.001}}),
                                                      random_upn(false, {{"staged", 1}})};
    for (const auto& network : networks) {
        const auto policy = nmo::Policy::load(network);
        for (const auto fen : {std::string_view(chess::constants::STARTPOS), free_queen, poisoned_rook, castling,
                               en_passant, promotion}) {
            CAPTURE(network.string());
            CAPTURE(fen);
            const auto board = nmo::parse_board(fen);
            nmo::MoveOrderer orderer;
            orderer.reset(nmo::OrderingStrategy::neural, policy);
            orderer.set_root(board);
            auto ordered = legal_moves(board);
            orderer.order(board, ordered);
            CHECK(move_set(ordered) == move_set(legal_moves(board)));

            nmo::AlphaBetaSearcher reference({0, false});
            const auto expected = reference.search(board, 4).score_cp;
            nmo::AlphaBetaSearcher searcher;
            searcher.set_policy(policy);
            const auto result = searcher.search(board, 4, nmo::OrderingStrategy::neural);
            CHECK(result.score_cp == expected);
            CHECK(board.isLegal(result.best_move));
        }
    }
}

TEST_CASE("Neural ordering without a network is rejected") {
    nmo::AlphaBetaSearcher searcher;
    CHECK_THROWS_AS((void)searcher.search(chess::Board{}, 2, nmo::OrderingStrategy::neural), std::logic_error);
}

TEST_CASE("Malformed weight files are rejected") {
    const auto path = std::filesystem::temp_directory_path() / "nmo-test-bad.nmo";
    {
        std::ofstream out(path, std::ios::binary);
        out << "not a network";
    }
    CHECK_THROWS_AS((void)nmo::Policy::load(path), std::runtime_error);
    CHECK_THROWS_AS((void)nmo::Policy::load(path.string() + ".missing"), std::runtime_error);
    const auto unknown = random_network("unknown", {}, {});
    CHECK_THROWS_AS((void)nmo::Policy::load(unknown), std::runtime_error);
}
