#include <stdexcept>
#include <string>

#include <doctest.h>

#include "support.hpp"

using namespace test_support;

TEST_CASE("Parsing a position preserves its playable chess state") {
    for (const auto fen : {std::string_view(chess::constants::STARTPOS), castling, en_passant}) {
        CAPTURE(fen);
        const chess::Board expected(fen);
        const auto parsed = nmo::parse_board(fen);
        CHECK(parsed.getFen() == expected.getFen());
        CHECK(move_set(legal_moves(parsed)) == move_set(legal_moves(expected)));
    }
}

TEST_CASE("An optional UCI label identifies a legal move in the supplied position") {
    const auto position = nmo::parse_position(std::string(en_passant) + " | e5d6");
    REQUIRE(position.label.has_value());
    CHECK(*position.label == move_from_uci(position.board, "e5d6"));
    CHECK_FALSE(nmo::parse_position(en_passant).label.has_value());
}

TEST_CASE("Malformed positions and illegal move labels are rejected") {
    CHECK_THROWS_AS((void)nmo::parse_position("not a FEN"), std::invalid_argument);
    CHECK_THROWS_AS((void)nmo::parse_position("8/8/8/8/8/8/8/8 w - - 0 1"), std::invalid_argument);
    CHECK_THROWS_AS((void)nmo::parse_position(std::string(en_passant) + " | a1a8"), std::invalid_argument);
}

TEST_CASE("Position files may contain comments, blank lines and labeled records") {
    nmo::PositionReader reader("tests/fixtures/positions.fen");
    const auto first = reader.next();
    REQUIRE(first.has_value());
    CHECK(first->board.getFen() == chess::Board{}.getFen());
    CHECK_FALSE(first->label.has_value());

    const auto second = reader.next();
    REQUIRE(second.has_value());
    REQUIRE(second->label.has_value());
    CHECK(*second->label == move_from_uci(second->board, "e5d6"));
    CHECK_FALSE(reader.next().has_value());
}
