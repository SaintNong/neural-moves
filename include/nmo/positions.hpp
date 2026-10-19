#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>

#include <chess.hpp>

namespace nmo {

struct Position {
    chess::Board board;
    std::string fen;
    std::optional<chess::Move> label;
    std::size_t line_number = 0;
};

// Parse six-field standard-chess FEN and validate placement, castling, en passant,
// and that the previous mover did not leave its king in check. This does not
// prove reachability from the starting position. Fullmove must be in [1, 32700];
// halfmove clocks >= 150 are normalized to 150 for automatic draw adjudication.
// Throws std::invalid_argument on invalid input.
[[nodiscard]] chess::Board parse_board(std::string_view fen);
// Parse FEN optionally followed by "| legal_move_uci". Preserves trimmed input
// FEN text; line_number is zero until assigned by a reader. Invalid FEN or an
// illegal/malformed label throws std::invalid_argument.
[[nodiscard]] Position parse_position(std::string_view line);

// Streaming input keeps memory use independent of the number of positions.
class PositionReader {
public:
    // Throws std::runtime_error if the file cannot be opened.
    explicit PositionReader(const std::filesystem::path& path);
    // Skip blank/comment (#) lines; return nullopt at EOF. Records use one-based
    // physical line numbers. Parse errors include path and line number in
    // std::invalid_argument; read failures throw std::runtime_error.
    [[nodiscard]] std::optional<Position> next();

private:
    std::filesystem::path path_;
    std::ifstream input_;
    std::size_t line_number_ = 0;
};

}  // namespace nmo

