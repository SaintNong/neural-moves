#include "nmo/positions.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace nmo {
namespace {
std::string_view trim(std::string_view value) {
    constexpr std::string_view whitespace = " \t\r\n";
    const auto first = value.find_first_not_of(whitespace);
    if (first == std::string_view::npos) return {};
    return value.substr(first, value.find_last_not_of(whitespace) - first + 1);
}

std::uint64_t parse_counter(std::string_view value, std::string_view name) {
    std::uint64_t number = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid " + std::string(name) + " counter");
    }
    return number;
}

std::array<char, 64> validate_placement(std::string_view placement) {
    std::array<char, 64> squares{};
    std::array<int, 2> kings{};
    std::array<int, 2> pawns{};
    std::array<int, 2> pieces{};
    int rank = 7;
    int file = 0;
    bool previous_digit = false;
    for (const char symbol : placement) {
        if (symbol == '/') {
            if (file != 8 || rank == 0) throw std::invalid_argument("FEN must contain eight ranks of eight squares");
            --rank;
            file = 0;
            previous_digit = false;
            continue;
        }
        if (symbol >= '1' && symbol <= '8') {
            if (previous_digit) throw std::invalid_argument("consecutive empty-square counts in FEN");
            file += symbol - '0';
            previous_digit = true;
        } else {
            if (std::string_view("PNBRQKpnbrqk").find(symbol) == std::string_view::npos || file >= 8) {
                throw std::invalid_argument("invalid piece placement in FEN");
            }
            const std::size_t color = symbol >= 'a' && symbol <= 'z' ? 1 : 0;
            ++pieces[color];
            if (symbol == 'K' || symbol == 'k') ++kings[color];
            if (symbol == 'P' || symbol == 'p') {
                ++pawns[color];
                if (rank == 0 || rank == 7) throw std::invalid_argument("pawn on a back rank");
            }
            squares[static_cast<std::size_t>(rank * 8 + file)] = symbol;
            ++file;
            previous_digit = false;
        }
        if (file > 8) throw std::invalid_argument("too many squares in FEN rank");
    }
    if (rank != 0 || file != 8) throw std::invalid_argument("FEN must contain eight ranks of eight squares");
    if (kings[0] != 1 || kings[1] != 1) throw std::invalid_argument("position requires exactly one king per side");
    if (pieces[0] > 16 || pieces[1] > 16 || pawns[0] > 8 || pawns[1] > 8) {
        throw std::invalid_argument("too many pieces or pawns");
    }
    return squares;
}

void validate_castling(std::string_view rights, const std::array<char, 64>& squares) {
    if (rights == "-") return;
    std::string seen;
    for (const char right : rights) {
        if (std::string_view("KQkq").find(right) == std::string_view::npos || seen.find(right) != std::string::npos) {
            throw std::invalid_argument("invalid or duplicate castling rights");
        }
        seen += right;
        const bool white = right == 'K' || right == 'Q';
        const bool kingside = right == 'K' || right == 'k';
        const std::size_t base = white ? 0 : 56;
        if (squares[base + 4] != (white ? 'K' : 'k') ||
            squares[base + (kingside ? 7 : 0)] != (white ? 'R' : 'r')) {
            throw std::invalid_argument("castling rights require a king and rook on their initial squares");
        }
    }
}

void validate_en_passant(std::string_view ep, std::string_view side, const std::array<char, 64>& squares) {
    if (ep == "-") return;
    const bool white = side == "w";
    if (ep.size() != 2 || ep[0] < 'a' || ep[0] > 'h' || ep[1] != (white ? '6' : '3')) {
        throw std::invalid_argument("invalid en passant square");
    }
    const auto square = static_cast<std::size_t>((ep[1] - '1') * 8 + ep[0] - 'a');
    const auto pawn_square = white ? square - 8 : square + 8;
    const auto origin = white ? square + 8 : square - 8;
    if (squares[square] != 0 || squares[pawn_square] != (white ? 'p' : 'P') || squares[origin] != 0) {
        throw std::invalid_argument("en passant square does not follow a double pawn push");
    }
}
}  // namespace

chess::Board parse_board(std::string_view fen) {
    std::istringstream stream{std::string(fen)};
    std::array<std::string, 6> fields;
    for (auto& field : fields) {
        if (!(stream >> field)) throw std::invalid_argument("FEN requires six fields");
    }
    std::string extra;
    if (stream >> extra) throw std::invalid_argument("unexpected fields after FEN");
    const auto squares = validate_placement(fields[0]);
    if (fields[1] != "w" && fields[1] != "b") throw std::invalid_argument("invalid side to move");
    validate_castling(fields[2], squares);
    validate_en_passant(fields[3], fields[1], squares);
    const auto halfmove = parse_counter(fields[4], "halfmove");
    const auto fullmove = parse_counter(fields[5], "fullmove");
    // Leave room in the library's 16-bit ply counter for a complete search.
    if (fullmove < 1 || fullmove > 32700) throw std::invalid_argument("fullmove number must be between 1 and 32700");
    // All clocks >= 150 are already drawn; cap before the library's 8-bit conversion.
    fields[4] = std::to_string(std::min<std::uint64_t>(halfmove, 150));
    std::string normalized = fields.front();
    for (std::size_t index = 1; index < fields.size(); ++index) normalized += ' ' + fields[index];
    chess::Board board;
    if (!board.setFen(normalized)) throw std::invalid_argument("chess library rejected FEN");
    const auto previous_side = ~board.sideToMove();
    if (board.isAttacked(board.kingSq(previous_side), board.sideToMove())) {
        throw std::invalid_argument("the side that just moved left its king in check");
    }
    return board;
}

Position parse_position(std::string_view line) {
    line = trim(line);
    const auto separator = line.find('|');
    const auto fen = trim(line.substr(0, separator));
    Position position{parse_board(fen), std::string(fen), std::nullopt, 0};
    if (separator != std::string_view::npos) {
        const auto label = trim(line.substr(separator + 1));
        if (label.size() != 4 && label.size() != 5) throw std::invalid_argument("label must be a legal UCI move");
        // Match text against generated legal moves before calling any low-level parser.
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, position.board);
        for (const auto move : moves) {
            if (chess::uci::moveToUci(move) == label) {
                position.label = move;
                break;
            }
        }
        if (!position.label) throw std::invalid_argument("label must be a legal UCI move");
    }
    return position;
}

PositionReader::PositionReader(const std::filesystem::path& path) : path_(path), input_(path) {
    if (!input_) throw std::runtime_error("cannot open positions file: " + path.string());
}

std::optional<Position> PositionReader::next() {
    std::string line;
    while (std::getline(input_, line)) {
        ++line_number_;
        const auto text = trim(line);
        if (text.empty() || text.front() == '#') continue;
        try {
            auto position = parse_position(text);
            position.line_number = line_number_;
            return position;
        } catch (const std::invalid_argument& error) {
            throw std::invalid_argument(path_.string() + ':' + std::to_string(line_number_) + ": " + error.what());
        }
    }
    if (input_.bad()) throw std::runtime_error("failed to read positions file: " + path_.string());
    return std::nullopt;
}

}  // namespace nmo

