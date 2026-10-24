"""Select quiet positions (like the benchmark's quiet-labeled sets) for search validation.

A position is quiet when the side to move is not in check and a material-only
quiescence search returns the static material balance, i.e. no capture
sequence changes the material.

    python quiet_positions.py data/roots/roots-validation.fen --count 200 --out data/search-validation-200.fen
"""

import argparse
import random
from pathlib import Path

import chess

VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def material(board: chess.Board) -> int:
    score = sum(VALUES[p.piece_type] * (1 if p.color == board.turn else -1) for p in board.piece_map().values())
    return score


def quiescence(board: chess.Board, alpha: int, beta: int) -> int:
    stand = material(board)
    if stand >= beta:
        return stand
    alpha = max(alpha, stand)
    for move in board.legal_moves:
        if not board.is_capture(move) and move.promotion is None:
            continue
        board.push(move)
        score = -quiescence(board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("positions", type=Path)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    fens = [line.split("|")[0].strip() for line in args.positions.read_text().splitlines() if line.strip()]
    random.Random(args.seed).shuffle(fens)
    chosen = []
    for fen in fens:
        board = chess.Board(fen)
        if board.is_check() or not any(board.legal_moves):
            continue
        if quiescence(board, -100_000, 100_000) == material(board):
            chosen.append(fen)
            if len(chosen) == args.count:
                break
    args.out.write_text(f"# {len(chosen)} quiet positions from {args.positions.name} (seed {args.seed}).\n" +
                        "\n".join(chosen) + "\n")
    print(f"wrote {len(chosen)} positions to {args.out}")


if __name__ == "__main__":
    main()
