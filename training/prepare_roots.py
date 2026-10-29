"""Sample root positions for data generation from Lichess PGN archives.

One position is drawn from each game at a uniformly random ply, skipping the
opening and positions in check. Games are split into train/validation by a hash
of their index, so no game contributes to both. Any position that also appears
in the benchmark position files is dropped.

    python prepare_roots.py data/lichess/*.pgn.zst --out data/roots
"""

import argparse
import hashlib
import io
import random
import subprocess
from pathlib import Path

import chess
import chess.pgn


def position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def benchmark_keys(directory: Path) -> set[str]:
    keys = set()
    for path in directory.glob("*.fen"):
        for line in path.read_text().splitlines():
            line = line.split("|")[0].strip()
            if line and not line.startswith("#"):
                keys.add(position_key(line))
    return keys


def games(path: Path):
    with subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE) as process:
        stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace")
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            yield game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(__file__).resolve().parents[1] / "positions")
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    excluded = benchmark_keys(args.benchmark_dir)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {split: open(args.out / f"roots-{split}.fen", "w") for split in ("train", "validation")}
    counts = {"train": 0, "validation": 0, "excluded": 0}
    game_index = 0
    for archive in args.archives:
        for game in games(archive):
            game_index += 1
            moves = list(game.mainline_moves())
            if len(moves) < args.min_ply + 2:
                continue
            ply = rng.randint(args.min_ply, len(moves) - 1)
            board = game.board()
            for move in moves[:ply]:
                board.push(move)
            if board.is_check() or board.is_game_over():
                continue
            fen = board.fen()
            if position_key(fen) in excluded:
                counts["excluded"] += 1
                continue
            digest = hashlib.blake2b(f"{archive.name}:{game_index}".encode(), digest_size=4).digest()
            split = "validation" if int.from_bytes(digest) % 100 < args.validation_percent else "train"
            outputs[split].write(f"{fen}\n")
            counts[split] += 1
            if game_index % 20000 == 0:
                print(f"{game_index} games, {counts}", flush=True)
    for output in outputs.values():
        output.close()
    print(f"done: {game_index} games, {counts}")


if __name__ == "__main__":
    main()
