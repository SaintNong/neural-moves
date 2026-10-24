"""Label positions with Stockfish's best move at a fixed node budget.

This reproduces the labelling planned in the project proposal (one thread,
fixed nodes, so labels are deterministic) as a baseline for the search-derived
targets. Output lines are "FEN | best_move_uci".

    python label_stockfish.py data/roots/roots-train.fen --engine STOCKFISH --out sf.fen
"""

import argparse
import multiprocessing
import subprocess
from pathlib import Path


class Engine:
    def __init__(self, path: str, nodes: int):
        self.nodes = nodes
        self.process = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 16")
        self.send("isready")
        self.wait("readyok")

    def send(self, command: str) -> None:
        self.process.stdin.write(command + "\n")

    def wait(self, prefix: str) -> str:
        for line in self.process.stdout:
            if line.startswith(prefix):
                return line
        raise RuntimeError("engine exited")

    def best_move(self, fen: str) -> str:
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send(f"go nodes {self.nodes}")
        return self.wait("bestmove").split()[1]


_engine = None


def _init(path: str, nodes: int) -> None:
    global _engine
    _engine = Engine(path, nodes)


def _label(fen: str) -> str:
    return f"{fen} | {_engine.best_move(fen)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("positions", type=Path)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    fens = [line.split("|")[0].strip() for line in args.positions.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    if args.limit:
        fens = fens[:args.limit]
    with multiprocessing.Pool(args.workers, _init, (args.engine, args.nodes)) as pool, open(args.out, "w") as out:
        for index, line in enumerate(pool.imap(_label, fens, chunksize=64), 1):
            out.write(line + "\n")
            if index % 20000 == 0:
                print(f"{index}/{len(fens)}", flush=True)
    print(f"labelled {len(fens)} positions")


if __name__ == "__main__":
    main()
