"""Run the C++ benchmark for several strategies and print a comparison table.

    python bench.py ../positions/quiet-100.fen --depth 5 mvv-lva-history neural:upn

Nodes and time are also shown relative to the first strategy.
"""

import argparse
import csv
import io
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(positions: Path, depth: int, strategies: list[str], binary: Path, extra: list[str]) -> list[dict]:
    command = [str(binary), str(positions), "--depth", str(depth), "--csv"] + extra
    for strategy in strategies:
        command += ["--strategy", strategy]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(output)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("positions", type=Path)
    parser.add_argument("strategies", nargs="+")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/release/neural-move-ordering")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="further benchmark options")
    args = parser.parse_args()
    rows = run(args.positions, args.depth, args.strategies, args.binary, args.extra)
    base_nodes, base_time = float(rows[0]["total_nodes"]), float(rows[0]["seconds"])
    print(f"{args.positions.name}, depth {args.depth}")
    print(f"| {'strategy':<28} | {'mean EBF':>8} | {'total nodes':>12} | {'nodes':>7} | {'time (s)':>8} | {'time':>7} |")
    print(f"|{'-' * 30}|{'-' * 10}|{'-' * 14}|{'-' * 9}|{'-' * 10}|{'-' * 9}|")
    for row in rows:
        nodes, seconds = float(row["total_nodes"]), float(row["seconds"])
        print(f"| {row['strategy']:<28} | {float(row['mean_ebf']):8.3f} | {int(nodes):12d} | {nodes / base_nodes:6.1%} "
              f"| {seconds:8.2f} | {seconds / base_time:6.1%} |")


if __name__ == "__main__":
    main()
