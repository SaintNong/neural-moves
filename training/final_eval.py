"""Benchmark every strategy on the benchmark position sets and write tables.

    python final_eval.py --out results

Runs the C++ benchmark once per (position set, depth, strategy) so each
strategy's time is measured on its own, saves the summary rows as CSV and
prints markdown tables with nodes and time relative to mvv-lva-history.
Run it on an otherwise idle machine; times are wall-clock.
"""

import argparse
import csv
from pathlib import Path

from bench import ROOT, run

HEURISTICS = ["mvv-lva", "history", "history-gravity", "capture-hist", "mvv-lva-history"]
NEURAL = ["neural:movemap", "neural:nnue-policy", "neural:sf-nnue", "neural:sf-upn", "neural:tactical",
          "neural:upn-pure", "neural:upn-max", "neural:upn"]
SLOW = ["neural:cnn", "neural:attention"]  # Full-board networks: about 100x slower per node.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--sets", nargs="+", default=["test-10", "quiet-100"])
    parser.add_argument("--depths", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--slow-max-depth", type=int, default=5)
    parser.add_argument("--only", nargs="+", help="rerun just these strategies, replacing their saved rows")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.only:
        with open(args.out / "summary.csv", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["strategy"] not in args.only]
    for name in args.sets:
        positions = ROOT / "positions" / f"{name}.fen"
        for depth in args.depths:
            strategies = HEURISTICS + NEURAL + (SLOW if depth <= args.slow_max_depth else [])
            for strategy in strategies:
                if args.only and strategy not in args.only:
                    continue
                (row,) = run(positions, depth, [strategy], ROOT / "build/release/neural-move-ordering", [])
                row["set"] = name
                rows.append(row)
                print(f"{name} d{depth} {strategy}: {row['total_nodes']} nodes, {float(row['seconds']):.2f}s", flush=True)
            with open(args.out / "summary.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    lines = []
    for name in args.sets:
        for depth in args.depths:
            order = HEURISTICS + NEURAL + SLOW
            selected = sorted((r for r in rows if r["set"] == name and int(r["depth"]) == depth),
                              key=lambda r: order.index(r["strategy"]))
            base = next(r for r in selected if r["strategy"] == "mvv-lva-history")
            lines.append(f"\n### {name}, depth {depth}\n")
            lines.append("| strategy | mean EBF | total nodes | nodes vs mvv-lva-history | time (s) | time vs mvv-lva-history |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for r in selected:
                nodes = int(r["total_nodes"]) / int(base["total_nodes"])
                seconds = float(r["seconds"]) / float(base["seconds"])
                lines.append(f"| {r['strategy']} | {float(r['mean_ebf']):.3f} | {int(r['total_nodes']):,} | {nodes:.1%} "
                             f"| {float(r['seconds']):.2f} | {seconds:.1%} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
