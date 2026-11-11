"""Summarise the final benchmark runs (training/final/run_final.sh). Usage: analyse.py RUNS_DIR > stats.json

Reads ho-d{4..7}-r{1..3}-{summary,results}.csv (held-out set), q100-d*.csv, noq-*, slow-*, nott-*.
Node counts are deterministic, so they come from repeat 1; times are the median over repeats.
"""
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path(sys.argv[1])
BASE = "strong"
OLD = "mvv-lva-history"
UPN = "neural:upn"


def summary(path):
    with open(path) as f:
        return {r["strategy"]: r for r in csv.DictReader(f)}


def done(path):
    """A run's summary is written only when it finishes; the file exists (empty) while it runs."""
    return path.exists() and path.stat().st_size > 0


def per_position(path):
    rows = defaultdict(dict)  # position -> strategy -> row
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[int(r["position"])][r["strategy"]] = r
    return rows


def bootstrap(pos_rows, num, den, reps=10000, seed=1):
    ids = sorted(pos_rows)
    a = [int(pos_rows[i][num]["nodes"]) for i in ids]
    b = [int(pos_rows[i][den]["nodes"]) for i in ids]
    rng = random.Random(seed)
    ratios = []
    n = len(ids)
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        ratios.append(sum(a[i] for i in idx) / sum(b[i] for i in idx))
    ratios.sort()
    return ratios[int(0.025 * reps)], ratios[int(0.975 * reps)]


def check_scores(pos_rows):
    bad = 0
    for i, by in pos_rows.items():
        if len({r["score_cp"] for r in by.values()}) != 1:
            bad += 1
    return bad


out = {"held_out": {}, "quiet100": {}, "no_quiescence": {}, "slow": {}, "phase": {}}
nodes = {}  # depth -> strategy -> total nodes (held-out)
for d in (4, 5, 6, 7):
    reps = [summary(RUNS / f"ho-d{d}-r{r}-summary.csv") for r in (1, 2, 3) if done(RUNS / f"ho-d{d}-r{r}-summary.csv")]
    if not reps:
        continue
    first = reps[0]
    for other in reps[1:]:
        assert all(other[s]["total_nodes"] == first[s]["total_nodes"] for s in first), "node counts differ between repeats"
    pos = per_position(RUNS / f"ho-d{d}-r1-results.csv")
    times = {s: statistics.median(float(r[s]["seconds"]) for r in reps) for s in first}
    spread = {s: (min(float(r[s]["seconds"]) for r in reps), max(float(r[s]["seconds"]) for r in reps)) for s in first}
    nodes[d] = {s: int(first[s]["total_nodes"]) for s in first}
    rows = {}
    for s, r in first.items():
        rows[s] = {
            "nodes": nodes[d][s],
            "nodes_vs_strong": nodes[d][s] / nodes[d][BASE],
            "nodes_vs_old": nodes[d][s] / nodes[d][OLD],
            "time": times[s],
            "time_range": spread[s],
            "time_vs_strong": times[s] / times[BASE],
            "time_vs_old": times[s] / times[OLD],
            "first_move_cut_rate": int(r["first_move_cutoffs"]) / int(r["cutoffs"]),
            "us_per_node": times[s] / nodes[d][s] * 1e6,
        }
    for s in (UPN, "neural:upn-max", "neural:upn-pure", "neural:tactical", "neural:upn+ch"):
        rows[s]["ci_vs_strong"] = bootstrap(pos, s, BASE)
        rows[s]["wins_vs_strong"] = sum(int(pos[i][s]["nodes"]) < int(pos[i][BASE]["nodes"]) for i in pos)
    rows[BASE]["ci_vs_old"] = bootstrap(pos, BASE, OLD)
    heuristics = ["mvv-lva", OLD, "history-gravity", "capture-hist", "killers", "cont-hist", BASE]
    best = min(heuristics, key=lambda s: nodes[d][s])
    rows[UPN]["best_heuristic"] = best
    rows[UPN]["nodes_vs_best"] = nodes[d][UPN] / nodes[d][best]
    rows[UPN]["time_vs_best"] = times[UPN] / times[best]
    rows[UPN]["ci_vs_best"] = bootstrap(pos, UPN, best)
    out["held_out"][d] = {"repeats": len(reps), "score_disagreements": check_scores(pos), "rows": rows}
    # Per phase: positions 1-100 opening, 101-200 middlegame, 201-300 endgame (file order).
    phases = {"opening": range(1, 101), "middlegame": range(101, 201), "endgame": range(201, 301)}
    out["phase"][d] = {
        name: {s: sum(int(pos[i][s]["nodes"]) for i in ids) / sum(int(pos[i][BASE]["nodes"]) for i in ids)
               for s in (UPN, "neural:tactical", OLD, "mvv-lva")}
        for name, ids in phases.items()
    }
for d in (6, 7):
    if d in nodes and d - 2 in nodes:
        for s in nodes[d]:
            out["held_out"][d]["rows"][s]["ebf_ratio"] = math.sqrt(nodes[d][s] / nodes[d - 2][s])

for d in (4, 5, 6, 7):
    p = RUNS / f"q100-d{d}-summary.csv"
    if done(p):
        q = summary(p)
        heur = ["mvv-lva", OLD, "history-gravity", "capture-hist", "killers", "cont-hist", BASE]
        best = min(heur, key=lambda s: int(q[s]["total_nodes"]))
        out.setdefault("quiet100_best", {})[d] = {"best": best,
            "upn_vs_best": int(q[UPN]["total_nodes"]) / int(q[best]["total_nodes"]),
            "upn_time_vs_best": float(q[UPN]["seconds"]) / float(q[best]["seconds"])}
        out["quiet100"][d] = {s: {"nodes_vs_strong": int(r["total_nodes"]) / int(q[BASE]["total_nodes"]),
                                  "nodes_vs_old": int(r["total_nodes"]) / int(q[OLD]["total_nodes"]),
                                  "time_vs_strong": float(r["seconds"]) / float(q[BASE]["seconds"])}
                              for s, r in q.items()}
for d in (5, 6):
    p = RUNS / f"noq-d{d}-summary.csv"
    if done(p):
        q = summary(p)
        out["no_quiescence"][d] = {s: int(r["total_nodes"]) / int(q[BASE]["total_nodes"]) for s, r in q.items()}
for d in (4, 5):
    p = RUNS / f"slow-d{d}-summary.csv"
    if done(p):
        q = summary(p)
        out["slow"][d] = {s: {"nodes_vs_strong": int(r["total_nodes"]) / int(q[BASE]["total_nodes"]),
                              "time_vs_strong": float(r["seconds"]) / float(q[BASE]["seconds"])} for s, r in q.items()}
p = RUNS / "nott-d5-results.csv"
if done(p):
    nott = per_position(p)
    tt = per_position(RUNS / "ho-d5-r1-results.csv")
    out["no_tt_score_mismatches"] = sum(nott[i][s]["score_cp"] != tt[i][s]["score_cp"] for i in nott for s in nott[i])
    out["no_tt_positions"] = len(nott)
json.dump(out, sys.stdout, indent=1, default=list)
