"""Turn stats.json into LaTeX inputs shared by the report and the poster.

Usage: gen_tex.py stats.json OUTDIR  -> OUTDIR/numbers.tex, tab-heuristics.tex, tab-neural.tex, tab-phase.tex,
                                        plot-depth.tex, plot-scatter.tex
"""
import json
import sys
from pathlib import Path

stats = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
H = stats["held_out"]
DEPTHS = [d for d in ("4", "5", "6", "7") if d in H]
TOP = DEPTHS[-1]

HEUR = [("mvv-lva", "MVV-LVA", "captures by MVV-LVA"),
        ("mvv-lva-history", "+ history", "quiet moves by history"),
        ("history-gravity", "+ gravity and penalty", "bounded history, failed moves penalised"),
        ("capture-hist", "+ capture history", "captures by victim and capture history"),
        ("killers", "+ killers", "two killer moves per ply"),
        ("cont-hist", "+ continuation history", "history given the previous move"),
        ("strong", "\\textbf{strong}", "+ capture history and SEE")]
NEUR = [("neural:sf-nnue", "NNUE, Stockfish labels (proposal)"),
        ("neural:nnue-policy", "NNUE, search labels"),
        ("neural:movemap", "MoveMap, search labels"),
        ("neural:sf-upn", "UPN, Stockfish labels"),
        ("neural:tactical", "Tactical tables only"),
        ("neural:upn-pure", "UPN, network only"),
        ("neural:upn-max", "UPN, network everywhere"),
        ("neural:upn", "\\textbf{UPN cascade}"),
        ("neural:upn+ch", "UPN cascade + cont.\\ history")]


def pct(x):
    return f"{100 * x:.0f}\\%"


def pct1(x):
    return f"{100 * x:.1f}\\%"


def row(s, d):
    return H[d]["rows"][s]


def cmd(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


# ------------------------------------------------------------------ numbers.tex
upn = row("neural:upn", TOP)
st = row("strong", TOP)
old = row("mvv-lva-history", TOP)
n = ""
n += cmd("RESULTdepth", TOP)
n += cmd("RESULTupnVsStrong", pct(upn["nodes_vs_strong"]))
n += cmd("RESULTupnTimeVsStrong", pct(upn["time_vs_strong"]))
n += cmd("RESULTupnVsOld", pct(upn["nodes_vs_old"]))
n += cmd("RESULTupnTimeVsOld", pct(upn["time_vs_old"]))
lo, hi = upn["ci_vs_strong"]
n += cmd("RESULTupnCI", f"[{pct(lo)}, {pct(hi)}]")
n += cmd("RESULTupnWins", str(upn["wins_vs_strong"]))
n += cmd("RESULTstrongVsOld", pct(st["nodes_vs_old"]))
lo, hi = st["ci_vs_old"]
n += cmd("RESULTstrongCI", f"[{pct(lo)}, {pct(hi)}]")
n += cmd("RESULTstrongTimeVsOld", pct(st["time_vs_old"]))
upn_range = [row("neural:upn", d)["nodes_vs_strong"] for d in DEPTHS]
n += cmd("RESULTupnRange", f"{100 * min(upn_range):.0f}--{pct(max(upn_range))}")
upn_trange = [row("neural:upn", d)["time_vs_strong"] for d in DEPTHS]
n += cmd("RESULTupnTimeRange", f"{100 * min(upn_trange):.0f}--{pct(max(upn_trange))}")
for key, s in (("Max", "neural:upn-max"), ("Pure", "neural:upn-pure"), ("Tactical", "neural:tactical"),
               ("Ch", "neural:upn+ch"), ("Sfupn", "neural:sf-upn"), ("Sfnnue", "neural:sf-nnue")):
    r = row(s, TOP)
    n += cmd(f"RESULT{key}VsStrong", pct(r["nodes_vs_strong"]))
    n += cmd(f"RESULT{key}TimeVsStrong", pct(r["time_vs_strong"]))
lo, hi = row("neural:tactical", TOP)["ci_vs_strong"]
n += cmd("RESULTtacticalCI", f"[{pct(lo)}, {pct(hi)}]")
n += cmd("RESULTfmcOld", pct1(old["first_move_cut_rate"]))
n += cmd("RESULTfmcStrong", pct1(st["first_move_cut_rate"]))
n += cmd("RESULTfmcUpn", pct1(upn["first_move_cut_rate"]))
n += cmd("RESULTusStrong", f"{st['us_per_node']:.2f}")
n += cmd("RESULTusUpn", f"{upn['us_per_node']:.2f}")
n += cmd("RESULTusPure", f"{row('neural:upn-pure', TOP)['us_per_node']:.2f}")
if "ebf_ratio" in upn:
    n += cmd("RESULTebfUpn", f"{upn['ebf_ratio']:.2f}")
    n += cmd("RESULTebfStrong", f"{st['ebf_ratio']:.2f}")
    n += cmd("RESULTebfOld", f"{old['ebf_ratio']:.2f}")
WORD = {"4": "Four", "5": "Five", "6": "Six", "7": "Seven"}
KEYS = {"mvv-lva-history": "Old", "history-gravity": "Gravity", "killers": "Killers", "cont-hist": "Cont",
        "strong": "Strong", "neural:upn": "Upn", "neural:upn-max": "Max", "neural:upn-pure": "Pure",
        "neural:tactical": "Tactical", "neural:sf-upn": "Sfupn", "neural:sf-nnue": "Sfnnue", "neural:upn+ch": "Ch"}
for d in DEPTHS:
    for s, key in KEYS.items():
        n += cmd(f"N{key}{WORD[d]}", pct(row(s, d)["nodes_vs_strong"]))
        n += cmd(f"T{key}{WORD[d]}", pct(row(s, d)["time_vs_strong"]))
    n += cmd(f"NStrongOld{WORD[d]}", pct(row("strong", d)["nodes_vs_old"]))
    u_ = row("neural:upn", d)
    n += cmd(f"NUpnBest{WORD[d]}", pct(u_["nodes_vs_best"]))
    n += cmd(f"TUpnBest{WORD[d]}", pct(u_["time_vs_best"]))
    n += cmd(f"BestName{WORD[d]}", "\\code{" + u_["best_heuristic"] + "}")
    lo, hi = u_["ci_vs_strong"]
    n += cmd(f"CIUpn{WORD[d]}", f"[{pct(lo)}, {pct(hi)}]")
    n += cmd(f"WinsUpn{WORD[d]}", str(u_["wins_vs_strong"]))
qb = stats.get("quiet100_best", {})
for d, v in qb.items():
    n += cmd(f"QBest{WORD[d]}", "\\code{" + v["best"] + "}")
    n += cmd(f"QUpnBest{WORD[d]}", pct(v["upn_vs_best"]))
    n += cmd(f"QTUpnBest{WORD[d]}", pct(v["upn_time_vs_best"]))
for d, v in stats.get("quiet100", {}).items():
    n += cmd(f"QStrongOld{WORD[d]}", pct(v["strong"]["nodes_vs_old"]))
reps = min(H[d]["repeats"] for d in DEPTHS)
n += cmd("RESULTrepeats", str(reps))
n += cmd("RESULTdisagreements", str(sum(H[d]["score_disagreements"] for d in DEPTHS)))
nq = stats.get("no_quiescence", {})
for d in ("5", "6"):
    if d in nq:
        best = min(("mvv-lva-history", "history-gravity", "strong"), key=lambda s: nq[d][s])
        n += cmd(f"RESULTnoqBest{'Five' if d == '5' else 'Six'}", "\\code{" + best + "}")
        for key, s in (("Upn", "neural:upn"), ("Pure", "neural:upn-pure"), ("Tactical", "neural:tactical"),
                       ("Strong", "strong"), ("Old", "mvv-lva-history")):
            n += cmd(f"RESULTnoqVsBest{key}{'Five' if d == '5' else 'Six'}", pct(nq[d][s] / nq[d][best]))
    if d in nq:
        for key, s in (("Upn", "neural:upn"), ("Pure", "neural:upn-pure"), ("Tactical", "neural:tactical"),
                       ("Old", "mvv-lva-history")):
            n += cmd(f"RESULTnoq{key}{'Five' if d == '5' else 'Six'}", pct(nq[d][s]))
sl = stats.get("slow", {})
for d in ("4", "5"):
    if d in sl:
        w = "Four" if d == "4" else "Five"
        for key, s in (("Cnn", "neural:cnn"), ("Att", "neural:attention")):
            n += cmd(f"RESULT{key}Nodes{w}", pct(sl[d][s]["nodes_vs_strong"]))
            n += cmd(f"RESULT{key}Time{w}", f"{sl[d][s]['time_vs_strong']:.0f}$\\times$")
q = stats.get("quiet100", {})
if q:
    qd = max(q, key=int)
    n += cmd("RESULTqDepth", qd)
    n += cmd("RESULTqUpnVsStrong", pct(q[qd]["neural:upn"]["nodes_vs_strong"]))
    n += cmd("RESULTqUpnTimeVsStrong", pct(q[qd]["neural:upn"]["time_vs_strong"]))
    n += cmd("RESULTqStrongVsOld", pct(q[qd]["strong"]["nodes_vs_old"]))
if "no_tt_score_mismatches" in stats:
    n += cmd("RESULTnoTTcheck", f"{stats['no_tt_score_mismatches']} differences on {stats['no_tt_positions']} positions at depth 5")
log = Path(sys.argv[1]).parent / "runs" / "progress.log"
if len(sys.argv) > 3:
    log = Path(sys.argv[3])
if log.exists():
    stamps = [line.split()[0] for line in log.read_text().splitlines() if line.strip()]
    h0, m0, _ = map(int, stamps[0].split(":")); h1, m1, _ = map(int, stamps[-1].split(":"))
    minutes = (h1 * 60 + m1 - h0 * 60 - m0) % (24 * 60)
    n += cmd("RESULTruntime", f"{minutes / 60:.1f} hours" if minutes >= 90 else f"{minutes} minutes")
(out / "numbers.tex").write_text(n)

# ------------------------------------------------------------------ tables
cols = " & ".join(pct(row(s, d)["nodes_vs_strong"]) for s in [] for d in DEPTHS)


def body(items, label_of):
    lines = []
    for item in items:
        s, label = item[0], label_of(item)
        bold = s in ("strong", "neural:upn")
        cells = [pct(row(s, d)["nodes_vs_strong"]) for d in DEPTHS]
        r = row(s, TOP)
        cells += [pct(r["time_vs_strong"]), pct1(r["first_move_cut_rate"])]
        if "ebf_ratio" in r:
            cells.append(f"{r['ebf_ratio']:.2f}")
        if bold:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(f"      \\quad {label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


header = " & ".join(["Orderer"] + [f"Depth {d}" for d in DEPTHS] + [f"Time, d{TOP}", "First-move cuts"] +
                   (["EBF"] if "ebf_ratio" in upn else []))
spec = "@{}l" + "r" * (header.count("&")) + "@{}"


def tabular(rows):
    return (f"\\setlength{{\\tabcolsep}}{{4pt}}\n\\begin{{tabular}}{{{spec}}}\n  \\toprule\n  {header} \\\\\n  \\midrule\n" + rows +
            "  \\bottomrule\n\\end{tabular}\n")


(out / "tab-heuristics.tex").write_text(tabular(body(HEUR, lambda i: i[1])))
(out / "tab-neural.tex").write_text(tabular(body(NEUR, lambda i: i[1])))

ph = stats["phase"][TOP]
lines = []
for s, label in (("mvv-lva", "MVV-LVA"), ("mvv-lva-history", "MVV-LVA + history"), ("neural:tactical", "Tactical tables"),
                 ("neural:upn", "\\textbf{UPN cascade}")):
    cells = [pct(ph[p][s]) for p in ("opening", "middlegame", "endgame")]
    lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
(out / "tab-phase.tex").write_text("\\begin{tabular}{@{}lrrr@{}}\n  \\toprule\n  Orderer & Opening & Middlegame & Endgame \\\\\n"
                                   "  \\midrule\n" + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ plots
plot = ""
for s, name in (("mvv-lva-history", "old"), ("strong", "strong"), ("neural:tactical", "tactical"),
                ("neural:upn-pure", "pure"), ("neural:upn", "upn"), ("neural:upn-max", "max")):
    coords = " ".join(f"({d},{100 * row(s, d)['nodes_vs_strong']:.1f})" for d in DEPTHS)
    plot += f"\\def\\PLOT{name}{{{coords}}}\n"
heur = [100 * row(h[0], TOP)["time_vs_strong"] for h in HEUR], [100 * row(h[0], TOP)["nodes_vs_strong"] for h in HEUR]
neur = [100 * row(x[0], TOP)["time_vs_strong"] for x in NEUR], [100 * row(x[0], TOP)["nodes_vs_strong"] for x in NEUR]
plot += "\\def\\SCATTERheur{" + " ".join(f"({t:.1f},{v:.1f})" for t, v in zip(*heur)) + "}\n"
plot += "\\def\\SCATTERneural{" + " ".join(f"({t:.1f},{v:.1f})" for t, v in zip(*neur)) + "}\n"
plot += f"\\def\\UPNTIME{{{100 * upn['time_vs_strong']:.1f}}}\\def\\UPNNODES{{{100 * upn['nodes_vs_strong']:.1f}}}\n"
times, vals = heur[0] + neur[0], heur[1] + neur[1]
plot += f"\\def\\XMIN{{{10 * int(min(times) // 10)}}}\\def\\XMAX{{{10 * int(max(times) // 10 + 1)}}}\n"
plot += f"\\def\\YMIN{{{10 * int((min(vals) - 12) // 10)}}}\\def\\YMAX{{{10 * int(max(vals) // 10 + 1)}}}\n"
deps = [100 * row(x, d)["nodes_vs_strong"] for x in ("mvv-lva-history", "strong", "neural:tactical", "neural:upn-pure", "neural:upn") for d in DEPTHS]
plot += f"\\def\\DYMIN{{{10 * int(min(deps) // 10)}}}\\def\\DYMAX{{{10 * int(max(deps) // 10 + 1)}}}\n"
(out / "plots.tex").write_text(plot)
print("wrote", out)
