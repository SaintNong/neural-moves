#!/bin/bash
# Run every benchmark configuration in the final report, one at a time so timings are not contended.
# Usage (from the repository root, after building build/release): training/final/run_final.sh OUTDIR
# Then: python3 training/final/analyse.py OUTDIR > stats.json
#       python3 training/final/gen_tex.py stats.json REPORT/generated
set -uo pipefail
OUT=${1:?usage: run_final.sh OUTDIR}
BIN=${BIN:-build/release/neural-move-ordering}
mkdir -p "$OUT"
MAIN="mvv-lva mvv-lva-history history-gravity capture-hist killers cont-hist strong neural:nnue-policy neural:sf-nnue
      neural:movemap neural:sf-upn neural:tactical neural:upn-pure neural:upn-max neural:upn neural:upn+ch"
H=positions/held-out-300.fen

bench() {  # bench POSITIONS DEPTH NAME "STRATEGIES" [extra options...]
    local positions=$1 depth=$2 name=$3 strategies=$4; shift 4
    local args=()
    for s in $strategies; do args+=(--strategy "$s"); done
    echo "$(date +%H:%M:%S) $name" >> "$OUT/progress.log"
    "$BIN" "$positions" --depth "$depth" --tt-mb 256 --csv --results "$OUT/$name-results.csv" "${args[@]}" "$@" \
        > "$OUT/$name-summary.csv" 2> "$OUT/$name-stderr.txt"
}

for rep in 1 2 3; do for d in 4 5 6 7; do bench $H $d ho-d$d-r$rep "$MAIN"; done; done
for d in 4 5; do bench $H $d slow-d$d "mvv-lva-history strong neural:upn neural:cnn neural:attention"; done
for d in 5 6; do
    bench $H $d noq-d$d "mvv-lva-history history-gravity strong neural:tactical neural:upn-pure neural:upn" --no-quiescence
done
bench $H 5 nott-d5 "mvv-lva-history strong neural:upn" --tt-mb 0
for d in 4 5 6 7; do bench positions/quiet-100.fen $d q100-d$d "$MAIN"; done
echo "$(date +%H:%M:%S) done" >> "$OUT/progress.log"
