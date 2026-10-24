#!/bin/bash
# End-to-end reproduction of the shipped networks in ../networks.
# Takes a few hours on an 8-core CPU with a small CUDA GPU. Run from training/.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
DATAGEN=../build/release/nmo-datagen
train() { $PY -m nmo_train.train "$@"; }

# 1. Root positions: one random position per game, split by game, benchmark positions excluded.
mkdir -p data/lichess
for month in 01 02 03 04 05; do
    file=data/lichess/lichess_2013-$month.pgn.zst
    [ -f $file ] || curl -sSfL -o $file https://database.lichess.org/standard/lichess_db_standard_rated_2013-$month.pgn.zst
done
$PY prepare_roots.py data/lichess/lichess_2013-0{1,2}.pgn.zst --out data/roots
$PY prepare_roots.py data/lichess/lichess_2013-0{3,4,5}.pgn.zst --out data/roots2 --seed 2
$PY quiet_positions.py data/roots/roots-validation.fen --count 200 --out data/search-validation-200.fen

# 2. Generation 1: search-tree decisions sampled from mvv-lva-history searches, every move probed.
SAMPLING="--depth 5 --rate 0.0015 --boost 3 --qrate 0.0005"
mkdir -p data/gen1
for split in train validation; do
    $DATAGEN label data/roots/roots-$split.fen --out data/gen1/$split.raw $SAMPLING
    $DATAGEN featurize data/gen1/$split.raw --out data/gen1/$split
done

# 3. Generation 2 (on-policy): searched and probed with a generation-1 UPN.
mkdir -p runs data/gen2
train --train data/gen1/train --val data/gen1/validation --arch upn \
    --model-args '{"hidden":64,"depth_aware":false}' --epochs 8 --out runs/upn-gen1.nmo
$PY -m nmo_train.export runs/upn-gen1.nmo runs/gen2-policy.nmo quiescence=1 history_weight=0.0003 quantized=0
POLICY="--strategy neural:runs/gen2-policy.nmo --label-strategy neural:runs/gen2-policy.nmo"
$DATAGEN label data/roots2/roots-validation.fen --out data/gen2/validation.raw $SAMPLING $POLICY
$DATAGEN label data/roots2/roots-train.fen --limit 300000 --seed 2 --out data/gen2/train.raw $SAMPLING $POLICY
for split in train validation; do $DATAGEN featurize data/gen2/$split.raw --out data/gen2/$split; done

# 4. Stockfish labels on game positions (the proposal's methodology), for comparison.
mkdir -p data/stockfish data/sf
SF=data/stockfish/stockfish/stockfish-linux-x86-64-universal
[ -x $SF ] || (cd data/stockfish && curl -sSfL https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz | tar xz)
$PY label_stockfish.py data/roots/roots-train.fen --engine $SF --nodes 10000 --out data/stockfish/roots-train-sf.fen
$DATAGEN import data/stockfish/roots-train-sf.fen --out data/sf/train.raw
$DATAGEN featurize data/sf/train.raw --out data/sf/train

# 5. Networks.
G1="--train data/gen1/train --val data/gen1/validation --epochs 8"
G12="--train data/gen1/train data/gen2/train --val data/gen1/validation"
UPN='{"hidden":64,"depth_aware":false}'
train $G1 --arch movemap --model-args '{"hidden":128}' --out runs/movemap.nmo
train $G1 --arch nnue-policy --model-args '{"hidden":64}' --out runs/nnue-policy.nmo
train $G1 --arch cnn --model-args '{"filters":32,"blocks":4}' --out runs/cnn.nmo
train $G1 --arch attention --model-args '{"dim":64,"layers":2,"heads":4,"ffn":128}' --out runs/attention.nmo
train $G1 --arch tactical --lr 0.02 --out runs/tactical.nmo
train --train data/sf/train --val data/gen1/validation --epochs 8 --loss best --arch nnue-policy \
    --model-args '{"hidden":64}' --out runs/sf-nnue.nmo
train --train data/sf/train --val data/gen1/validation --epochs 8 --loss best --arch upn \
    --model-args "$UPN" --out runs/sf-upn.nmo
# UPN: a network for every depth (cost-weighted), one for depths >= 2, and a tactical leaf head.
train $G12 --epochs 8 --arch upn --model-args "$UPN" --cost-weight 0.5 --out runs/upn-pure.nmo
train $G12 --epochs 12 --arch upn --model-args "$UPN" --depths 2,3,4,5,6 --out runs/upn-deep.nmo
train $G1 --epochs 10 --arch tactical --depths 0,1 --lr 0.02 --out runs/upn-leaf.nmo

mkdir -p ../networks
for name in movemap nnue-policy cnn attention tactical sf-upn upn-pure; do cp runs/$name.nmo ../networks/; done
$PY -m nmo_train.export runs/sf-nnue.nmo ../networks/sf-nnue.nmo staged=1
$PY -m nmo_train.export runs/upn-deep.nmo ../networks/upn.nmo history_weight=0.0003 \
    --leaf-head runs/upn-leaf.nmo --leaf-depth 1
$PY -m nmo_train.export runs/upn-pure.nmo ../networks/upn-max.nmo history_weight=0.0003 \
    --leaf-head runs/upn-leaf.nmo --leaf-depth 0

# 6. Benchmark.
$PY final_eval.py --out results
