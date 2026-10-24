# Training neural move orderers

This folder builds training data from the benchmark's own search trees, trains
move-ordering networks with PyTorch, and exports them as `.nmo` weight files
that the C++ benchmark loads with `--strategy neural:NAME` (searching
`networks/NAME.nmo`) or `--strategy neural:path/to/file.nmo`.

`reproduce.sh` rebuilds every shipped network from scratch.

## Setup

```sh
cd training
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt   # CUDA wheels: see requirements.txt
cmake --build ../build/release --target nmo-datagen neural-move-ordering
```

## Results

Nodes and time relative to `mvv-lva-history`, the strongest heuristic in the
benchmark (lower is better; full tables in `results/`). Node counts are
deterministic; times are single-threaded wall-clock on a Ryzen 7 5800H.

| quiet-100 | depth 4 | depth 5 | depth 6 | depth 7 |
|---|---|---|---|---|
| `mvv-lva` | 119% nodes, 113% time | 114%, 111% | 130%, 131% | 139%, 136% |
| `neural:movemap` (Kocsis et al. 2003) | 102%, 152% | 98%, 141% | 105%, 170% | |
| `neural:nnue-policy` (Monty-style) | 121%, 143% | 112%, 133% | 138%, 167% | |
| `neural:cnn` (AlphaZero-style) | 87%, 171× | 89%, 126× | | |
| `neural:attention` (Lc0-style) | 74%, 238× | 83%, 187× | | |
| `neural:sf-nnue` (proposal: Stockfish labels) | 115%, 137% | 106%, 127% | 118%, 145% | |
| `neural:sf-upn` (Stockfish labels, our network) | 85%, 130% | 86%, 119% | 88%, 138% | 82%, 114% |
| `neural:tactical` (tables only) | 85%, 95% | 82%, 92% | 85%, 101% | 85%, 99% |
| `neural:upn-pure` | 76%, 119% | 80%, 113% | 81%, 133% | 79%, 115% |
| `neural:upn-max` | **71%**, 116% | **76%**, 112% | **72%**, 124% | **75%**, 112% |
| `neural:upn` | 75%, **99%** | 77%, **95%** | 74%, **101%** | 80%, **100%** |

On test-10, `upn` searches 61% / 77% / 70% of the baseline's nodes at depths
4 / 5 / 6 in 86% / 100% / 99% of the time, and `upn-max` 55% / 71% / 63%.
The other heuristics (`history`, `history-gravity`, `capture-hist`) need
3.5–22 times as many nodes as `mvv-lva-history` (see `results/summary.md`).

* `upn` is the recommended orderer: it searches about a quarter fewer nodes
  than the best heuristic in the same or less time.
* `upn-max` searches the fewest nodes, at a 10–25% time cost.
* The full-board networks (CNN, attention) order well but are 100–250 times
  slower; positional networks without tactical inputs (MoveMap, NNUE policy)
  do not beat MVV-LVA + history.

## Methodology

The benchmark searches with material-only evaluation, so the moves that cut
off in *this* search are what an orderer must find, and most ordering
decisions are made deep inside the tree, in positions that never occur in
games (for example straight after a blunder). The data therefore comes from
the search itself rather than from engine labels on game positions:

1. **Root positions** (`prepare_roots.py`): one random position per game from
   the Lichess 2013 archives, split into train and validation by game.
   Positions in `../positions/` are excluded so the benchmark sets stay held
   out; `quiet_positions.py` picks 200 quiet validation positions for
   choosing between models in the real search.
2. **Sampling decisions** (`nmo-datagen label`): each root is searched to
   depth 5 by `AlphaBetaSearcher`. A `SearchObserver` samples nodes at the
   moment their moves are ordered, at a rate growing with remaining depth
   (shallow decisions are plentiful and cheap to label, deep ones rare and
   expensive); quiescence decisions are sampled too.
3. **Labelling**: every legal move of a sampled node is searched with the
   node's real (alpha, beta) window by a fresh searcher
   (`AlphaBetaSearcher::probe`), recording whether it cuts off (score >= beta)
   and how many nodes it costs. The search's hash move is recorded too.
4. **Featurizing** (`nmo-datagen featurize`): positions and moves are encoded
   by `include/nmo/features.hpp`, the same code used at inference time, and
   `nmo_train.check` confirms C++ logits match PyTorch.

Targets (`nmo_train/data.py`): at a cut node the *efficiency* of searching
move m first is `c_min / c_m` if it cuts (c_min is the cheapest refutation)
and `c_min / (c_m + c_min)` if it fails. The loss is `-log sum_m p_m r_m`, the
negative log of the expected first-pick efficiency under the network's
softmax. All-nodes are dropped (their order does not change their cost), as
are nodes whose hash move already cuts; the hash move is masked out elsewhere
because the search tries it first anyway.

Generation 1 (2.97M sampled nodes, 81M labelled moves) comes from
`mvv-lva-history` searches; generation 2 (3.59M nodes) was searched and
probed with a generation-1 network (DAgger-style), and is mixed in for the
final UPN networks.

## The Updatable Policy Network (UPN)

`upn` combines three parts, each trained on the data above:

* **Network** (`UPN` in `models.py`, `src/neural/nnue_policy.cpp`): an
  NNUE-style feature transformer (768 piece-square inputs to 64 units per
  perspective) whose accumulators are updated incrementally and lazily along
  the search path. The clipped activations feed a per-move output: an int8
  row per (from, to) slot, a learned bias per (piece, victim, threat bits),
  and a low-rank interaction between the position and the move's tactical
  context. Threat bits are cheap bitboard tests: is the from- or to-square
  attacked, attacked by a lower-valued piece, or defended; does the move give
  direct check; does another own piece stay attacked.
* **Leaf head** (`TacticalTable`): the tactical and move biases alone. Most
  ordering decisions sit at remaining depth 1 and in quiescence, where
  tactics decide the cheapest refutation, so `upn` uses the leaf head there
  and the network only at depth >= 2, where each decision is rarer and
  costlier. Subtrees ordered by the leaf head skip accumulator updates.
* **History**: 0.0003 logits per unit of gravity history are added to quiet
  moves, giving the static network the search's dynamic cutoff statistics.

`upn-max` uses the network at every depth >= 1 and the leaf head in
quiescence; `upn-pure` is the network alone (MVV-LVA in quiescence, no history).

The network orders captures and quiet moves together. Forcing captures first,
as most engines and `mvv-lva-history` do, raised UPN's node count from 79% to
95% of the baseline: at depth-1 cut nodes the cheapest refutation is usually a
quiet move that leaves the opponent no capture, which ends the quiescence
search at once.

## What was tried

Offline numbers are validation ordering costs at cut nodes relative to
MVV-LVA (`rel`, lower is better); search numbers are nodes relative to
`mvv-lva-history` at depth 5. Early comparisons used quiet-100; later choices
were made on the separate 200 validation positions.

| Experiment | Result |
|---|---|
| Labels: search-derived vs Stockfish 19 best move (10k nodes), same UPN | 80% vs 86% nodes on quiet-100 (79% vs 82% at depth 7) |
| Labels on a positional NNUE policy | Stockfish labels 138% (106% with captures first); search labels 112% |
| Loss: efficiency vs cost-agnostic set vs best-move cross-entropy | 79% / 82% / 90% nodes |
| Loss variants: τ=2, no failing credit, cost weight 0.5, cost weight 1 | 79–80%; cost weight 0.5 best for `upn-max` (67% vs 68% at depth 6) |
| Network depth input (4 depth buckets) | better offline (0.505 vs 0.520) but worse in search (82% vs 79%) |
| Hidden size 32 vs 64 (validation positions); 128 | 79.3% vs 78.6%; 128 overfits (rel 0.518 vs 0.505) |
| Without the low-rank context interaction | 82% vs 79% |
| Tactical tables alone (no network) | 82–87% nodes, faster than the baseline |
| CNN (4 residual blocks × 32) / attention (2 layers × 64) | 89% / 83%, 126× / 187× slower |
| Network in quiescence instead of MVV-LVA | 76% vs 79%; the leaf head matches the network there (75.0% vs 75.1%, both with history) at a fraction of the cost |
| Generation 2 data alone vs generation 1 alone | 83% vs 79%: on-policy labels did not help on their own |
| Network for depth >= 2 trained only on those decisions | 76.4% vs 77.6% in the cascade |
| Node-context leaf tables, higher table learning rate | no gain |
| MVV-LVA at depth-1 nodes whose material is below beta | 1.5% more nodes, 3–5% less time; not adopted (removed) |

## Making it fast

UPN's cost per ordering decision started at 1.4 µs, against about 0.23 µs per
node for the whole baseline search. The changes that recovered the time:

* `features::describe` rewritten as an inline, branch-light function that
  builds its result in a register (10.7 → 5.9 ns per move).
* int8 move table with per-row scales and 7-bit activations (`maddubs`).
* Deferred evaluation: with a legal hash move, the search tries it before
  asking the network to order the rest (no change to the order).
* The depth cascade and untracked accumulator pushes described above.
* Lazy best-first move selection in the search (`OrderedMoves`): most nodes
  cut after one or two moves, and a full sort of distinct network scores
  costs 310 ns against 77 ns to select the best. This speeds up every
  strategy and leaves node counts unchanged.

## Commands

```sh
# Roots from Lichess archives (download from database.lichess.org)
.venv/bin/python prepare_roots.py data/lichess/*.pgn.zst --out data/roots

# Label and featurize
../build/release/nmo-datagen label data/roots/roots-train.fen --out data/gen1/train.raw \
    --depth 5 --rate 0.0015 --boost 3 --qrate 0.0005
../build/release/nmo-datagen featurize data/gen1/train.raw --out data/gen1/train

# Train and export (also writes a .pt checkpoint next to the .nmo file)
.venv/bin/python -m nmo_train.train --train data/gen1/train --val data/gen1/validation \
    --arch upn --model-args '{"hidden": 64, "depth_aware": false}' --out runs/upn.nmo

# Check that C++ inference matches PyTorch
.venv/bin/python -m nmo_train.check runs/upn.pt data/gen1/validation

# Combine a network with a leaf head and change orderer settings
.venv/bin/python -m nmo_train.export runs/upn.nmo ../networks/my-upn.nmo history_weight=0.0003 \
    --leaf-head runs/leaf.nmo --leaf-depth 1

# Benchmark
.venv/bin/python bench.py ../positions/quiet-100.fen --depth 5 mvv-lva-history neural:upn
.venv/bin/python final_eval.py --out results
```

Settings stored in a `.nmo` file control how the orderer uses the network:
`history_weight` (blend in gravity history), `quiescence` (order quiescence
captures with the network or leaf head instead of MVV-LVA), `leaf_depth`,
`staged` (captures first by MVV-LVA, network for quiet moves only) and
`quantized`.

`nmo-datagen stats ROOTS.fen` counts ordering decisions per depth,
`nmo-datagen evaluate RAW --strategy NET.nmo --out LOGITS --repeat 5` times
inference, and `nmo-datagen import` converts `FEN | move` labels (such as
`label_stockfish.py` output) into training records.
