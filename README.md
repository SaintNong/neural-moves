# Neural Move Ordering

[![CI](https://github.com/SaintNong/neural-moves/actions/workflows/ci.yml/badge.svg)](https://github.com/SaintNong/neural-moves/actions/workflows/ci.yml)

A C++20 benchmark for comparing chess move ordering strategies using
alpha–beta search, iterative deepening, quiescence search, and a transposition
table. It uses [Disservin's chess library](https://disservin.github.io/chess-library/).

Available strategies: `mvv-lva`, `history`, `history-gravity`,
`capture-hist`, and `mvv-lva-history`. All five run by default. Neural
orderers are selected with `--strategy neural:NAME` (see
[Neural move ordering](#neural-move-ordering)).

## Build and run

Requires a C++20 compiler, CMake 3.20+, and a build tool (Make or Ninja). The
indicators headers and license are in `third_party/indicators/`.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel

# Search the starting position
./build/release/neural-move-ordering positions/start.fen --depth 4

# Disable the quiescence search
./build/release/neural-move-ordering positions/start.fen --depth 4 --no-quiescence

# Disable transposition-table move ordering
./build/release/neural-move-ordering positions/start.fen --depth 4 --no-hash-move-ordering

# Build and run the full dataset at depth 4
cmake --build build/release --target run

# Search the 100-position sample at depth 7
./build/release/neural-move-ordering positions/quiet-100.fen --depth 7

# Compare selected strategies on a smaller sample
./build/release/neural-move-ordering positions/quiet-labeled-5000.fen --depth 4 --limit 100 --strategy mvv-lva --strategy capture-hist

# Compare the best heuristic with the neural orderers
./build/release/neural-move-ordering positions/quiet-100.fen --depth 6 --strategy mvv-lva-history --strategy neural:upn --strategy neural:upn-max

# Show all options
./build/release/neural-move-ordering --help
```

Commands assume the repository root and a single-configuration generator such as
Make or Ninja. Set `-DNMO_RUN_DEPTH=5` when configuring to change the `run` depth.

Position files contain one FEN per line, optionally followed by
`| best_move_uci`. Blank lines and lines beginning with `#` are ignored.
FEN datasets are in `positions/`; neural training code is in `training/`.

## Results

By default, results appear in the terminal and are not saved. The summary
reports mean effective branching factor (EBF), node counts, cutoffs,
transposition table hits, and search time.

Save the summary and detailed results as CSV:

```sh
./build/release/neural-move-ordering positions/test-10.fen --depth 4 --csv --results results.csv > summary.csv
```

`summary.csv` contains one aggregate row per strategy. `results.csv` contains
one row per position and strategy, including the FEN, best UCI move, optional
input label, score in centipawns, depth, node counts, cutoffs, EBF, and time.
Quiescence search is enabled by default; `--no-quiescence` restores material-only
leaf evaluation at the requested depth. Transposition-table move ordering is
also enabled by default; `--no-hash-move-ordering` disables only move
prioritization from the table.
Interactive runs show one progress bar for the full dataset, with elapsed and
estimated total time in `(MM:SSs / ~MM:SSs)` format. It writes to stderr and is
hidden when stderr is redirected.

## Neural move ordering

`--strategy neural:NAME` orders moves with the network in `networks/NAME.nmo`
(or `--strategy neural:path/to/file.nmo`; `--network-dir` changes the
directory). The networks were trained on decisions sampled from this
benchmark's own search trees: every move of a sampled node is searched with
the node's real window, and networks learn to pick cheap refutations.
`training/README.md` describes the method, the experiments and how to
reproduce every network.

| Network | Description |
|---|---|
| `upn` | Updatable Policy Network (ours): incrementally updated NNUE-style accumulator plus per-move tactical features at depth >= 2, learned tactical tables near the leaves, history blending. Recommended. |
| `upn-max` | The same network at every depth >= 1: fewest nodes, slower |
| `upn-pure` | The network alone (MVV-LVA in quiescence, no history) |
| `tactical` | Learned tactical tables only, no network |
| `movemap` | Neural MoveMap (Kocsis et al., 2003) |
| `nnue-policy` | Efficiently updatable policy in the style of Monty |
| `cnn` | AlphaZero/Lc0-style residual CNN policy (about 100x slower) |
| `attention` | Transformer with an Lc0-style attention policy head (about 200x slower) |
| `sf-nnue` | The proposal's plan: an NNUE policy trained on Stockfish 19 best moves for game positions, captures first by MVV-LVA |
| `sf-upn` | Our network architecture trained on Stockfish 19 best moves |

Nodes and time relative to `mvv-lva-history` on `positions/quiet-100.fen`:

| Strategy | Depth 4 | Depth 5 | Depth 6 | Depth 7 |
|---|---|---|---|---|
| `neural:upn` | 75% nodes, 99% time | 77%, 95% | 74%, 101% | 80%, 100% |
| `neural:upn-max` | 71%, 116% | 76%, 112% | 72%, 124% | 75%, 112% |
| `neural:tactical` | 85%, 95% | 82%, 92% | 85%, 101% | 85%, 99% |
| `neural:sf-upn` | 85%, 130% | 86%, 119% | 88%, 138% | 82%, 114% |
| `neural:attention` | 74%, 238x | 83%, 187x | | |
| `neural:movemap` | 102%, 152% | 98%, 141% | 105%, 170% | |

Neural strategies return the same minimax scores as the heuristics; they only
change the order in which moves are searched.

## Development

```sh
# Run the doctest suite after building
ctest --test-dir build/release --output-on-failure

# Debug build
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug --parallel

# AddressSanitizer and UndefinedBehaviorSanitizer (GCC or Clang)
cmake -S . -B build/sanitize -DCMAKE_BUILD_TYPE=Debug -DNMO_SANITIZERS=ON
cmake --build build/sanitize --parallel
ctest --test-dir build/sanitize --output-on-failure

# Remove compiled outputs for one configuration
cmake --build build/release --target clean
```

`NMO_NATIVE` (on by default) compiles with `-march=native` so network
inference can use AVX2; configure with `-DNMO_NATIVE=OFF` for portable
binaries. `nmo-datagen` generates and featurizes training data and times
network inference (see `training/README.md`).

CMake tracks configured flag changes and rebuilds affected targets. Reconfigure
with `-DCMAKE_CXX_FLAGS="..."` to change flags; use a new build directory when
switching compilers (select with `-DCMAKE_CXX_COMPILER=clang++`). Tests are enabled
by default; configure with `-DBUILD_TESTING=OFF` to omit them. Search contracts and
metric definitions are documented in `include/nmo/search.hpp`.
