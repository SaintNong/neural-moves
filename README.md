# Neural Move Ordering

[![CI](https://github.com/SaintNong/neural-moves/actions/workflows/ci.yml/badge.svg)](https://github.com/SaintNong/neural-moves/actions/workflows/ci.yml)

A C++20 benchmark for comparing chess move ordering strategies using
alpha–beta search, iterative deepening, and a transposition table. It uses
[Disservin's chess library](https://disservin.github.io/chess-library/).

Available strategies: `captures` (MVV-LVA), `history`, `history-gravity`,
`capture-hist`, and `mvv-lva-history`. All five run by default.

## Build and run

Requires a C++20 compiler, CMake 3.20+, and a build tool (Make or Ninja). The
indicators headers and license are in `third_party/indicators/`.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel

# Search the starting position
./build/release/neural-move-ordering positions/start.fen --depth 4

# Build and run the full dataset at depth 4
cmake --build build/release --target run

# Search the 100-position sample at depth 7
./build/release/neural-move-ordering positions/quiet-100.fen --depth 7

# Compare selected strategies on a smaller sample
./build/release/neural-move-ordering positions/quiet-labeled-5000.fen --depth 4 --limit 100 --strategy captures --strategy capture-hist

# Show all options
./build/release/neural-move-ordering --help
```

Commands assume the repository root and a single-configuration generator such as
Make or Ninja. Set `-DNMO_RUN_DEPTH=5` when configuring to change the `run` depth.

Position files contain one FEN per line, optionally followed by
`| best_move_uci`. Blank lines and lines beginning with `#` are ignored.
FEN datasets are in `positions/`; prepared training data is in
`training_examples/`.

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
Interactive runs show one progress bar for the full dataset, with elapsed and
estimated total time in `(MM:SSs / ~MM:SSs)` format. It writes to stderr and is
hidden when stderr is redirected.

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

CMake tracks configured flag changes and rebuilds affected targets. Reconfigure
with `-DCMAKE_CXX_FLAGS="..."` to change flags; use a new build directory when
switching compilers (select with `-DCMAKE_CXX_COMPILER=clang++`). Tests are enabled
by default; configure with `-DBUILD_TESTING=OFF` to omit them. Search contracts and
metric definitions are documented in `include/nmo/search.hpp`.
