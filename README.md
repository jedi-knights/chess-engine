# chess-engine

A C++20 chess engine built as a validated milestone sequence — small enough to read in an afternoon, tested end-to-end.

[![CI](https://github.com/jedi-knights/chess-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/jedi-knights/chess-engine/actions/workflows/ci.yml)
[![Badge](https://github.com/jedi-knights/chess-engine/actions/workflows/badge.yaml/badge.svg)](https://github.com/jedi-knights/chess-engine/actions/workflows/badge.yaml)
[![Nightly SPRT](https://github.com/jedi-knights/chess-engine/actions/workflows/nightly-sprt.yml/badge.svg)](https://github.com/jedi-knights/chess-engine/actions/workflows/nightly-sprt.yml)
[![Baseline bump](https://github.com/jedi-knights/chess-engine/actions/workflows/baseline-bump.yml/badge.svg)](https://github.com/jedi-knights/chess-engine/actions/workflows/baseline-bump.yml)
[![Coverage](https://img.shields.io/badge/Coverage-94.3%25-brightgreen)](https://jedi-knights.github.io/chess-engine/?v=50)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)
![Status](https://img.shields.io/badge/status-early--development-orange.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <a href="#installation">Install</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#nnue">NNUE</a> ·
  <a href="#development">Development</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#contributing">Contributing</a>
</p>

## Overview

Chess engines are the AI behind chess apps — accept a board position, output a best move. Production engines (Stockfish, Leela, Ethereal) are 50k+ lines optimized over decades; reading them end-to-end to learn how the pieces fit is impractical.

Beginner tutorials swing the other way: they wire up move generation and a search in a few hundred lines but skip perft, the standard move-generation correctness gate. Subtle bugs — off-by-one on en passant, missed castling-rights update, promotion emitting the wrong piece — then ship silently and only surface as "the engine plays odd moves sometimes."

Building a chess engine correctly requires getting each of these right: bitboard representation, make/unmake with full undo, all pawn special cases (single/double push, ep, promotion, capture-promotion), sliding-piece attacks, castling rights bookkeeping, a legal-move filter, search, and evaluation. Any one of them wrong silently corrupts every game the engine plays.

This project builds each layer as a validated milestone. Every commit is a well-scoped piece of the engine with tests that prove correctness before the next milestone starts. Perft is the gate for move generation; doctest covers everything else.

```bash
$ make test
[doctest] test cases:    180 |    180 passed | 0 failed | 0 skipped
[doctest] assertions: 274847 | 274847 passed | 0 failed |

$ ./engine perft 2 | grep "Startpos" -A 2
=== Startpos ===
  [OK  ] depth 1: got 20, expected 20
  [OK  ] depth 2: got 400, expected 400
```

## Features

Currently implemented (all 8 milestones plus post-roadmap search / eval / UCI work):

- Bitboard position representation (piece mailbox + per-color/per-type bitboards + occupancy)
- FEN parsing and emission (round-tripped by the test suite)
- `Position::make_move` / `unmake_move` with a caller-owned `UndoInfo` — supports normal, capture, en passant, castling, and promotion move types. Incremental Zobrist key + incremental piece-square accumulators + repetition-key history stack.
- **Fully legal** move generation for all piece types (knights, king, pawns with all special cases, sliders, castling) with a pin-aware legality filter (make/unmake only when the shortcut can't rule the move in/out) and a precomputed enemy-attack bitboard for king-move legality. Perft matches all six standard positions through depth 5 (~200M node checks); `make perft-cert` extends to depth 6 (~8B nodes) as an offline correctness gate.
- Precomputed leaper attack tables (knight, king, pawn) + **magic bitboards for sliders** (O(1) bishop/rook/queen attack lookups; magic numbers found at init via seeded random search)
- Perft driver and 6-position standard test suite
- Evaluation: material + **piece-square tables** (Simplified Evaluation Function) with **tapered eval** (king PST interpolates linearly between middlegame safety and endgame centralization by non-pawn phase), **safe mobility** (per-piece weighted attack squares excluding enemy pawn attacks), **passed pawns** (separate MG/EG rank bonuses via precomputed masks), and a **bishop pair** bonus. Incremental PSQ so the hot path pays no per-piece loop.
- Search: **iterative-deepening negamax** with alpha-beta + **quiescence** (captures + promotions, in-check evasion, SEE-pruned) + **aspiration windows** (±75 cp, doubling on fail) + **Zobrist-hashed TT** (~1M entries, EXACT/LOWER/UPPER, mate-score ply-adjusted) + **PVS** (root and internal) + **null-window LMR** + **null-move pruning** (R=3, zugzwang-guarded) + **check extensions** + **reverse futility** + **razoring** + **SEE**-scored capture/promotion ordering (winning above killers, losing below) + **killer moves** + **history heuristic** (capped) + **repetition + 50-move** draw detection. Startpos reaches depth 10 in ~27 ms / ~301k nodes with a full 10-ply PV.
- **NNUE evaluation** — HalfKP → 256 → 1 architecture with a per-`Position` **incremental accumulator** (per-side dirty flags on king moves; non-king pieces update in O(features-per-piece) via `put_piece`/`remove_piece` hooks). **SIMD kernels** for the three hot loops — NEON on AArch64 (Apple Silicon), AVX2 on x86-64, scalar reference always compiled and pinned by SIMD-vs-reference equivalence tests. Custom **JNN1** binary file format (~20 MiB) with header validation. Off by default — enabled per-run via UCI `UseNNUE` + `EvalFile`; a companion **Python training pipeline** (`training/`, uv-scripts) turns self-play data into a loadable network.
- UCI protocol (`uci`, `isready`, `ucinewgame`, `position [startpos | fen ...] [moves ...]`, `go` with `depth`/`movetime`/`wtime`/`btime`/`winc`/`binc`/`movestogo`/`infinite`, `stop`, `d`, `quit`) on a background `std::thread`; per-iteration `info` lines emit `depth score cp nodes nps time pv <full line walked from the TT>`; `ucinewgame` clears the TT; `go infinite` runs asynchronously; sending `position` mid-search surfaces an `info string` before canceling. UCI `option` block exposes `UseNNUE` (check) + `EvalFile` (string) for enabling NNUE and pointing at a `.jnn1` file.
- doctest unit test suite (180 cases / 274k assertions) compiled with AddressSanitizer + UndefinedBehaviorSanitizer

Post-roadmap ideas still open (see `CLAUDE.md` non-goals): opening book / endgame tablebases, multi-threading (Lazy SMP), pondering, MultiPV output.

## Requirements

- **`clang++`** with C++20 support (or any C++20 compiler; override `CXX` in `Makefile` to swap)
- **`make`**
- macOS or Linux — verified on macOS 15 (Darwin 25); Linux CI is planned

## Installation

```bash
git clone https://github.com/jedi-knights/chess-engine.git
cd chess-engine
make
```

The release build produces a single binary at `./engine` compiled with `-O3 -march=native`.

## Usage

The engine speaks [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface) on stdin/stdout. Handshake:

```bash
$ ./engine
uci
id name jedi-engine 0.0.1
id author omar
uciok
quit
```

Any UCI-compatible GUI can drive it as an engine binary. Point the GUI at the compiled `./engine`. Common choices: [Arena](http://www.playwitharena.de/), [Cute Chess](https://cutechess.com/), [Banksia GUI](https://banksiagui.com/), [ChessBase](https://en.chessbase.com/).

The `go` command runs iterative-deepening alpha-beta search. Supported forms:

- `go depth N` — search to a fixed depth
- `go movetime N` — search until N milliseconds elapse (any-time; deepest completed iteration wins)
- `go wtime W btime B [winc I] [binc I] [movestogo N]` — derive movetime from the clock for the side to move. Sudden death (no `movestogo`) assumes ~30 more moves. Increment is spent generously since it refills the clock.
- `go infinite` — search until `stop`. Runs asynchronously on a background thread; the engine keeps reading commands.
- `go` (no args) — default depth 4

Each completed depth emits a UCI `info` line with `depth`, `score cp`, `nodes`, `nps`, `time`, and the full principal variation (walked from the TT), followed by `bestmove`.

## Examples

**1. Print the starting board.** Useful sanity check that FEN parsing and rendering work.

```bash
$ printf 'position startpos\nd\nquit\n' | ./engine
  +---+---+---+---+---+---+---+---+
8 | r | n | b | q | k | b | n | r |
  +---+---+---+---+---+---+---+---+
7 | p | p | p | p | p | p | p | p |
  +---+---+---+---+---+---+---+---+
...
1 | R | N | B | Q | K | B | N | R |
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

**2. Load a specific FEN.** Any position parseable via `set_from_fen` works.

```bash
$ printf 'position fen 4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1\nd\nquit\n' | ./engine | tail -3
    a   b   c   d   e   f   g   h
FEN: 4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1
```

**3. Run perft as a correctness gate.** Perft counts the number of leaf nodes in the move tree at a given depth. An engine whose perft numbers match all six standard positions has a correct move generator.

```bash
$ ./engine perft 2
=== Startpos ===
  [OK  ] depth 1: got 20, expected 20
  [OK  ] depth 2: got 400, expected 400
=== Kiwipete ===
  [OK  ] depth 1: got 48, expected 48
  [OK  ] depth 2: got 2039, expected 2039
...
```

All six positions match through depth 5 (`make perft`, ~200M nodes). `make perft-cert` runs depth 6 (~8B nodes, ~10 minutes) as an offline gate before shipping movegen changes.

## Configuration

No environment variables, no config files. The engine reads UCI on stdin and writes on stdout. Two per-run knobs live behind the UCI `option` block:

| Option     | Type     | Default    | Meaning |
|------------|----------|------------|---------|
| `UseNNUE`  | `check`  | `false`    | Route `evaluate()` through the NNUE forward pass. Falls back to classical eval when `false` or when no network is loaded. |
| `EvalFile` | `string` | `<empty>`  | Absolute path to a `.jnn1` file. Loads on assignment; a malformed / mistyped path keeps the previous state (safety net). |

Set them from any UCI GUI, or via stdin:

```bash
$ ./engine
setoption name EvalFile value /abs/path/to/net.jnn1
setoption name UseNNUE value true
position startpos
go depth 8
```

## NNUE

Classical evaluation adds up handcrafted terms — material, piece-square tables, mobility, pawn structure, king safety. Every extra term needs a human to (a) name a chess concept, (b) hand-tune its weight, (c) prove via SPRT it actually adds Elo. Progress is linear in the author's chess knowledge and time.

**NNUE** (Efficiently Updatable Neural Network) skips the handcrafting: a small neural network reads the raw board and outputs a score. The trick is the "efficiently updatable" part — the input to a chess network is 41,024 sparse features per side (HalfKP: one feature per `(king_square, piece_square, piece_type, piece_color)` combination), and recomputing all of them per position would be far slower than any classical eval. Instead the network's first-layer output is kept as a **per-position accumulator**: when a piece moves, only the affected feature-weight columns change, so `O(features per move)` scalar updates in `Position::put_piece` / `remove_piece` keep the accumulator in sync — cheaper than the classical mobility loop it replaces.

This repo's NNUE stack (all shipped, all off by default):

| Layer                        | What it does                                                                                                                                                  |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Runtime** (`src/nnue.*`)   | HalfKP → 256 → 1 forward pass. Per-side accumulator on `Position` with dirty flags for king moves (only the moving side's perspective invalidates).           |
| **SIMD** (`src/nnue_simd.h`) | NEON kernels for AArch64 (Apple Silicon), AVX2 kernels for x86-64. Scalar reference always compiled; SIMD-vs-reference equivalence is unit-tested.            |
| **JNN1 format** (`src/nnue.cpp`) | Custom int16 quantized binary — `"JNN1"` magic + `uint32 {version, hidden, features}` header + weights. Header-validated on load; not Stockfish-compatible.   |
| **Trainer** (`training/`)    | Python `torch` pipeline. Reads `fen;wdl` / `fen\|cp` self-play data → HalfKP encoding → MSE-on-sigmoid loss → int16 export in the runtime's format.           |

### Loading a network at runtime

```bash
$ ./engine
setoption name EvalFile value /abs/path/to/net.jnn1
setoption name UseNNUE value true
position startpos
go depth 8
```

The engine logs `nnue: loaded '<path>' (256 hidden units, 41024 features)` on success and emits `info string EvalFile loaded: <path>`. A missing / mistyped / architecture-mismatched file is rejected and the engine keeps whatever eval mode was active before.

### Training a network

See [`training/README.md`](training/README.md) for the full workflow. Short form:

```bash
# 1. Produce self-play data (all flags default; --help for the full surface)
scripts/gen_selfplay_data.py --games 500 --output data/selfplay.txt
# Or reuse the checked-in tests/data/selfplay_500.txt (49,772 positions).

# 2. Train — uv resolves torch + python-chess on first run and caches
training/train.py --data data/selfplay.txt --out net.jnn1 --epochs 20

# 3. Point the engine at the .jnn1 (see "Loading a network" above)
```

Correctness bridge: the Python `feature_index()` is pinned against the C++ `nnue::feature_index()` values in `tests/test_nnue.cpp` — a divergence would silently make trained networks unusable at inference.

## Development

```bash
make            # release build (-O3 -march=native) → ./engine
make debug      # -O0 -g with AddressSanitizer + UndefinedBehaviorSanitizer
make test       # compile + run the doctest suite under ASan            → ./tests/run
make perft      # ./engine perft 5 (~200M nodes; standard correctness gate)
make perft-cert # ./engine perft 6 (~8B nodes, ~10 min; offline gate before shipping movegen changes)
make run        # ./engine
make clean
```

### Layout

```
src/
  types.h            Bitboard / Move / Piece / Square + encoding helpers
  bitboard.[h|cpp]   popcount, lsb, pretty-print
  attacks.[h|cpp]    precomputed knight / king / pawn attack tables
  magic.[h|cpp]      magic bitboards — init-time search + O(1) slider attacks
  zobrist.[h|cpp]    Zobrist keys + init + full-recompute reference
  position.[h|cpp]   Position, FEN, make_move / unmake_move + UndoInfo;
                     incremental Zobrist + PSQ + NNUE accumulator hooks;
                     repetition-key stack
  movegen.[h|cpp]    generate_moves (fully legal) + in_check; pin-aware +
                     enemy-attack shortcuts skip most make/unmake round-trips
  perft.[h|cpp]      perft driver + 6-position standard suite
  eval.[h|cpp]       material + PST + tapered + mobility + passed pawn +
                     bishop pair; NNUE-aware entry point routes to
                     `nnue::evaluate` when UseNNUE + a loaded network
  nnue_types.h       Accumulator + shape constants (HIDDEN_SIZE, etc.);
                     kept here so `Position` can hold an Accumulator by
                     value without cycling nnue.h ↔ position.h
  nnue.[h|cpp]       NNUE runtime — HalfKP feature index, per-side
                     incremental accumulator with dirty flags, JNN1
                     binary loader/saver, forward pass
  nnue_simd.h        NEON + AVX2 kernels for add_column / sub_column /
                     forward_side, scalar reference always compiled
  tt.[h|cpp]         transposition table (fixed-size direct-mapped)
  search.[h|cpp]     iterative-deepening negamax + qsearch + TT + SEE + PVS
                     + LMR + null-move + check extensions + RFP + razoring
                     + killers + capped history + aspiration + repetition
  notation.[h|cpp]   UCI move ↔ Move (move_to_uci, parse_uci_move)
  uci.[h|cpp]        UCI protocol loop on a background std::thread
  main.cpp           entry point (dispatches `perft` or falls into UCI)
tests/               doctest suite; one file per src unit under test
training/            Python NNUE trainer (uv-scripts) — see training/README.md
third_party/
  doctest.h          v2.4.11 (pinned single-header)
```

### Perft: the correctness gate

Perft (**Per**formance **T**est) counts leaf nodes in the move tree at exactly `depth` plies from a starting position. Any off-by-one in move generation — a missing en-passant capture, an incorrectly-updated castling right, a promotion emitting the wrong piece — will make perft diverge from the standard values.

The standard values used here come from the [Chess Programming Wiki perft results page](https://www.chessprogramming.org/Perft_Results). See `src/perft.cpp` for the six-position suite.

## Evaluation

The engine's strength is measured by playing it against another engine (usually a saved snapshot of itself before a proposed change) and reading off the Elo delta. This is what confirms that a "tuner said MSE went down" or "search change looked good on Kiwipete" translates into actual wins over the board. Two tools cover it:

- **[Cute Chess](https://cutechess.com/)** — the standard toolkit for automating chess-engine matches. Cross-platform, C++/Qt, actively developed since 2008. Ships a GUI (`cutechess`) plus a headless tournament runner (`cutechess-cli`). We use `cutechess-cli`.
- **[fastchess](https://github.com/Disservin/fastchess)** — a drop-in `cutechess-cli` replacement with a single-Makefile build (no Qt). Faster at running large tournaments. `scripts/sprt.py` picks it up automatically if it's on `PATH`.

### Installing Cute Chess

Cute Chess is not in Homebrew. Options:

- **Debian / Ubuntu**: `sudo apt install cutechess-cli` — packaged, older but functional.
- **Release binaries**: [github.com/cutechess/cutechess/releases](https://github.com/cutechess/cutechess/releases) (macOS `.dmg`, Windows `.exe`, Linux `.tar.gz`).
- **From source** (macOS, Linux, Windows): requires Qt 6.8+, cmake, and a C++17 compiler:
  ```bash
  git clone https://github.com/cutechess/cutechess.git
  cd cutechess
  cmake -S . -B build
  cmake --build build
  # cutechess-cli binary at: build/cutechess-cli
  ```

If the Qt dependency is painful, install fastchess instead — same command-line surface, no Qt:

```bash
git clone https://github.com/Disservin/fastchess.git
cd fastchess
make -j
# fastchess binary at: ./fastchess
```

### Running a match

Play 10 games between two copies of this engine at 40 moves in 60 seconds each side:

```bash
cutechess-cli \
    -engine cmd=./engine name=engine1 \
    -engine cmd=./engine name=engine2 \
    -each proto=uci tc=40/60 \
    -rounds 10
```

The `-each` block applies to every engine — set the UCI protocol (`proto=uci`) and time control once instead of duplicating per `-engine`. Cutechess prints per-game results to stdout; add `-pgnout games.pgn` to save the games as PGN for later analysis.

Time control syntax:

| Form | Meaning |
|---|---|
| `tc=40/60` | 40 moves in 60 seconds |
| `tc=10+0.1` | 10 seconds plus 0.1s per-move increment |
| `tc=inf` | Unlimited (engine controls its own thinking time) |
| `st=1` | 1 second per move, no increment |

### SPRT — the eval regression gate

For validating any proposed change (tuned eval weights, refactored search, new pruning technique), what matters is whether the change adds Elo — not that it just plays differently. Running 10 games is noise; you need hundreds or thousands. Sequential Probability Ratio Test (Wald 1945) stops as soon as enough evidence accumulates to accept either **H0** ("no gain") or **H1** ("meaningful gain"), so a decisive change resolves in a few hundred games while a marginal one runs to the cap:

```bash
# 1. Snapshot baseline before applying the change
make && cp engine engine.baseline

# 2. Apply the change and rebuild
#    ...edit src/eval.cpp / src/search.cpp ...
make

# 3. Run SPRT — usually stops in 200-2000 games depending on effect size
scripts/sprt.py --baseline engine.baseline --tuned engine \
                --tc 10+0.1 --elo0 0 --elo1 5
```

`scripts/sprt.py` is a thin wrapper around `cutechess-cli` / `fastchess` that adds preflight validation (both binaries exist, opening book exists, game manager on `PATH`), a checked-in demo opening book (`tests/data/openings_demo.pgn`), and streams results as they arrive. See `scripts/sprt.py --help` for the full option surface (concurrency, max-games, PGN output).

Cutechess handles the SPRT statistic natively (log-likelihood ratio test); the harness just wraps the invocation and reports the verdict.

### Nightly CI SPRT

CI runs SPRT overnight (07:00 UTC) between `HEAD` and the commit currently tagged `baseline` — see `.github/workflows/nightly-sprt.yml`. When SPRT accepts H0 (regression), the workflow auto-opens a labeled GitHub issue with SHAs, config, and a log excerpt so the regression surfaces without a human noticing red X's. Once you're confident a change genuinely added Elo (SPRT-confirmed H1), advance the baseline tag:

```bash
git tag -f baseline <sha>
git push -f origin baseline
```

The tag-push triggers `.github/workflows/baseline-bump.yml`, which auto-closes any stale open regression issues.

## Contributing

Contributions welcome. The design is milestone-driven — see `CLAUDE.md` for the current roadmap, current state, and coding conventions.

The workflow:

1. Pick the next unstarted milestone (or file an issue proposing something else)
2. Fork and branch: `git checkout -b feat/milestone-N-<name>`
3. Implement, run `make test`, run `make perft` (or `make perft-cert` for anything touching movegen or `Position`) — verify no regression in already-passing positions and depths
4. Open a PR describing what the milestone adds and what changes in the perft output

Coding conventions: no comments that restate what code obviously does; keep milestones scoped to one concern per commit; every non-trivial function carries at least one invariant assertion; do not weaken existing assertions to silence a spurious failure — investigate the position.

## License

MIT — see [LICENSE](LICENSE).

## References

- [Chess Programming Wiki — Getting Started](https://chessprogramming.org/Getting_Started) — the canonical reference for chess engine implementation techniques. Every algorithm and data structure in this repo has a corresponding article there.
- [Chess Programming Wiki — Perft Results](https://www.chessprogramming.org/Perft_Results) — source of the standard perft counts used by `src/perft.cpp`.
- [UCI protocol specification](https://backscattering.de/chess/uci/) — mirror of the Stefan Meyer-Kahlen UCI spec.
- [doctest documentation](https://github.com/doctest/doctest) — testing framework used here.
