# chess-engine

A C++20 chess engine built as a validated milestone sequence — small enough to read in an afternoon, tested end-to-end.

[![CI](https://github.com/jedi-knights/chess-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/jedi-knights/chess-engine/actions/workflows/ci.yml)
[![Badge](https://github.com/jedi-knights/chess-engine/actions/workflows/badge.yaml/badge.svg)](https://github.com/jedi-knights/chess-engine/actions/workflows/badge.yaml)
[![Coverage](https://img.shields.io/badge/Coverage-98.5%25-brightgreen)](https://jedi-knights.github.io/chess-engine/?v=13)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)
![Status](https://img.shields.io/badge/status-early--development-orange.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Overview

Chess engines are the AI behind chess apps — accept a board position, output a best move. Production engines (Stockfish, Leela, Ethereal) are 50k+ lines optimized over decades; reading them end-to-end to learn how the pieces fit is impractical.

Beginner tutorials swing the other way: they wire up move generation and a search in a few hundred lines but skip perft, the standard move-generation correctness gate. Subtle bugs — off-by-one on en passant, missed castling-rights update, promotion emitting the wrong piece — then ship silently and only surface as "the engine plays odd moves sometimes."

Building a chess engine correctly requires getting each of these right: bitboard representation, make/unmake with full undo, all pawn special cases (single/double push, ep, promotion, capture-promotion), sliding-piece attacks, castling rights bookkeeping, a legal-move filter, search, and evaluation. Any one of them wrong silently corrupts every game the engine plays.

This project builds each layer as a validated milestone. Every commit is a well-scoped piece of the engine with tests that prove correctness before the next milestone starts. Perft is the gate for move generation; doctest covers everything else.

```bash
$ make test
[doctest] test cases:    144 |    144 passed | 0 failed | 0 skipped
[doctest] assertions: 270026 | 270026 passed | 0 failed |

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
- UCI protocol (`uci`, `isready`, `ucinewgame`, `position [startpos | fen ...] [moves ...]`, `go` with `depth`/`movetime`/`wtime`/`btime`/`winc`/`binc`/`movestogo`/`infinite`, `stop`, `d`, `quit`) on a background `std::thread`; per-iteration `info` lines emit `depth score cp nodes nps time pv <full line walked from the TT>`; `ucinewgame` clears the TT; `go infinite` runs asynchronously; sending `position` mid-search surfaces an `info string` before canceling.
- doctest unit test suite (144 cases / 270k assertions) compiled with AddressSanitizer + UndefinedBehaviorSanitizer

Post-roadmap ideas still open (see `CLAUDE.md` non-goals): opening book / endgame tablebases, multi-threading (Lazy SMP), pondering, MultiPV output, NNUE eval.

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

None. No environment variables, no config files. The engine reads UCI on stdin and writes on stdout.

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
                     incremental Zobrist + PSQ; repetition-key stack
  movegen.[h|cpp]    generate_moves (fully legal) + in_check; pin-aware +
                     enemy-attack shortcuts skip most make/unmake round-trips
  perft.[h|cpp]      perft driver + 6-position standard suite
  eval.[h|cpp]       material + PST + tapered + mobility + passed pawn +
                     bishop pair
  tt.[h|cpp]         transposition table (fixed-size direct-mapped)
  search.[h|cpp]     iterative-deepening negamax + qsearch + TT + SEE + PVS
                     + LMR + null-move + check extensions + RFP + razoring
                     + killers + capped history + aspiration + repetition
  notation.[h|cpp]   UCI move ↔ Move (move_to_uci, parse_uci_move)
  uci.[h|cpp]        UCI protocol loop on a background std::thread
  main.cpp           entry point (dispatches `perft` or falls into UCI)
tests/               doctest suite; one file per src unit under test
third_party/
  doctest.h          v2.4.11 (pinned single-header)
```

### Perft: the correctness gate

Perft (**Per**formance **T**est) counts leaf nodes in the move tree at exactly `depth` plies from a starting position. Any off-by-one in move generation — a missing en-passant capture, an incorrectly-updated castling right, a promotion emitting the wrong piece — will make perft diverge from the standard values.

The standard values used here come from the [Chess Programming Wiki perft results page](https://www.chessprogramming.org/Perft_Results). See `src/perft.cpp` for the six-position suite.

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
