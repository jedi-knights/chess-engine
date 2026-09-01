# chess-engine

A C++20 chess engine built as a validated milestone sequence — small enough to read in an afternoon, tested end-to-end.

![CI](https://github.com/jedi-knights/chess-engine/actions/workflows/ci.yml/badge.svg)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)
![Status](https://img.shields.io/badge/status-early--development-orange.svg)

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
[doctest] test cases:    11 |    11 passed | 0 failed | 0 skipped
[doctest] assertions: 37456 | 37456 passed | 0 failed |

$ ./engine perft 2 | grep "Startpos" -A 2
=== Startpos ===
  [OK  ] depth 1: got 20, expected 20
  [OK  ] depth 2: got 400, expected 400
```

## Features

Currently implemented (all 8 milestones complete):

- Bitboard position representation (piece mailbox + per-color/per-type bitboards + occupancy)
- FEN parsing and emission (round-tripped by the test suite)
- `Position::make_move` / `unmake_move` with a caller-owned `UndoInfo` — supports normal, capture, en passant, castling, and promotion move types
- **Fully legal** move generation for all piece types (knights, king, pawns with all special cases, sliders, castling) with a king-not-in-check legality filter. Perft matches the six standard positions through depth 4 (~10.7M node checks).
- Precomputed leaper attack tables (knight, king, pawn) + **magic bitboards for sliders** (O(1) bishop/rook/queen attack lookups; magic numbers found at init via seeded random search)
- Perft driver and 6-position standard test suite
- Material evaluation with **piece-square tables** (Simplified Evaluation Function values) and **tapered eval** — king PST interpolates linearly between middlegame (safety) and endgame (centralization) tables by remaining non-pawn material. Score in centipawns from side-to-move perspective.
- **Iterative-deepening negamax with alpha-beta pruning** + **quiescence search** at leaves (extends captures until quiet, resolves horizon-effect blunders) + **MVV-LVA move ordering** + **Zobrist-hashed transposition table** (~1M entries, EXACT/LOWER/UPPER bounds, mate-score ply adjustment) + **killer moves and history heuristic** (order quiet-move beta-cutoffs first); ~11.6× speedup over baseline alpha-beta at depth 6; supports `go movetime N` with mid-iteration cancellation (any-time property); `ucinewgame` clears the TT
- UCI protocol (`uci`, `isready`, `ucinewgame`, `position [startpos | fen ...] [moves ...]`, `go` with `depth`/`movetime`/`wtime`/`btime`/`winc`/`binc`/`movestogo`/`infinite`, `stop`, `d`, `quit`) with per-iteration `info` lines and `bestmove` output; `go infinite` runs asynchronously so the engine keeps reading commands
- doctest unit test suite (64 cases) compiled with AddressSanitizer + UndefinedBehaviorSanitizer

Post-roadmap ideas (see CLAUDE.md): iterative deepening + time management, move ordering (MVV-LVA / killers / history), quiescence search, transposition table with Zobrist hashing, magic bitboards, piece-square tables, `position ... moves e2e4 ...` UCI extension.

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

Any UCI-compatible GUI (Arena, Cute Chess, Banksia, ChessBase) can drive it as an engine binary. Point the GUI at the compiled `./engine`.

The `go` command runs iterative-deepening alpha-beta search. Supported forms:

- `go depth N` — search to a fixed depth
- `go movetime N` — search until N milliseconds elapse (any-time; deepest completed iteration wins)
- `go wtime W btime B [winc I] [binc I] [movestogo N]` — derive movetime from the clock for the side to move. Sudden death (no `movestogo`) assumes ~30 more moves. Increment is spent generously since it refills the clock.
- `go infinite` — search until `stop`. Runs asynchronously on a background thread; the engine keeps reading commands.
- `go` (no args) — default depth 4

Each completed depth emits a UCI `info` line with depth, centipawn score, node count, and the root move as PV, followed by `bestmove`.

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
  [FAIL] depth 1: got 21, expected 48
...
```

Startpos matches exactly at milestone 4; other positions require sliding pieces + castling + a legality filter to reach their targets.

## Configuration

None. No environment variables, no config files. The engine reads UCI on stdin and writes on stdout.

## Development

```bash
make          # release build (-O3 -march=native) → ./engine
make debug    # -O0 -g with AddressSanitizer + UndefinedBehaviorSanitizer
make test     # compile + run the doctest suite under ASan            → ./tests/run
make perft    # ./engine perft 5
make run      # ./engine
make clean
```

### Layout

```
src/
  types.h            Bitboard / Move / Piece / Square + encoding helpers
  bitboard.[h|cpp]   popcount, lsb, pretty-print
  attacks.[h|cpp]    precomputed knight / king / pawn attack tables
  position.[h|cpp]   Position, FEN parsing, make_move / unmake_move + UndoInfo
  movegen.[h|cpp]    generate_moves — knights, king, pawns as of milestone 4
  perft.[h|cpp]      perft driver + 6-position standard suite
  uci.[h|cpp]        UCI protocol loop
  main.cpp           entry point
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
3. Implement, run `make test`, run `make perft 5` — verify no regression in already-passing positions and depths
4. Open a PR describing what the milestone adds and what changes in the perft output

Coding conventions: no comments that restate what code obviously does; keep milestones scoped to one concern per commit; every non-trivial function carries at least one invariant assertion; do not weaken existing assertions to silence a spurious failure — investigate the position.

## License

Not yet chosen. A `LICENSE` file will land before the first tagged release.

## References

- [Chess Programming Wiki — Getting Started](https://chessprogramming.org/Getting_Started) — the canonical reference for chess engine implementation techniques. Every algorithm and data structure in this repo has a corresponding article there.
- [Chess Programming Wiki — Perft Results](https://www.chessprogramming.org/Perft_Results) — source of the standard perft counts used by `src/perft.cpp`.
- [UCI protocol specification](https://backscattering.de/chess/uci/) — mirror of the Stefan Meyer-Kahlen UCI spec.
- [doctest documentation](https://github.com/doctest/doctest) — testing framework used here.
