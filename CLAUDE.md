# chess-engine

A C++20 chess engine (bitboard movegen + UCI). Author: single dev, hobby project.

## Build & test

```
make          # release build (-O3 -march=native)         → ./engine
make debug    # -O0 -g + ASan/UBSan                        → ./engine
make test     # compile + run doctest suite under ASan     → ./tests/run
make perft    # ./engine perft 5 (correctness gate)
make run      # ./engine  (UCI stdin loop)
make clean
```

Every change that touches movegen or `Position` must pass **both** `make test` and `make perft` at the highest depth that currently works. Add tests when adding behavior; extend the perft target depth as movegen milestones complete.

## Layout

```
src/                engine sources
  types.h           Bitboard/Move/Piece/Square + encoding helpers
  bitboard.[h|cpp]  popcount, lsb, pretty
  attacks.[h|cpp]   precomputed leaper attack tables (knight/king/pawn)
  position.[h|cpp]  Position, FEN, make_move/unmake_move + UndoInfo
  movegen.[h|cpp]   generate_moves — currently knights only, milestones in header
  perft.[h|cpp]     perft driver + 6-position standard suite
  uci.[h|cpp]       UCI protocol loop
  main.cpp          entry (dispatches `perft` subcommand or falls into UCI)
tests/              doctest suite; one file per src unit under test
third_party/
  doctest.h         v2.4.11 (pinned single-header). Update via curl from the release tag.
```

## Movegen milestone roadmap

Tracked in `src/movegen.h`. Each milestone is committed separately and validated by extending the perft depth that passes.

1. ✅ `Position::make_move` / `unmake_move` (supports all move types including ep, castling, promotion)
2. ✅ Knight moves (no legality filter yet)
3. ✅ King moves (no legality filter, no castling — those are 7 and 6)
4. ✅ Pawn moves (pushes, double, captures, ep, promotions incl. underpromotion + capture-promotion)
5. ✅ Sliding pieces — bishop, rook, queen (naive per-step rays; magic bitboards deferred)
6. ✅ Castling (rights + emptiness + king start/transit/land squares not attacked)
7. ⬜ Legality filter (king-not-in-check after move)
8. ⬜ Search in `cmd_go` (negamax + alpha-beta; blocked on full movegen + eval)

Do not skip a milestone. Perft numbers stay artificially low until every piece type generates, but each milestone's *round-trip* invariants (see `tests/test_position.cpp`) must hold before advancing.

## Testing conventions

- Framework: **doctest** (single header, no CMake). `TEST_CASE` / `SUBCASE` / `CHECK` / `REQUIRE`. See existing `tests/test_position.cpp` for style.
- Every new pure-logic behavior gets a test. Round-trip invariants (make → unmake, encode → decode, FEN parse → emit) are the highest-value shape.
- Test files: `tests/test_<unit>.cpp` mirroring `src/`. Include `"doctest.h"` first; `#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN` lives only in `tests/test_main.cpp`.
- Tests link the whole `src/` tree (except `main.cpp`) — see Makefile `$(TEST_SRCS)`. Attack tables must be initialized once per binary; `test_position.cpp` shows the static-init idiom.
- Do not mock `Position` internals. Verify through `to_fen()` / public accessors.

## Perft gotcha

Position 4 in the standard suite has **no white king** — it's a contrived position for isolating castling/promotion. `make_move`'s king-count assertion is `<= 1`, not `== 1`, to accommodate this. Do not tighten it.

## Move encoding (types.h)

16 bits: `from(6) | to(6) | promo(2) | movetype(2)`. `promo` is offset from KNIGHT, so it always fits in 2 bits. `NULL_MOVE == 0` is safe because a legal move never has `from == to == A1`.

## References

- [Chess Programming Wiki — Getting Started](https://chessprogramming.org/Getting_Started) is the canonical reference for every technique in this repo (bitboards, move generation, perft, search, evaluation). Consult it before designing a new subsystem.
- [Perft Results](https://www.chessprogramming.org/Perft_Results) — source of the six-position suite in `src/perft.cpp`. Any change to move generation must be validated against these.
- [UCI protocol](https://backscattering.de/chess/uci/) — spec for the stdin/stdout interface `src/uci.cpp` implements.

## Non-goals (for now)

- Zobrist hashing / transposition table
- Move ordering, killers, history heuristics
- Time management
- Opening book / endgame tablebases
- Multi-threading

These belong after milestone 8. Do not add them speculatively.
