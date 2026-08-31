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
7. ✅ Legality filter — make/unmake round-trip + `is_square_attacked` on own king. **All 6 standard perft positions match through depth 4** (~10.7M node checks) — move generation is provably correct.
8. ⬜ Search in `cmd_go` (negamax + alpha-beta; needs an evaluation function)

Do not skip a milestone. Perft numbers stay artificially low until every piece type generates, but each milestone's *round-trip* invariants (see `tests/test_position.cpp`) must hold before advancing.

## Testing conventions

- Framework: **doctest** (single header, no CMake). `TEST_CASE` / `SUBCASE` / `CHECK` / `REQUIRE`. See any existing `tests/test_*.cpp` for style.
- Every new pure-logic behavior gets a test. Round-trip invariants (make → unmake, encode → decode, FEN parse → emit) are the highest-value shape.
- **One test file per src unit under test**, mirroring `src/`:
    - `tests/test_bitboard.cpp` ↔ `src/bitboard.[h|cpp]`
    - `tests/test_attacks.cpp`  ↔ `src/attacks.[h|cpp]`
    - `tests/test_position.cpp` ↔ `src/position.[h|cpp]` (FEN + make/unmake forward-correctness only)
    - `tests/test_movegen.cpp`  ↔ `src/movegen.[h|cpp]` (generator shape + movegen-driven make/unmake walks)
    - `tests/test_perft.cpp`    ↔ `src/perft.[h|cpp]`
    - `tests/test_uci.cpp`      ↔ `src/uci.[h|cpp]` (protocol via stringstream — `uci_loop` takes `std::istream&/std::ostream&` for exactly this reason; do not reintroduce `std::cin`/`std::cout` inside the loop)
  New src units require a matching `tests/test_<unit>.cpp`. Shared fixtures / helpers live in `tests/support.h`.
- `tests/test_main.cpp` uses `DOCTEST_CONFIG_IMPLEMENT` and provides `main()` — this is the single place where `init_attacks()` is called so per-TU static-init hacks are unnecessary.
- Tests link the whole `src/` tree (except `main.cpp`) — see Makefile `$(TEST_SRCS)`.
- Do not mock `Position` internals. Verify through `to_fen()` / public accessors.

## Perft gotcha

`make_move`'s king-count assertion is `<= 1`, not `== 1`. Real games and every current perft position have exactly one king per side, but the loose form defends against hand-constructed test positions that omit a king (bishop-only slider tests, etc.). Do not tighten it without auditing every test FEN.

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
