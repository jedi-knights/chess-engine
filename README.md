# jedi-engine

A C++20 chess engine with bitboard move generation and a UCI protocol front-end.

Early-stage — move generation and perft validation are in place; search is stubbed.

## Build

Requires `clang++` (or any C++20 compiler; edit `CXX` in the `Makefile` to swap) and `make`.

```sh
make          # release build (-O3 -march=native) → ./engine
make debug    # -O0 -g with ASan + UBSan
make clean
```

## Run

```sh
./engine              # UCI mode — talks to any UCI-compatible GUI (Arena, Cute Chess, etc.)
./engine perft 5      # run the standard perft suite up to depth 5
make run              # ./engine
make perft            # ./engine perft 5
```

## UCI commands supported

`uci`, `isready`, `ucinewgame`, `position [startpos | fen ...]`, `go`, `d` (print board), `quit`.

`go` currently returns `bestmove 0000` — search is not implemented yet.

## Layout

```
src/
  bitboard.[h|cpp]   64-bit board representation and helpers
  attacks.[h|cpp]    precomputed attack tables
  position.[h|cpp]   Position, FEN parsing, make/unmake
  movegen.[h|cpp]    pseudo-legal + legal move generation
  perft.[h|cpp]      perft driver + standard test suite
  uci.[h|cpp]        UCI protocol loop
  main.cpp           entry point
```

Perft is the correctness gate for move generation — an engine that matches all six standard positions through depth 5+ has a correct generator.

## License

TBD.
