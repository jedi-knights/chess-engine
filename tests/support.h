#pragma once
// Fixtures and helpers shared across per-module test TUs.

#include "position.h"
#include "types.h"

#include <algorithm>
#include <string>
#include <vector>

// The six standard perft positions from chessprogramming.org, minus the
// ones exercised only by test_perft.cpp — the subset here is used for
// position-layer and generator round-trip coverage.
inline const std::vector<std::string> STANDARD_FENS = {
    STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
};

inline bool contains_move(const MoveList& moves, Move needle) {
    return std::find(moves.begin(), moves.end(), needle) != moves.end();
}

inline int count_moves_from(const MoveList& moves, Square from) {
    int n = 0;
    for (Move m : moves) if (move_from(m) == from) ++n;
    return n;
}
