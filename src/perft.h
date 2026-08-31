#pragma once
#include "position.h"
#include <cstdint>

// Count leaf nodes at exactly `depth` from `pos`. Correctness gate for
// move generation: an engine whose perft matches all 6 standard positions
// through depth 5+ has a correct move generator. One off-by-one anywhere
// (en passant, promotion undo, castling rights update) will diverge.
uint64_t perft(Position& pos, int depth);

// Run the standard perft suite; print pass/fail per position and depth.
// Returns true only if every checked value matched.
bool run_perft_suite(int max_depth);
