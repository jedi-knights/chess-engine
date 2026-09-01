#pragma once
#include "position.h"

#include <cstdint>
#include <iosfwd>

// Count leaf nodes at exactly `depth` from `pos`. Correctness gate for
// move generation: an engine whose perft matches all 6 standard positions
// through depth 5+ has a correct move generator. One off-by-one anywhere
// (en passant, promotion undo, castling rights update) will diverge.
uint64_t perft(Position& pos, int depth);

// Run the standard perft suite; write pass/fail per position and depth
// to `out`. Returns true only if every checked value matched. Taking
// an ostream (rather than reaching for std::cout) keeps the driver
// testable via std::ostringstream — no fd redirection dance needed.
bool run_perft_suite(int max_depth, std::ostream& out);
