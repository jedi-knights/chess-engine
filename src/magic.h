#pragma once
#include "types.h"

// Magic bitboards for slider attack generation. A magic bitboard hashes
// the "relevant blocker" bits of a slider's ray into a small table index
// via one multiply and one shift, replacing gen_ray's per-square walk
// with a single memory load in the hot search loop.
//
// The multiplier ("magic number") is chosen such that every possible
// blocker pattern for the piece on that square hashes to a distinct
// index — or to an index that shares an attack set with another
// pattern, which is acceptable ("constructive collision"). Magics are
// discovered at init time by seeded random search, so the code stays
// portable and no huge magic-number constant table lives in the source.
//
// After init_magic() has run, bishop/rook/queen attack sets are:
//
//   Bitboard atk = rook_attacks(sq, occupancy);
//   // 3 loads + 1 multiply + 1 shift ≈ few nanoseconds per lookup
//
// Move generation gets these attack sets and masks out own-piece squares.
void init_magic();

// Return the set of squares a bishop/rook/queen on `s` attacks given the
// full-board occupancy `occ`. Occupancy MUST include the slider itself;
// it doesn't matter which color owns each blocker (the returned bitboard
// includes the blocker squares themselves — caller masks out own pieces).
Bitboard bishop_attacks(Square s, Bitboard occ);
Bitboard rook_attacks  (Square s, Bitboard occ);

inline Bitboard queen_attacks(Square s, Bitboard occ) {
    return bishop_attacks(s, occ) | rook_attacks(s, occ);
}
