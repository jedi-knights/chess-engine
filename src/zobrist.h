#pragma once
#include "position.h"
#include "types.h"

#include <cstdint>

// Zobrist hashing: assign a random 64-bit key to every (piece, color, square)
// combination plus a few auxiliary keys, so a Position's key is the XOR of
// all its state-defining features. Because XOR is its own inverse, keys
// update incrementally: XOR out the removed feature, XOR in the added one.
// A transposition (same position reached via different move orders) then
// produces the same key and can share a search result.
namespace zobrist {

// Populate the random tables. Must run once at startup, before any
// Position uses set_from_fen (which calls compute()). Deterministic —
// no OS randomness — so keys are reproducible across runs and machines.
void init();

// Recompute a position's Zobrist key from scratch. Called by
// set_from_fen; used to verify incremental-update correctness in tests.
uint64_t compute(const Position& pos);

// Keys exposed for incremental update in make_move / unmake_move. Anything
// that modifies Position state must XOR the corresponding key(s) into the
// position's `key` field so it stays in sync with what compute() would produce.
extern uint64_t PIECE_SQ  [NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES];
extern uint64_t CASTLING  [16];    // indexed by castling rights bitmask
extern uint64_t EP_FILE   [8];     // XOR only when ep_is_capturable(pos)
extern uint64_t SIDE;              // XOR when it becomes black's turn

// True iff an en-passant capture is pseudo-legally available in `pos`:
// side_to_move has a pawn on one of the two squares diagonally adjacent
// to ep_square. Guards the EP_FILE XOR — positions with a "phantom" ep
// flag (double-push landing next to no enemy pawn) must not hash it, or
// they'd collide differently than an otherwise-identical position where
// the ep flag was never set.
bool ep_is_capturable(const Position& pos);

}  // namespace zobrist
