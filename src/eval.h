#pragma once
#include "position.h"
#include "types.h"

// Static evaluation from the side-to-move's perspective — positive scores
// mean the position is better for whoever is about to move.
//
// Material + PST is O(1) via Position::psq_mg / psq_eg (incremental).
// Mobility and pawn structure are computed per-call (small — pawn
// structure is pawn-hash cached; mobility is the dominant cost).
//
// Lazy eval: when `alpha` and `beta` bracket a window and the cheap
// material + PST + phase score is already OUTSIDE that window by more
// than EVAL_LAZY_MARGIN (500 cp), skip the expensive terms and return
// the lazy score early. Callers pass their alpha-beta window; tests use
// the wide default (equivalent to no lazy pruning) so pinned values
// stay stable.
constexpr int EVAL_UNBOUNDED = 1'000'000;
int evaluate(const Position& pos,
             int alpha = -EVAL_UNBOUNDED,
             int beta  =  EVAL_UNBOUNDED);

// Populate per-square passed-pawn masks. Must be called once at startup
// before any evaluate() call.
namespace eval { void init(); }

// Piece-square + material tables exposed so Position can maintain
// psq_mg / psq_eg incrementally. Not intended for other consumers.
namespace eval {

extern const int PIECE_VALUE[NUM_PIECE_TYPES];
extern const int* const PST_MG_TABLE[NUM_PIECE_TYPES];
extern const int* const PST_EG_TABLE[NUM_PIECE_TYPES];

// Combined material + PST for one piece at one square, from `c`'s
// perspective (black's tables are the vertical mirror of white's).
// Inline so the make/unmake hot path pays no function-call cost.
inline int psq_mg(Color c, PieceType pt, Square sq) {
    Square lookup = (c == WHITE) ? sq : Square(int(sq) ^ 56);
    return PIECE_VALUE[pt] + PST_MG_TABLE[pt][lookup];
}
inline int psq_eg(Color c, PieceType pt, Square sq) {
    Square lookup = (c == WHITE) ? sq : Square(int(sq) ^ 56);
    return PIECE_VALUE[pt] + PST_EG_TABLE[pt][lookup];
}

}  // namespace eval
