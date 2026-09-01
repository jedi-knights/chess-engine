#pragma once
#include "position.h"
#include "types.h"

// Static evaluation from the side-to-move's perspective — positive scores
// mean the position is better for whoever is about to move.
//
// The heavy lifting (material sum + PST accumulation) is done incrementally
// by Position::put_piece / remove_piece so `evaluate` itself is O(1): read
// the four cached sums, subtract, phase-interpolate, sign-flip.
int evaluate(const Position& pos);

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
