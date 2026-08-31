#pragma once
#include "types.h"
#include "position.h"
#include <vector>

// Produces the full set of LEGAL moves for the side to move — pseudo-legal
// generation followed by an is_legal filter that rejects any move leaving
// the mover's king in check. Takes Position by non-const reference because
// the filter uses make/unmake internally; the position is restored before
// return, so it's logically const from the caller's perspective.
//
// Milestones (validate each against the perft suite):
//   1. Add Position::make_move / unmake_move with an undo record
//   2. Knight moves      (uses KNIGHT_ATTACKS)
//   3. King moves        (uses KING_ATTACKS)
//   4. Pawn moves        (pushes, double-push, captures, en passant, promotions)
//   5. Sliding pieces    (naive ray attacks first, then magic bitboards)
//   6. Castling          (all four squares-not-attacked / not-in-check rules)
//   7. Legality filter   (king not left in check)
//
// Reference: https://www.chessprogramming.org/Move_Generation
void generate_moves(Position& pos, std::vector<Move>& moves);
