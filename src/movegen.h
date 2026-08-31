#pragma once
#include "types.h"
#include "position.h"
#include <vector>

// Move generation. Currently a stub — you implement this.
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
void generate_moves(const Position& pos, std::vector<Move>& moves);
