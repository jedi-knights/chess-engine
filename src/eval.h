#pragma once
#include "position.h"

// Static evaluation from the side-to-move's perspective — positive scores
// mean the position is better for whoever is about to move. Returning from
// this perspective (rather than absolute white-good/black-bad) is what
// makes negamax work uniformly at every ply.
//
// Currently pure material (centipawns). Positional terms (piece-square
// tables, pawn structure, mobility, king safety) are deliberately absent
// so the initial search is easy to reason about.
int evaluate(const Position& pos);
