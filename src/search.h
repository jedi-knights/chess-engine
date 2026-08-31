#pragma once
#include "position.h"
#include "types.h"
#include <cstdint>

struct SearchResult {
    Move     best_move = NULL_MOVE;   // NULL_MOVE only if no legal moves exist
    int      score     = 0;           // centipawns from side_to_move perspective
    int      depth     = 0;           // requested search depth
    uint64_t nodes     = 0;           // total nodes visited (including root)
};

// Fixed-depth alpha-beta negamax search. Returns the best move and its
// score from `pos.side_to_move`'s perspective. Mutates `pos` internally
// via make/unmake but restores the original state before returning.
SearchResult search_best(Position& pos, int depth);
