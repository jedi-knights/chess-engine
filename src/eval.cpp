#include "eval.h"
#include "bitboard.h"
#include "types.h"

// Larry Kaufman's classic centipawn values. King has no material value —
// it's always on the board and losing it means the game is already over.
static constexpr int PIECE_VALUE[NUM_PIECE_TYPES] = {
    0,     // NO_PIECE_TYPE
    100,   // PAWN
    320,   // KNIGHT
    330,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
    0,     // KING
};

int evaluate(const Position& pos) {
    int white = 0, black = 0;
    for (int pt = PAWN; pt <= QUEEN; ++pt) {
        white += PIECE_VALUE[pt] * popcount(pos.pieces[WHITE][pt]);
        black += PIECE_VALUE[pt] * popcount(pos.pieces[BLACK][pt]);
    }
    int score = white - black;
    return (pos.side_to_move == WHITE) ? score : -score;
}
