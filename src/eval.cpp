#include "eval.h"
#include "bitboard.h"
#include "types.h"

namespace eval {

// Larry Kaufman's classic centipawn values. King has no material value —
// it's always on the board and losing it means the game is already over.
const int PIECE_VALUE[NUM_PIECE_TYPES] = {
    0,     // NO_PIECE_TYPE
    100,   // PAWN
    320,   // KNIGHT
    330,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
    0,     // KING
};

// Piece-square tables from the Chess Programming Wiki's "Simplified
// Evaluation Function" (Michniewski). Values are white-perspective and
// indexed by Square (A1=0, H1=7, A8=56, H8=63). Black looks up its own
// PST value by XOR'ing the square with 56 (rank flip).

static const int PST_PAWN[NUM_SQUARES] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
};

static const int PST_KNIGHT[NUM_SQUARES] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
};

static const int PST_BISHOP[NUM_SQUARES] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
};

static const int PST_ROOK[NUM_SQUARES] = {
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
};

static const int PST_QUEEN[NUM_SQUARES] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
};

// King has two tables — middlegame (safety) and endgame (activity) —
// blended by tapered eval. The middlegame rewards castled positions and
// penalizes advance; the endgame inverts, favoring centralization.

static const int PST_KING_MG[NUM_SQUARES] = {
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
};

static const int PST_KING_EG[NUM_SQUARES] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
};

// Per-piece-type PST pointers. Only KING differs between MG and EG.
const int* const PST_MG_TABLE[NUM_PIECE_TYPES] = {
    nullptr, PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_MG,
};
const int* const PST_EG_TABLE[NUM_PIECE_TYPES] = {
    nullptr, PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_EG,
};

}  // namespace eval

// Phase weights from Fruit / Stockfish. Startpos non-pawn material = 24
// (Q=4, 2R=4, 2B=2, 2N=2 per side). Fewer pieces → smaller phase →
// more endgame weighting. Pawns don't count — the transition is driven
// by "big pieces" coming off.
constexpr int PHASE_KNIGHT = 1;
constexpr int PHASE_BISHOP = 1;
constexpr int PHASE_ROOK   = 2;
constexpr int PHASE_QUEEN  = 4;
constexpr int PHASE_MAX    = 24;

static int compute_phase(const Position& pos) {
    int phase = 0;
    for (int c = 0; c < NUM_COLORS; ++c) {
        phase += PHASE_KNIGHT * popcount(pos.pieces[c][KNIGHT]);
        phase += PHASE_BISHOP * popcount(pos.pieces[c][BISHOP]);
        phase += PHASE_ROOK   * popcount(pos.pieces[c][ROOK]);
        phase += PHASE_QUEEN  * popcount(pos.pieces[c][QUEEN]);
    }
    // Cap defends against post-promotion positions with more than the
    // starting material. Prevents over-weighting MG.
    return phase > PHASE_MAX ? PHASE_MAX : phase;
}

int evaluate(const Position& pos) {
    // Position maintains psq_mg / psq_eg incrementally via put_piece /
    // remove_piece — the loop over pieces this used to do is gone.
    int mg_diff = pos.psq_mg[WHITE] - pos.psq_mg[BLACK];
    int eg_diff = pos.psq_eg[WHITE] - pos.psq_eg[BLACK];
    int phase   = compute_phase(pos);
    int score   = (mg_diff * phase + eg_diff * (PHASE_MAX - phase)) / PHASE_MAX;
    return (pos.side_to_move == WHITE) ? score : -score;
}
