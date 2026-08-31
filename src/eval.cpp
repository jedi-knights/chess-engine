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

// Piece-square tables from the Chess Programming Wiki's "Simplified
// Evaluation Function" (Michniewski). Values are white-perspective and
// indexed by Square (A1=0, H1=7, A8=56, H8=63) — the visible row order
// matches board layout when the array is read as an 8x8 grid.
//
// Black looks up its own PST value by XOR'ing the square with 56, which
// flips only the rank bits (a1↔a8, e2↔e7, etc.). See psq_bonus below.
//
// Only middlegame values are used — an endgame king ideally centralizes,
// but tapered eval (interpolating MG/EG by remaining material) is a
// separate concern. Named in CLAUDE.md as future work.

static constexpr int PST_PAWN[NUM_SQUARES] = {
     0,  0,  0,  0,  0,  0,  0,  0,   // rank 1: no pawns land here
     5, 10, 10,-20,-20, 10, 10,  5,   // rank 2: encourages central push
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,   // rank 7: about to promote
     0,  0,  0,  0,  0,  0,  0,  0,   // rank 8: promoted, no pawns
};

static constexpr int PST_KNIGHT[NUM_SQUARES] = {
    -50,-40,-30,-30,-30,-30,-40,-50,   // "a knight on the rim is dim"
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
};

static constexpr int PST_BISHOP[NUM_SQUARES] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,   // fianchetto squares (b2/g2) get +5
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
};

static constexpr int PST_ROOK[NUM_SQUARES] = {
     0,  0,  0,  5,  5,  0,  0,  0,   // rank 1: mild bonus on d/e files
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,   // rank 7: classical "rook on the 7th"
     0,  0,  0,  0,  0,  0,  0,  0,
};

static constexpr int PST_QUEEN[NUM_SQUARES] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
};

// Middlegame king safety: heavy penalty for advancing into the middle,
// bonus for castled positions. In pure endgames this pushes the king
// the wrong way — but endgames without a phase-aware interpolation are
// on the follow-up list, not this PR.
static constexpr int PST_KING[NUM_SQUARES] = {
     20, 30, 10,  0,  0, 10, 30, 20,   // rank 1: castled positions best
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
};

static constexpr const int* PST[NUM_PIECE_TYPES] = {
    nullptr,     // NO_PIECE_TYPE — never indexed
    PST_PAWN,
    PST_KNIGHT,
    PST_BISHOP,
    PST_ROOK,
    PST_QUEEN,
    PST_KING,
};

// Return the PST value for `pt` at `sq` from `c`'s perspective. Black's
// tables are the vertical mirror of white's — same intuition, opposite
// direction — obtained by XOR'ing the square with 56 (flips the rank).
static inline int psq_bonus(Color c, PieceType pt, Square sq) {
    Square lookup = (c == WHITE) ? sq : Square(int(sq) ^ 56);
    return PST[pt][lookup];
}

// Absolute score for one side (positive = good for that side).
static int side_score(Color c, const Position& pos) {
    int total = 0;
    for (int pt = PAWN; pt <= KING; ++pt) {
        Bitboard bb = pos.pieces[c][pt];
        total += PIECE_VALUE[pt] * popcount(bb);
        while (bb) {
            Square s = pop_lsb(bb);
            total += psq_bonus(c, PieceType(pt), s);
        }
    }
    return total;
}

int evaluate(const Position& pos) {
    int score = side_score(WHITE, pos) - side_score(BLACK, pos);
    return (pos.side_to_move == WHITE) ? score : -score;
}
