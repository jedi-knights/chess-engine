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

// King has TWO tables — middlegame (safety) and endgame (activity) —
// blended by tapered eval below. The middlegame table rewards castled
// positions and penalizes king advance; the endgame table inverts this
// so the king centralizes and helps push pawns / mate the opponent king.

static constexpr int PST_KING_MG[NUM_SQUARES] = {
     20, 30, 10,  0,  0, 10, 30, 20,   // rank 1: castled positions best
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
};

static constexpr int PST_KING_EG[NUM_SQUARES] = {
    -50,-40,-30,-20,-20,-30,-40,-50,   // corners bad in the endgame
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,   // center squares peak
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
};

// Only the king has a phase-dependent table for now — the biggest
// tapered-eval win is fixing endgame king activity. MG/EG pawn tables
// (encourage advance in the endgame) are a natural next step; other
// pieces show smaller phase-dependence and are commonly identical.
static constexpr const int* PST[NUM_PIECE_TYPES] = {
    nullptr,          // NO_PIECE_TYPE — never indexed
    PST_PAWN,
    PST_KNIGHT,
    PST_BISHOP,
    PST_ROOK,
    PST_QUEEN,
    PST_KING_MG,      // KING placeholder — psq_bonus branches on phase for the king
};

// Phase weights from Fruit / Stockfish. Startpos has 24 non-pawn phase
// units per side pair (Q=4, 2R=4, 2B=2, 2N=2 = 12 per side * 2 sides / 2
// ... actually let me spell it out): both sides start with 1 queen, 2
// rooks, 2 bishops, 2 knights. That's 4+4+2+2 = 12 per side, 24 total.
// Fewer pieces on the board → smaller phase → more endgame weighting.
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
    // starting material (e.g., 2+ queens per side after under-defended
    // promotion runs). Prevents the interpolation from over-weighting MG.
    return phase > PHASE_MAX ? PHASE_MAX : phase;
}

// Return the phase-interpolated PST bonus for a piece at a square. Only
// the king has separate MG/EG tables; every other piece falls through
// to a single-table lookup.
static inline int psq_bonus(Color c, PieceType pt, Square sq, int phase) {
    Square lookup = (c == WHITE) ? sq : Square(int(sq) ^ 56);
    if (pt == KING) {
        int mg = PST_KING_MG[lookup];
        int eg = PST_KING_EG[lookup];
        // Linear interpolation: full phase → all MG; zero phase → all EG.
        return (mg * phase + eg * (PHASE_MAX - phase)) / PHASE_MAX;
    }
    return PST[pt][lookup];
}

// Absolute score for one side (positive = good for that side).
static int side_score(Color c, const Position& pos, int phase) {
    int total = 0;
    for (int pt = PAWN; pt <= KING; ++pt) {
        Bitboard bb = pos.pieces[c][pt];
        total += PIECE_VALUE[pt] * popcount(bb);
        while (bb) {
            Square s = pop_lsb(bb);
            total += psq_bonus(c, PieceType(pt), s, phase);
        }
    }
    return total;
}

int evaluate(const Position& pos) {
    int phase = compute_phase(pos);
    int score = side_score(WHITE, pos, phase) - side_score(BLACK, pos, phase);
    return (pos.side_to_move == WHITE) ? score : -score;
}
