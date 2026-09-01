#include "eval.h"
#include "attacks.h"
#include "bitboard.h"
#include "magic.h"
#include "types.h"

#include <algorithm>

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

// --- Passed pawns -------------------------------------------------------
// For each (color, square): bitboard of squares on the same file or an
// adjacent file, on ranks IN FRONT of the pawn (from that color's
// perspective). If no enemy pawns intersect, our pawn is passed.
static Bitboard PASSED_PAWN_MASK[NUM_COLORS][NUM_SQUARES];

// Bonus per rank a passed pawn has advanced. Indexed by "how far from
// starting rank" — rank 2 = 0 advance, rank 7 = 5 advances. MG values
// are conservative (a passed pawn is nice but the middlegame is about
// pieces); EG values are large (a passed pawn IS the endgame).
static constexpr int PASSED_MG[8] = { 0,  5, 10, 20, 35, 60, 90, 0 };
static constexpr int PASSED_EG[8] = { 0, 10, 25, 45, 75, 120, 200, 0 };

namespace eval {
void init() {
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int sq = 0; sq < NUM_SQUARES; ++sq) {
            Bitboard mask = 0;
            int f = file_of(Square(sq));
            int r = rank_of(Square(sq));
            for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ++ff) {
                if (c == WHITE) {
                    for (int rr = r + 1; rr <= 7; ++rr)
                        mask |= square_bb(make_square(File(ff), Rank(rr)));
                } else {
                    for (int rr = r - 1; rr >= 0; --rr)
                        mask |= square_bb(make_square(File(ff), Rank(rr)));
                }
            }
            PASSED_PAWN_MASK[c][sq] = mask;
        }
    }
}
}  // namespace eval

// Sum passed-pawn bonuses for `us`. Returns a pair of (mg_bonus, eg_bonus)
// packed into two accumulator refs — cheaper than one call per phase.
static void passed_pawn_bonus(const Position& pos, Color us,
                              int& mg_out, int& eg_out) {
    int mg = 0, eg = 0;
    const Bitboard enemy_pawns = pos.pieces[Color(us ^ 1)][PAWN];
    Bitboard our_pawns = pos.pieces[us][PAWN];
    while (our_pawns) {
        Square s = pop_lsb(our_pawns);
        if ((PASSED_PAWN_MASK[us][s] & enemy_pawns) == 0) {
            // Advance rank from the pawn's starting side.
            int adv = (us == WHITE) ? (rank_of(s) - 1) : (6 - rank_of(s));
            if (adv >= 0 && adv < 8) {
                mg += PASSED_MG[adv];
                eg += PASSED_EG[adv];
            }
        }
    }
    mg_out += mg;
    eg_out += eg;
}

// --- Bishop pair --------------------------------------------------------
// Two bishops are worth more than the sum of the parts — they cover both
// color complexes together, so tactical possibilities compound. Standard
// engine bonus is ~30 cp MG, ~50 cp EG (endgame value is higher because
// open positions give bishops more scope). Guards on "at least two"
// rather than "exactly two" so promotions to bishop still count, though
// two same-color bishops after underpromotion is a rare degenerate case.
constexpr int BISHOP_PAIR_MG = 30;
constexpr int BISHOP_PAIR_EG = 50;

// --- Mobility -----------------------------------------------------------
// Cheap and effective: count squares each piece can move to, minus own
// pieces AND enemy pawn attack squares ("safe mobility"). Excluding
// pawn-attacked squares is standard — a knight on a square a pawn can
// hit is not really mobile there, it'll get traded off. Different
// weights per piece type reflect diminishing returns (queens usually
// have plenty of mobility regardless).
static constexpr int MOB_KNIGHT = 4;
static constexpr int MOB_BISHOP = 3;
static constexpr int MOB_ROOK   = 2;
static constexpr int MOB_QUEEN  = 1;

// Local file masks — same as movegen's file-local constants, duplicated
// here to keep bitboard.h lean. Only pawn attack computation needs them.
static constexpr Bitboard EVAL_FILE_A = 0x0101010101010101ULL;
static constexpr Bitboard EVAL_FILE_H = 0x8080808080808080ULL;

// All squares attacked by `us` pawns (as a shifted bitboard, no per-pawn
// loop needed). Kept local — only mobility uses it.
static Bitboard pawn_attacks_of(Color us, Bitboard pawns) {
    if (us == WHITE) {
        return ((pawns & ~EVAL_FILE_A) << 7) | ((pawns & ~EVAL_FILE_H) << 9);
    } else {
        return ((pawns & ~EVAL_FILE_H) >> 7) | ((pawns & ~EVAL_FILE_A) >> 9);
    }
}

static int mobility(const Position& pos, Color us) {
    const Bitboard occ  = pos.occupied;
    const Bitboard our  = pos.colors[us];
    const Bitboard their_pawn_atk =
        pawn_attacks_of(Color(us ^ 1), pos.pieces[Color(us ^ 1)][PAWN]);
    const Bitboard mask = ~our & ~their_pawn_atk;
    int score = 0;

    Bitboard b = pos.pieces[us][KNIGHT];
    while (b) { Square s = pop_lsb(b);
                score += MOB_KNIGHT * popcount(KNIGHT_ATTACKS[s] & mask); }
    b = pos.pieces[us][BISHOP];
    while (b) { Square s = pop_lsb(b);
                score += MOB_BISHOP * popcount(bishop_attacks(s, occ) & mask); }
    b = pos.pieces[us][ROOK];
    while (b) { Square s = pop_lsb(b);
                score += MOB_ROOK   * popcount(rook_attacks(s, occ)   & mask); }
    b = pos.pieces[us][QUEEN];
    while (b) { Square s = pop_lsb(b);
                score += MOB_QUEEN  * popcount(
                    (bishop_attacks(s, occ) | rook_attacks(s, occ)) & mask); }

    return score;
}

int evaluate(const Position& pos) {
    int mg_diff = pos.psq_mg[WHITE] - pos.psq_mg[BLACK];
    int eg_diff = pos.psq_eg[WHITE] - pos.psq_eg[BLACK];

    // Mobility — favor active pieces. Applied at half weight in the
    // endgame because open positions dominate mobility numbers there
    // and can drown out material.
    int mob_diff = mobility(pos, WHITE) - mobility(pos, BLACK);
    mg_diff += mob_diff;
    eg_diff += mob_diff / 2;

    // Passed pawns — separate MG and EG tables since a passed pawn is
    // a mild bonus in the middlegame and often decisive in the endgame.
    int passed_w_mg = 0, passed_w_eg = 0;
    int passed_b_mg = 0, passed_b_eg = 0;
    passed_pawn_bonus(pos, WHITE, passed_w_mg, passed_w_eg);
    passed_pawn_bonus(pos, BLACK, passed_b_mg, passed_b_eg);
    mg_diff += passed_w_mg - passed_b_mg;
    eg_diff += passed_w_eg - passed_b_eg;

    // Bishop pair — flat bonus per side with two or more bishops.
    if (popcount(pos.pieces[WHITE][BISHOP]) >= 2) {
        mg_diff += BISHOP_PAIR_MG;
        eg_diff += BISHOP_PAIR_EG;
    }
    if (popcount(pos.pieces[BLACK][BISHOP]) >= 2) {
        mg_diff -= BISHOP_PAIR_MG;
        eg_diff -= BISHOP_PAIR_EG;
    }

    int phase = compute_phase(pos);
    int score = (mg_diff * phase + eg_diff * (PHASE_MAX - phase)) / PHASE_MAX;
    return (pos.side_to_move == WHITE) ? score : -score;
}
