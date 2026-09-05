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

// --- Pawn structure ----------------------------------------------------
// Passed pawn masks: for each (color, square), the bitboard of enemy
// pawn squares that would prevent this pawn from being passed —
// same file or an adjacent file, on any rank IN FRONT of the pawn.
static Bitboard PASSED_PAWN_MASK[NUM_COLORS][NUM_SQUARES];

// Per-file bitboards for doubled-pawn counting.
static constexpr Bitboard PAWN_FILE_MASK[8] = {
    0x0101010101010101ULL, 0x0202020202020202ULL,
    0x0404040404040404ULL, 0x0808080808080808ULL,
    0x1010101010101010ULL, 0x2020202020202020ULL,
    0x4040404040404040ULL, 0x8080808080808080ULL,
};

// For each file, the OR of its two adjacent files' masks (or one, at the
// A/H edges). An isolated pawn is one whose adjacent-file mask contains
// no friendly pawns.
static constexpr Bitboard ADJACENT_FILE_MASK[8] = {
    PAWN_FILE_MASK[1],
    PAWN_FILE_MASK[0] | PAWN_FILE_MASK[2],
    PAWN_FILE_MASK[1] | PAWN_FILE_MASK[3],
    PAWN_FILE_MASK[2] | PAWN_FILE_MASK[4],
    PAWN_FILE_MASK[3] | PAWN_FILE_MASK[5],
    PAWN_FILE_MASK[4] | PAWN_FILE_MASK[6],
    PAWN_FILE_MASK[5] | PAWN_FILE_MASK[7],
    PAWN_FILE_MASK[6],
};

// Bonus per rank a passed pawn has advanced. Indexed by "how far from
// starting rank" — rank 2 = 0 advance, rank 7 = 5 advances. MG values
// are conservative (a passed pawn is nice but the middlegame is about
// pieces); EG values are large (a passed pawn IS the endgame).
static constexpr int PASSED_MG[8] = { 0,  5, 10, 20, 35, 60, 90, 0 };
static constexpr int PASSED_EG[8] = { 0, 10, 25, 45, 75, 120, 200, 0 };

// Isolated pawn penalty: no friendly pawn on adjacent files. Standard
// starter values from the Chess Programming Wiki — isolated pawns are
// harder to defend and easier to blockade, worse in the endgame where
// the piece cover thins out.
static constexpr int ISOLATED_MG = -15;
static constexpr int ISOLATED_EG = -20;

// Doubled pawn penalty: applied per EXTRA pawn on a file (two pawns
// stacked → one penalty, three → two, etc.). Doubled pawns block each
// other and can't defend each other diagonally.
static constexpr int DOUBLED_MG = -10;
static constexpr int DOUBLED_EG = -20;

namespace eval {
void init() {
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int sq = 0; sq < NUM_SQUARES; ++sq) {
            Bitboard mask = 0;
            int f = file_of(Square(sq));
            int r = rank_of(Square(sq));
            for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ++ff) {
                if (c == WHITE) {
                    for (int rr = r + 1; rr <= 7; ++rr) {
                        mask |= square_bb(make_square(File(ff), Rank(rr)));
                    }
                } else {
                    for (int rr = r - 1; rr >= 0; --rr) {
                        mask |= square_bb(make_square(File(ff), Rank(rr)));
                    }
                }
            }
            PASSED_PAWN_MASK[c][sq] = mask;
        }
    }
}
}  // namespace eval

// Sum passed + isolated + doubled contributions for `us` into the
// accumulator refs. One pass over the pawns handles per-pawn terms
// (passed, isolated); a per-file loop handles doubled counting.
static void pawn_structure_side(const Position& pos, Color us,
                                int& mg_out, int& eg_out) {
    int mg = 0;
    int eg = 0;
    const Bitboard our_pawns   = pos.pieces[us][PAWN];
    const Bitboard enemy_pawns = pos.pieces[Color(us ^ 1)][PAWN];

    for (int f = 0; f < 8; ++f) {
        int count = popcount(PAWN_FILE_MASK[f] & our_pawns);
        if (count > 1) {
            mg += DOUBLED_MG * (count - 1);
            eg += DOUBLED_EG * (count - 1);
        }
    }

    Bitboard b = our_pawns;
    while (b != 0U) {
        Square s = pop_lsb(b);
        if ((ADJACENT_FILE_MASK[file_of(s)] & our_pawns) == 0) {
            mg += ISOLATED_MG;
            eg += ISOLATED_EG;
        }
        if ((PASSED_PAWN_MASK[us][s] & enemy_pawns) == 0) {
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

// --- Pawn hash table ----------------------------------------------------
// Cache the (mg, eg) diff (WHITE - BLACK) of the pawn structure eval,
// keyed on Position::pawn_key. Direct-mapped, always-replace. Collisions
// are resolved by comparing the full 64-bit key — a mismatch triggers a
// recompute rather than trusting a stale entry. Small enough to sit in
// L2 (16k * 16 bytes = 256KB). No clearing between games: pawn-structure
// eval is a pure function of pawn positions, so entries stay valid
// across `ucinewgame` boundaries.
constexpr int    PAWN_HASH_BITS = 14;
constexpr size_t PAWN_HASH_SIZE = 1UL << PAWN_HASH_BITS;

struct PawnEntry {
    uint64_t key;
    int16_t  mg;
    int16_t  eg;
};

static PawnEntry pawn_hash[PAWN_HASH_SIZE];

// Populate `mg_diff` / `eg_diff` with the WHITE - BLACK pawn structure
// score, either from the hash or by computing and storing it.
static void pawn_structure_eval(const Position& pos,
                                int& mg_diff, int& eg_diff) {
    PawnEntry& slot = pawn_hash[pos.pawn_key & (PAWN_HASH_SIZE - 1)];
    if (slot.key == pos.pawn_key) {
        mg_diff += slot.mg;
        eg_diff += slot.eg;
        return;
    }
    int mg_w = 0;
    int eg_w = 0;
    int mg_b = 0;
    int eg_b = 0;
    pawn_structure_side(pos, WHITE, mg_w, eg_w);
    pawn_structure_side(pos, BLACK, mg_b, eg_b);
    int mg = mg_w - mg_b;
    int eg = eg_w - eg_b;
    slot.key = pos.pawn_key;
    slot.mg  = static_cast<int16_t>(mg);
    slot.eg  = static_cast<int16_t>(eg);
    mg_diff += mg;
    eg_diff += eg;
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
    }
    return ((pawns & ~EVAL_FILE_H) >> 7) | ((pawns & ~EVAL_FILE_A) >> 9);
}

static int mobility(const Position& pos, Color us) {
    const Bitboard occ  = pos.occupied;
    const Bitboard our  = pos.colors[us];
    const Bitboard their_pawn_atk =
        pawn_attacks_of(Color(us ^ 1), pos.pieces[Color(us ^ 1)][PAWN]);
    const Bitboard mask = ~our & ~their_pawn_atk;
    int score = 0;

    Bitboard b = pos.pieces[us][KNIGHT];
    while (b != 0U) { Square s = pop_lsb(b);
                score += MOB_KNIGHT * popcount(KNIGHT_ATTACKS[s] & mask); }
    b = pos.pieces[us][BISHOP];
    while (b != 0U) { Square s = pop_lsb(b);
                score += MOB_BISHOP * popcount(bishop_attacks(s, occ) & mask); }
    b = pos.pieces[us][ROOK];
    while (b != 0U) { Square s = pop_lsb(b);
                score += MOB_ROOK   * popcount(rook_attacks(s, occ)   & mask); }
    b = pos.pieces[us][QUEEN];
    while (b != 0U) { Square s = pop_lsb(b);
                score += MOB_QUEEN  * popcount(
                    (bishop_attacks(s, occ) | rook_attacks(s, occ)) & mask); }

    return score;
}

// --- King safety --------------------------------------------------------
// Rebel-style king danger: for each enemy attacker (N/B/R/Q) count the
// number of squares in our king ring it attacks, weight by piece type,
// sum, and look up the total in a saturating quadratic table. The ring
// is the 8 squares around the king (KING_ATTACKS[king_sq]); pawns and
// enemy king are excluded from the attacker set (pawns are cheap
// attackers whose threat is subsumed by pawn structure; kings can't
// legally sit adjacent to the enemy king). Applied to the middlegame
// term only — endgame play favors active kings, so a king in a "ring
// under attack" is already captured by tapered king PST.
//
// Weights chosen so a lone rook or minor grazing the ring gives a
// modest penalty (~5-15 cp) while a coordinated multi-attacker assault
// saturates at 500 cp — an entire queen's worth of pressure.
static constexpr int KING_ATTACK_WEIGHT[NUM_PIECE_TYPES] = {
    0,   // NO_PIECE_TYPE
    0,   // PAWN — excluded (see above)
    2,   // KNIGHT
    2,   // BISHOP
    3,   // ROOK
    5,   // QUEEN
    0,   // KING — excluded (see above)
};

// Saturating quadratic penalty in centipawns, indexed by attack units
// clamped to [0, 99]. Standard Rebel / Chess Programming Wiki curve —
// tuned across ~decades of real engines. Small unit counts give ~0-30
// cp; heavy attack (60+ units) saturates at 500 cp.
static constexpr int KING_SAFETY_TABLE[100] = {
      0,   0,   1,   2,   3,   5,   7,   9,  12,  15,
     18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
     68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
    140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
    260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
    377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
    494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
};

// Penalty in centipawns for `us`'s king being attacked. Returns 0 if
// the opponent has no queen — the queen is by far the dominant king
// attacker (weight 5), and without one the raw attack units cap out
// at ~15 for extreme cases (still only ~40 cp penalty). Skipping
// avoids the ~40-80 ns per node cost of the piece scans in queenless
// endgames, where king safety impact is minimal anyway.
static int king_safety_penalty_side(const Position& pos, Color us) {
    const Bitboard king_bb = pos.pieces[us][KING];
    if (king_bb == 0U) {
        return 0;
    }
    const Color them = Color(us ^ 1);
    if (pos.pieces[them][QUEEN] == 0U) {
        return 0;
    }
    const Square   king_sq   = lsb(king_bb);
    const Bitboard king_ring = KING_ATTACKS[king_sq];
    const Bitboard occ       = pos.occupied;

    int units = 0;

    Bitboard b = pos.pieces[them][KNIGHT];
    while (b != 0U) {
        Square s = pop_lsb(b);
        units += KING_ATTACK_WEIGHT[KNIGHT] *
                 popcount(KNIGHT_ATTACKS[s] & king_ring);
    }
    b = pos.pieces[them][BISHOP];
    while (b != 0U) {
        Square s = pop_lsb(b);
        units += KING_ATTACK_WEIGHT[BISHOP] *
                 popcount(bishop_attacks(s, occ) & king_ring);
    }
    b = pos.pieces[them][ROOK];
    while (b != 0U) {
        Square s = pop_lsb(b);
        units += KING_ATTACK_WEIGHT[ROOK] *
                 popcount(rook_attacks(s, occ) & king_ring);
    }
    b = pos.pieces[them][QUEEN];
    while (b != 0U) {
        Square s = pop_lsb(b);
        units += KING_ATTACK_WEIGHT[QUEEN] *
                 popcount((bishop_attacks(s, occ) | rook_attacks(s, occ))
                          & king_ring);
    }

    if (units > 99) {
        units = 99;
    }
    return KING_SAFETY_TABLE[units];
}

// --- Pawn shield --------------------------------------------------------
// Bonus/penalty for pawns directly in front of a castled king. Only
// meaningful when the king has actually castled (wing files A-C or F-H
// and on its home rank); a central-file or advanced king is scored by
// mobility / king PST instead. For each of the three files near the
// king (king_file - 1, king_file, king_file + 1), find the nearest
// friendly pawn moving toward the enemy: pawn one rank forward = full
// shield bonus, two ranks = partial, three = trace, none = missing
// penalty. Applied to the middlegame term only — an active endgame
// king shouldn't be trying to hide behind pawns.
//
// The shield term does NOT participate in the king safety attack
// arithmetic (weights, table) — pawn shield is a positional feature
// of the king's placement, independent of specific attackers. In a
// good real position both fire; they compose additively.
static constexpr int SHIELD_BONUS[3]         = { 15, 8, 2 };  // distance 1, 2, 3
static constexpr int SHIELD_MISSING_PENALTY  = 12;

static int pawn_shield_side(const Position& pos, Color us) {
    const Bitboard king_bb   = pos.pieces[us][KING];
    if (king_bb == 0U) {
        return 0;
    }
    const Bitboard our_pawns = pos.pieces[us][PAWN];
    // Pawnless positions (tactical test setups, endgames with all pawns
    // traded) aren't meaningfully described by a "shield" concept. Skip
    // to avoid pathological huge penalties on artificial positions.
    if (popcount(our_pawns) < 3) {
        return 0;
    }
    const Square king_sq   = lsb(king_bb);
    const int    king_file = file_of(king_sq);
    const int    king_rank = rank_of(king_sq);

    // Central-file king (D/E) has no meaningful shield concept — score
    // via king PST and mobility only. Off-home-rank king (advanced,
    // uncastled) similarly.
    if (king_file >= FILE_D && king_file <= FILE_E) {
        return 0;
    }
    const int home_rank = (us == WHITE) ? int(RANK_1) : int(RANK_8);
    if (king_rank != home_rank) {
        return 0;
    }

    int score = 0;
    for (int df = -1; df <= 1; ++df) {
        const int f = king_file + df;
        if (f < 0 || f > 7) {
            continue;
        }
        // Scan king's forward direction for the nearest own pawn on
        // this file. Distance = ranks between king and the pawn (1 =
        // right in front, 3 = still shielding but advanced).
        int distance = 0;
        for (int d = 1; d <= 3; ++d) {
            const int r = (us == WHITE) ? (home_rank + d) : (home_rank - d);
            const Square s = make_square(File(f), Rank(r));
            if ((our_pawns & square_bb(s)) != 0U) {
                distance = d;
                break;
            }
        }
        if (distance == 0) {
            score -= SHIELD_MISSING_PENALTY;
        } else {
            score += SHIELD_BONUS[distance - 1];
        }
    }
    return score;
}

// Bound on the total swing the non-lazy terms can contribute:
// mobility ~200, pawn structure ~200, bishop pair ~50, king safety
// caps at ±500, pawn shield caps at ±(3 * max(BONUS,MISSING)) = ±45.
// 1200 cp comfortably covers the combined worst case; lazy triggers
// only when material + PST alone is already unambiguously outside the
// alpha-beta window even after the remaining terms fire.
constexpr int EVAL_LAZY_MARGIN = 1200;

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — alpha/beta is standard evaluation-window naming, swapping would be caught by the assertion `alpha <= beta` at the top of the search loop.
int evaluate(const Position& pos, int alpha, int beta) {
    int mg_diff = pos.psq_mg[WHITE] - pos.psq_mg[BLACK];
    int eg_diff = pos.psq_eg[WHITE] - pos.psq_eg[BLACK];
    int phase   = compute_phase(pos);

    // Lazy eval: if material + PST alone is already so far outside the
    // window that even the maximum possible swing from the remaining
    // terms couldn't bring it back inside, return the lazy score. The
    // caller's fail-high / fail-low branches will fire on the returned
    // value exactly as they would on the full score.
    int lazy_score = ((mg_diff * phase) + (eg_diff * (PHASE_MAX - phase))) / PHASE_MAX;
    int lazy_persp = (pos.side_to_move == WHITE) ? lazy_score : -lazy_score;
    if (lazy_persp - EVAL_LAZY_MARGIN >= beta ||
        lazy_persp + EVAL_LAZY_MARGIN <= alpha) {
        return lazy_persp;
    }

    // Mobility — favor active pieces. Applied at half weight in the
    // endgame because open positions dominate mobility numbers there
    // and can drown out material.
    int mob_diff = mobility(pos, WHITE) - mobility(pos, BLACK);
    mg_diff += mob_diff;
    eg_diff += mob_diff / 2;

    // Pawn structure — passed / isolated / doubled, cached in the pawn
    // hash by pos.pawn_key so sibling positions with the same pawn
    // skeleton share the result.
    pawn_structure_eval(pos, mg_diff, eg_diff);

    // Bishop pair — flat bonus per side with two or more bishops.
    if (popcount(pos.pieces[WHITE][BISHOP]) >= 2) {
        mg_diff += BISHOP_PAIR_MG;
        eg_diff += BISHOP_PAIR_EG;
    }
    if (popcount(pos.pieces[BLACK][BISHOP]) >= 2) {
        mg_diff -= BISHOP_PAIR_MG;
        eg_diff -= BISHOP_PAIR_EG;
    }

    // King safety — penalty for each side's king under attack, applied
    // only to the middlegame term (endgame values active kings, which
    // the king PST already rewards; layering a separate safety term
    // on top would double-count). Diff is BLACK_penalty - WHITE_penalty:
    // a heavier attack on the black king means a higher BLACK penalty,
    // which favors WHITE, so it adds to the WHITE-BLACK diff. Each
    // side's penalty short-circuits to 0 if the opposing side has no
    // queen (see king_safety_penalty_side).
    mg_diff += king_safety_penalty_side(pos, BLACK) -
               king_safety_penalty_side(pos, WHITE);

    // Pawn shield — positive for a well-defended castled king. WHITE's
    // shield bonus is good for white → adds to the WHITE-BLACK diff.
    // BLACK's shield bonus is good for black → subtracts. Composes
    // additively with king safety (attack pressure vs. structural
    // shelter — different signals).
    mg_diff += pawn_shield_side(pos, WHITE) -
               pawn_shield_side(pos, BLACK);

    int score = ((mg_diff * phase) + (eg_diff * (PHASE_MAX - phase))) / PHASE_MAX;
    return (pos.side_to_move == WHITE) ? score : -score;
}
