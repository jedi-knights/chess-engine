#pragma once
#include "position.h"
#include "types.h"

// Static evaluation from the side-to-move's perspective — positive scores
// mean the position is better for whoever is about to move.
//
// Material + PST is O(1) via Position::psq_mg / psq_eg (incremental).
// Mobility and pawn structure are computed per-call (small — pawn
// structure is pawn-hash cached; mobility is the dominant cost).
//
// Lazy eval: when `alpha` and `beta` bracket a window and the cheap
// material + PST + phase score is already OUTSIDE that window by more
// than EVAL_LAZY_MARGIN (500 cp), skip the expensive terms and return
// the lazy score early. Callers pass their alpha-beta window; tests use
// the wide default (equivalent to no lazy pruning) so pinned values
// stay stable.
constexpr int EVAL_UNBOUNDED = 1'000'000;
int evaluate(const Position& pos,
             int alpha = -EVAL_UNBOUNDED,
             int beta  =  EVAL_UNBOUNDED);

// Populate per-square passed-pawn masks. Must be called once at startup
// before any evaluate() call.
namespace eval { void init(); }

// Tunable weights exposed for coordinate-descent tuning. Non-PSQ terms
// (mobility, pawn structure, king safety) can be tuned without any
// Position invalidation. PSQ terms (piece_values[]) feed the
// incremental psq_mg / psq_eg accumulators — callers that mutate them
// must call Position::recompute_psq() on every position whose eval
// they care about, or the accumulators go stale.
//
// Default values match the previous constexpr literals so eval
// semantics are unchanged out-of-the-box; the tuner mutates them at
// runtime and calls recompute_psq() when needed.
namespace eval {
struct TuningParams {
    // PSQ-affecting: mutating requires Position::recompute_psq(). Index
    // 0 (NO_PIECE_TYPE) and 6 (KING) stay 0 — the king has no material
    // value (losing it means the game is already lost), and no-piece is
    // a sentinel. Kaufman classical values elsewhere.
    int piece_values[NUM_PIECE_TYPES] = { 0, 100, 320, 330, 500, 900, 0 };

    // Non-PSQ terms.
    int isolated_mg                 = -15;
    int isolated_eg                 = -20;
    int doubled_mg                  = -10;
    int doubled_eg                  = -20;
    int bishop_pair_mg              =  30;
    int bishop_pair_eg              =  50;
    int mob_knight                  =   4;
    int mob_bishop                  =   3;
    int mob_rook                    =   2;
    int mob_queen                   =   1;
    int shield_missing_penalty      =  12;
    int king_open_file_penalty      =  30;
    int king_semi_open_file_penalty =  15;
};
extern TuningParams params;
}  // namespace eval

// Piece-square + material tables exposed so Position can maintain
// psq_mg / psq_eg incrementally. Not intended for other consumers.
namespace eval {

// MG and EG PSTs, split per piece type. Michniewski originally used
// identical MG/EG for P/N/B/R/Q (only KING differed); the split lets
// the SPSA tuner discover phase-specific placement values. Mutable so
// the tuner can adjust — callers must invoke Position::recompute_psq()
// on any position whose eval they consume after a change.
extern int pst_mg[NUM_PIECE_TYPES][NUM_SQUARES];
extern int pst_eg[NUM_PIECE_TYPES][NUM_SQUARES];

// Combined material + PST for one piece at one square, from `c`'s
// perspective (black's tables are the vertical mirror of white's).
// Inline so the make/unmake hot path pays no function-call cost.
inline int psq_mg(Color c, PieceType pt, Square sq) {
    Square lookup = (c == WHITE) ? sq : Square(int(sq) ^ 56);
    return params.piece_values[pt] + pst_mg[pt][lookup];
}
inline int psq_eg(Color c, PieceType pt, Square sq) {
    Square lookup = (c == WHITE) ? sq : Square(int(sq) ^ 56);
    return params.piece_values[pt] + pst_eg[pt][lookup];
}

}  // namespace eval
