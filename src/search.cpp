#include "search.h"
#include "attacks.h"
#include "bitboard.h"
#include "eval.h"
#include "magic.h"
#include "movegen.h"
#include "tt.h"
#include "zobrist.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

namespace {

constexpr int INF        = 1'000'000;   // sentinel — never a real eval score
constexpr int MATE_SCORE = 100'000;     // large but distinguishable from INF

// Module-static TT — lives for the process, warm across `go` commands so
// iterative deepening's later iterations can hit entries from earlier
// iterations. Sized at 2^20 entries (~24 MB with 24-byte TTEntry).
TranspositionTable& tt() {
    static TranspositionTable inst(20);
    return inst;
}

// Mate scores encode "distance from THIS node to mate," but the TT is
// shared across nodes at different plies, so a stored mate score must be
// converted to a ply-invariant "distance to mate from the storing node"
// on store and back to "distance from probing node" on probe.
int score_to_tt(int score, int ply) {
    if (score >  MATE_SCORE - 1000) {
        return score + ply;
    }
    if (score < -MATE_SCORE + 1000) {
        return score - ply;
    }
    return score;
}
int score_from_tt(int score, int ply) {
    if (score >  MATE_SCORE - 1000) {
        return score - ply;
    }
    if (score < -MATE_SCORE + 1000) {
        return score + ply;
    }
    return score;
}

using Clock = std::chrono::steady_clock;

// Ceiling on ply depth for indexing killer arrays. Any real search stays
// well under this; qsearch can nominally exceed it but is bounded by the
// number of pieces on the board (~30). Callers clamp before indexing.
constexpr int MAX_PLY = 128;

struct SearchContext {
    SearchLimits           limits;
    Clock::time_point      start;
    bool                   stopped = false;
    uint64_t               nodes   = 0;

    // Quiet moves that produced a beta cutoff earlier at this ply. Two
    // slots so a new killer bumps the old one to slot 1 rather than
    // losing it — most searches benefit from the two most-recent killers.
    Move killers[MAX_PLY][2] = {};

    // Butterfly-board history: keyed by (side, piece_type, dest_square).
    // Incremented by depth² on quiet-move beta cutoffs — deeper cutoffs
    // are stronger evidence the move is a general good idea for that
    // (piece, destination) combination. Capped at HISTORY_MAX so entries
    // can't crawl above the losing-capture ordering band (100k) or the
    // killer bands (800k, 900k) over long searches with many cutoffs.
    int  history[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES] = {};
};

// Ceiling on any single (side, piece, dest) history slot. Sits well
// below the losing-capture band (100k) so band ordering is preserved:
// TT > cap/promo > killers > losing cap/promo > history-quiets.
constexpr int HISTORY_MAX = 16'000;

// Poll for cancellation reasons every ~1024 nodes. Both the wall-clock
// deadline and the external `stop` flag use this cadence; on every-node
// checks the syscall/atomic-load overhead would dominate the search on
// short movetimes.
bool should_stop(SearchContext& ctx) {
    if ((ctx.nodes & 0x3FF) != 0) {
        return false;
    }
    if ((ctx.limits.external_stop != nullptr) &&
        ctx.limits.external_stop->load(std::memory_order_relaxed)) {
        return true;
    }
    if (ctx.limits.movetime_ms == 0) {
        return false;
    }
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - ctx.start).count();
    return elapsed_ms >= ctx.limits.movetime_ms;
}

// A move captures iff it removes an enemy piece from the board — either
// the destination square is occupied (normal capture) or the move is en
// passant (destination empty; captured pawn sits on an adjacent square).
bool is_capture(const Position& pos, Move m) {
    if (move_type(m) == MT_EN_PASSANT) {
        return true;
    }
    return pos.board[move_to(m)] != NO_PIECE;
}

// All pieces (either color) currently attacking `sq` given occupancy
// `occ`. Uses the same leaper-attack / magic-slider lookups as the
// legality check. Symmetric — no side awareness.
Bitboard attackers_to(const Position& pos, Square sq, Bitboard occ) {
    Bitboard bq = pos.pieces[WHITE][BISHOP] | pos.pieces[BLACK][BISHOP]
                | pos.pieces[WHITE][QUEEN]  | pos.pieces[BLACK][QUEEN];
    Bitboard rq = pos.pieces[WHITE][ROOK]   | pos.pieces[BLACK][ROOK]
                | pos.pieces[WHITE][QUEEN]  | pos.pieces[BLACK][QUEEN];
    return  (PAWN_ATTACKS[BLACK][sq] & pos.pieces[WHITE][PAWN])
          | (PAWN_ATTACKS[WHITE][sq] & pos.pieces[BLACK][PAWN])
          | (KNIGHT_ATTACKS[sq]      & (pos.pieces[WHITE][KNIGHT] | pos.pieces[BLACK][KNIGHT]))
          | (KING_ATTACKS[sq]        & (pos.pieces[WHITE][KING]   | pos.pieces[BLACK][KING]))
          | (bishop_attacks(sq, occ) & bq)
          | (rook_attacks  (sq, occ) & rq);
}

// SEE piece values — same shape as PIECE_ORDER_VALUE but with king
// low (a king is worth nothing in a swap: if we're capturing the king,
// the position was illegal to begin with).
constexpr int SEE_VALUE[NUM_PIECE_TYPES] = {
    0, 100, 320, 330, 500, 900, 20000,
};

// Static Exchange Evaluation: net material change (in centipawns) after
// all captures on `move_to(m)` play out with each side always using its
// least-valuable available attacker. Positive means we come out ahead.
//
// Handles both captures and promotions (including non-capture promotions):
// a promoting pawn contributes `promo - PAWN` to the immediate gain and
// enters the swap AS the promoted piece, so recaptures value it correctly.
//
// Reference: Chess Programming Wiki, "SEE" — the classic gain[] array
// with minimax backup. Xray attackers (pieces revealed after a blocker
// captures away) are handled by recomputing slider attacks against the
// remaining occupancy after each removal.
int see(const Position& pos, Move m) {
    const Square to      = move_to(m);
    const Square from    = move_from(m);
    const bool   is_ep   = move_type(m) == MT_EN_PASSANT;
    const bool   is_promo= move_type(m) == MT_PROMOTION;

    // Victim value. En passant captures a pawn even though the target
    // square is empty. A non-capture promotion has no victim but still
    // has a promotion gain, so we allow it through. Anything else with
    // no victim shouldn't be passed to see(); return 0 defensively.
    int victim_value;
    if (is_ep) {
        victim_value = SEE_VALUE[PAWN];
    } else if (pos.board[to] != NO_PIECE) {
        victim_value = SEE_VALUE[type_of(pos.board[to])];
    } else if (is_promo) {
        victim_value = 0;
    } else {
        return 0;
    }

    // A promoting pawn leaves the square worth its promoted piece, and
    // gains (promoted - pawn) material immediately. Subsequent recaptures
    // are captures OF the promoted piece, so the attacker type upgrades too.
    PieceType attacker_type = type_of(pos.board[from]);
    int promo_bonus = 0;
    if (is_promo) {
        PieceType promo = move_promotion(m);
        promo_bonus     = SEE_VALUE[promo] - SEE_VALUE[PAWN];
        attacker_type   = promo;
    }

    int gain[32];
    int d = 0;
    gain[d] = victim_value + promo_bonus;

    Bitboard occ = pos.occupied ^ square_bb(from);
    if (is_ep) {
        // The captured pawn sits on an adjacent square, not `to` — remove
        // it from occupancy so xray computations see through the gap.
        Square cap_sq = (pos.side_to_move == WHITE)
                        ? Square(int(to) - 8) : Square(int(to) + 8);
        occ ^= square_bb(cap_sq);
    }

    // Pieces that can attack via a slider ray once a blocker leaves —
    // used to add newly-visible xray attackers after each capture.
    const Bitboard bq_all = pos.pieces[WHITE][BISHOP] | pos.pieces[BLACK][BISHOP]
                          | pos.pieces[WHITE][QUEEN]  | pos.pieces[BLACK][QUEEN];
    const Bitboard rq_all = pos.pieces[WHITE][ROOK]   | pos.pieces[BLACK][ROOK]
                          | pos.pieces[WHITE][QUEEN]  | pos.pieces[BLACK][QUEEN];

    Bitboard attackers = attackers_to(pos, to, occ) & occ;
    Color side = Color(pos.side_to_move ^ 1);   // opponent moves next in the swap

    while (true) {
        Bitboard side_atk = attackers & pos.colors[side];
        if (side_atk == 0U) {
            break;
        }

        // Least-valuable attacker of `side`. Piece iteration order
        // is PAWN..KING, so the first match is the cheapest attacker.
        Square    lva_sq = NO_SQUARE;
        PieceType lva_pt = NO_PIECE_TYPE;
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard cands = side_atk & pos.pieces[side][pt];
            if (cands != 0U) {
                lva_sq = lsb(cands);
                lva_pt = PieceType(pt);
                break;
            }
        }

        ++d;
        gain[d] = SEE_VALUE[attacker_type] - gain[d - 1];
        // Speculative pruning: if the current best guaranteed outcome
        // is already losing, stop — deeper captures can't rescue us.
        if (std::max(-gain[d - 1], gain[d]) < 0) {
            break;
        }

        // Remove the attacker from occupancy; uncover any xray sliders.
        occ ^= square_bb(lva_sq);
        attackers &= ~square_bb(lva_sq);
        attackers |= (bishop_attacks(to, occ) & bq_all);
        attackers |= (rook_attacks  (to, occ) & rq_all);
        attackers &= occ;   // keep only still-present pieces

        attacker_type = lva_pt;
        side = Color(side ^ 1);
    }

    // Minimax backup: at each level, the side to move chooses whether
    // to make the capture or stop the swap. `-max(-parent, child)`
    // encodes "opponent picks the worse of (stop here) vs (continue)."
    // Guarded on d > 0 — a single capture with no follow-up (side had
    // no attackers on first iteration) leaves d = 0 and no backup work
    // to do.
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}

// Ordering bands, driven by SEE for anything material-touching:
//
//   TT move                       : 10,000,000
//   Winning cap/promo (SEE >= 0)  :  1,000,000 + SEE
//   Killer 1                      :    900,000
//   Killer 2                      :    800,000
//   Losing cap/promo (SEE  < 0)   :    100,000 + SEE   (still above quiets)
//   Quiet (history)               :          0 .. ~HISTORY_MAX
//
// Non-capture promotions go through the SEE path too — a non-capture
// promotion to queen is worth +800 raw, and SEE catches the case where
// an opponent piece can recapture the promoted queen for a net loss.
int move_ordering_score(const Position& pos, Move m, Move tt_move,
                        const SearchContext& ctx, int ply) {
    if (m == tt_move) {
        return 10'000'000;
    }

    const bool cap   = is_capture(pos, m);
    const bool promo = move_type(m) == MT_PROMOTION;

    if (cap || promo) {
        int see_score = see(pos, m);
        return (see_score >= 0)
            ? (1'000'000 + see_score)
            : (  100'000 + see_score);
    }

    const int p = (ply < MAX_PLY) ? ply : MAX_PLY - 1;
    if      (m == ctx.killers[p][0]) {
        return 900'000;
    }
    if (m == ctx.killers[p][1]) {
        return 800'000;
    }

    PieceType pt = type_of(pos.board[move_from(m)]);
    return ctx.history[pos.side_to_move][pt][move_to(m)];
}

// Precompute ordering scores in parallel with the moves list. Called
// once per node; the score for each move is then used by pick_move_to_front
// without recomputing.
void score_moves(const Position& pos, const MoveList& moves,
                 int* scores, Move tt_move,
                 const SearchContext& ctx, int ply) {
    for (int i = 0; i < moves.size(); ++i) {
        scores[i] = move_ordering_score(pos, moves[i], tt_move, ctx, ply);
    }
}

// Selection-style "swap the highest-scoring remaining move into position
// `start`" — the classic lazy alternative to full sort. When beta cutoff
// ends the loop early, we never scan the tail; typical well-ordered
// searches cut off on the first or second move, so most of the sort work
// std::sort would have done is avoided outright.
void pick_move_to_front(MoveList& moves, int* scores, int start) {
    int best_i = start;
    int best_s = scores[start];
    for (int i = start + 1; i < moves.size(); ++i) {
        if (scores[i] > best_s) { best_i = i; best_s = scores[i]; }
    }
    if (best_i != start) {
        Move tm = moves[start];  moves[start]  = moves[best_i];  moves[best_i]  = tm;
        int  ts = scores[start]; scores[start] = scores[best_i]; scores[best_i] = ts;
    }
}

// Quiescence search: extend the main search at leaf nodes with capture
// sequences only, until the position is "quiet" (no more captures to
// consider). This fixes the horizon effect — without it, a search that
// stops mid-exchange evaluates as if the exchange were resolved in the
// mover's favor, and the engine plays speculative captures that get
// refuted one ply beyond its horizon.
//
// The trick: "stand-pat" assumes the mover can decline to capture (a
// reasonable proxy since at a real interior node they could always play
// a quiet move). Static eval becomes a lower bound on the true value;
// any capture that fails to improve it can be pruned. Convergence is
// guaranteed because every recursive step removes a piece from the board.
//
// Exception: when the side to move is in check, standing pat is unsound
// (we're not allowed to "do nothing"), so we consider all legal moves
// and let the recursion resolve the check.
int qsearch(Position& pos, int alpha, int beta, int ply, SearchContext& ctx) {
    ++ctx.nodes;
    if (ctx.stopped || should_stop(ctx)) { ctx.stopped = true; return 0; }

    const bool checked = in_check(pos);

    if (!checked) {
        int stand_pat = evaluate(pos);
        if (stand_pat >= beta) {
            return beta;
        }
        alpha = std::max(alpha, stand_pat);
    }

    // In-check nodes need every legal move (evasion may require a quiet
    // block or king step). Quiet nodes only need captures + promotions —
    // stand_pat already handled the "do nothing" baseline.
    MoveList moves;
    if (checked) {
        generate_moves   (pos, moves);
    }
    else {
        generate_captures(pos, moves);
    }

    if (moves.empty()) {
        // Two very different meanings here:
        //  - checked + no evasion → checkmate (mate-in-ply score)
        //  - not checked + no captures → quiet, just return stand_pat via alpha
        //    (stand-pat has already set alpha via the earlier update)
        return checked ? (-MATE_SCORE + ply) : alpha;
    }

    // SEE-scored ordering: winning captures first, losing captures below
    // killers/history. Also enables the SEE prune below.
    int scores[MoveList::MAX_MOVES];
    score_moves(pos, moves, scores, NULL_MOVE, ctx, ply);

    for (int i = 0; i < moves.size(); ++i) {
        pick_move_to_front(moves, scores, i);
        Move m = moves[i];

        // SEE prune: when not in check, skip losing captures — they can't
        // improve alpha (down material after resolution). This is what
        // makes qsearch actually cheap on tactical positions.
        // Also skip further quiet moves at qsearch — they're only here
        // because we're in check, and if we've dropped below the capture
        // band, we've exhausted meaningful evasions from ordering.
        if (!checked && scores[i] < 100'000) {
            break;
        }

        UndoInfo u;
        pos.make_move(m, u);
        int score = -qsearch(pos, -beta, -alpha, ply + 1, ctx);
        pos.unmake_move(m, u);
        if (ctx.stopped) {
            return 0;
        }
        if (score >= beta) {
            return beta;
        }
        alpha = std::max(alpha, score);
    }
    return alpha;
}

// Negamax with alpha-beta + TT + PVS + LMR + check extensions.
// `ply` is distance from the root so we can prefer shorter mates
// (deeper mate scores are penalized) and shorter paths out of a
// losing position (later mates score less negative).
// NOLINTNEXTLINE(readability-function-cognitive-complexity) — fused PVS+null-move+LMR+razoring+RFP+killer/history hot path; every branch is a documented pruning technique with a measured node-count contribution (see CLAUDE.md search perf stack).
int negamax(Position& pos, int depth, int alpha, int beta,
            int ply, SearchContext& ctx) {
    ++ctx.nodes;
    if (ctx.stopped || should_stop(ctx)) { ctx.stopped = true; return 0; }

    // Draw detection — never at the root itself (ply > 0) so search_root's
    // fallback still returns a legal move. Both draws score 0: repetition
    // opponent will claim, 50-move rule is enforceable by law of the game.
    if (ply > 0 && (pos.is_repetition() || pos.halfmove_clock >= 100)) {
        return 0;
    }

    // Check extension: when the side to move is in check, tactical lines
    // are often deeper than the requested depth. Extend by one ply so
    // mate combinations don't fall off the horizon. Computed once here
    // and reused for LMR gating below.
    const bool node_in_check = in_check(pos);
    if (node_in_check) {
        depth += 1;
    }

    // Leaf: hand off to quiescence rather than static-evaluating a
    // position that may be mid-exchange.
    if (depth <= 0) {
        return qsearch(pos, alpha, beta, ply, ctx);
    }

    // TT probe: may return an immediately-usable score, and always hands
    // back a move to try first if the key was seen before.
    int  tt_score = 0;
    Move tt_move  = NULL_MOVE;
    if (tt().probe(pos.key, depth, alpha, beta, tt_score, tt_move)) {
        return score_from_tt(tt_score, ply);
    }

    // Static-eval-based pruning at non-check nodes with non-mate windows.
    // We compute evaluate() once and use it for both reverse futility and
    // razoring — they hit opposite ends of the window, so both can apply.
    //
    //   Reverse futility (aka static null pruning): if we're ALREADY well
    //   above beta after subtracting a generous margin-per-ply, giving
    //   the opponent even a big move can't bring us below beta — prune.
    //
    //   Razoring: if we're WAY below alpha at very shallow depth, hand
    //   off to qsearch and if even qsearch (which sees captures) can't
    //   pull us back to alpha, return that as an upper bound.
    if (!node_in_check
        && std::abs(beta)  < MATE_SCORE - 1000
        && std::abs(alpha) < MATE_SCORE - 1000) {
        int se = evaluate(pos);

        constexpr int RFP_MARGIN   = 80;    // cp per depth ply
        constexpr int RAZOR_MARGIN = 200;   // cp

        if (depth <= 6 && se - (RFP_MARGIN * depth) >= beta) {
            return se;
        }
        if (depth <= 2 && se + RAZOR_MARGIN <= alpha) {
            int qs = qsearch(pos, alpha, beta, ply, ctx);
            if (qs <= alpha) {
                return qs;
            }
        }
    }

    // Null-move pruning: at non-PV interior nodes with sufficient
    // material, we let the opponent play two moves in a row (i.e., pass
    // our turn) and search at reduced depth. If the position is STILL
    // good enough to beat beta, our real move can only be better —
    // prune. Reduction R = 3 is the classical default.
    //
    // Guards:
    //  - Not in check (can't "pass" in check — it's illegal).
    //  - depth >= 3 (below that the reduced search is qsearch anyway).
    //  - Beta isn't a mate score (mate detection needs the real search).
    //  - Side to move has at least one non-pawn non-king piece
    //    (zugzwang guard: in pure king-pawn endgames, passing is often
    //    strictly worse than any real move, so null-move gives false
    //    prunings that lose the game).
    const bool has_non_pawn = (pos.pieces[pos.side_to_move][KNIGHT] |
                               pos.pieces[pos.side_to_move][BISHOP] |
                               pos.pieces[pos.side_to_move][ROOK]   |
                               pos.pieces[pos.side_to_move][QUEEN]) != 0U;
    if (depth >= 3 && !node_in_check && has_non_pawn &&
        std::abs(beta) < MATE_SCORE - 1000) {
        // Make null move inline (no ~30-byte UndoInfo, no board changes):
        // just flip side, clear ep, adjust Zobrist for those two.
        const Square   saved_ep  = pos.ep_square;
        const uint64_t saved_key = pos.key;
        if (saved_ep != NO_SQUARE) {
            pos.key ^= zobrist::EP_FILE[file_of(saved_ep)];
        }
        pos.ep_square    = NO_SQUARE;
        pos.side_to_move = Color(pos.side_to_move ^ 1);
        pos.key         ^= zobrist::SIDE;

        constexpr int R = 3;
        int null_score = -negamax(pos, depth - 1 - R,
                                  -beta, -beta + 1, ply + 1, ctx);

        pos.side_to_move = Color(pos.side_to_move ^ 1);
        pos.ep_square    = saved_ep;
        pos.key          = saved_key;

        if (ctx.stopped) {
            return 0;
        }
        if (null_score >= beta) {
            return beta;      // fail-high: prune the whole subtree
        }
    }

    MoveList moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        if (node_in_check) {
            return -MATE_SCORE + ply;
        }
        return 0;   // stalemate
    }

    int scores[MoveList::MAX_MOVES];
    score_moves(pos, moves, scores, tt_move, ctx, ply);

    const int original_alpha = alpha;
    int  best      = -INF;
    Move best_move = NULL_MOVE;
    for (int i = 0; i < moves.size(); ++i) {
        pick_move_to_front(moves, scores, i);
        Move m = moves[i];

        const bool is_cap    = is_capture(pos, m);
        const bool is_promo  = move_type(m) == MT_PROMOTION;

        UndoInfo u;
        pos.make_move(m, u);

        // PVS + LMR:
        //   - First move (i==0) is our best guess — search at full window
        //     and full depth to establish the PV.
        //   - Later moves are much less likely to beat alpha. Probe with
        //     a NULL WINDOW (`-alpha-1, -alpha`) — cheapest possible way
        //     to answer "does this move improve alpha?" On quiet non-
        //     promotion moves at deep-enough depths we also REDUCE depth
        //     (LMR) since late quiet moves rarely change the eval much.
        //   - If the null-window probe fails high (score > alpha), the
        //     move actually might be good — pay full window + full depth
        //     to get a real score.
        int score;
        if (i == 0) {
            score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, ctx);
        } else {
            const bool can_reduce = depth >= 3 && !is_cap && !is_promo
                                  && !node_in_check;
            int reduction = 0;
            if (can_reduce) {
                reduction = (i >= 12) ? 2 : 1;
            }
            score = -negamax(pos, depth - 1 - reduction,
                             -alpha - 1, -alpha, ply + 1, ctx);
            if (score > alpha && (reduction > 0 || score < beta)) {
                // Verified at full depth + full window. Two triggers:
                // reduced probe found something (need real depth), or
                // null-window failed high inside the ACTUAL window
                // (need real score, not just "> alpha" upper bound).
                score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, ctx);
            }
        }

        pos.unmake_move(m, u);
        if (ctx.stopped) {
            return 0;                // bubble up cancellation
        }
        if (score > best)  { best = score; best_move = m; }
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            // Beta cutoff on a quiet, non-promotion move: record as a
            // killer at this ply and bump history. Captures and promos
            // already order well via SEE, so we skip them to avoid
            // polluting the tables with forcing moves.
            if (!is_cap && !is_promo && ply < MAX_PLY) {
                if (ctx.killers[ply][0] != m) {
                    ctx.killers[ply][1] = ctx.killers[ply][0];
                    ctx.killers[ply][0] = m;
                }
                PieceType pt = type_of(pos.board[move_from(m)]);
                int& h = ctx.history[pos.side_to_move][pt][move_to(m)];
                h += depth * depth;
                h = std::min(h, HISTORY_MAX);
            }
            break;
        }
    }

    // TT store: classify by how the search terminated relative to the
    // original window. `best >= beta` means a fail-high (LOWER bound);
    // `best <= original_alpha` means no move improved on alpha (UPPER
    // bound); otherwise the score is exact.
    TTBound bound = TT_EXACT;
    if (best >= beta) {
        bound = TT_LOWER;
    } else if (best <= original_alpha) {
        bound = TT_UPPER;
    }
    tt().store(pos.key, depth, score_to_tt(best, ply), best_move, bound);
    return best;
}

// Single-depth root search bounded by (alpha_init, beta_root). Populates
// `out` if the depth completes; leaves it untouched if the search stopped
// mid-iteration. The window parameters enable aspiration search — when
// the returned score lands outside the window, the caller re-searches
// with a wider one.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — (depth, alpha, beta) is standard chess-engine search-node argument order.
bool search_root(Position& pos, int depth,
                 int alpha_init, int beta_root,
                 SearchContext& ctx, SearchResult& out) {
    ++ctx.nodes;

    MoveList moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        out.best_move = NULL_MOVE;
        out.score     = in_check(pos) ? -MATE_SCORE : 0;
        out.depth     = depth;
        return true;
    }

    // Root uses the TT-move hint from any prior iteration or earlier
    // `go` call — iterative deepening's biggest ordering win comes from
    // trying last iteration's best move first at the next depth.
    int  tt_score = 0;
    Move tt_move  = NULL_MOVE;
    tt().probe(pos.key, /*depth=*/0, -INF, INF, tt_score, tt_move);

    int scores[MoveList::MAX_MOVES];
    score_moves(pos, moves, scores, tt_move, ctx, /*ply=*/0);

    int  alpha     = alpha_init;
    int  best      = -INF;
    Move best_move = NULL_MOVE;
    for (int i = 0; i < moves.size(); ++i) {
        pick_move_to_front(moves, scores, i);
        Move m = moves[i];
        UndoInfo u;
        pos.make_move(m, u);
        // Root PVS: the first move (best guess from prior iteration's TT
        // hint or move ordering) gets a full-window search to establish
        // the PV. Later moves get a null-window probe first — if they
        // don't beat alpha, no need to spend the full-window cost. Only
        // re-search when the probe genuinely lands inside the aspiration
        // window (score > alpha AND score < beta_root); a probe result
        // >= beta_root is a fail-high that the aspiration wrapper will
        // widen and re-search anyway.
        int score;
        if (i == 0) {
            score = -negamax(pos, depth - 1, -beta_root, -alpha, 1, ctx);
        } else {
            score = -negamax(pos, depth - 1, -alpha - 1, -alpha, 1, ctx);
            if (score > alpha && score < beta_root) {
                score = -negamax(pos, depth - 1, -beta_root, -alpha, 1, ctx);
            }
        }
        pos.unmake_move(m, u);
        if (ctx.stopped) {
            return false;
        }
        if (score > best)  { best = score; best_move = m; }
        alpha = std::max(alpha, score);
        // Fail-high at root: with aspiration windows beta_root is finite
        // and a move that beats it means our score estimate was too low —
        // the caller widens and re-searches, so no point iterating the
        // remaining moves (they can't lower the fail-high score).
        if (alpha >= beta_root) {
            break;
        }
    }
    out.best_move = best_move;
    out.score     = best;
    out.depth     = depth;
    // Only store as EXACT when the score landed inside the window; the
    // wrapper handles fail-high/low re-searches and stores the exact
    // result once it converges.
    if (best > alpha_init && best < beta_root) {
        tt().store(pos.key, depth, score_to_tt(best, 0), best_move, TT_EXACT);
    }
    return true;
}

// Walk the TT starting from `best_move` at `pos` to reconstruct the
// principal variation. Bounded by `max_len` and by defensive checks:
// TT miss, TT-move illegal in the current position, or a cycle in the
// walk all cause an early return.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — (position, move, depth-limit, out) is the natural signature; Move is uint16_t but semantically distinct from max_len.
void build_pv(Position pos, Move best_move, int max_len,
              std::vector<Move>& out) {
    out.clear();
    if (best_move == NULL_MOVE) {
        return;
    }

    // Fresh MoveList each generate_moves call — the generator appends, so
    // reusing across iterations would accumulate entries from prior plies.
    auto legal_in = [](Position& p, Move want) {
        MoveList legal;
        generate_moves(p, legal);
        return std::find(legal.begin(), legal.end(), want) != legal.end();
    };

    if (!legal_in(pos, best_move)) {
        return;
    }

    out.push_back(best_move);
    UndoInfo u;
    pos.make_move(best_move, u);

    std::vector<uint64_t> seen{pos.key};
    for (int i = 1; i < max_len; ++i) {
        int  dummy_score;
        Move next;
        if (!tt().probe(pos.key, /*depth=*/0, -INF, INF, dummy_score, next)) {
            break;
        }
        if (next == NULL_MOVE) {
            break;
        }
        if (!legal_in(pos, next)) {
            break;
        }

        out.push_back(next);
        UndoInfo u2;
        pos.make_move(next, u2);
        // Cycle guard: TT collisions or transpositions can point us back
        // to a position already in the walk. Stop rather than loop.
        if (std::find(seen.begin(), seen.end(), pos.key) != seen.end()) {
            break;
        }
        seen.push_back(pos.key);
    }
}

}  // namespace

SearchResult search_best(Position& pos, int depth) {
    SearchContext ctx;
    ctx.start = Clock::now();
    SearchResult r;
    search_root(pos, depth, -INF, INF, ctx, r);
    r.nodes = ctx.nodes;
    build_pv(pos, r.best_move, depth, r.pv);
    return r;
}

void clear_transposition_table() {
    tt().clear();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) — iterative-deepening driver fuses aspiration windows, time-management, PV extraction, and stop polling; splitting hurts readability more than it helps. See CLAUDE.md perf table.
SearchResult search_iterative(Position& pos, SearchLimits limits,
                              const InfoCallback& on_iter) {
    SearchContext ctx;
    ctx.limits = limits;
    ctx.start  = Clock::now();

    SearchResult best;

    // Aspiration search: from d=2 onward we guess the score will be close
    // to last iteration's, so search inside a narrow window (±25 cp) and
    // fall back to a wider one only if we fail-high or fail-low. Cheap
    // when the guess is right (most iterations), self-correcting when
    // wrong. Delta doubles per re-search up to a threshold, then opens
    // the window fully.
    // Initial window sized above the typical depth-parity score oscillation
    // (~40 cp on startpos-style positions). Too narrow and we re-search
    // every iteration; too wide and we lose the pruning benefit. 75 is
    // conservative for our current eval; larger MAX bounds re-search work
    // when the score genuinely moves (tactical positions).
    constexpr int ASPIRATION_INITIAL = 75;
    constexpr int ASPIRATION_MAX     = 2000;

    for (int d = 1; d <= limits.max_depth; ++d) {
        SearchResult r;
        bool completed;

        if (d == 1) {
            completed = search_root(pos, d, -INF, INF, ctx, r);
        } else {
            int delta = ASPIRATION_INITIAL;
            int alpha = best.score - delta;
            int beta  = best.score + delta;
            while (true) {
                completed = search_root(pos, d, alpha, beta, ctx, r);
                if (!completed) {
                    break;
                }
                if (r.score <= alpha) {           // fail-low: widen alpha
                    delta *= 2;
                    alpha = (delta >= ASPIRATION_MAX) ? -INF : best.score - delta;
                } else if (r.score >= beta) {     // fail-high: widen beta
                    delta *= 2;
                    beta = (delta >= ASPIRATION_MAX) ? INF : best.score + delta;
                } else {
                    break;                        // score inside window — done
                }
            }
        }

        // For d > 1, only accept fully-completed iterations — a
        // partial deeper result is unreliable (best-move might come
        // from a losing subtree we hadn't refuted yet).
        if (!completed && d > 1) {
            break;
        }
        r.nodes = ctx.nodes;
        // Reconstruct PV before firing the callback so UCI can print
        // the full line. Bounded by iteration depth so we don't chase
        // TT transpositions past what we actually searched.
        build_pv(pos, r.best_move, d, r.pv);
        best = r;
        if (on_iter) {
            on_iter(best);
        }
        if (ctx.stopped) {
            break;
        }
    }

    // Final guarantee: bestmove must be a legal move if any exist.
    // A stop signal arriving before d=1 completes a single root move
    // can leave best.best_move as NULL_MOVE despite legal moves being
    // available. Fall back to the first legal move so the UCI contract
    // holds. (Genuine mate/stalemate → generate_moves is empty → keep
    // NULL_MOVE, which serializes as "0000" per protocol.)
    if (best.best_move == NULL_MOVE) {
        MoveList moves;
        generate_moves(pos, moves);
        if (!moves.empty()) {
            best.best_move = moves[0];
            if (best.depth == 0) {
                best.depth = 1;
            }
        }
    }
    return best;
}
