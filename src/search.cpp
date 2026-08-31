#include "search.h"
#include "eval.h"
#include "movegen.h"
#include "tt.h"

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
    if (score >  MATE_SCORE - 1000) return score + ply;
    if (score < -MATE_SCORE + 1000) return score - ply;
    return score;
}
int score_from_tt(int score, int ply) {
    if (score >  MATE_SCORE - 1000) return score - ply;
    if (score < -MATE_SCORE + 1000) return score + ply;
    return score;
}

using Clock = std::chrono::steady_clock;

struct SearchContext {
    SearchLimits           limits;
    Clock::time_point      start;
    bool                   stopped = false;
    uint64_t               nodes   = 0;
};

// Poll the wall clock every ~1024 nodes. Checking on every node makes
// clock_gettime dominate short searches; the granularity chosen keeps
// syscall overhead under ~1% while still stopping within a millisecond
// or two of the movetime target.
bool time_up(SearchContext& ctx) {
    if (ctx.limits.movetime_ms == 0)   return false;
    if ((ctx.nodes & 0x3FF) != 0)      return false;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - ctx.start).count();
    return elapsed_ms >= ctx.limits.movetime_ms;
}

// A move captures iff it removes an enemy piece from the board — either
// the destination square is occupied (normal capture) or the move is en
// passant (destination empty; captured pawn sits on an adjacent square).
bool is_capture(const Position& pos, Move m) {
    if (move_type(m) == MT_EN_PASSANT) return true;
    return pos.board[move_to(m)] != NO_PIECE;
}

// Ordering scores. Centipawn values chosen so the "victim" bump dominates
// even the largest attacker adjustment: PxQ (900*10 - 100 = 8900) beats
// QxP (100*10 - 900 = 100) by nearly two orders of magnitude, so full
// MVV-LVA ordering falls out of a single subtraction.
constexpr int PIECE_ORDER_VALUE[NUM_PIECE_TYPES] = {
    0, 100, 320, 330, 500, 900, 20000,   // K high — captures on king shouldn't occur but guard anyway
};

// Assign a comparable score to a move for ordering. Captures score in the
// tens of thousands (well above any non-capture), promotions add on top,
// non-captures score 0.
int move_ordering_score(const Position& pos, Move m) {
    int score = 0;
    if (is_capture(pos, m)) {
        PieceType victim = (move_type(m) == MT_EN_PASSANT)
            ? PAWN
            : type_of(pos.board[move_to(m)]);
        PieceType attacker = type_of(pos.board[move_from(m)]);
        // Base ensures every capture outranks every quiet move even when
        // victim < attacker (e.g., QxP: 100*10 - 900 = 100, still + 100000).
        score = 100000 + PIECE_ORDER_VALUE[victim] * 10 - PIECE_ORDER_VALUE[attacker];
    }
    if (move_type(m) == MT_PROMOTION) {
        // Prefer queen promotions first; treat under-promotions as tie-broken
        // by the promotion piece's own value.
        score += PIECE_ORDER_VALUE[move_promotion(m)];
    }
    return score;
}

// In-place descending sort by move_ordering_score. `tt_move`, if it's a
// real move present in `moves`, is bumped to the very top — a hash-table
// hit from a previous iteration is a much better first-move guess than
// MVV-LVA alone can produce.
void order_moves(const Position& pos, std::vector<Move>& moves, Move tt_move) {
    std::sort(moves.begin(), moves.end(),
              [&](Move a, Move b) {
                  int sa = move_ordering_score(pos, a);
                  int sb = move_ordering_score(pos, b);
                  if (a == tt_move) sa += 10'000'000;
                  if (b == tt_move) sb += 10'000'000;
                  return sa > sb;
              });
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
    if (ctx.stopped || time_up(ctx)) { ctx.stopped = true; return 0; }

    const bool checked = in_check(pos);

    if (!checked) {
        int stand_pat = evaluate(pos);
        if (stand_pat >= beta)   return beta;
        if (stand_pat > alpha)   alpha = stand_pat;
    }

    std::vector<Move> moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        // A terminal position reached via qsearch — same mate/stalemate
        // resolution as full negamax at a leaf with no legal moves.
        return checked ? (-MATE_SCORE + ply) : 0;
    }

    // MVV-LVA: try highest-victim captures first so the strong replies
    // trigger beta cutoffs immediately. Qsearch doesn't probe the TT
    // (leaves change rapidly and add little), so no move hint.
    order_moves(pos, moves, NULL_MOVE);

    for (Move m : moves) {
        // Non-captures are only searched when we're evading check; otherwise
        // they don't help resolve the tactical sequence we entered qsearch
        // to analyze, and searching them would defeat the point.
        if (!checked && !is_capture(pos, m)) continue;

        UndoInfo u;
        pos.make_move(m, u);
        int score = -qsearch(pos, -beta, -alpha, ply + 1, ctx);
        pos.unmake_move(m, u);
        if (ctx.stopped)   return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// Negamax with alpha-beta + TT. `ply` is distance from the root so we can
// prefer shorter mates (deeper mate scores are penalized), and shorter
// paths out of a losing position (later mates score less negative).
int negamax(Position& pos, int depth, int alpha, int beta,
            int ply, SearchContext& ctx) {
    ++ctx.nodes;
    if (ctx.stopped || time_up(ctx)) { ctx.stopped = true; return 0; }
    // Leaf: hand off to quiescence rather than static-evaluating a
    // position that may be mid-exchange. See qsearch for the rationale.
    if (depth == 0) return qsearch(pos, alpha, beta, ply, ctx);

    // TT probe: may return an immediately-usable score, and always hands
    // back a move to try first if the key was seen before.
    int  tt_score = 0;
    Move tt_move  = NULL_MOVE;
    if (tt().probe(pos.key, depth, alpha, beta, tt_score, tt_move)) {
        return score_from_tt(tt_score, ply);
    }

    std::vector<Move> moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        if (in_check(pos)) return -MATE_SCORE + ply;
        return 0;   // stalemate
    }

    order_moves(pos, moves, tt_move);

    const int original_alpha = alpha;
    int  best      = -INF;
    Move best_move = NULL_MOVE;
    for (Move m : moves) {
        UndoInfo u;
        pos.make_move(m, u);
        int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, ctx);
        pos.unmake_move(m, u);
        if (ctx.stopped) return 0;                // bubble up cancellation
        if (score > best)  { best = score; best_move = m; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;                 // beta cutoff
    }

    // TT store: classify by how the search terminated relative to the
    // original window. `best >= beta` means a fail-high (LOWER bound);
    // `best <= original_alpha` means no move improved on alpha (UPPER
    // bound); otherwise the score is exact.
    TTBound bound = (best >= beta)            ? TT_LOWER
                  : (best <= original_alpha)  ? TT_UPPER
                                              : TT_EXACT;
    tt().store(pos.key, depth, score_to_tt(best, ply), best_move, bound);
    return best;
}

// Single-depth root search. Populates `out` if the depth completes; leaves
// it untouched if the search stopped mid-iteration so the caller can fall
// back to the previous iteration's result.
bool search_root(Position& pos, int depth, SearchContext& ctx,
                 SearchResult& out) {
    ++ctx.nodes;   // count the root

    std::vector<Move> moves;
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
    order_moves(pos, moves, tt_move);

    int  alpha     = -INF;
    int  best      = -INF;
    Move best_move = NULL_MOVE;
    for (Move m : moves) {
        UndoInfo u;
        pos.make_move(m, u);
        int score = -negamax(pos, depth - 1, -INF, -alpha, 1, ctx);
        pos.unmake_move(m, u);
        if (ctx.stopped) return false;
        if (score > best)  { best = score; best_move = m; }
        if (score > alpha) alpha = score;
    }
    out.best_move = best_move;
    out.score     = best;
    out.depth     = depth;
    // Store the root result so the next ID iteration (and future `go`
    // calls on the same position) benefit from move ordering.
    tt().store(pos.key, depth, score_to_tt(best, 0), best_move, TT_EXACT);
    return true;
}

}  // namespace

SearchResult search_best(Position& pos, int depth) {
    SearchContext ctx;
    ctx.start = Clock::now();
    SearchResult r;
    search_root(pos, depth, ctx, r);
    r.nodes = ctx.nodes;
    return r;
}

void clear_transposition_table() {
    tt().clear();
}

SearchResult search_iterative(Position& pos, SearchLimits limits,
                              InfoCallback on_iter) {
    SearchContext ctx;
    ctx.limits = limits;
    ctx.start  = Clock::now();

    SearchResult best;

    for (int d = 1; d <= limits.max_depth; ++d) {
        SearchResult r;
        bool completed = search_root(pos, d, ctx, r);
        // Guarantee at least the depth-1 result even if movetime is
        // absurdly short — a "no move" bestmove would violate the UCI
        // contract with the GUI.
        if (!completed && d > 1) break;
        r.nodes = ctx.nodes;
        best = r;
        if (on_iter) on_iter(best);
        if (ctx.stopped) break;
    }
    return best;
}
