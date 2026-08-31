#include "search.h"
#include "eval.h"
#include "movegen.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace {

constexpr int INF        = 1'000'000;   // sentinel — never a real eval score
constexpr int MATE_SCORE = 100'000;     // large but distinguishable from INF

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

// In-place descending sort by move_ordering_score. std::sort is O(N log N)
// per call — fine at chess branching factors (~30-40 moves). Could later
// be replaced with a lazy "pick next best" that avoids sorting the full
// list when a beta cutoff happens early.
void order_moves(const Position& pos, std::vector<Move>& moves) {
    std::sort(moves.begin(), moves.end(),
              [&](Move a, Move b) {
                  return move_ordering_score(pos, a) > move_ordering_score(pos, b);
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
    // trigger beta cutoffs immediately.
    order_moves(pos, moves);

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

// Negamax with alpha-beta. `ply` is distance from the root so we can prefer
// shorter mates (deeper mate scores are penalized), and shorter paths out
// of a losing position (later mates score less negative).
int negamax(Position& pos, int depth, int alpha, int beta,
            int ply, SearchContext& ctx) {
    ++ctx.nodes;
    if (ctx.stopped || time_up(ctx)) { ctx.stopped = true; return 0; }
    // Leaf: hand off to quiescence rather than static-evaluating a
    // position that may be mid-exchange. See qsearch for the rationale.
    if (depth == 0) return qsearch(pos, alpha, beta, ply, ctx);

    std::vector<Move> moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        if (in_check(pos)) return -MATE_SCORE + ply;
        return 0;   // stalemate
    }

    order_moves(pos, moves);

    int best = -INF;
    for (Move m : moves) {
        UndoInfo u;
        pos.make_move(m, u);
        int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, ctx);
        pos.unmake_move(m, u);
        if (ctx.stopped) return 0;                // bubble up cancellation
        if (score > best)  best  = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;                 // beta cutoff
    }
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

    order_moves(pos, moves);

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
