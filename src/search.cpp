#include "search.h"
#include "eval.h"
#include "movegen.h"

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

// Negamax with alpha-beta. `ply` is distance from the root so we can prefer
// shorter mates (deeper mate scores are penalized), and shorter paths out
// of a losing position (later mates score less negative).
int negamax(Position& pos, int depth, int alpha, int beta,
            int ply, SearchContext& ctx) {
    ++ctx.nodes;
    if (ctx.stopped || time_up(ctx)) { ctx.stopped = true; return 0; }
    if (depth == 0) return evaluate(pos);

    std::vector<Move> moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        if (in_check(pos)) return -MATE_SCORE + ply;
        return 0;   // stalemate
    }

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
