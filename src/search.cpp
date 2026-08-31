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
    // (piece, destination) combination.
    int  history[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES] = {};
};

// Poll for cancellation reasons every ~1024 nodes. Both the wall-clock
// deadline and the external `stop` flag use this cadence; on every-node
// checks the syscall/atomic-load overhead would dominate the search on
// short movetimes.
bool should_stop(SearchContext& ctx) {
    if ((ctx.nodes & 0x3FF) != 0) return false;
    if (ctx.limits.external_stop &&
        ctx.limits.external_stop->load(std::memory_order_relaxed)) {
        return true;
    }
    if (ctx.limits.movetime_ms == 0) return false;
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

// Ordering scores. Bands are chosen so higher-priority classes always
// outrank lower ones, even with the intra-band spread:
//
//   TT move           : 10,000,000
//   Any capture       :  1,000,000 + MVV-LVA
//   Killer 1          :    900,000
//   Killer 2          :    800,000
//   Quiet (history)   :     0..~500,000 in practice
//
// The "victim*10 - attacker" MVV-LVA span (a few thousand) fits comfortably
// inside the 100k gap between capture-base and killer-base.
constexpr int PIECE_ORDER_VALUE[NUM_PIECE_TYPES] = {
    0, 100, 320, 330, 500, 900, 20000,   // K high — captures on king shouldn't occur but guard anyway
};

int move_ordering_score(const Position& pos, Move m, Move tt_move,
                        const SearchContext& ctx, int ply) {
    if (m == tt_move) return 10'000'000;

    int score = 0;

    if (is_capture(pos, m)) {
        PieceType victim = (move_type(m) == MT_EN_PASSANT)
            ? PAWN
            : type_of(pos.board[move_to(m)]);
        PieceType attacker = type_of(pos.board[move_from(m)]);
        score = 1'000'000 + PIECE_ORDER_VALUE[victim] * 10 - PIECE_ORDER_VALUE[attacker];
    } else {
        // Quiet moves: killer > killer2 > history-ranked. Clamp ply to
        // keep qsearch (which can nominally exceed MAX_PLY) safe.
        const int p = (ply < MAX_PLY) ? ply : MAX_PLY - 1;
        if      (m == ctx.killers[p][0]) score = 900'000;
        else if (m == ctx.killers[p][1]) score = 800'000;
        else {
            PieceType pt = type_of(pos.board[move_from(m)]);
            score = ctx.history[pos.side_to_move][pt][move_to(m)];
        }
    }

    if (move_type(m) == MT_PROMOTION) {
        // Applies on top of both branches; a capture-promotion earns both
        // its capture score and its promotion piece value.
        score += PIECE_ORDER_VALUE[move_promotion(m)];
    }
    return score;
}

// In-place descending sort by move_ordering_score with TT-move / killer /
// history hints woven in. std::sort is O(N log N) per call — cheap at
// chess branching factors — but could later be replaced with a lazy
// selection-sort that avoids sorting the tail after a beta cutoff.
void order_moves(const Position& pos, std::vector<Move>& moves,
                 Move tt_move, const SearchContext& ctx, int ply) {
    std::sort(moves.begin(), moves.end(),
              [&](Move a, Move b) {
                  return move_ordering_score(pos, a, tt_move, ctx, ply)
                       > move_ordering_score(pos, b, tt_move, ctx, ply);
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
    if (ctx.stopped || should_stop(ctx)) { ctx.stopped = true; return 0; }

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
    // (leaves change rapidly and add little), so no move hint. Killers
    // and history are quiet-move heuristics — qsearch only searches
    // captures (except when in check) so they don't affect ordering here.
    order_moves(pos, moves, NULL_MOVE, ctx, ply);

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
    if (ctx.stopped || should_stop(ctx)) { ctx.stopped = true; return 0; }
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

    order_moves(pos, moves, tt_move, ctx, ply);

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
        if (alpha >= beta) {
            // Beta cutoff on a quiet move: record it as a killer at this
            // ply and bump its history. Captures already order well via
            // MVV-LVA so we don't polute those tables with them.
            if (!is_capture(pos, m) && ply < MAX_PLY) {
                if (ctx.killers[ply][0] != m) {
                    ctx.killers[ply][1] = ctx.killers[ply][0];
                    ctx.killers[ply][0] = m;
                }
                PieceType pt = type_of(pos.board[move_from(m)]);
                ctx.history[pos.side_to_move][pt][move_to(m)] += depth * depth;
            }
            break;
        }
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
    // Killers/history at ply 0 are populated by prior root iterations of
    // this same search_iterative call.
    int  tt_score = 0;
    Move tt_move  = NULL_MOVE;
    tt().probe(pos.key, /*depth=*/0, -INF, INF, tt_score, tt_move);
    order_moves(pos, moves, tt_move, ctx, /*ply=*/0);

    int  alpha     = -INF;
    int  best      = -INF;
    Move best_move = NULL_MOVE;
    for (Move m : moves) {
        UndoInfo u;
        pos.make_move(m, u);
        int score = -negamax(pos, depth - 1, -INF, -alpha, 1, ctx);
        pos.unmake_move(m, u);
        if (ctx.stopped) {
            // Cancelled mid-iteration. Preserve whatever `best_move` we
            // found so far — the caller can salvage it as a partial
            // result. If we hadn't finished any move yet, best_move
            // stays NULL_MOVE and the fallback in search_iterative
            // picks the first legal move.
            return false;
        }
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
        // For d > 1, only accept fully-completed iterations — a
        // partial deeper result is unreliable (best-move might come
        // from a losing subtree we hadn't refuted yet).
        if (!completed && d > 1) break;
        r.nodes = ctx.nodes;
        best = r;
        if (on_iter) on_iter(best);
        if (ctx.stopped) break;
    }

    // Final guarantee: bestmove must be a legal move if any exist.
    // A stop signal arriving before d=1 completes a single root move
    // can leave best.best_move as NULL_MOVE despite legal moves being
    // available. Fall back to the first legal move so the UCI contract
    // holds. (Genuine mate/stalemate → generate_moves is empty → keep
    // NULL_MOVE, which serializes as "0000" per protocol.)
    if (best.best_move == NULL_MOVE) {
        std::vector<Move> moves;
        generate_moves(pos, moves);
        if (!moves.empty()) {
            best.best_move = moves[0];
            if (best.depth == 0) best.depth = 1;
        }
    }
    return best;
}
