#include "search.h"
#include "eval.h"
#include "movegen.h"

#include <vector>

namespace {

constexpr int INF        = 1'000'000;   // sentinel — never a real eval score
constexpr int MATE_SCORE = 100'000;     // large but distinguishable from INF

// Negamax with alpha-beta pruning. `ply` is distance from the root so we
// can prefer shorter mates (deeper mate scores are penalized), and shorter
// paths out of a losing position (later mates score less negative).
int negamax(Position& pos, int depth, int alpha, int beta,
            int ply, uint64_t& nodes) {
    ++nodes;
    if (depth == 0) return evaluate(pos);

    std::vector<Move> moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        // Terminal: distinguish checkmate from stalemate.
        if (in_check(pos)) return -MATE_SCORE + ply;
        return 0;
    }

    int best = -INF;
    for (Move m : moves) {
        UndoInfo u;
        pos.make_move(m, u);
        int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, nodes);
        pos.unmake_move(m, u);
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;   // beta cutoff
    }
    return best;
}

}  // namespace

SearchResult search_best(Position& pos, int depth) {
    SearchResult r;
    r.depth = depth;

    std::vector<Move> moves;
    generate_moves(pos, moves);

    if (moves.empty()) {
        r.score = in_check(pos) ? -MATE_SCORE : 0;
        return r;
    }

    // Track best move separately at the root — negamax alone returns only
    // the score. Any move is better than the sentinel, so first iteration
    // always sets best_move.
    int alpha = -INF;
    int best  = -INF;
    ++r.nodes;   // count the root
    for (Move m : moves) {
        UndoInfo u;
        pos.make_move(m, u);
        int score = -negamax(pos, depth - 1, -INF, -alpha, 1, r.nodes);
        pos.unmake_move(m, u);
        if (score > best) {
            best         = score;
            r.best_move  = m;
        }
        if (score > alpha) alpha = score;
    }
    r.score = best;
    return r;
}
