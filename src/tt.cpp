#include "tt.h"

TranspositionTable::TranspositionTable(size_t entries_pow2)
    : entries_(size_t{1} << entries_pow2),
      mask_((size_t{1} << entries_pow2) - 1) {}

void TranspositionTable::clear() {
    for (auto& e : entries_) {
        e = TTEntry{};
    }
}

// (key, depth, alpha, beta) is chess-engine TT convention; matches every reference implementation. Out-params grouped by convention too.
bool TranspositionTable::probe(uint64_t key, int depth, int alpha, int beta,  // NOLINT(bugprone-easily-swappable-parameters)
                                int& out_score, Move& out_move) const {       // NOLINT(bugprone-easily-swappable-parameters)
    const TTEntry& e = entries_[key & mask_];
    if (e.key != key || e.bound == TT_NONE) {
        return false;
    }

    // Move hint is always safe to hand back — it's just an ordering tip.
    out_move = e.move;

    // Score is only trustworthy if the stored search went at least as
    // deep as we're asking for; a shallower entry might have pruned
    // subtrees the current window needs to see.
    if (e.depth < depth) {
        return false;
    }

    switch (e.bound) {
        case TT_EXACT:
            out_score = e.score;
            return true;
        case TT_LOWER:                    // score is a lower bound → beta cutoff usable
            if (e.score >= beta) { out_score = e.score; return true; }
            return false;
        case TT_UPPER:                    // score is an upper bound → alpha cutoff usable
            if (e.score <= alpha) { out_score = e.score; return true; }
            return false;
        default:
            return false;
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — (key, depth, score, move, bound) is chess-engine TT convention; matches every reference implementation.
void TranspositionTable::store(uint64_t key, int depth, int score,
                                Move move, TTBound bound) {
    TTEntry& e = entries_[key & mask_];
    // Always replace — simplest policy. Depth-preferred replacement is a
    // known improvement but has enough edge cases (aging, PV protection)
    // that it belongs in a later pass.
    e.key   = key;
    e.score = score;
    e.move  = move;
    e.depth = static_cast<uint8_t>(depth);
    e.bound = static_cast<uint8_t>(bound);
}
