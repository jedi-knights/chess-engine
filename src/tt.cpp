#include "tt.h"

TranspositionTable::TranspositionTable(size_t entries_pow2)
    : buckets_(size_t{1} << (entries_pow2 - 1)),
      mask_((size_t{1} << (entries_pow2 - 1)) - 1) {}

void TranspositionTable::clear() {
    for (auto& b : buckets_) {
        for (auto& e : b.entries) {
            e = TTEntry{};
        }
    }
    generation_ = 0;
}

void TranspositionTable::new_search() {
    // Wraps in 6 bits — eviction tests generation for equality with
    // current, not ordering, so wrap is harmless. Worst case after
    // wrap: an entry whose original generation happened to equal the
    // new current-generation value looks "fresh" for one more search;
    // it still gets evicted the moment a real fresh store beats its
    // depth.
    generation_ = uint8_t((generation_ + 1) & 0x3F);
}

// (key, depth, alpha, beta) is chess-engine TT convention; matches every reference implementation. Out-params grouped by convention too.
bool TranspositionTable::probe(uint64_t key, int depth, int alpha, int beta,  // NOLINT(bugprone-easily-swappable-parameters)
                                int& out_score, Move& out_move) const {       // NOLINT(bugprone-easily-swappable-parameters)
    const TTBucket& b = buckets_[key & mask_];

    for (const TTEntry& e : b.entries) {
        if (e.key != key || e.bound() == TT_NONE) {
            continue;
        }

        // Move hint is always safe to hand back — it's just an ordering tip.
        out_move = e.move;

        // Score is only trustworthy if the stored search went at least
        // as deep as we're asking for; a shallower entry might have
        // pruned subtrees the current window needs to see.
        if (e.depth < depth) {
            return false;
        }

        switch (e.bound()) {
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
    return false;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — (key, depth, score, move, bound) is chess-engine TT convention; matches every reference implementation.
void TranspositionTable::store(uint64_t key, int depth, int score,
                                Move move, TTBound bound) {
    TTBucket& b = buckets_[key & mask_];

    auto write = [&](TTEntry& e) {
        e.key       = key;
        e.score     = score;
        e.move      = move;
        e.depth     = static_cast<uint8_t>(depth);
        e.gen_bound = static_cast<uint8_t>((generation_ << 2) | uint8_t(bound));
    };

    // Key match on either slot wins — refresh in place so repeated
    // stores on the same key stay in the same slot and don't split
    // the bucket's capacity across two copies of the same key.
    for (TTEntry& e : b.entries) {
        if (e.key == key) {
            write(e);
            return;
        }
    }

    // No key match. Evict the slot with the lowest priority, where:
    //   priority = depth - AGE_BONUS * (entry.gen != current_gen)
    // AGE_BONUS is chosen so a one-generation-old entry of depth D
    // looks like depth D-8 to the policy — a fresh store of depth
    // D-7 or better evicts it, but a fresh very-shallow store does
    // not (protecting the previous root iteration's deep work).
    // Empty slots (bound == TT_NONE, depth = 0, generation = 0)
    // score lowest against any real fresh entry, so they fill first
    // as the table warms.
    constexpr int AGE_BONUS = 8;
    auto priority = [this](const TTEntry& e) -> int {
        return int(e.depth) - AGE_BONUS * int(e.generation() != generation_);
    };

    TTEntry* victim = &b.entries[0];
    if (priority(b.entries[1]) < priority(b.entries[0])) {
        victim = &b.entries[1];
    }
    write(*victim);
}
