#pragma once
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Alpha-beta produces three flavors of score for a completed subtree:
//   EXACT: the search saw every move; the returned score is the true value.
//   LOWER: a beta cutoff — the true value is >= score (fail-high).
//   UPPER: no move raised alpha — the true value is <= score (fail-low).
// The bound determines when a stored entry is usable on later probes.
enum TTBound : uint8_t {
    TT_NONE  = 0,
    TT_EXACT = 1,
    TT_LOWER = 2,
    TT_UPPER = 3,
};

struct TTEntry {
    uint64_t key   = 0;
    int      score = 0;
    Move     move  = NULL_MOVE;
    uint8_t  depth = 0;
    uint8_t  bound = TT_NONE;
};

// Power-of-two sized direct-mapped transposition table. Collisions are
// resolved by "always replace" — simple, and the sizing (1M+ entries)
// keeps aliasing negligible in typical searches. Not thread-safe; single-
// threaded engine.
class TranspositionTable {
public:
    explicit TranspositionTable(size_t entries_pow2 = 20);  // default 2^20 = ~1M

    void clear();

    // Look up `key`. If a usable score exists for the given (depth, alpha,
    // beta) window, populates `out_score` and returns true. `out_move` is
    // populated on any key match (even if the score isn't usable) so it
    // can seed move ordering.
    bool probe(uint64_t key, int depth, int alpha, int beta,
               int& out_score, Move& out_move) const;

    void store(uint64_t key, int depth, int score, Move move, TTBound bound);

    size_t size() const { return entries_.size(); }

private:
    std::vector<TTEntry> entries_;
    size_t               mask_;   // (size - 1); key & mask_ is the slot index
};
