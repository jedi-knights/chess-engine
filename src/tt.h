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
    uint64_t key       = 0;
    int      score     = 0;
    Move     move      = NULL_MOVE;
    uint8_t  depth     = 0;
    // Packed: high 6 bits = generation, low 2 bits = TTBound. Packing
    // keeps the entry at 16 bytes (no padding), so two entries fit in
    // one bucket / half a cache line.
    uint8_t  gen_bound = 0;

    TTBound bound()      const { return TTBound(gen_bound & 0x3); }
    uint8_t generation() const { return uint8_t(gen_bound >> 2); }
};

// Two-slot bucket. Both slots participate in a depth-preferred + aging
// replacement policy: a key match wins outright, otherwise the slot with
// the lowest (depth - age_penalty) is evicted. Keeps deep entries safe
// from shallow leaf/qsearch stores that would thrash a single-slot
// table. Two slots fit in 32 bytes (half a cache line), so probe and
// prefetch still touch exactly one line.
struct TTBucket {
    static constexpr int SLOTS = 2;
    TTEntry entries[SLOTS];
};

// Power-of-two sized direct-mapped transposition table. Not thread-safe;
// single-threaded engine.
class TranspositionTable {
public:
    // entries_pow2 sets total ENTRIES (not buckets), preserving the
    // caller-visible size semantics from the original single-slot
    // layout. Internally we allocate 2^(entries_pow2 - 1) buckets, each
    // holding TTBucket::SLOTS entries. Requires entries_pow2 >= 1.
    explicit TranspositionTable(size_t entries_pow2 = 20);

    void clear();

    // Bump the per-search generation counter. Called once at the top of
    // search_iterative so entries stored by earlier `go` calls preferentially
    // age out as new stores arrive. The bump is a wrapping increment in
    // 6 bits (values 0-63) — wrap does not corrupt eviction because the
    // freshness test is equality against current, not ordering.
    void new_search();

    // Look up `key`. If a usable score exists for the given (depth, alpha,
    // beta) window, populates `out_score` and returns true. `out_move` is
    // populated on any key match (even if the score isn't usable) so it
    // can seed move ordering. Scans both bucket slots.
    bool probe(uint64_t key, int depth, int alpha, int beta,
               int& out_score, Move& out_move) const;

    void store(uint64_t key, int depth, int score, Move move, TTBound bound);

    // Issue a non-blocking cache-line prefetch for the bucket that `key`
    // maps to. Callers invoke this right after `make_move` so the child
    // node's TT bucket starts loading from memory in parallel with the
    // ~100+ cycles of setup work (node counter, stop check, in_check
    // computation, static eval) before the actual probe runs. On a hit
    // this hides most or all of the ~200-cycle main-memory latency.
    void prefetch(uint64_t key) const {
        __builtin_prefetch(&buckets_[key & mask_], /*rw=*/0, /*locality=*/3);
    }

    // Total entries (buckets * slots), preserving the semantic of the
    // constructor's entries_pow2 argument.
    size_t size() const { return buckets_.size() * TTBucket::SLOTS; }

private:
    std::vector<TTBucket> buckets_;
    size_t                mask_;             // (buckets_.size() - 1); key & mask_ is the bucket index
    uint8_t               generation_ = 0;   // 6 low bits; bumped by new_search(), wraps at 64
};
