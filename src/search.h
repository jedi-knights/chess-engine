#pragma once
#include "position.h"
#include "types.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

struct SearchResult {
    Move     best_move = NULL_MOVE;   // NULL_MOVE only if no legal moves exist
    int      score     = 0;           // centipawns from side_to_move perspective
    int      depth     = 0;           // deepest fully-completed depth
    uint64_t nodes     = 0;           // cumulative nodes visited
    // Principal variation extracted by walking the TT after each iteration.
    // pv[0] == best_move; length is bounded by iteration depth. UCI wants
    // this in the `info ... pv ...` line so GUIs can display the engine's
    // predicted line of play.
    std::vector<Move> pv;
};

struct SearchLimits {
    int max_depth   = 64;   // hard ceiling — search never exceeds this
    int movetime_ms = 0;    // 0 = no time limit; stops mid-iteration when reached
    // External stop signal — polled on the same cadence as the movetime
    // deadline. Non-null enables UCI `stop` handling and `go infinite`;
    // caller owns the atomic and is responsible for its lifetime being
    // at least as long as the search.
    std::atomic<bool>* external_stop = nullptr;
};

using InfoCallback = std::function<void(const SearchResult&)>;

// Fixed-depth alpha-beta negamax. Kept as a primitive for tests that pin
// a specific depth's behavior; production searches should prefer
// search_iterative so time controls apply.
SearchResult search_best(Position& pos, int depth);

// Iterative deepening driver: search depth 1, 2, ..., min(limits.max_depth,
// natural stop). Calls `on_iter` after each COMPLETED depth so a UCI
// wrapper can emit per-iteration info lines. If movetime_ms is set and
// hit mid-iteration, the partial iteration is discarded and the last
// fully-completed result is returned — the any-time property.
SearchResult search_iterative(Position& pos, SearchLimits limits,
                              const InfoCallback& on_iter = nullptr);

// Reset the transposition table. UCI's `ucinewgame` calls this so entries
// from the previous game don't bias the new one; tests use it to isolate
// search behavior from prior-run state.
void clear_transposition_table();
