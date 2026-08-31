// Search tests. The tricky part of any negamax test is picking positions
// where the "right" move is unambiguous — otherwise we're testing a
// specific tie-breaking heuristic rather than the search itself. These
// cases use tactical setups where any correct engine picks the same move.

#include "doctest.h"
#include "support.h"

#include "movegen.h"
#include "position.h"
#include "search.h"

#include <chrono>
#include <cstdlib>
#include <vector>

TEST_CASE("search returns a legal move for the starting position") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchResult r = search_best(pos, 2);
    CHECK(r.best_move != NULL_MOVE);
    CHECK(r.nodes > 0);

    // The chosen move must be one the generator would emit — anything else
    // means search invented an illegal move.
    std::vector<Move> legal;
    generate_moves(pos, legal);
    CHECK(contains_move(legal, r.best_move));
}

TEST_CASE("search returns NULL_MOVE when there are no legal moves") {
    // Back-rank mate — same position used in the legality tests.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/6PP/5r1K w - - 0 1"));
    SearchResult r = search_best(pos, 3);
    CHECK(r.best_move == NULL_MOVE);
    // Being checkmated must score negatively (the losing side's perspective).
    CHECK(r.score < 0);
}

TEST_CASE("search returns NULL_MOVE and score 0 on stalemate") {
    Position pos;
    REQUIRE(pos.set_from_fen("8/8/8/8/8/4k3/4p3/4K3 w - - 0 1"));
    SearchResult r = search_best(pos, 3);
    CHECK(r.best_move == NULL_MOVE);
    CHECK(r.score == 0);
}

TEST_CASE("search picks the free capture when material is at stake") {
    // White knight on b5 attacks black queen on d4 (undefended). Any
    // correct engine plays Nxd4 — the only capture available out of the
    // ~15 knight moves. Depth 2 is enough: capture is one ply, no recapture.
    // Score is the *absolute* eval after the capture (white K+N, black K)
    // = +320, not +900 (which would be the *delta* — a common trap).
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/1N6/3q4/8/8/4K3 w - - 0 1"));
    SearchResult r = search_best(pos, 2);
    CHECK(r.best_move == ::make_move(B5, D4));
    CHECK(r.score > 0);   // holding material after the capture
}

TEST_CASE("search finds a unique mate-in-1") {
    // Rook-mate corner setup with a UNIQUE mating move. Black king a8,
    // white king b6 (covers a7 and b7/b8, boxing the black king), white
    // rook h1. Rh8# is the only checking move (the rook has to reach the
    // 8th rank; h-file's only 8th-rank square not blocked by our own king
    // is h8), and it's mate because every black king escape is covered.
    //
    // Prior fixture "7k/8/6KQ/8/8/8/8/8" was ambiguous: black king was
    // already in check with all escapes covered, so any white move that
    // didn't unblock the check was mate — test asserted one specific move
    // out of many equally-mating options.
    Position pos;
    REQUIRE(pos.set_from_fen("k7/8/1K6/8/8/8/8/7R w - - 0 1"));
    SearchResult r = search_best(pos, 2);
    CHECK(r.best_move == ::make_move(H1, H8));
    CHECK(r.score > 50000);   // mate score family — well above material ranges
}

TEST_CASE("search node count grows with depth") {
    // Not a correctness assertion so much as a smoke test: deeper searches
    // should visit strictly more nodes on a non-terminal position.
    // TT must be cleared — a warm table from a prior test can make a
    // depth-1 search visit fewer nodes than depth-2 in isolation would.
    clear_transposition_table();
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchResult d1 = search_best(pos, 1);
    clear_transposition_table();
    SearchResult d2 = search_best(pos, 2);
    clear_transposition_table();
    SearchResult d3 = search_best(pos, 3);
    CHECK(d1.nodes < d2.nodes);
    CHECK(d2.nodes < d3.nodes);
}

// --- Iterative deepening -----------------------------------------------

TEST_CASE("iterative deepening reaches the requested max_depth") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchLimits limits;
    limits.max_depth = 3;
    SearchResult r = search_iterative(pos, limits);
    CHECK(r.depth == 3);
    CHECK(r.best_move != NULL_MOVE);
}

TEST_CASE("iterative deepening fires callback once per completed depth") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchLimits limits;
    limits.max_depth = 4;

    std::vector<int> depths_seen;
    search_iterative(pos, limits, [&](const SearchResult& r) {
        depths_seen.push_back(r.depth);
    });

    REQUIRE(depths_seen.size() == 4);
    for (int i = 0; i < 4; ++i) CHECK(depths_seen[i] == i + 1);
}

TEST_CASE("iterative deepening accumulates nodes across iterations") {
    // Each iteration re-searches from the root, so cumulative node count
    // must strictly increase across callbacks. TT reuse would still leave
    // this monotone (each depth touches new deeper subtrees), but clearing
    // it isolates from any warm state left by other tests.
    clear_transposition_table();
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchLimits limits;
    limits.max_depth = 3;

    std::vector<uint64_t> node_snapshots;
    search_iterative(pos, limits, [&](const SearchResult& r) {
        node_snapshots.push_back(r.nodes);
    });
    REQUIRE(node_snapshots.size() == 3);
    CHECK(node_snapshots[0] < node_snapshots[1]);
    CHECK(node_snapshots[1] < node_snapshots[2]);
}

TEST_CASE("iterative deepening returns at least the depth-1 result") {
    // Even with a 1 ms movetime — likely far too short to finish depth 1
    // under ASan — the driver must still return a legal move so UCI's
    // "bestmove is required" contract holds.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchLimits limits;
    limits.max_depth   = 8;
    limits.movetime_ms = 1;   // absurdly short
    SearchResult r = search_iterative(pos, limits);
    CHECK(r.best_move != NULL_MOVE);
    CHECK(r.depth >= 1);
}

// --- Quiescence -------------------------------------------------------

TEST_CASE("quiescence resolves the horizon on a defended-piece capture") {
    // White queen on e4 is attacked by a black knight on d6. The knight
    // is defended by the pawn on c7. Without quiescence, a depth-1 search
    // would see Qxd6 as "free" (+320 for winning the knight) because it
    // stops evaluating before black's cxd6 recapture. With quiescence,
    // depth 1 resolves the exchange: black recaptures, white loses queen
    // for knight+pawn, and Qxd6 scores as losing.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/2p5/3n4/8/4Q3/8/8/4K3 w - - 0 1"));
    SearchResult r = search_best(pos, 1);
    // Correct play: move the queen out of danger. Qxd6 must NOT be picked.
    CHECK_FALSE(r.best_move == ::make_move(E4, D6));
    // Best move should keep white ahead — white starts with K+Q vs K+N+P
    // (900 vs 420), so a queen escape leaves the score near +480.
    CHECK(r.score > 200);
}

TEST_CASE("quiescence stabilizes score across depths on a quiet position") {
    // Startpos evaluates to 0 statically. Without quiescence, deeper
    // searches oscillated around 0 (±100 at successive plies — the
    // horizon caught a pawn move mid-exchange). With quiescence, the
    // eval at every completed depth should stay near 0 because leaves
    // resolve any speculative captures rather than banking their gain.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchLimits limits;
    limits.max_depth = 4;
    std::vector<int> scores;
    search_iterative(pos, limits, [&](const SearchResult& r) {
        scores.push_back(r.score);
    });
    REQUIRE(scores.size() == 4);
    for (int s : scores) {
        INFO("iteration score: " << s);
        CHECK(std::abs(s) < 50);   // near-zero, no oscillation
    }
}

TEST_CASE("quiescence still detects mate at qsearch leaves") {
    // Same back-rank mate as the negamax test. Terminal detection has to
    // work at qsearch's terminal too, otherwise a mated leaf would
    // stand-pat to a material eval and the engine would think it's fine.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/6PP/5r1K w - - 0 1"));
    SearchResult r = search_best(pos, 3);
    CHECK(r.best_move == NULL_MOVE);
    CHECK(r.score < 0);
}

// --- Transposition table ------------------------------------------------

TEST_CASE("TT: warm table reduces node count on a re-search of the same position") {
    // Search once from cold, once from warm. The warm search must find
    // the same best move at the same score with fewer (or equal) nodes.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));

    clear_transposition_table();
    SearchResult cold = search_best(pos, 4);
    SearchResult warm = search_best(pos, 4);   // TT populated by `cold`

    CHECK(warm.best_move == cold.best_move);
    CHECK(warm.score     == cold.score);
    CHECK(warm.nodes     <= cold.nodes);
}

TEST_CASE("TT: clear resets prior state (same result cold-vs-cold)") {
    // Two cold searches must be identical — TT clear is the reset signal
    // UCI's `ucinewgame` relies on to prevent stale scores biasing a new game.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));

    clear_transposition_table();
    SearchResult first = search_best(pos, 3);
    clear_transposition_table();
    SearchResult second = search_best(pos, 3);

    CHECK(first.best_move == second.best_move);
    CHECK(first.score     == second.score);
    CHECK(first.nodes     == second.nodes);
}

TEST_CASE("iterative deepening honors movetime_ms as an upper bound") {
    // A generous cap (2 seconds) that any startpos search finishes well
    // within — the point of the assertion is that we don't run away past
    // the deadline, not that we barely make it.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchLimits limits;
    limits.max_depth   = 64;    // would take forever without the time cap
    limits.movetime_ms = 500;

    auto start = std::chrono::steady_clock::now();
    SearchResult r = search_iterative(pos, limits);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    CHECK(r.best_move != NULL_MOVE);
    // Slack for the 1024-node polling granularity and doctest overhead.
    CHECK(elapsed < 2000);
}
