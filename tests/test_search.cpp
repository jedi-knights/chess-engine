// Search tests. The tricky part of any negamax test is picking positions
// where the "right" move is unambiguous — otherwise we're testing a
// specific tie-breaking heuristic rather than the search itself. These
// cases use tactical setups where any correct engine picks the same move.

#include "doctest.h"
#include "support.h"

#include "movegen.h"
#include "position.h"
#include "search.h"

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
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    SearchResult d1 = search_best(pos, 1);
    SearchResult d2 = search_best(pos, 2);
    SearchResult d3 = search_best(pos, 3);
    CHECK(d1.nodes < d2.nodes);
    CHECK(d2.nodes < d3.nodes);
}
