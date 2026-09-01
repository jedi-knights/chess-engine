// Transposition-table tests. The TT is a pure data structure — no chess
// knowledge — so these are ordinary hashtable tests plus the alpha-beta
// bound-usability rules. The one thing worth remembering when reading:
// TT semantics distinguish between "found a matching entry" (returns the
// move hint always) and "the stored score is usable in this window"
// (only when the bound + score allow it).

#include "doctest.h"

#include "tt.h"
#include "types.h"

// Small dedicated table for these tests — 2^4 = 16 entries. Small enough
// to make deliberate collisions cheap to construct, big enough to be a
// real hashtable rather than a special case.
static constexpr int  TEST_TT_BITS = 4;
static constexpr size_t TEST_TT_SIZE = size_t{1} << TEST_TT_BITS;

TEST_CASE("probe on an empty TT misses") {
    TranspositionTable tt(TEST_TT_BITS);
    int score = -1;
    Move move = ::make_move(A1, H8);   // sentinel — should be untouched
    CHECK(tt.probe(0xDEADBEEF, 3, -100, 100, score, move) == false);
    CHECK(move == ::make_move(A1, H8));   // no key match → move hint left alone
}

TEST_CASE("store then probe returns the stored score for an EXACT entry") {
    TranspositionTable tt(TEST_TT_BITS);
    tt.store(0xC0FFEE, /*depth=*/5, /*score=*/42, ::make_move(E2, E4), TT_EXACT);

    int score = 0;
    Move move = NULL_MOVE;
    CHECK(tt.probe(0xC0FFEE, /*depth=*/5, -100, 100, score, move) == true);
    CHECK(score == 42);
    CHECK(move  == ::make_move(E2, E4));
}

TEST_CASE("probe with wrong key misses") {
    TranspositionTable tt(TEST_TT_BITS);
    tt.store(0xC0FFEE, 5, 42, ::make_move(E2, E4), TT_EXACT);
    int score = 0;
    Move move = NULL_MOVE;
    CHECK(tt.probe(0xBADCAB1E, 5, -100, 100, score, move) == false);
}

TEST_CASE("insufficient-depth probe still hands back the move hint") {
    // Move ordering benefits from a hash-hit at ANY depth, even one
    // shallower than requested — the score can't be trusted, but the
    // move is a fine "try this first" suggestion.
    TranspositionTable tt(TEST_TT_BITS);
    tt.store(0xC0FFEE, /*stored_depth=*/2, 42, ::make_move(E2, E4), TT_EXACT);

    int  score = 0;
    Move move  = NULL_MOVE;
    // Ask for depth 5 — only depth 2 is stored → score not usable, but
    // move should come back.
    CHECK(tt.probe(0xC0FFEE, /*depth=*/5, -100, 100, score, move) == false);
    CHECK(move == ::make_move(E2, E4));
}

TEST_CASE("bound rules: EXACT is always usable, LOWER needs score>=beta, UPPER needs score<=alpha") {
    TranspositionTable tt(TEST_TT_BITS);

    // Reused across the sub-cases with distinct keys so they don't
    // collide in the tiny table.
    tt.store(0x1, 5, 50, NULL_MOVE, TT_EXACT);
    tt.store(0x2, 5, 50, NULL_MOVE, TT_LOWER);   // true score >= 50
    tt.store(0x3, 5, 50, NULL_MOVE, TT_UPPER);   // true score <= 50

    int score = 0;
    Move move = NULL_MOVE;

    SUBCASE("EXACT is usable in any window") {
        CHECK(tt.probe(0x1, 5, -100, 100, score, move) == true);
        CHECK(score == 50);
    }

    SUBCASE("LOWER usable only when stored score >= beta") {
        CHECK(tt.probe(0x2, 5, -100, /*beta=*/40, score, move) == true);   // 50 >= 40 → cutoff
        CHECK(score == 50);
        CHECK(tt.probe(0x2, 5, -100, /*beta=*/60, score, move) == false);  // 50 < 60 → not usable
    }

    SUBCASE("UPPER usable only when stored score <= alpha") {
        CHECK(tt.probe(0x3, 5, /*alpha=*/60, 100, score, move) == true);   // 50 <= 60 → cutoff
        CHECK(score == 50);
        CHECK(tt.probe(0x3, 5, /*alpha=*/40, 100, score, move) == false);  // 50 > 40 → not usable
    }
}

TEST_CASE("clear wipes all entries") {
    TranspositionTable tt(TEST_TT_BITS);
    tt.store(0xC0FFEE, 5, 42, ::make_move(E2, E4), TT_EXACT);

    int  score = 0;
    Move move  = NULL_MOVE;
    REQUIRE(tt.probe(0xC0FFEE, 5, -100, 100, score, move) == true);

    tt.clear();
    CHECK(tt.probe(0xC0FFEE, 5, -100, 100, score, move) == false);
}

TEST_CASE("collision: different key in the same slot doesn't cross-contaminate") {
    // With a 16-slot table (mask = 0xF), keys 0x10 and 0x20 both hash
    // to slot 0. Store one, then check the other doesn't spuriously hit.
    TranspositionTable tt(TEST_TT_BITS);
    REQUIRE((0x10 & (TEST_TT_SIZE - 1)) == (0x20 & (TEST_TT_SIZE - 1)));

    tt.store(0x10, 5, 42, ::make_move(E2, E4), TT_EXACT);

    int  score = 0;
    Move move  = NULL_MOVE;
    CHECK(tt.probe(0x20, 5, -100, 100, score, move) == false);
}

TEST_CASE("collision with always-replace: new store overwrites the previous slot entry") {
    // Same slot collision → the second store wins (always-replace policy).
    // A subsequent probe of the FIRST key must miss.
    TranspositionTable tt(TEST_TT_BITS);
    tt.store(0x10, 5, 42, ::make_move(E2, E4), TT_EXACT);
    tt.store(0x20, 5, 99, ::make_move(D2, D4), TT_EXACT);   // same slot as 0x10

    int  score = 0;
    Move move  = NULL_MOVE;
    CHECK(tt.probe(0x10, 5, -100, 100, score, move) == false);   // evicted
    CHECK(tt.probe(0x20, 5, -100, 100, score, move) == true);    // still there
    CHECK(score == 99);
    CHECK(move  == ::make_move(D2, D4));
}

TEST_CASE("table size is a power of two matching the constructor argument") {
    TranspositionTable tt(TEST_TT_BITS);
    CHECK(tt.size() == TEST_TT_SIZE);
}
