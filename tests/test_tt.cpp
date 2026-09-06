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

TEST_CASE("two-slot bucket: two colliding keys coexist across both slots") {
    // With 2-slot buckets, two same-bucket colliding keys occupy both
    // slots — probing either key still hits. This replaces the previous
    // single-slot always-replace behavior.
    TranspositionTable tt(TEST_TT_BITS);
    tt.store(0x10, 5, 42, ::make_move(E2, E4), TT_EXACT);
    tt.store(0x20, 5, 99, ::make_move(D2, D4), TT_EXACT);   // same bucket as 0x10

    int  score = 0;
    Move move  = NULL_MOVE;
    CHECK(tt.probe(0x10, 5, -100, 100, score, move) == true);
    CHECK(score == 42);
    CHECK(move  == ::make_move(E2, E4));
    CHECK(tt.probe(0x20, 5, -100, 100, score, move) == true);
    CHECK(score == 99);
    CHECK(move  == ::make_move(D2, D4));
}

TEST_CASE("depth-preferred eviction: 3rd colliding store evicts the shallowest slot") {
    // Both bucket slots occupied by same-generation entries; a third
    // store on the same bucket evicts the lower-depth slot. Keys 0x10,
    // 0x20, 0x30 all hit bucket 0 (low 3 bits zero) once TEST_TT_BITS=4
    // resolves to 8 two-slot buckets.
    TranspositionTable tt(TEST_TT_BITS);
    tt.new_search();
    tt.store(0x10, /*depth=*/9, 42, ::make_move(E2, E4), TT_EXACT);   // deep
    tt.store(0x20, /*depth=*/2, 99, ::make_move(D2, D4), TT_EXACT);   // shallow
    tt.store(0x30, /*depth=*/5, 77, ::make_move(C2, C4), TT_EXACT);   // medium — evicts the depth=2 slot

    int  score = 0;
    Move move  = NULL_MOVE;
    CHECK(tt.probe(0x10, 9, -100, 100, score, move) == true);
    CHECK(score == 42);                                              // deep entry survives
    CHECK(tt.probe(0x20, 2, -100, 100, score, move) == false);       // shallow entry evicted
    CHECK(tt.probe(0x30, 5, -100, 100, score, move) == true);
    CHECK(score == 77);
}

TEST_CASE("generation aging: stale deep entries lose to fresh shallow ones after new_search bumps") {
    // A fresh depth-3 store beats a stale depth-9 entry because the
    // age penalty (8) drops the stale entry's effective priority to
    // 9 - 8 = 1. A fresh depth-2 store would NOT beat it (2 < 9-8=1
    // wait: 2 vs 1 — the fresh depth-2 would still beat it; the test
    // uses depth-3 to make the margin obvious).
    TranspositionTable tt(TEST_TT_BITS);
    tt.new_search();
    tt.store(0x10, /*depth=*/9, 42, ::make_move(E2, E4), TT_EXACT);
    tt.store(0x20, /*depth=*/9, 99, ::make_move(D2, D4), TT_EXACT);

    tt.new_search();   // both entries above are now one generation stale
    tt.store(0x30, /*depth=*/3, 77, ::make_move(C2, C4), TT_EXACT);

    int  score = 0;
    Move move  = NULL_MOVE;
    CHECK(tt.probe(0x30, 3, -100, 100, score, move) == true);
    CHECK(score == 77);
    // One of the two stale entries got evicted. Both had equal
    // (depth, generation) so tie-breaking is implementation defined
    // (first-slot preference); we just check exactly one survives.
    bool hit10 = tt.probe(0x10, 9, -100, 100, score, move);
    bool hit20 = tt.probe(0x20, 9, -100, 100, score, move);
    CHECK((hit10 != hit20));
}

TEST_CASE("depth-preferred eviction protects the deepest entry under repeated shallow pressure") {
    // Ethereal/Stockfish-style guarantee: the deepest entry in a
    // bucket survives repeated shallow leaf-store pressure. Store one
    // deep entry, then many shallow stores at the same bucket; the
    // deep entry must remain probeable throughout.
    TranspositionTable tt(TEST_TT_BITS);
    tt.new_search();
    tt.store(0x10, /*depth=*/12, 42, ::make_move(E2, E4), TT_EXACT);

    // 20 shallow stores at same-bucket keys. Each pair fills slot 1
    // then evicts itself; the deep entry in slot 0 is never touched.
    for (int i = 0; i < 20; ++i) {
        uint64_t shallow_key = 0x20 + (uint64_t(i) << 8);   // same bucket, distinct keys
        tt.store(shallow_key, /*depth=*/1, 0, NULL_MOVE, TT_EXACT);
    }

    int  score = 0;
    Move move  = NULL_MOVE;
    CHECK(tt.probe(0x10, 12, -100, 100, score, move) == true);
    CHECK(score == 42);
    CHECK(move  == ::make_move(E2, E4));
}

TEST_CASE("table size is a power of two matching the constructor argument") {
    TranspositionTable tt(TEST_TT_BITS);
    CHECK(tt.size() == TEST_TT_SIZE);
}
