// Bitboard helper tests: popcount, lsb, pop_lsb, square_bb, pretty.

#include "doctest.h"

#include "bitboard.h"
#include "types.h"

#include <algorithm>
#include <string>

TEST_CASE("popcount") {
    CHECK(popcount(0)         == 0);
    CHECK(popcount(1)         == 1);
    CHECK(popcount(0xFFULL)   == 8);
    CHECK(popcount(~0ULL)     == 64);
}

TEST_CASE("square_bb sets exactly one bit at the right index") {
    for (int i = 0; i < NUM_SQUARES; ++i) {
        Bitboard bb = square_bb(Square(i));
        CHECK(popcount(bb) == 1);
        CHECK(bb == (Bitboard(1) << i));
    }
}

TEST_CASE("lsb identifies the least-significant set bit as a Square") {
    CHECK(lsb(square_bb(A1)) == A1);
    CHECK(lsb(square_bb(H8)) == H8);
    CHECK(lsb(square_bb(E4)) == E4);
    CHECK(lsb(square_bb(A1) | square_bb(H8)) == A1);   // lower index wins
}

TEST_CASE("pop_lsb returns the lsb AND clears it from the input") {
    Bitboard bb = square_bb(A1) | square_bb(E4) | square_bb(H8);
    CHECK(pop_lsb(bb) == A1);
    CHECK(pop_lsb(bb) == E4);
    CHECK(pop_lsb(bb) == H8);
    CHECK(bb == 0);
}

TEST_CASE("iterating a bitboard with pop_lsb visits each square exactly once") {
    Bitboard bb = square_bb(A1) | square_bb(D4) | square_bb(H8);
    int count = 0;
    while (bb) {
        pop_lsb(bb);
        ++count;
    }
    CHECK(count == 3);
}

// --- pretty(Bitboard) — debug renderer ---------------------------------
// Not on the hot path — only used when a human wants to eyeball a
// bitboard. But it's still code that should be exercised so we notice
// if it ever segfaults or misrenders.

TEST_CASE("pretty(0) renders all squares as '.'") {
    std::string s = pretty(Bitboard(0));
    CHECK(std::count(s.begin(), s.end(), 'X') == 0);
    CHECK(std::count(s.begin(), s.end(), '.') == 64);
}

TEST_CASE("pretty(single-square) renders exactly one 'X'") {
    // Every square in isolation must produce exactly one 'X' — pins down
    // the mapping between a bitboard bit and its grid cell.
    for (int i = 0; i < NUM_SQUARES; ++i) {
        Bitboard bb = square_bb(Square(i));
        std::string s = pretty(bb);
        INFO("square: " << i);
        CHECK(std::count(s.begin(), s.end(), 'X') == 1);
        CHECK(std::count(s.begin(), s.end(), '.') == 63);
    }
}

TEST_CASE("pretty(~0) renders all squares as 'X'") {
    std::string s = pretty(Bitboard{~0ULL});
    CHECK(std::count(s.begin(), s.end(), 'X') == 64);
    CHECK(std::count(s.begin(), s.end(), '.') == 0);
}

TEST_CASE("pretty layout includes rank labels 1-8 and file labels a-h") {
    // Sanity that the human-readable coordinate axes actually appear —
    // catches regressions where the renderer drops labels.
    std::string s = pretty(Bitboard(0));
    for (char rank = '1'; rank <= '8'; ++rank) {
        INFO("rank label: " << rank);
        CHECK(s.find(rank) != std::string::npos);
    }
    CHECK(s.find("a   b   c   d   e   f   g   h") != std::string::npos);
}
