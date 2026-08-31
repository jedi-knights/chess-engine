// Bitboard helper tests: popcount, lsb, pop_lsb, square_bb.

#include "doctest.h"

#include "bitboard.h"
#include "types.h"

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
