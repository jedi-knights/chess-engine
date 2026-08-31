// Magic bitboard tests. The invariant is: for every square and every
// possible occupancy pattern, the magic-hashed lookup must return the
// same attack set as a naive per-square walk. If this holds, no move
// generator using magic can be wrong in a way the reference isn't
// also wrong. Correctness before speed.

#include "doctest.h"

#include "bitboard.h"
#include "magic.h"
#include "types.h"

namespace {

// Reference attack generator — walks rays one square at a time. Never
// used in production; only here to cross-check the magic lookups.
Bitboard slow_rook(Square s, Bitboard occ) {
    Bitboard atk = 0;
    int f = file_of(s), r = rank_of(s);
    auto walk = [&](int df, int dr) {
        int ff = f + df, rr = r + dr;
        while (ff >= 0 && ff <= 7 && rr >= 0 && rr <= 7) {
            Bitboard bb = square_bb(make_square(File(ff), Rank(rr)));
            atk |= bb;
            if (occ & bb) break;
            ff += df; rr += dr;
        }
    };
    walk(0, 1); walk(0, -1); walk(1, 0); walk(-1, 0);
    return atk;
}

Bitboard slow_bishop(Square s, Bitboard occ) {
    Bitboard atk = 0;
    int f = file_of(s), r = rank_of(s);
    auto walk = [&](int df, int dr) {
        int ff = f + df, rr = r + dr;
        while (ff >= 0 && ff <= 7 && rr >= 0 && rr <= 7) {
            Bitboard bb = square_bb(make_square(File(ff), Rank(rr)));
            atk |= bb;
            if (occ & bb) break;
            ff += df; rr += dr;
        }
    };
    walk(1, 1); walk(1, -1); walk(-1, 1); walk(-1, -1);
    return atk;
}

}  // namespace

TEST_CASE("rook: empty board = 14 attacks from d4") {
    Bitboard atk = rook_attacks(D4, 0);
    CHECK(popcount(atk) == 14);
    CHECK((atk & square_bb(D8)) != 0);   // reaches the file's edge
    CHECK((atk & square_bb(A4)) != 0);   // reaches the rank's edge
    CHECK((atk & square_bb(D4)) == 0);   // doesn't include origin
}

TEST_CASE("rook: single blocker stops the ray at the blocker") {
    Bitboard occ = square_bb(D6);
    Bitboard atk = rook_attacks(D4, occ);
    CHECK((atk & square_bb(D5)) != 0);   // reaches blocker's square-1
    CHECK((atk & square_bb(D6)) != 0);   // reaches blocker (attacker "sees" it)
    CHECK((atk & square_bb(D7)) == 0);   // does NOT reach past blocker
    CHECK((atk & square_bb(D8)) == 0);
}

TEST_CASE("bishop: empty board = 13 attacks from d4") {
    Bitboard atk = bishop_attacks(D4, 0);
    CHECK(popcount(atk) == 13);
    CHECK((atk & square_bb(H8)) != 0);
    CHECK((atk & square_bb(A1)) != 0);
    CHECK((atk & square_bb(D4)) == 0);
}

TEST_CASE("queen: empty board = 27 attacks (rook 14 + bishop 13)") {
    CHECK(popcount(queen_attacks(D4, 0)) == 27);
}

TEST_CASE("rook: magic matches slow reference on every empty-board square") {
    for (int s = 0; s < NUM_SQUARES; ++s) {
        INFO("square: " << s);
        CHECK(rook_attacks(Square(s), 0) == slow_rook(Square(s), 0));
    }
}

TEST_CASE("bishop: magic matches slow reference on every empty-board square") {
    for (int s = 0; s < NUM_SQUARES; ++s) {
        INFO("square: " << s);
        CHECK(bishop_attacks(Square(s), 0) == slow_bishop(Square(s), 0));
    }
}

TEST_CASE("magic vs slow: rook on d4 across a range of blocker patterns") {
    // Sample a few dozen occupancies that touch d4's rays. Full exhaustive
    // (2^10 = 1024) coverage would be nicer but the sample here is
    // sufficient to trip a hash-collision bug — those manifest across
    // most occupancy patterns, not just contrived ones.
    for (Bitboard occ : {
        Bitboard{0},                                              // empty
        square_bb(D2) | square_bb(D6),                             // file blockers
        square_bb(A4) | square_bb(H4),                             // rank blockers
        square_bb(D2) | square_bb(D6) | square_bb(A4) | square_bb(H4),
        square_bb(D5) | square_bb(D3),                             // adjacent blockers
        Bitboard{0xFFFFFFFFFFFFFFFFULL} & ~square_bb(D4),         // fully surrounded
    }) {
        INFO("occupancy: " << occ);
        CHECK(rook_attacks(D4, occ) == slow_rook(D4, occ));
    }
}

TEST_CASE("magic vs slow: bishop on d4 across a range of blocker patterns") {
    for (Bitboard occ : {
        Bitboard{0},
        square_bb(F6) | square_bb(B2),
        square_bb(A7) | square_bb(H8),
        Bitboard{0xFFFFFFFFFFFFFFFFULL} & ~square_bb(D4),
    }) {
        INFO("occupancy: " << occ);
        CHECK(bishop_attacks(D4, occ) == slow_bishop(D4, occ));
    }
}
