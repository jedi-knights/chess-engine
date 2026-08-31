// Attack-table tests: KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS.
// Movegen tests exercise these indirectly; the tests here isolate the
// tables so a bad entry surfaces here first with a clear "table" symptom
// rather than as a mysterious movegen miscount downstream.

#include "doctest.h"

#include "attacks.h"
#include "bitboard.h"
#include "types.h"

TEST_CASE("KNIGHT_ATTACKS from a1: exactly {b3, c2}") {
    Bitboard a = KNIGHT_ATTACKS[A1];
    CHECK(popcount(a) == 2);
    CHECK((a & square_bb(B3)) != 0);
    CHECK((a & square_bb(C2)) != 0);
}

TEST_CASE("KNIGHT_ATTACKS from h8: exactly {g6, f7}") {
    Bitboard a = KNIGHT_ATTACKS[H8];
    CHECK(popcount(a) == 2);
    CHECK((a & square_bb(G6)) != 0);
    CHECK((a & square_bb(F7)) != 0);
}

TEST_CASE("KNIGHT_ATTACKS from e4: all 8 L-moves present") {
    Bitboard a = KNIGHT_ATTACKS[E4];
    CHECK(popcount(a) == 8);
    for (Square s : {D2, F2, C3, G3, C5, G5, D6, F6}) {
        INFO("target square index " << int(s));
        CHECK((a & square_bb(s)) != 0);
    }
}

TEST_CASE("KING_ATTACKS from a1: exactly {a2, b1, b2}") {
    Bitboard a = KING_ATTACKS[A1];
    CHECK(popcount(a) == 3);
    CHECK((a & square_bb(A2)) != 0);
    CHECK((a & square_bb(B1)) != 0);
    CHECK((a & square_bb(B2)) != 0);
}

TEST_CASE("KING_ATTACKS from e4: all 8 adjacent squares present") {
    Bitboard a = KING_ATTACKS[E4];
    CHECK(popcount(a) == 8);
    for (Square s : {D3, E3, F3, D4, F4, D5, E5, F5}) {
        CHECK((a & square_bb(s)) != 0);
    }
}

TEST_CASE("PAWN_ATTACKS[WHITE] goes forward-diagonal; [BLACK] backward-diagonal") {
    // White pawn at e4 attacks d5, f5. Black pawn at e4 attacks d3, f3.
    Bitboard wa = PAWN_ATTACKS[WHITE][E4];
    Bitboard ba = PAWN_ATTACKS[BLACK][E4];
    CHECK(wa == (square_bb(D5) | square_bb(F5)));
    CHECK(ba == (square_bb(D3) | square_bb(F3)));
}

TEST_CASE("PAWN_ATTACKS respects file bounds — no wraparound") {
    // White pawn at a2 attacks only b3 (never h3).
    CHECK(PAWN_ATTACKS[WHITE][A2] == square_bb(B3));
    // Black pawn at h7 attacks only g6 (never a6).
    CHECK(PAWN_ATTACKS[BLACK][H7] == square_bb(G6));
}
