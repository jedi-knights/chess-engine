// Evaluation tests. Eval is (material + piece-square table) from side-to-
// move perspective, so tests here pin (a) material values, (b) PST
// contributions for each piece type, (c) the perspective-flip convention,
// (d) symmetry: mirror positions with matched material+PST score 0.

#include "doctest.h"

#include "eval.h"
#include "position.h"

TEST_CASE("startpos evaluates to 0 — every piece has a color-symmetric counterpart") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    CHECK(evaluate(pos) == 0);
}

TEST_CASE("empty-material position (kings on e-file) evaluates to 0") {
    // Kings on E1 / E8 both look up PST_KING[E1] = 0, so no PST asymmetry.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    CHECK(evaluate(pos) == 0);
}

TEST_CASE("perspective: same board, different side-to-move flips the sign") {
    // Queen on D1: material 900 + PST_QUEEN[D1]=-5 → 895.
    const char* w_to_move = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";
    const char* b_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1";
    Position p1, p2;
    REQUIRE(p1.set_from_fen(w_to_move));
    REQUIRE(p2.set_from_fen(b_to_move));
    CHECK(evaluate(p1) == 895);
    CHECK(evaluate(p2) == -895);
    CHECK(evaluate(p1) == -evaluate(p2));
}

TEST_CASE("piece values + PST contribution on a fixed square") {
    // Kings on E1/E8 have PST=0 so they don't skew the material comparison.
    // Each case's extra piece sits on D1 (or D2 for pawn); expected =
    // piece_material + PST_piece[destination_square].
    struct Case { const char* fen; int expected; const char* label; };
    const Case cases[] = {
        {"4k3/8/8/8/8/8/8/3QK3 w - - 0 1",  895, "queen  (900 -  5)"},
        {"4k3/8/8/8/8/8/8/3RK3 w - - 0 1",  505, "rook   (500 +  5)"},
        {"4k3/8/8/8/8/8/8/3BK3 w - - 0 1",  320, "bishop (330 - 10)"},
        {"4k3/8/8/8/8/8/8/3NK3 w - - 0 1",  290, "knight (320 - 30)"},
        {"4k3/8/8/8/8/8/3P4/4K3 w - - 0 1",  80, "pawn   (100 - 20)"},
    };
    for (const auto& c : cases) {
        Position pos;
        REQUIRE(pos.set_from_fen(c.fen));
        INFO("piece: " << std::string(c.label));
        CHECK(evaluate(pos) == c.expected);
    }
}

TEST_CASE("material differences aggregate linearly (with PST)") {
    // White has extra rook (+500 + PST_ROOK[D1]= +5) and extra pawn
    // (+100 + PST_PAWN[D2]= -20). Total = 500 + 5 + 100 - 20 = 585.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/3P4/3RK3 w - - 0 1"));
    CHECK(evaluate(pos) == 585);
}

// --- PST-specific behaviors ---------------------------------------------

TEST_CASE("central knight scores higher than knight on the edge") {
    // "A knight on the rim is dim." PST_KNIGHT[A1] = -50; PST_KNIGHT[E4] = 20.
    Position central, edge;
    REQUIRE(central.set_from_fen("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1"));  // N on E4
    REQUIRE(edge.set_from_fen   ("4k3/8/8/8/8/8/8/N3K3 w - - 0 1"));   // N on A1
    CHECK(evaluate(central) > evaluate(edge));
    // Delta = PST_KNIGHT[E4] - PST_KNIGHT[A1] = 20 - (-50) = 70.
    CHECK((evaluate(central) - evaluate(edge)) == 70);
}

TEST_CASE("castled king scores higher than king in the center") {
    // Middlegame PST rewards g1 / b1 (castled positions), penalizes any
    // advance. PST_KING[G1] = +30, PST_KING[E4] = -40 → delta 70.
    Position castled, exposed;
    REQUIRE(castled.set_from_fen("4k3/8/8/8/8/8/8/6K1 w - - 0 1"));  // K on G1
    REQUIRE(exposed.set_from_fen("4k3/8/8/8/4K3/8/8/8 w - - 0 1"));  // K on E4
    CHECK(evaluate(castled) > evaluate(exposed));
    CHECK((evaluate(castled) - evaluate(exposed)) == 70);
}

TEST_CASE("PST is mirrored for black") {
    // White knight on E4 and black knight on E5 are geometrically equivalent
    // (each is centralized in their own half). Both should contribute the
    // same PST bonus via the rank-flip lookup, so scores match sign and
    // magnitude when kings are placed symmetrically.
    Position wknight, bknight;
    REQUIRE(wknight.set_from_fen("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1"));  // N on E4, white to move
    REQUIRE(bknight.set_from_fen("4k3/8/8/4n3/8/8/8/4K3 b - - 0 1"));  // n on E5, black to move
    CHECK(evaluate(wknight) == evaluate(bknight));
}
