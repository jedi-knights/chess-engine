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
    // Queen on D1: material + PST + mobility (Q sees 17 squares, weight 1)
    // + tapered king PST. Actual pinned value below; update when eval
    // terms shift.
    const char* w_to_move = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";
    const char* b_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1";
    Position p1, p2;
    REQUIRE(p1.set_from_fen(w_to_move));
    REQUIRE(p2.set_from_fen(b_to_move));
    CHECK(evaluate(p1) == 904);
    CHECK(evaluate(p2) == -904);
    CHECK(evaluate(p1) == -evaluate(p2));
}

TEST_CASE("piece values + PST contribution on a fixed square") {
    // Kings on E1/E8 have PST=0 so they don't skew the material comparison.
    // Each case's extra piece sits on D1 (or D2 for pawn); expected =
    // piece_material + PST_piece[destination_square].
    // Expected values also fold in mobility (weighted by piece type) and
    // the tapered king PST — the raw material+PST numbers in the labels
    // are only the dominant term; the totals below are what evaluate()
    // actually returns after mobility + phase blending.
    struct Case { const char* fen; int expected; const char* label; };
    const Case cases[] = {
        {"4k3/8/8/8/8/8/8/3QK3 w - - 0 1",  904, "queen  (900 -  5)"},
        {"4k3/8/8/8/8/8/8/3RK3 w - - 0 1",  515, "rook   (500 +  5)"},
        {"4k3/8/8/8/8/8/8/3BK3 w - - 0 1",  330, "bishop (330 - 10)"},
        {"4k3/8/8/8/8/8/8/3NK3 w - - 0 1",  298, "knight (320 - 30)"},
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
    // (+100 + PST_PAWN[D2]= -20). Raw material+PST = 585. Rook on D1
    // also gets 3 mobility squares (weight 2) blended by phase, taking
    // the pinned total to 588.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/3P4/3RK3 w - - 0 1"));
    CHECK(evaluate(pos) == 588);
}

// --- PST-specific behaviors ---------------------------------------------

TEST_CASE("central knight scores higher than knight on the edge") {
    // "A knight on the rim is dim." PST_KNIGHT[A1] = -50; PST_KNIGHT[E4] = 20.
    Position central, edge;
    REQUIRE(central.set_from_fen("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1"));  // N on E4
    REQUIRE(edge.set_from_fen   ("4k3/8/8/8/8/8/8/N3K3 w - - 0 1"));   // N on A1
    CHECK(evaluate(central) > evaluate(edge));
    // Delta = PST_KNIGHT[E4] - PST_KNIGHT[A1] = 20 - (-50) = 70 PST,
    // plus mobility: E4 knight attacks 8 squares vs A1 knight's 2
    // (weight 4). Blended by phase (1 knight → mostly EG), pinned = 82.
    CHECK((evaluate(central) - evaluate(edge)) == 82);
}

TEST_CASE("middlegame: castled king scores higher than king in the center") {
    // Force full middlegame phase (24 = both sides at full non-pawn
    // material) so PST_KING_MG carries 100% of the king's weight and
    // isn't diluted by the EG table's center bonus. The two positions
    // share identical material — the F1 bishop moved to E2 and the G1
    // knight to F3 in both — so the only PST difference is the king's
    // square. Delta = MG[G1] - MG[E4] = 30 - (-40) = 70 in PST alone.
    // But the castled king blocks the F1 rook's east ray and the F3
    // knight's G1 square, costing 2 rook squares (weight 2 = -4) plus
    // 1 knight square (weight 4 = -4). Net pinned delta = 70 - 8 = 62.
    Position castled, exposed;
    REQUIRE(castled.set_from_fen("rnbqkbnr/8/8/8/8/5N2/4B3/RNBQ1RK1 w - - 0 1"));  // K on G1
    REQUIRE(exposed.set_from_fen("rnbqkbnr/8/8/8/4K3/5N2/4B3/RNBQ1R2 w - - 0 1"));  // K on E4
    CHECK(evaluate(castled) > evaluate(exposed));
    CHECK((evaluate(castled) - evaluate(exposed)) == 62);
}

TEST_CASE("endgame: king in center scores higher than king in corner") {
    // Pure K+K → phase 0 → 100% endgame weighting. This is the point of
    // tapered eval: PST_KING_EG rewards centralization (which the MG
    // table punishes), so the ordering flips relative to the MG test.
    Position center, corner;
    REQUIRE(center.set_from_fen("4k3/8/8/8/4K3/8/8/8 w - - 0 1"));  // K on E4
    REQUIRE(corner.set_from_fen("4k3/8/8/8/8/8/8/K7 w - - 0 1"));   // K on A1
    CHECK(evaluate(center) > evaluate(corner));
    // Delta = PST_KING_EG[E4] - PST_KING_EG[A1] = 40 - (-50) = 90.
    CHECK((evaluate(center) - evaluate(corner)) == 90);
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
