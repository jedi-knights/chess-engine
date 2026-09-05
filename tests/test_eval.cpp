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
    // + tapered king PST + king safety (queen attacks D8 in the black
    // king ring: 5 * 1 = 5 units → table[5] = 5 cp MG penalty for black,
    // blended by phase = 4 to +~1 cp for white in the final score).
    // Actual pinned value below; update when eval terms shift.
    const char* w_to_move = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";
    const char* b_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1";
    Position p1, p2;
    REQUIRE(p1.set_from_fen(w_to_move));
    REQUIRE(p2.set_from_fen(b_to_move));
    CHECK(evaluate(p1) == 907);
    CHECK(evaluate(p2) == -907);
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
    // King-safety pinned totals: only the queen case fires the safety
    // penalty. The queen on D1 attacks D8 in the black king ring
    // (D7/D8/E7/F7/F8 around E8): 5*1 = 5 units → table[5] = 5 cp MG,
    // blended by phase = 4 → +~1 cp for white in the final score. The
    // rook/bishop/knight/pawn cases have no white queen, so the safety
    // short-circuit returns 0 for both sides and their totals are
    // material + PST + mobility only.
    const Case cases[] = {
        {"4k3/8/8/8/8/8/8/3QK3 w - - 0 1",  907, "queen  (900 -  5, K-safety +3 MG blended)"},
        {"4k3/8/8/8/8/8/8/3RK3 w - - 0 1",  515, "rook   (500 +  5)"},
        {"4k3/8/8/8/8/8/8/3BK3 w - - 0 1",  330, "bishop (330 - 10)"},
        {"4k3/8/8/8/8/8/8/3NK3 w - - 0 1",  298, "knight (320 - 30)"},
        {"4k3/8/8/8/8/8/3P4/4K3 w - - 0 1",  60, "pawn   (100 - 20, isolated)"},
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
    // also gets 3 mobility squares (weight 2). The D2 pawn is isolated
    // (no C or E pawn) — MG -15, EG -20. Phase 2/24, so the final
    // blended total lands at 568.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/3P4/3RK3 w - - 0 1"));
    CHECK(evaluate(pos) == 568);
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
    // 1 knight square (weight 4 = -4). Base mobility delta = 70 - 8 = 62.
    // King safety piles on: black's queen/rooks/knights/bishops all
    // attack the exposed E4 king ring, adding a large penalty for the
    // exposed side. Pinned delta below reflects the combined effect.
    Position castled, exposed;
    REQUIRE(castled.set_from_fen("rnbqkbnr/8/8/8/8/5N2/4B3/RNBQ1RK1 w - - 0 1"));  // K on G1
    REQUIRE(exposed.set_from_fen("rnbqkbnr/8/8/8/4K3/5N2/4B3/RNBQ1R2 w - - 0 1"));  // K on E4
    CHECK(evaluate(castled) > evaluate(exposed));
    CHECK((evaluate(castled) - evaluate(exposed)) == 105);
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

// --- Pawn structure ------------------------------------------------------

TEST_CASE("isolated pawn: same pawn count scores higher when the pawns support each other") {
    // Both positions have two white pawns. In `connected` they sit on
    // adjacent files (D2 + E2) — neither isolated. In `isolated_pair`
    // they sit on files a gap apart (D2 + F2) — both isolated. Same
    // material, only the isolated penalty differs.
    Position connected, isolated_pair;
    REQUIRE(connected    .set_from_fen("4k3/8/8/8/8/8/3PP3/4K3 w - - 0 1"));
    REQUIRE(isolated_pair.set_from_fen("4k3/8/8/8/8/8/3P1P2/4K3 w - - 0 1"));
    CHECK(evaluate(connected) > evaluate(isolated_pair));
}

TEST_CASE("doubled pawns: same pawn count scores lower when stacked on one file") {
    // Both positions have two white pawns. In `abreast` they sit on
    // D2 + E3 — adjacent files, supporting each other, one passed with
    // an advancement bonus. In `doubled` they stack on D2 + D3 —
    // isolated pair, plus a doubled-file penalty on top.
    Position abreast, doubled;
    REQUIRE(abreast.set_from_fen("4k3/8/8/8/8/4P3/3P4/4K3 w - - 0 1"));
    REQUIRE(doubled.set_from_fen("4k3/8/8/8/8/3P4/3P4/4K3 w - - 0 1"));
    CHECK(evaluate(abreast) > evaluate(doubled));
}

TEST_CASE("passed pawn: advanced pawn with no enemy pawn ahead scores higher than a blockaded one") {
    // Both positions have one white pawn advanced to rank 5. In
    // `passed` the file is clear. In `blocked` a black pawn sits on
    // the same file directly ahead. The passed pawn is worth more.
    Position passed, blocked;
    REQUIRE(passed .set_from_fen("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1"));
    REQUIRE(blocked.set_from_fen("4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1"));
    CHECK(evaluate(passed) > evaluate(blocked));
}

TEST_CASE("pawn hash: repeat evaluate calls agree with fresh evaluate on the same position") {
    // Sanity: cached path (second call, hash hit) returns the same
    // score as the miss path (first call). Combined with startpos == 0
    // above, this proves the hash isn't drifting.
    Position pos;
    REQUIRE(pos.set_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));
    int first  = evaluate(pos);
    int second = evaluate(pos);
    CHECK(first == second);
}

// --- King safety ---------------------------------------------------------

TEST_CASE("king safety: queenless positions skip the penalty entirely") {
    // Neither side has a queen — the per-side gate returns 0 for both
    // king-safety queries regardless of what pieces are attacking the
    // rings. This test pins that behavior by comparing a position
    // where a rook DOES attack the black king ring against one where
    // no attackers are present: the eval delta must equal only the
    // material+PST+mobility contribution of the rook, with no safety
    // component.
    Position rook_attacking, no_rook;
    REQUIRE(rook_attacking.set_from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 0 1"));  // R on D1 attacks D8 in ring
    REQUIRE(no_rook       .set_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));

    // Without king safety the delta is just the rook's material + PST
    // (500 + 5 = 505) plus its mobility contribution. Adding a king-
    // safety term would push this higher; the gate must keep it fixed.
    CHECK((evaluate(rook_attacking) - evaluate(no_rook)) == 515);
}

TEST_CASE("king safety: exposed king with a queen attacker scores worse than a sheltered king") {
    // Same material on both sides. In `sheltered`, black's king is
    // castled at G8. In `exposed`, black's king walked out to F5 into
    // the white queen's range. White to move — an exposed BLACK king
    // shows up as a higher (better-for-white) eval.
    Position sheltered, exposed;
    REQUIRE(sheltered.set_from_fen("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1"));
    REQUIRE(exposed  .set_from_fen("r1bq1r2/pppp1ppp/2n2n2/2b1pk2/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1"));
    CHECK(evaluate(exposed) > evaluate(sheltered));
}

TEST_CASE("king safety: more ring attackers produce a strictly higher penalty") {
    // Three positions with identical material (K + Q per side) but
    // different queen placements against a fixed black king at E8.
    // The black king ring is {D7, E7, F7, D8, F8}.
    //   none: white Q at H1 — attacks no ring square.
    //   one : white Q at D1 — attacks D8 only (1 ring square).
    //   many: white Q at D5 — attacks D7, D8, F7 (3 ring squares).
    // Same material means the eval difference is purely king safety.
    Position none, one, many;
    REQUIRE(none.set_from_fen("4k3/8/8/8/8/8/8/4K2Q w - - 0 1"));
    REQUIRE(one .set_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"));
    REQUIRE(many.set_from_fen("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1"));
    CHECK(evaluate(many) > evaluate(one));
    CHECK(evaluate(one)  > evaluate(none));
}

// --- Pawn shield ---------------------------------------------------------

TEST_CASE("pawn shield: castled king with intact shield beats one with a gap") {
    // Same material and identical piece placement except for the G-pawn:
    // in `intact`, pawns sit on F2, G2, H2 (full kingside shield). In
    // `gap`, the G-pawn is gone — same king square, weaker shelter.
    // The intact-shield position must evaluate higher for white.
    Position intact, gap;
    REQUIRE(intact.set_from_fen("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1"));
    REQUIRE(gap   .set_from_fen("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1P1P/R1BQ1RK1 w - - 0 1"));
    CHECK(evaluate(intact) > evaluate(gap));
}

TEST_CASE("pawn shield: advanced shield pawn scores lower than one right in front") {
    // Same material, same king. In `intact`, G-pawn on G2 (distance 1
    // from king on G1 = full bonus). In `advanced`, G-pawn on G3
    // (distance 2 = partial bonus). Intact should score higher.
    Position intact, advanced;
    REQUIRE(intact  .set_from_fen("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1"));
    REQUIRE(advanced.set_from_fen("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2NP1/PPPP1P1P/R1BQ1RK1 w - - 0 1"));
    CHECK(evaluate(intact) > evaluate(advanced));
}

TEST_CASE("pawn shield: king on a central file gets no shield contribution") {
    // Two positions with white king on E1. In `with_pawns` there are
    // shield-shaped pawns on F2, G2, H2; in `no_pawns` those squares
    // are empty. If the shield gate fired on a central-file king, the
    // `with_pawns` position would gain ~+45 cp from shield alone on top
    // of the 3-pawn material — but shield returns 0 (E is central),
    // so the delta is purely material + PST + mobility. The bound
    // below fits comfortably above the expected material-driven delta
    // (~300 cp) but below what a shield contribution would add.
    Position with_pawns, no_pawns;
    REQUIRE(with_pawns.set_from_fen("4k3/8/8/8/8/8/5PPP/4K3 w - - 0 1"));
    REQUIRE(no_pawns  .set_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    const int delta = evaluate(with_pawns) - evaluate(no_pawns);
    // 3 pawns × ~100 cp material + PST tweaks + mobility. Should sit
    // well under 380 — a shield contribution would push it past 400.
    CHECK(delta < 380);
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
