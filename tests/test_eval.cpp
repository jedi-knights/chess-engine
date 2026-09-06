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
    // blended by phase). King-on-open-file skips (both kings on the
    // E-file, which the wing-file gate excludes as central).
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
    // penalty (queen attacks D8 in black king ring → +3 MG blended).
    // The rook/bishop/knight/pawn cases have no white queen, so the
    // safety short-circuit returns 0. King-on-open-file skips all
    // cases: kings are on the central E-file which the wing-file gate
    // excludes.
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
    // square. King safety fires on both (attackers on ring), and
    // king-on-open-file fires on the CASTLED king (G-file has no pawns
    // in this pawnless test position) but not on the exposed E4 king
    // (central-file gate). That flips 30 cp back toward the exposed
    // king, so the pinned delta reflects the combined effect and is
    // smaller than the shield-and-safety-only case. The important
    // invariant is directional: castled > exposed.
    Position castled, exposed;
    REQUIRE(castled.set_from_fen("rnbqkbnr/8/8/8/8/5N2/4B3/RNBQ1RK1 w - - 0 1"));  // K on G1
    REQUIRE(exposed.set_from_fen("rnbqkbnr/8/8/8/4K3/5N2/4B3/RNBQ1R2 w - - 0 1"));  // K on E4
    CHECK(evaluate(castled) > evaluate(exposed));
    CHECK((evaluate(castled) - evaluate(exposed)) == 75);
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

TEST_CASE("king safety: queenless positions skip the ring-attack penalty entirely") {
    // Neither side has a queen — the king-safety ring-attack gate
    // returns 0 regardless of what pieces are attacking. The king-on-
    // open-file term also skips (its wing-file gate excludes the
    // central E-file kings here). Delta reflects only the rook's own
    // material + PST + mobility.
    Position rook_attacking, no_rook;
    REQUIRE(rook_attacking.set_from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 0 1"));  // R on D1 attacks D8 in ring
    REQUIRE(no_rook       .set_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
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

// --- King on open / half-open file --------------------------------------

TEST_CASE("king on open file: king with a defending pawn on its file scores better than one without") {
    // Same material on both sides, both kings castled at G1/G8. In
    // `defended`, white's G-pawn sits on G2 (king shielded by own
    // pawn on king's file). In `undefended`, the G-pawn is gone
    // entirely — G-file is half-open toward white (only black's G7
    // pawn on the file). Black has a rook, so the open-file gate
    // fires. Defended > undefended for white.
    Position defended, undefended;
    REQUIRE(defended  .set_from_fen("r1bq1rk1/pppppppp/8/8/8/8/PPPPPPPP/R1BQ1RK1 w - - 0 1"));
    REQUIRE(undefended.set_from_fen("r1bq1rk1/pppppppp/8/8/8/8/PPPPPP1P/R1BQ1RK1 w - - 0 1"));
    CHECK(evaluate(defended) > evaluate(undefended));
}

TEST_CASE("king on open file: skips when opponent has no queen") {
    // Enemy has no queen — the queen-only gate returns 0 regardless
    // of file structure. Even a fully-open king file with a rook
    // attacker doesn't fire the penalty, because the rook's file
    // threat is already captured by mobility and search extends far
    // enough for a rook-only pressure line without needing the eval
    // hint.
    Position no_pawn, with_pawn;
    // Only pieces: kings + one white rook (no queen anywhere). Both
    // white kings sit on E-file for the wing-file gate to skip too;
    // even without that, the queen gate would suffice.
    REQUIRE(no_pawn  .set_from_fen("4k3/8/8/8/8/8/8/4KR2 w - - 0 1"));  // K + R only
    REQUIRE(with_pawn.set_from_fen("4k3/8/8/8/8/8/4P3/4KR2 w - - 0 1"));
    const int delta = evaluate(with_pawn) - evaluate(no_pawn);
    // 1 pawn = ~100 cp material + PST tweaks. Should sit under 130.
    // A spurious open-file penalty would push the "no pawn" position
    // ~30 cp lower, driving delta above 130.
    CHECK(delta < 130);
}

// --- Pawn storm ----------------------------------------------------------

TEST_CASE("pawn storm: enemy pawn near our castled king scores worse than one far away") {
    // Two positions with identical material and an intact white kingside
    // shield (F2, G2, H2, king on G1). The difference is a single black
    // pawn's rank on the H-file:
    //   quiet: black H-pawn on H7 (unmoved) — no storm.
    //   storm: black H-pawn on H3 (rank_of 2 = distance 2 from white king) — imminent.
    // The storm position should score worse for white.
    Position quiet, storm;
    REQUIRE(quiet.set_from_fen("r1bq1rk1/pppppp1p/2n2np1/2b5/2B5/2N2N2/PPPPPPPP/R1BQ1RK1 w - - 0 1"));
    REQUIRE(storm.set_from_fen("r1bq1rk1/pppppp2/2n2np1/2b5/2B5/2N2N1p/PPPPPPPP/R1BQ1RK1 w - - 0 1"));
    CHECK(evaluate(quiet) > evaluate(storm));
}

TEST_CASE("pawn storm: closer enemy pawn is worse than a farther one") {
    // Same castled positions; only difference is how advanced black's
    // H-pawn is. Both positions have same material.
    //   far  : black H-pawn on H5 (distance 4 from white king rank 1).
    //   close: black H-pawn on H3 (distance 2).
    // Closer storm pawn = larger penalty for white.
    Position far_pawn, close_pawn;
    REQUIRE(far_pawn  .set_from_fen("r1bq1rk1/ppppppp1/2n2np1/2b4p/2B5/2N2N2/PPPPPPPP/R1BQ1RK1 w - - 0 1"));
    REQUIRE(close_pawn.set_from_fen("r1bq1rk1/ppppppp1/2n2np1/2b5/2B5/2N2N1p/PPPPPPPP/R1BQ1RK1 w - - 0 1"));
    CHECK(evaluate(far_pawn) > evaluate(close_pawn));
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
