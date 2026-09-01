// Perft-based validation. Perft is the aggregate correctness gate for
// move generation — a passing count at a given depth means every piece
// type's rules, every special-move handler, and every state-update is
// mutually consistent to that depth.

#include "doctest.h"

#include "perft.h"
#include "position.h"

#include <cstdint>
#include <sstream>
#include <string>

// After milestone 4, startpos is fully covered by knight+king+pawn moves:
// king is blocked so contributes 0, knights contribute 4, pawns 16 = 20.
// Depth 2 = 20 * 20 because all 20 white replies leave a legal position
// where black has 20 responses, and no legality-filtering is needed
// (kings can't be captured within the search tree here).
TEST_CASE("perft startpos matches after milestones 1-4") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    CHECK(perft(pos, 1) == 20);
    CHECK(perft(pos, 2) == 400);
}

// After milestone 7 (legality filter), all six standard positions match
// at every depth the perft driver checks. Depth 3 is chosen for the doctest
// suite because it exercises the full move-generation tree (~600k node
// applications total across the six positions) while still finishing in a
// few seconds under ASan+UBSan. `make perft 5` in the release binary is
// available for deeper validation.
TEST_CASE("perft matches on all standard positions through depth 3") {
    struct Case {
        const char* name;
        const char* fen;
        uint64_t    counts[4];   // index = depth (1..3), [0] unused
    };
    const Case cases[] = {
        {"Startpos",   STARTPOS_FEN,
         {0, 20, 400, 8902}},
        {"Kiwipete",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         {0, 48, 2039, 97862}},
        {"Position 3",
         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
         {0, 14, 191, 2812}},
        {"Position 4",
         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
         {0, 6, 264, 9467}},
        {"Position 5",
         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
         {0, 44, 1486, 62379}},
        {"Position 6",
         "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
         {0, 46, 2079, 89890}},
    };
    for (const auto& c : cases) {
        for (int d = 1; d <= 3; ++d) {
            Position pos;
            REQUIRE(pos.set_from_fen(c.fen));
            INFO("position: " << std::string(c.name) << " depth: " << d);
            CHECK(perft(pos, d) == c.counts[d]);
        }
    }
}

// --- perft() edge cases -------------------------------------------------

TEST_CASE("perft depth 0 always returns 1") {
    // Base case: no moves to enumerate, exactly one node (the starting
    // position itself). This is what makes the recurrence sum correctly.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    CHECK(perft(pos, 0) == 1);
}

TEST_CASE("perft returns 0 on a mated position") {
    // Back-rank mate — same fixture the legality tests use. No legal
    // moves at depth 1 means the sum over children is empty.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/6PP/5r1K w - - 0 1"));
    CHECK(perft(pos, 1) == 0);
    CHECK(perft(pos, 2) == 0);  // still 0 deeper — no children to enumerate
}

TEST_CASE("perft returns 0 on a stalemate position") {
    // Classic king+pawn stalemate.
    Position pos;
    REQUIRE(pos.set_from_fen("8/8/8/8/8/4k3/4p3/4K3 w - - 0 1"));
    CHECK(perft(pos, 1) == 0);
}

// --- run_perft_suite() driver -------------------------------------------

TEST_CASE("run_perft_suite: depth 1 all positions pass and print expected shape") {
    // Drive the suite via an ostringstream so the printf-style output
    // stays out of test noise. Verifies both the return value AND the
    // rendered report format that a user staring at `./engine perft N`
    // relies on.
    std::ostringstream out;
    bool ok = run_perft_suite(1, out);
    CHECK(ok);

    const std::string s = out.str();
    // Every one of the 6 standard-suite entries should appear in a
    // header line. The suite renders "=== <name> ===" per position, so
    // "===" appears 2 times per position = 12 total.
    int trip_count = 0;
    size_t pos = 0;
    while ((pos = s.find("===", pos)) != std::string::npos) { ++trip_count; ++pos; }
    CHECK(trip_count == 12);

    // Depth-1 counts all pass with our current legality-correct movegen,
    // so no [FAIL] lines and at least one [OK  ] line.
    CHECK(s.find("[FAIL]") == std::string::npos);
    CHECK(s.find("[OK  ]") != std::string::npos);

    // Spot-check that specific standard-suite names made it into the output.
    CHECK(s.find("Startpos")   != std::string::npos);
    CHECK(s.find("Kiwipete")   != std::string::npos);
    CHECK(s.find("Position 6") != std::string::npos);
}

TEST_CASE("run_perft_suite: max_depth=0 emits headers but no depth lines") {
    // Boundary — the depth loop's condition (`d <= max_depth`) never
    // enters the body when max_depth is 0, but headers still print
    // per-position. Confirms we don't crash on the trivial input.
    std::ostringstream out;
    CHECK(run_perft_suite(0, out) == true);
    CHECK(out.str().find("Startpos") != std::string::npos);
    CHECK(out.str().find("depth")    == std::string::npos);
}
