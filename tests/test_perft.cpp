// Perft-based validation. Perft is the aggregate correctness gate for
// move generation — a passing count at a given depth means every piece
// type's rules, every special-move handler, and every state-update is
// mutually consistent to that depth.

#include "doctest.h"

#include "perft.h"
#include "position.h"

#include <cstdint>
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
