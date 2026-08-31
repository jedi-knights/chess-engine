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

// After milestone 6 (castling), Kiwipete gains its two castling moves at
// ply 1 and Position 5 gains its one. Both hit their standard perft depth-1
// targets even without the general legality filter — the child positions
// they generate are only reached at depth 2+.
TEST_CASE("perft depth 1 matches for positions that don't need legality filter") {
    struct Case { const char* name; const char* fen; uint64_t expected; };
    const Case cases[] = {
        {"Startpos",   STARTPOS_FEN, 20},
        {"Position 6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 46},
        {"Kiwipete",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 48},
        {"Position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 44},
    };
    for (const auto& c : cases) {
        Position pos;
        REQUIRE(pos.set_from_fen(c.fen));
        INFO("position: " << std::string(c.name));
        CHECK(perft(pos, 1) == c.expected);
    }
}
