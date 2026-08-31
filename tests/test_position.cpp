#include "doctest.h"

#include "attacks.h"
#include "movegen.h"
#include "position.h"
#include <string>
#include <vector>

// Perft's standard test-suite FENs — reused here to exercise make/unmake
// across the same variety of positions perft covers.
static const std::vector<std::string> STANDARD_FENS = {
    STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
};

struct AttackInit {
    AttackInit() { init_attacks(); }
};
static AttackInit s_init;   // ensure attack tables are ready for every TU that includes this

TEST_CASE("FEN round-trip is idempotent") {
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        CHECK(pos.to_fen() == fen);
    }
}

TEST_CASE("make_move then unmake_move restores identical FEN (knight moves)") {
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        const std::string before = pos.to_fen();

        std::vector<Move> moves;
        generate_moves(pos, moves);

        for (Move m : moves) {
            UndoInfo u;
            pos.make_move(m, u);
            pos.unmake_move(m, u);
            CHECK(pos.to_fen() == before);
        }
    }
}

TEST_CASE("nested make/unmake at depth 3 restores FEN") {
    // Recursively make/unmake to depth 3 across the whole knight-move tree.
    // Catches any bug where a nested undo silently drops or duplicates state.
    auto walk = [](auto& self, Position& pos, int depth) -> void {
        if (depth == 0) return;
        const std::string before = pos.to_fen();
        std::vector<Move> moves;
        generate_moves(pos, moves);
        for (Move m : moves) {
            UndoInfo u;
            pos.make_move(m, u);
            self(self, pos, depth - 1);
            pos.unmake_move(m, u);
            CHECK(pos.to_fen() == before);
        }
    };
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        walk(walk, pos, 3);
    }
}
