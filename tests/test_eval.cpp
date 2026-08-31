// Evaluation tests. Eval is pure material for now, so tests here pin
// (a) the material values in centipawns, (b) the side-to-move perspective
// convention (positive = good for whoever is about to move), and
// (c) symmetry — mirror positions with matched material score 0.

#include "doctest.h"

#include "eval.h"
#include "position.h"

TEST_CASE("startpos evaluates to 0 — symmetric material") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    CHECK(evaluate(pos) == 0);
}

TEST_CASE("empty-material position (kings only) evaluates to 0") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    CHECK(evaluate(pos) == 0);
}

TEST_CASE("perspective: same board, different side-to-move flips the sign") {
    // White up a queen. Positive when white to move, negative when black.
    const char* w_to_move = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";
    const char* b_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1";
    Position p1, p2;
    REQUIRE(p1.set_from_fen(w_to_move));
    REQUIRE(p2.set_from_fen(b_to_move));
    CHECK(evaluate(p1) == 900);
    CHECK(evaluate(p2) == -900);
    CHECK(evaluate(p1) == -evaluate(p2));
}

TEST_CASE("piece values: Q=900 R=500 B=330 N=320 P=100") {
    // Each case: an extra white piece of the given type vs bare kings,
    // white to move → eval equals that piece's value.
    struct Case { const char* fen; int expected; const char* label; };
    const Case cases[] = {
        {"4k3/8/8/8/8/8/8/3QK3 w - - 0 1",  900, "queen"},
        {"4k3/8/8/8/8/8/8/3RK3 w - - 0 1",  500, "rook"},
        {"4k3/8/8/8/8/8/8/3BK3 w - - 0 1",  330, "bishop"},
        {"4k3/8/8/8/8/8/8/3NK3 w - - 0 1",  320, "knight"},
        {"4k3/8/8/8/8/8/3P4/4K3 w - - 0 1", 100, "pawn"},
    };
    for (const auto& c : cases) {
        Position pos;
        REQUIRE(pos.set_from_fen(c.fen));
        INFO("piece: " << std::string(c.label));
        CHECK(evaluate(pos) == c.expected);
    }
}

TEST_CASE("material differences aggregate linearly") {
    // White has an extra rook (+500) AND an extra pawn (+100) = +600.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/3P4/3RK3 w - - 0 1"));
    CHECK(evaluate(pos) == 600);
}
