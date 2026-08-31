// Position-layer tests: FEN round-trip, make_move / unmake_move forward
// correctness, unmake round-trip on every special move type.
// Movegen-driven walks live in test_movegen.cpp.

#include "doctest.h"
#include "support.h"

#include "position.h"

#include <string>

TEST_CASE("FEN round-trip is idempotent") {
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        CHECK(pos.to_fen() == fen);
    }
}

// Forward-correctness: hand-constructed (position + move) → expected FEN.
// Round-trip alone would pass with a no-op make_move; these pin the actual
// outcome for every move type and every state-component the move touches.
TEST_CASE("make_move produces expected FEN for every move type") {
    struct MoveCase {
        const char* name;
        const char* fen_before;
        Move        move;
        const char* fen_after;
    };
    const MoveCase cases[] = {
        {"quiet knight Ng1-f3 (halfmove++, castling preserved)",
         STARTPOS_FEN,
         ::make_move(G1, F3),
         "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"},

        {"knight capture Nc3xd4 (halfmove resets on capture)",
         "4k3/8/8/8/3n4/2N5/8/4K3 w - - 0 1",
         ::make_move(C3, D4),
         "4k3/8/8/8/3N4/8/8/4K3 b - - 0 1"},

        {"pawn double push e2-e4 sets ep_square, resets halfmove",
         STARTPOS_FEN,
         ::make_move(E2, E4),
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"},

        {"pawn single push e2-e3 does NOT set ep_square",
         STARTPOS_FEN,
         ::make_move(E2, E3),
         "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1"},

        {"en passant d5xc6 removes captured pawn from c5 (not c6)",
         "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1",
         ::make_move(D5, C6, MT_EN_PASSANT),
         "4k3/8/2P5/8/8/8/8/4K3 b - - 0 1"},

        {"kingside castling moves rook H1->F1, clears both white rights",
         "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
         ::make_move(E1, G1, MT_CASTLING),
         "4k3/8/8/8/8/8/8/R4RK1 b - - 1 1"},

        {"queenside castling moves rook A1->D1",
         "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
         ::make_move(E1, C1, MT_CASTLING),
         "4k3/8/8/8/8/8/8/2KR3R b - - 1 1"},

        {"promotion to queen replaces pawn on 8th rank",
         "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
         ::make_move(A7, A8, MT_PROMOTION, QUEEN),
         "Q3k3/8/8/8/8/8/8/4K3 b - - 0 1"},

        {"underpromotion to knight (verifies promo bits reach make_move)",
         "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
         ::make_move(A7, A8, MT_PROMOTION, KNIGHT),
         "N3k3/8/8/8/8/8/8/4K3 b - - 0 1"},

        {"king move clears own castling rights, preserves opponent's",
         "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
         ::make_move(E1, E2),
         "r3k2r/8/8/8/8/8/4K3/R6R b kq - 1 1"},

        {"queenside rook move clears WHITE_OOO only",
         "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
         ::make_move(A1, A2),
         "r3k2r/8/8/8/8/8/R7/4K2R b Kkq - 1 1"},

        {"capturing rook on original square clears victim's castling right",
         "r3k2r/8/8/8/8/1n6/8/R3K2R b KQkq - 0 1",
         ::make_move(B3, A1),
         "r3k2r/8/8/8/8/8/8/n3K2R w Kkq - 0 2"},

        {"fullmove increments after black's move",
         "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1",
         ::make_move(E8, E7),
         "8/4k3/8/8/8/8/4P3/4K3 w - - 1 2"},
    };
    for (const auto& c : cases) {
        Position pos;
        REQUIRE(pos.set_from_fen(c.fen_before));
        UndoInfo u;
        pos.make_move(c.move, u);
        // Cast char* → std::string so doctest stringifies contents (not pointer).
        INFO("case: " << std::string(c.name));
        CHECK(pos.to_fen() == std::string(c.fen_after));
    }
}

// Complements the forward-correctness table: verifies unmake also handles
// every special move type. Same cases, plus assertion that make actually
// changed the FEN (guards against no-op make_move surviving as a mutant).
TEST_CASE("unmake_move restores FEN for every special move type") {
    struct RoundTripCase { const char* name; const char* fen; Move m; };
    const RoundTripCase cases[] = {
        {"en passant",              "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1",
                                    ::make_move(D5, C6, MT_EN_PASSANT)},
        {"kingside castling",       "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
                                    ::make_move(E1, G1, MT_CASTLING)},
        {"queenside castling",      "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
                                    ::make_move(E1, C1, MT_CASTLING)},
        {"promotion (queen)",       "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                                    ::make_move(A7, A8, MT_PROMOTION, QUEEN)},
        {"promotion capture",       "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                                    ::make_move(A7, B8, MT_PROMOTION, QUEEN)},
        {"pawn double push",        STARTPOS_FEN, ::make_move(E2, E4)},
        {"rook capture",            "r3k2r/8/8/8/8/1n6/8/R3K2R b KQkq - 0 1",
                                    ::make_move(B3, A1)},
    };
    for (const auto& c : cases) {
        Position pos;
        REQUIRE(pos.set_from_fen(c.fen));
        const std::string before = pos.to_fen();

        UndoInfo u;
        pos.make_move(c.m, u);
        INFO("case: " << std::string(c.name));
        CHECK(pos.to_fen() != before);          // make actually mutated state

        pos.unmake_move(c.m, u);
        CHECK(pos.to_fen() == before);          // unmake restores
    }
}
