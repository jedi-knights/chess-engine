// Position-layer tests: FEN round-trip, make_move / unmake_move forward
// correctness, unmake round-trip on every special move type.
// Movegen-driven walks live in test_movegen.cpp.

#include "doctest.h"
#include "support.h"

#include "bitboard.h"
#include "position.h"
#include "zobrist.h"

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
TEST_CASE("is_repetition: fresh position has no repetition") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    CHECK_FALSE(pos.is_repetition());
}

TEST_CASE("is_repetition: shuttling a knight back and forth triggers repetition") {
    // Nf3 Nf6 Ng1 Ng8 → returns to the position AFTER 1.Nf3 Nf6? No — actually
    // returns to startpos-ish (same piece positions and side to move).
    // Play 1.Nf3 Nf6 2.Ng1 Ng8 → back to startpos AS BLACK to move... no,
    // white made two moves, black made two — side to move is white, same as
    // startpos. And piece positions all restored. Should detect repetition.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    UndoInfo u1, u2, u3, u4;
    pos.make_move(::make_move(G1, F3), u1);   // white Nf3
    pos.make_move(::make_move(G8, F6), u2);   // black Nf6
    pos.make_move(::make_move(F3, G1), u3);   // white Ng1 (back)
    pos.make_move(::make_move(F6, G8), u4);   // black Ng8 (back)
    // Back at startpos — repetition of the starting position.
    CHECK(pos.is_repetition());
}

TEST_CASE("is_repetition: distinct positions never trigger a false positive") {
    // Sequence never returns to a prior position — no square shared with any
    // earlier board. Belt-and-suspenders check that we don't misidentify a
    // hash collision or off-by-one scan.
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    UndoInfo u1, u2, u3, u4;
    pos.make_move(::make_move(E2, E4), u1);
    pos.make_move(::make_move(E7, E5), u2);
    pos.make_move(::make_move(G1, F3), u3);
    pos.make_move(::make_move(B8, C6), u4);
    CHECK_FALSE(pos.is_repetition());
}

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

TEST_CASE("pawn_key: matches recompute from pawn bitboards after every move type") {
    // pawn_key is an XOR of PIECE_SQ[color][PAWN][sq] over all pawns.
    // Recomputing from scratch after make/unmake at every special move
    // type verifies the incremental updates in put_piece / remove_piece
    // stay in sync with a full rebuild. Includes pawn captures,
    // promotions (pawn removed at from + promoted piece placed at to),
    // and non-pawn moves (pawn_key must be unchanged).
    auto recompute_pawn_key = [](const Position& pos) {
        uint64_t k = 0;
        for (int c = 0; c < NUM_COLORS; ++c) {
            Bitboard bb = pos.pieces[c][PAWN];
            while (bb != 0U) {
                Square s = pop_lsb(bb);
                k ^= zobrist::PIECE_SQ[c][PAWN][s];
            }
        }
        return k;
    };

    struct Case { const char* name; const char* fen; Move m; };
    const Case cases[] = {
        {"pawn double push",  STARTPOS_FEN, ::make_move(E2, E4)},
        {"en passant",        "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1",
                              ::make_move(D5, C6, MT_EN_PASSANT)},
        {"promotion",         "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                              ::make_move(A7, A8, MT_PROMOTION, QUEEN)},
        {"promotion capture", "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                              ::make_move(A7, B8, MT_PROMOTION, QUEEN)},
        {"pawn capture",      "4k3/8/8/2p5/3P4/8/8/4K3 w - - 0 1",
                              ::make_move(D4, C5)},
        {"knight move (no pawn change)", "4k3/8/8/8/8/8/PPPPPPPP/4K1N1 w - - 0 1",
                              ::make_move(G1, F3)},
    };
    for (const auto& c : cases) {
        Position pos;
        REQUIRE(pos.set_from_fen(c.fen));
        INFO("case: " << std::string(c.name));
        CHECK(pos.pawn_key == recompute_pawn_key(pos));   // set_from_fen seeded correctly

        UndoInfo u;
        pos.make_move(c.m, u);
        CHECK(pos.pawn_key == recompute_pawn_key(pos));   // incremental XOR is correct

        pos.unmake_move(c.m, u);
        CHECK(pos.pawn_key == recompute_pawn_key(pos));   // unmake snapshot restores
    }
}
