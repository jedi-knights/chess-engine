// UCI notation round-trip tests. move_to_uci and parse_uci_move together
// define the interface between our engine's internal Move representation
// and the wire format that GUIs speak. Round-trip correctness across every
// move type is the invariant that matters — if it holds, the engine can
// consume any move history a GUI hands it and reproduce any move history
// a GUI expects to receive.

#include "doctest.h"
#include "support.h"

#include "movegen.h"
#include "notation.h"
#include "position.h"

#include <string>
#include <vector>

// --- move_to_uci --------------------------------------------------------

TEST_CASE("move_to_uci renders quiet moves as <from><to>") {
    CHECK(move_to_uci(::make_move(E2, E4)) == "e2e4");
    CHECK(move_to_uci(::make_move(G1, F3)) == "g1f3");
    CHECK(move_to_uci(::make_move(A1, H8)) == "a1h8");
}

TEST_CASE("move_to_uci appends promotion piece letter") {
    CHECK(move_to_uci(::make_move(A7, A8, MT_PROMOTION, QUEEN))  == "a7a8q");
    CHECK(move_to_uci(::make_move(A7, A8, MT_PROMOTION, ROOK))   == "a7a8r");
    CHECK(move_to_uci(::make_move(A7, A8, MT_PROMOTION, BISHOP)) == "a7a8b");
    CHECK(move_to_uci(::make_move(A7, A8, MT_PROMOTION, KNIGHT)) == "a7a8n");
}

TEST_CASE("move_to_uci: castling and en passant emit no special suffix") {
    // UCI encodes both the same as any other move — the type is inferred
    // by the receiver from position state, not from the string.
    CHECK(move_to_uci(::make_move(E1, G1, MT_CASTLING))    == "e1g1");
    CHECK(move_to_uci(::make_move(D5, C6, MT_EN_PASSANT))  == "d5c6");
}

TEST_CASE("move_to_uci(NULL_MOVE) is \"0000\"") {
    CHECK(move_to_uci(NULL_MOVE) == "0000");
}

// --- parse_uci_move -----------------------------------------------------

TEST_CASE("parse_uci_move: normal move on startpos") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    Move m = parse_uci_move(pos, "e2e4");
    CHECK(move_from(m) == E2);
    CHECK(move_to(m)   == E4);
    CHECK(move_type(m) == MT_NORMAL);
}

TEST_CASE("parse_uci_move: promotion piece letter → MT_PROMOTION with correct piece") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
    Move q = parse_uci_move(pos, "a7a8q");
    CHECK(move_type(q)      == MT_PROMOTION);
    CHECK(move_promotion(q) == QUEEN);

    Move n = parse_uci_move(pos, "a7a8n");
    CHECK(move_type(n)      == MT_PROMOTION);
    CHECK(move_promotion(n) == KNIGHT);
}

TEST_CASE("parse_uci_move: king moving 2 files → MT_CASTLING") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"));
    CHECK(move_type(parse_uci_move(pos, "e1g1")) == MT_CASTLING);
    CHECK(move_type(parse_uci_move(pos, "e1c1")) == MT_CASTLING);
}

TEST_CASE("parse_uci_move: king-captures-own-rook form is castling in both classical and Chess960") {
    // Classical startpos layout: king on e1, rooks on a1/h1. The
    // king-captures-own-rook UCI form (e1h1, e1a1) is what Lichess's
    // Bot API sends; it must parse to MT_CASTLING regardless of whether
    // UCI_Chess960 has been toggled on, because the king's actual
    // target (g1 / c1) is derived from direction, not from the move
    // string.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"));
    Move ks = parse_uci_move(pos, "e1h1");
    CHECK(move_type(ks) == MT_CASTLING);
    CHECK(move_from(ks) == E1);
    CHECK(move_to(ks)   == G1);  // internal encoding normalizes to king_to
    Move qs = parse_uci_move(pos, "e1a1");
    CHECK(move_type(qs) == MT_CASTLING);
    CHECK(move_to(qs)   == C1);
}

TEST_CASE("parse_uci_move: FRC king-captures-own-rook with non-classical rook file") {
    // Chess960 position: king on D1, rooks on B1 and F1 (variant Shredder
    // castling-rights "BFbf"). Kingside castle in UCI is "d1f1" (king
    // captures own rook on F1); king's actual target is still G1.
    Position pos;
    REQUIRE(pos.set_from_fen("nrkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/NRKBNQBR w HBhb - 0 1"));
    Move qs = parse_uci_move(pos, "c1b1");  // king on C1, queenside rook on B1
    CHECK(move_type(qs) == MT_CASTLING);
    CHECK(move_from(qs) == C1);
    CHECK(move_to(qs)   == C1);  // king_to for queenside is C-file; king doesn't actually move
    Move ks = parse_uci_move(pos, "c1h1");  // king on C1, kingside rook on H1
    CHECK(move_type(ks) == MT_CASTLING);
    CHECK(move_to(ks)   == G1);
}

TEST_CASE("move_to_uci_output: FRC emits king-captures-own-rook when UCI_Chess960 is set") {
    Position pos;
    REQUIRE(pos.set_from_fen("nrkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/NRKBNQBR w HBhb - 0 1"));
    CHECK(pos.is_chess960);  // inferred from king not on E file
    // Kingside castle: internal from=C1, to=G1.
    Move ks = make_move(C1, G1, MT_CASTLING);
    CHECK(move_to_uci_output(ks, pos) == "c1h1");  // rook_from = H1
    // Queenside castle: internal from=C1, to=C1 (king stays).
    Move qs = make_move(C1, C1, MT_CASTLING);
    CHECK(move_to_uci_output(qs, pos) == "c1b1");  // rook_from = B1
}

TEST_CASE("move_to_uci_output: standard positions get classical form regardless") {
    Position pos;
    REQUIRE(pos.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
    CHECK(!pos.is_chess960);
    Move ks = make_move(E1, G1, MT_CASTLING);
    CHECK(move_to_uci_output(ks, pos) == "e1g1");
}

TEST_CASE("parse_uci_move: pawn to ep_square → MT_EN_PASSANT") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1"));
    CHECK(move_type(parse_uci_move(pos, "d5c6")) == MT_EN_PASSANT);
}

TEST_CASE("parse_uci_move: pawn to ep_square with no ep set → MT_NORMAL") {
    // Same board but no ep_square. Same UCI string parses to a normal
    // move, though it would be illegal in this position — parse doesn't
    // legality-check, the caller does.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/2pP4/8/8/8/4K3 w - - 0 1"));
    CHECK(move_type(parse_uci_move(pos, "d5c6")) == MT_NORMAL);
}

TEST_CASE("parse_uci_move: malformed input returns NULL_MOVE") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    CHECK(parse_uci_move(pos, "")        == NULL_MOVE);   // empty
    CHECK(parse_uci_move(pos, "e2")      == NULL_MOVE);   // too short
    CHECK(parse_uci_move(pos, "e2e4z9")  == NULL_MOVE);   // too long
    CHECK(parse_uci_move(pos, "z9z9")    == NULL_MOVE);   // out of range
    CHECK(parse_uci_move(pos, "e2e4z")   == NULL_MOVE);   // suffix on non-promo pawn move
    CHECK(parse_uci_move(pos, "e5e6")    == NULL_MOVE);   // empty from-square
}

TEST_CASE("parse_uci_move: pawn-to-back-rank without promotion char → NULL_MOVE") {
    // Explicit rejection: a bare "a7a8" for a pawn is ambiguous (queen?
    // knight?) and would silently under-promote to the make_move default.
    // Better to reject than guess.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
    CHECK(parse_uci_move(pos, "a7a8")  == NULL_MOVE);
    CHECK(parse_uci_move(pos, "a7a8x") == NULL_MOVE);  // unknown promo char
}

// --- Round-trip ---------------------------------------------------------

TEST_CASE("round-trip: parse_uci_move(move_to_uci(m)) == m for every legal move") {
    // The strongest invariant we can pin: every move the generator can
    // produce must survive a full round-trip through UCI notation.
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        MoveList moves;
        generate_moves(pos, moves);
        for (Move m : moves) {
            std::string s = move_to_uci(m);
            Move parsed = parse_uci_move(pos, s);
            INFO("fen: " << fen << " move: " << s);
            CHECK(parsed == m);
        }
    }
}


TEST_CASE("parse_uci_move: Crazyhouse drop notation parses to MT_DROP") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    Move m = parse_uci_move(pos, "P@e4");
    CHECK(move_type(m) == MT_DROP);
    CHECK(move_to(m)   == E4);
    CHECK(move_drop_piece_type(m) == PAWN);

    Move q = parse_uci_move(pos, "Q@d5");
    CHECK(move_type(q) == MT_DROP);
    CHECK(move_to(q)   == D5);
    CHECK(move_drop_piece_type(q) == QUEEN);
}

TEST_CASE("move_to_uci: MT_DROP emits P@sq / Q@sq form") {
    Move m = make_drop_move(E4, PAWN);
    CHECK(move_to_uci(m) == "P@e4");
    Move q = make_drop_move(D5, QUEEN);
    CHECK(move_to_uci(q) == "Q@d5");
}
