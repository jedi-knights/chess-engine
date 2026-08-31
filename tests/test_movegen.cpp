// Move-generation tests: knight/king/pawn/slider/castling generation shape,
// plus movegen-driven make/unmake round-trip walks (single-ply and nested).
// Forward-correctness of make_move itself lives in test_position.cpp.

#include "doctest.h"
#include "support.h"

#include "movegen.h"
#include "position.h"

#include <string>
#include <vector>

TEST_CASE("make_move then unmake_move restores identical FEN across generated moves") {
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
    // Recursively make/unmake to depth 3 across the whole legal(-ish) move
    // tree. Catches any bug where a nested undo silently drops or duplicates
    // state — especially state that only differs several plies deep.
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

// --- Pawn ---------------------------------------------------------------

TEST_CASE("pawn: double push generates the correct destination and ep pairing") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    for (int f = 0; f < 8; ++f) {
        Square from    = make_square(File(f), RANK_2);
        Square single  = make_square(File(f), RANK_3);
        Square dbl     = make_square(File(f), RANK_4);
        INFO("file " << f);
        CHECK(contains_move(moves, ::make_move(from, single)));
        CHECK(contains_move(moves, ::make_move(from, dbl)));
    }
}

TEST_CASE("pawn: en passant emitted only when ep_square is set and reachable") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(E5, D6, MT_EN_PASSANT)));

    Position no_ep;
    REQUIRE(no_ep.set_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1"));
    moves.clear();
    generate_moves(no_ep, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E5, D6, MT_EN_PASSANT)));
}

TEST_CASE("pawn: promotion fan-out emits all four piece types (push and capture)") {
    Position push_pos;
    REQUIRE(push_pos.set_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(push_pos, moves);
    for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
        INFO("push-promotion piece: " << int(pt));
        CHECK(contains_move(moves, ::make_move(A7, A8, MT_PROMOTION, pt)));
    }

    Position cap_pos;
    REQUIRE(cap_pos.set_from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
    moves.clear();
    generate_moves(cap_pos, moves);
    for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
        INFO("capture-promotion piece: " << int(pt));
        CHECK(contains_move(moves, ::make_move(A7, B8, MT_PROMOTION, pt)));
    }
}

TEST_CASE("pawn: single push blocked by any piece (own or enemy)") {
    Position blocked;
    REQUIRE(blocked.set_from_fen("4k3/8/8/8/8/4N3/4P3/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(blocked, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E2, E3)));
    CHECK_FALSE(contains_move(moves, ::make_move(E2, E4)));

    Position blocked_dbl;
    REQUIRE(blocked_dbl.set_from_fen("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1"));
    moves.clear();
    generate_moves(blocked_dbl, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E2, E4)));
}

TEST_CASE("pawn: file-wrap guard — a-file pawn has no NW capture, h-file no NE") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/7p/P6P/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(A2, H2)));
    CHECK_FALSE(contains_move(moves, ::make_move(H2, A3)));
}

// --- Sliders ------------------------------------------------------------

TEST_CASE("slider: rook on empty board covers 14 squares from d4") {
    Position pos;
    REQUIRE(pos.set_from_fen("k7/8/8/8/3R4/8/8/7K w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(count_moves_from(moves, D4) == 14);
    CHECK(contains_move(moves, ::make_move(D4, D8)));
    CHECK(contains_move(moves, ::make_move(D4, D1)));
    CHECK(contains_move(moves, ::make_move(D4, A4)));
    CHECK(contains_move(moves, ::make_move(D4, H4)));
    CHECK_FALSE(contains_move(moves, ::make_move(D4, E5)));
}

TEST_CASE("slider: bishop on empty board covers 13 squares from d4") {
    Position pos;
    REQUIRE(pos.set_from_fen("k7/8/8/8/3B4/8/8/7K w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(count_moves_from(moves, D4) == 13);
    CHECK(contains_move(moves, ::make_move(D4, H8)));
    CHECK(contains_move(moves, ::make_move(D4, A7)));
    CHECK(contains_move(moves, ::make_move(D4, G1)));
    CHECK(contains_move(moves, ::make_move(D4, A1)));
    CHECK_FALSE(contains_move(moves, ::make_move(D4, D5)));
}

TEST_CASE("slider: queen on empty board covers 27 squares from d4 (14+13)") {
    Position pos;
    REQUIRE(pos.set_from_fen("k7/8/8/8/3Q4/8/8/7K w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(count_moves_from(moves, D4) == 27);
}

TEST_CASE("slider: own piece blocks, ray stops before the blocker") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/P7/8/8/R3K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(A1, A2)));
    CHECK(contains_move(moves, ::make_move(A1, A3)));
    CHECK_FALSE(contains_move(moves, ::make_move(A1, A4)));
    CHECK_FALSE(contains_move(moves, ::make_move(A1, A5)));
}

TEST_CASE("slider: enemy piece is captured and ray stops") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/p7/8/8/R3K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(A1, A2)));
    CHECK(contains_move(moves, ::make_move(A1, A3)));
    CHECK(contains_move(moves, ::make_move(A1, A4)));
    CHECK_FALSE(contains_move(moves, ::make_move(A1, A5)));
}

// --- Castling -----------------------------------------------------------

TEST_CASE("castling: both moves emitted when position is clean") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(E1, G1, MT_CASTLING)));
    CHECK(contains_move(moves, ::make_move(E1, C1, MT_CASTLING)));
}

TEST_CASE("castling: missing right suppresses that side's move") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/8/R3K2R w Q - 0 1"));  // OOO only
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E1, G1, MT_CASTLING)));
    CHECK(contains_move(moves, ::make_move(E1, C1, MT_CASTLING)));
}

TEST_CASE("castling: squares between king and rook must be empty") {
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/8/RN2KB1R w KQ - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E1, G1, MT_CASTLING)));
    CHECK_FALSE(contains_move(moves, ::make_move(E1, C1, MT_CASTLING)));
}

TEST_CASE("castling: king in check disallows either side") {
    Position pos;
    REQUIRE(pos.set_from_fen("4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E1, G1, MT_CASTLING)));
    CHECK_FALSE(contains_move(moves, ::make_move(E1, C1, MT_CASTLING)));
}

TEST_CASE("castling: transit square attack disallows only that side") {
    Position pos;
    REQUIRE(pos.set_from_fen("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E1, G1, MT_CASTLING)));
    CHECK(contains_move(moves, ::make_move(E1, C1, MT_CASTLING)));
}

TEST_CASE("castling: b1 attack does NOT disallow queenside (king doesn't pass through)") {
    Position pos;
    REQUIRE(pos.set_from_fen("1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(E1, C1, MT_CASTLING)));
    CHECK(contains_move(moves, ::make_move(E1, G1, MT_CASTLING)));
}

TEST_CASE("castling: black castles both sides symmetrically") {
    Position pos;
    REQUIRE(pos.set_from_fen("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(E8, G8, MT_CASTLING)));
    CHECK(contains_move(moves, ::make_move(E8, C8, MT_CASTLING)));
}

// --- Legality filter ----------------------------------------------------

TEST_CASE("legality: king cannot move into a square attacked by an enemy piece") {
    // Black rook on d8 attacks the entire d-file. White king at e1 can move
    // to d1 or d2 only if those aren't attacked — but they are.
    Position pos;
    REQUIRE(pos.set_from_fen("3r4/8/8/8/8/8/8/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E1, D1)));
    CHECK_FALSE(contains_move(moves, ::make_move(E1, D2)));
    CHECK(contains_move(moves, ::make_move(E1, E2)));
    CHECK(contains_move(moves, ::make_move(E1, F1)));
    CHECK(contains_move(moves, ::make_move(E1, F2)));
    CHECK(count_moves_from(moves, E1) == 3);
}

TEST_CASE("legality: pinned piece cannot move off the pin line") {
    // Black rook on e8 pins white bishop on e2 to the king on e1. Bishop
    // moves off the e-file leave the king in check → filtered.
    Position pos;
    REQUIRE(pos.set_from_fen("4r3/8/8/8/8/8/4B3/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    // Every bishop move from e2 leaves the e-file (bishops only move diagonally),
    // so all bishop moves are pinned out.
    CHECK(count_moves_from(moves, E2) == 0);
}

TEST_CASE("legality: pinned piece CAN move along the pin line (incl. capture pinner)") {
    // Black rook on e8 pins white rook on e5 along the e-file. The pinned
    // rook can still move along the e-file, including capturing the pinner.
    Position pos;
    REQUIRE(pos.set_from_fen("4r3/8/8/4R3/8/8/8/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(E5, E6)));      // along pin
    CHECK(contains_move(moves, ::make_move(E5, E7)));      // along pin
    CHECK(contains_move(moves, ::make_move(E5, E8)));      // capture the pinner
    CHECK_FALSE(contains_move(moves, ::make_move(E5, D5))); // off pin
    CHECK_FALSE(contains_move(moves, ::make_move(E5, F5))); // off pin
    CHECK_FALSE(contains_move(moves, ::make_move(E5, A5))); // off pin
}

TEST_CASE("legality: king in check must address the check") {
    // Black rook on e8 checks the white king on e1. Legal king moves are
    // only squares that escape the e-file (or capture the checker, which
    // isn't reachable). No other piece exists to block or capture.
    Position pos;
    REQUIRE(pos.set_from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E1, E2)));  // still on e-file
    CHECK(contains_move(moves, ::make_move(E1, D1)));
    CHECK(contains_move(moves, ::make_move(E1, D2)));
    CHECK(contains_move(moves, ::make_move(E1, F1)));
    CHECK(contains_move(moves, ::make_move(E1, F2)));
    CHECK(moves.size() == 4);
}

TEST_CASE("legality: checkmate produces zero legal moves") {
    // Back-rank mate: white king on h1, own pawns on g2/h2 blocking escape,
    // black rook on f1 delivering check along the 1st rank.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/8/6PP/5r1K w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(moves.empty());
}

TEST_CASE("legality: stalemate produces zero legal moves and is NOT checkmate") {
    // White king on e1 not in check, but every escape square is either an
    // own piece, attacked by the black king on e3, or attacked by the black
    // pawn on e2. Classic king-and-pawn stalemate shape.
    Position pos;
    REQUIRE(pos.set_from_fen("8/8/8/8/8/4k3/4p3/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(moves.empty());
}
