#include "doctest.h"

#include "attacks.h"
#include "movegen.h"
#include "perft.h"
#include "position.h"
#include <algorithm>
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

// Generator-shape tests: the perft counts above prove aggregate correctness,
// but named tests document the specific behaviors that most commonly regress
// during pawn work (double-push semantics, promotion fan-out, ep availability).

static bool contains_move(const std::vector<Move>& moves, Move needle) {
    return std::find(moves.begin(), moves.end(), needle) != moves.end();
}

TEST_CASE("pawn: double push generates the correct destination and ep pairing") {
    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    // Every white pawn on rank 2 has both a single and a double push in startpos.
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
    // White pawn on e5, black pawn on d5, ep square d6 (i.e. black just played d7-d5).
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(E5, D6, MT_EN_PASSANT)));

    // Same board but no ep_square recorded — the ep move must not appear.
    Position no_ep;
    REQUIRE(no_ep.set_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1"));
    moves.clear();
    generate_moves(no_ep, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E5, D6, MT_EN_PASSANT)));
}

TEST_CASE("pawn: promotion fan-out emits all four piece types (push and capture)") {
    // Push promotion: white pawn on a7 can promote to Q/R/B/N on a8.
    Position push_pos;
    REQUIRE(push_pos.set_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(push_pos, moves);
    for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
        INFO("push-promotion piece: " << int(pt));
        CHECK(contains_move(moves, ::make_move(A7, A8, MT_PROMOTION, pt)));
    }

    // Capture-promotion: pawn on a7 captures rook on b8, promoting.
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
    // White pawn on e2, blocker on e3 (own knight) — no single or double push.
    Position blocked;
    REQUIRE(blocked.set_from_fen("4k3/8/8/8/8/4N3/4P3/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(blocked, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E2, E3)));
    CHECK_FALSE(contains_move(moves, ::make_move(E2, E4)));

    // Double push blocked by piece on e3 even though e4 is empty.
    Position blocked_dbl;
    REQUIRE(blocked_dbl.set_from_fen("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1"));
    moves.clear();
    generate_moves(blocked_dbl, moves);
    CHECK_FALSE(contains_move(moves, ::make_move(E2, E4)));
}

// Count how many generated moves originate from `from`. Useful for slider
// shape assertions where the position isolates one piece we care about.
static int count_moves_from(const std::vector<Move>& moves, Square from) {
    int n = 0;
    for (Move m : moves) if (move_from(m) == from) ++n;
    return n;
}

TEST_CASE("slider: rook on empty board covers 14 squares from d4") {
    // Kings tucked in corners so they don't interfere with the rook rays.
    Position pos;
    REQUIRE(pos.set_from_fen("k7/8/8/8/3R4/8/8/7K w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(count_moves_from(moves, D4) == 14);
    // Spot-check endpoints in each direction.
    CHECK(contains_move(moves, ::make_move(D4, D8)));   // N to edge
    CHECK(contains_move(moves, ::make_move(D4, D1)));   // S to edge
    CHECK(contains_move(moves, ::make_move(D4, A4)));   // W to edge
    CHECK(contains_move(moves, ::make_move(D4, H4)));   // E to edge
    // Diagonal move must NOT appear — this is a rook, not a queen.
    CHECK_FALSE(contains_move(moves, ::make_move(D4, E5)));
}

TEST_CASE("slider: bishop on empty board covers 13 squares from d4") {
    Position pos;
    REQUIRE(pos.set_from_fen("k7/8/8/8/3B4/8/8/7K w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(count_moves_from(moves, D4) == 13);
    CHECK(contains_move(moves, ::make_move(D4, H8)));   // NE
    CHECK(contains_move(moves, ::make_move(D4, A7)));   // NW
    CHECK(contains_move(moves, ::make_move(D4, G1)));   // SE
    CHECK(contains_move(moves, ::make_move(D4, A1)));   // SW
    // Orthogonal move must NOT appear.
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
    // White rook a1, white pawn a4. Rook should reach a2, a3 — not a4.
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
    // White rook a1, black pawn a4. Rook should reach a2, a3, a4 — not a5.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/p7/8/8/R3K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    CHECK(contains_move(moves, ::make_move(A1, A2)));
    CHECK(contains_move(moves, ::make_move(A1, A3)));
    CHECK(contains_move(moves, ::make_move(A1, A4)));      // capture
    CHECK_FALSE(contains_move(moves, ::make_move(A1, A5))); // ray stopped
}

TEST_CASE("pawn: file-wrap guard — a-file pawn has no NW capture, h-file no NE") {
    // Black pieces adjacent to hypothetical wrap targets to make wrap bugs visible.
    Position pos;
    REQUIRE(pos.set_from_fen("4k3/8/8/8/8/7p/P6P/4K3 w - - 0 1"));
    std::vector<Move> moves;
    generate_moves(pos, moves);
    // Would-be-wraparound bits: A2 << 7 = H2 (own piece here anyway),
    // H2 << 9 = A3 (empty). Neither should show up as a pawn capture.
    CHECK_FALSE(contains_move(moves, ::make_move(A2, H2)));
    CHECK_FALSE(contains_move(moves, ::make_move(H2, A3)));
}
