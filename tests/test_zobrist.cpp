// Zobrist hashing tests. The invariant we care about is that Position::key
// stays consistent with zobrist::compute(pos) across every make/unmake
// pair — otherwise the transposition table will lose entries or, worse,
// return stale scores under collision.

#include "doctest.h"
#include "support.h"

#include "movegen.h"
#include "position.h"
#include "zobrist.h"

#include <set>
#include <string>
#include <vector>

TEST_CASE("Position::key after set_from_fen equals zobrist::compute") {
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        CHECK(pos.key == zobrist::compute(pos));
    }
}

TEST_CASE("key returns to original after make/unmake across generated moves") {
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        const uint64_t before = pos.key;

        MoveList moves;
        generate_moves(pos, moves);
        for (Move m : moves) {
            UndoInfo u;
            pos.make_move(m, u);
            CHECK(pos.key != before);                   // make actually changed the key
            CHECK(pos.key == zobrist::compute(pos));    // incremental matches from-scratch
            pos.unmake_move(m, u);
            CHECK(pos.key == before);                   // unmake restores exactly
        }
    }
}

TEST_CASE("keys are distinct across the standard positions") {
    // Same-key collisions across genuinely different positions are astronomically
    // unlikely with a good PRNG but not impossible — spot-check the fixtures.
    std::set<uint64_t> keys;
    for (const auto& fen : STANDARD_FENS) {
        Position pos;
        REQUIRE(pos.set_from_fen(fen));
        keys.insert(pos.key);
    }
    CHECK(keys.size() == STANDARD_FENS.size());
}

TEST_CASE("side_to_move flip changes the key predictably") {
    // Same board, different side to move → keys differ by exactly SIDE.
    Position wpos, bpos;
    REQUIRE(wpos.set_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    REQUIRE(bpos.set_from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1"));
    CHECK((wpos.key ^ bpos.key) == zobrist::SIDE);
}

TEST_CASE("transpositions produce the same key") {
    // 1.Nf3 Nc6 2.Nc3 Nf6 and 1.Nc3 Nc6 2.Nf3 Nf6 reach the same board
    // via different move orders — Zobrist keys must match.
    auto play_sequence = [](const MoveList& seq) {
        Position pos;
        REQUIRE(pos.set_from_fen(STARTPOS_FEN));
        UndoInfo u;
        for (Move m : seq) pos.make_move(m, u);   // undo not needed
        return pos.key;
    };
    uint64_t k1 = play_sequence({::make_move(G1, F3),
                                 ::make_move(B8, C6),
                                 ::make_move(B1, C3),
                                 ::make_move(G8, F6)});
    uint64_t k2 = play_sequence({::make_move(B1, C3),
                                 ::make_move(B8, C6),
                                 ::make_move(G1, F3),
                                 ::make_move(G8, F6)});
    CHECK(k1 == k2);
}
