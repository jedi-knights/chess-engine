// NNUE scaffolding tests. Pins the runtime plumbing — load / gate /
// refresh / fallback — without asserting anything about eval quality
// (network is a zeroed placeholder in this scaffolding commit).
//
// What we verify:
//   1. Default state — not loaded, use_nnue() is false.
//   2. Bad path → load fails, loaded state unchanged.
//   3. Non-Stockfish magic → load fails.
//   4. SF magic header → load succeeds, use_nnue() gated on the toggle.
//   5. Feature-index layout follows HalfKP: king_sq * 641 + piece_sq * 10 + slot.
//   6. refresh_accumulator is deterministic for a given (position, network).
//
// Explicitly NOT verified: eval scores are meaningful (they're all 0
// with a zeroed placeholder net), or that the score matches classical.

#include "doctest.h"

#include "nnue.h"
#include "position.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

namespace {

// Writes a minimal file with the Stockfish HalfKP magic prefix.
// Padding is zeros — enough to satisfy the scaffolding loader; a real
// parse would demand the full binary layout.
void write_sf_stub(const std::string& path) {
    constexpr uint32_t SF_MAGIC = 0x7AF32F16;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fwrite(&SF_MAGIC, sizeof(SF_MAGIC), 1, f);
    // Extra zero padding to make the file look non-trivially sized.
    const std::array<uint8_t, 128> zeros{};
    std::fwrite(zeros.data(), zeros.size(), 1, f);
    std::fclose(f);
}

// RAII: snapshot NNUE global toggle state, restore on destruction so
// tests running after this one aren't affected. `is_loaded()` state
// is process-global and NOT restored — once you load a network in a
// test it stays loaded. That's fine for the current test set (all
// downstream tests either don't care or check the fallback path
// which gates on use_nnue not is_loaded).
struct NnueScope {
    bool prev_use;
    NnueScope() : prev_use(nnue::use_nnue()) { nnue::set_use_nnue(false); }
    ~NnueScope() { nnue::set_use_nnue(prev_use); }
};

}  // namespace

TEST_CASE("NNUE: default state is off + not loaded") {
    NnueScope scope;
    CHECK_FALSE(nnue::use_nnue());   // toggle off unless explicitly enabled
    // is_loaded() may be true if an earlier test loaded a stub; we
    // just verify the composite `use_nnue()` gate is false.
}

TEST_CASE("NNUE: load fails on missing file") {
    NnueScope scope;
    CHECK_FALSE(nnue::load_network("/tmp/definitely-does-not-exist.nnue"));
}

TEST_CASE("NNUE: load fails on file without Stockfish magic") {
    NnueScope scope;
    const std::string path = "/tmp/nnue_bad_magic.bin";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    const uint32_t not_magic = 0xDEADBEEF;
    std::fwrite(&not_magic, sizeof(not_magic), 1, f);
    std::fclose(f);
    CHECK_FALSE(nnue::load_network(path));
}

TEST_CASE("NNUE: load succeeds on Stockfish-magic stub") {
    NnueScope scope;
    write_sf_stub("/tmp/nnue_sf_stub.nnue");
    CHECK(nnue::load_network("/tmp/nnue_sf_stub.nnue"));
    CHECK(nnue::is_loaded());
    // Gate: use_nnue is false until explicitly toggled.
    CHECK_FALSE(nnue::use_nnue());
    nnue::set_use_nnue(true);
    CHECK(nnue::use_nnue());
    // Toggle back before scope exit restores; ensures the "and" gate.
    nnue::set_use_nnue(false);
    CHECK_FALSE(nnue::use_nnue());
}

TEST_CASE("NNUE: feature_index follows HalfKP layout") {
    // HalfKP: idx = king_sq * 641 + piece_sq * 10 + slot
    // Slot depends on (perspective, piece_color, piece_type).
    // Spot-check three cases.
    //
    // White king on E1, white pawn on E2, WHITE perspective:
    //   king_sq = E1 = 4, piece_sq = E2 = 12, slot = own pawn = 0
    //   idx = 4 * 641 + 12 * 10 + 0 = 2564 + 120 + 0 = 2684
    CHECK(nnue::feature_index(WHITE, E1, E2, PAWN, WHITE) == 2684);

    // Same board, BLACK perspective: king_sq oriented flips ranks,
    // piece_sq flips ranks, piece is "enemy" from black's POV.
    //   king_sq = E1 = 4  → oriented for black = 4 XOR 56 = 60 (E8)
    //   piece_sq = E2 = 12 → oriented = 12 XOR 56 = 52 (E7)
    //   slot = enemy pawn = 5 (PIECES_PER_SIDE)
    //   idx = 60 * 641 + 52 * 10 + 5 = 38460 + 520 + 5 = 38985
    CHECK(nnue::feature_index(BLACK, E1, E2, PAWN, WHITE) == 38985);

    // White knight on B1, WHITE perspective (king E1):
    //   slot = own knight = 1
    //   idx = 4 * 641 + 1 * 10 + 1 = 2564 + 10 + 1 = 2575
    CHECK(nnue::feature_index(WHITE, E1, B1, KNIGHT, WHITE) == 2575);
}

TEST_CASE("NNUE: refresh_accumulator is deterministic") {
    // With the scaffolding's zeroed network, both accumulators end
    // up all-zero after refresh (biases are 0, weight columns are 0).
    // The determinism check is that a second refresh produces
    // identical bits.
    NnueScope scope;
    write_sf_stub("/tmp/nnue_sf_stub.nnue");
    REQUIRE(nnue::load_network("/tmp/nnue_sf_stub.nnue"));

    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));

    nnue::Accumulator a, b;
    nnue::refresh_accumulator(pos, a);
    nnue::refresh_accumulator(pos, b);

    CHECK(a.computed);
    CHECK(b.computed);
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            CHECK(a.values[c][i] == b.values[c][i]);
        }
    }
}
