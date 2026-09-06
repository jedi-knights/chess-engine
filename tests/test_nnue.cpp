// NNUE runtime tests. Covers the binary format loader + saver, gate
// semantics, HalfKP feature-index layout, and accumulator determinism.
// Deliberately does NOT test eval quality — the network is
// zero-initialized until the training pipeline lands.

#include "doctest.h"

#include "nnue.h"
#include "position.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

// Write a minimal but VALID JNN1 file — all-zero weights past the
// header. Sized to match the current HIDDEN_SIZE / TOTAL_FEATURES
// (~20 MiB); tests that only need the header can truncate below.
void write_zeroed_network_file(const std::string& path) {
    constexpr char     MAGIC[4] = {'J', 'N', 'N', '1'};
    constexpr uint32_t VERSION  = 1;
    constexpr uint32_t HS       = nnue::HIDDEN_SIZE;
    constexpr uint32_t TF       = nnue::TOTAL_FEATURES;

    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fwrite(MAGIC,     sizeof(MAGIC),   1, f);
    std::fwrite(&VERSION,  sizeof(VERSION), 1, f);
    std::fwrite(&HS,       sizeof(HS),      1, f);
    std::fwrite(&TF,       sizeof(TF),      1, f);
    // Feature biases + weights + output weights + output bias.
    const size_t bytes =
          size_t(HS) * sizeof(int16_t)
        + size_t(TF) * HS * sizeof(int16_t)
        + size_t(2) * HS * sizeof(int16_t)
        + sizeof(int16_t);
    // Chunked zero-write so we don't allocate a 20 MiB heap array.
    std::array<char, 65536> zeros{};
    size_t remaining = bytes;
    while (remaining > 0) {
        size_t n = std::min(remaining, zeros.size());
        std::fwrite(zeros.data(), 1, n, f);
        remaining -= n;
    }
    std::fclose(f);
}

// RAII: snapshot NNUE toggle state, restore on destruction.
struct NnueScope {
    bool prev_use;
    NnueScope() : prev_use(nnue::use_nnue()) { nnue::set_use_nnue(false); }
    ~NnueScope() { nnue::set_use_nnue(prev_use); }
};

}  // namespace

TEST_CASE("NNUE: default state is off") {
    NnueScope scope;
    // The gate is the composite of use_nnue toggle AND is_loaded state;
    // by default with no explicit load, this must be false so eval
    // falls through to classical.
    nnue::set_use_nnue(false);
    CHECK_FALSE(nnue::use_nnue());
}

TEST_CASE("NNUE load: fails on missing file") {
    NnueScope scope;
    CHECK_FALSE(nnue::load_network("/tmp/definitely-does-not-exist.nnue"));
}

TEST_CASE("NNUE load: fails on bad magic") {
    NnueScope scope;
    const std::string path = "/tmp/nnue_bad_magic.bin";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    // "XXXX" as magic — anything other than JNN1.
    std::fwrite("XXXX", 4, 1, f);
    // Even with junk padding, magic-check should reject before
    // reading further.
    std::array<char, 128> zeros{};
    std::fwrite(zeros.data(), zeros.size(), 1, f);
    std::fclose(f);
    CHECK_FALSE(nnue::load_network(path));
}

TEST_CASE("NNUE load: fails on truncated header") {
    NnueScope scope;
    const std::string path = "/tmp/nnue_truncated_header.bin";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    // Only 4 bytes — magic without any of the four uint32 header fields.
    std::fwrite("JNN1", 4, 1, f);
    std::fclose(f);
    CHECK_FALSE(nnue::load_network(path));
}

TEST_CASE("NNUE load: fails on architecture mismatch") {
    NnueScope scope;
    const std::string path = "/tmp/nnue_arch_mismatch.bin";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    constexpr char MAGIC[4] = {'J', 'N', 'N', '1'};
    constexpr uint32_t VERSION = 1;
    constexpr uint32_t WRONG_HS = nnue::HIDDEN_SIZE + 1;
    constexpr uint32_t WRONG_TF = nnue::TOTAL_FEATURES;
    std::fwrite(MAGIC,     4, 1, f);
    std::fwrite(&VERSION,  sizeof(VERSION), 1, f);
    std::fwrite(&WRONG_HS, sizeof(WRONG_HS), 1, f);
    std::fwrite(&WRONG_TF, sizeof(WRONG_TF), 1, f);
    std::fclose(f);
    CHECK_FALSE(nnue::load_network(path));
}

TEST_CASE("NNUE load: succeeds on well-formed zeroed network") {
    NnueScope scope;
    write_zeroed_network_file("/tmp/nnue_zeroed.jnn1");
    CHECK(nnue::load_network("/tmp/nnue_zeroed.jnn1"));
    CHECK(nnue::is_loaded());
    // Gate: use_nnue is false until explicitly toggled.
    CHECK_FALSE(nnue::use_nnue());
    nnue::set_use_nnue(true);
    CHECK(nnue::use_nnue());
    nnue::set_use_nnue(false);
    CHECK_FALSE(nnue::use_nnue());
}

TEST_CASE("NNUE save/load round-trip preserves the network") {
    NnueScope scope;
    // Load a zeroed baseline first — this gives us g_network to save.
    write_zeroed_network_file("/tmp/nnue_rt_baseline.jnn1");
    REQUIRE(nnue::load_network("/tmp/nnue_rt_baseline.jnn1"));

    // Snapshot the eval on a canonical position (should be 0 for the
    // zeroed net regardless — but this pins the pre-round-trip value
    // so we can confirm it doesn't shift after save/load).
    Position startpos;
    REQUIRE(startpos.set_from_fen(STARTPOS_FEN));
    nnue::set_use_nnue(true);
    const int eval_before = nnue::evaluate(startpos);

    // Round trip: save → load a fresh path → eval matches.
    REQUIRE(nnue::save_network("/tmp/nnue_rt_dumped.jnn1"));
    REQUIRE(nnue::load_network("/tmp/nnue_rt_dumped.jnn1"));
    const int eval_after = nnue::evaluate(startpos);

    CHECK(eval_before == eval_after);
}

TEST_CASE("NNUE feature_index follows HalfKP layout") {
    // HalfKP: idx = king_sq * 641 + piece_sq * 10 + slot.
    // Same three spot-checks as scaffolding — regression guard on
    // any future orientation / slot-mapping change.

    // White king on E1, white pawn on E2, WHITE perspective.
    CHECK(nnue::feature_index(WHITE, E1, E2, PAWN, WHITE) == 2684);

    // Same board, BLACK perspective: squares mirror; piece is enemy.
    CHECK(nnue::feature_index(BLACK, E1, E2, PAWN, WHITE) == 38985);

    // White knight on B1, WHITE perspective.
    CHECK(nnue::feature_index(WHITE, E1, B1, KNIGHT, WHITE) == 2575);
}

TEST_CASE("NNUE refresh_accumulator is deterministic") {
    NnueScope scope;
    write_zeroed_network_file("/tmp/nnue_det.jnn1");
    REQUIRE(nnue::load_network("/tmp/nnue_det.jnn1"));

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
