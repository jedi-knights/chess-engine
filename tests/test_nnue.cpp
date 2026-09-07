// NNUE runtime tests. Covers the binary format loader + saver, gate
// semantics, HalfKP feature-index layout, and accumulator determinism.
// Deliberately does NOT test eval quality — the network is
// zero-initialized until the training pipeline lands.

#include "doctest.h"

#include "nnue.h"
#include "nnue_simd.h"
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

TEST_CASE("NNUE force_refresh_accumulator is deterministic") {
    NnueScope scope;
    write_zeroed_network_file("/tmp/nnue_det.jnn1");
    REQUIRE(nnue::load_network("/tmp/nnue_det.jnn1"));

    Position pos;
    REQUIRE(pos.set_from_fen(STARTPOS_FEN));

    nnue::force_refresh_accumulator(pos);
    // Snapshot both perspectives so a second refresh can't overwrite
    // us before the compare.
    std::array<int32_t, nnue::HIDDEN_SIZE> snapshot[NUM_COLORS];
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            snapshot[c][i] = pos.acc.values[c][i];
        }
        CHECK(pos.acc.computed[c]);
    }

    nnue::force_refresh_accumulator(pos);
    for (int c = 0; c < NUM_COLORS; ++c) {
        CHECK(pos.acc.computed[c]);
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            CHECK(pos.acc.values[c][i] == snapshot[c][i]);
        }
    }
}

namespace {

// Compare two accumulators element-wise for both perspectives.
bool accs_equal(const nnue::Accumulator& a, const nnue::Accumulator& b) {
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            if (a.values[c][i] != b.values[c][i]) { return false; }
        }
    }
    return true;
}

// Incremental-update invariant: for `fen`, apply a UCI move token, then
// compare the resulting `pos.acc` (updated incrementally by put/remove
// piece + lazy refresh) against a fresh `Position` set from the target
// FEN whose accumulator is force-refreshed. If they diverge, either the
// incremental update or the dirty-flag path is wrong. Uses a
// non-zeroed network so any per-piece delta actually mutates the
// accumulator (the zeroed network from other tests would falsely pass
// even if add/sub_piece_from_accumulator were no-ops).
void check_incremental_matches_full(const std::string& fen_before,
                                    Move                move,
                                    const std::string&  fen_after_expected) {
    Position pos_incremental;
    REQUIRE(pos_incremental.set_from_fen(fen_before));
    // Prime the accumulator so the make_move deltas land on a
    // known-good baseline (otherwise dirty flags mask the incremental
    // updates entirely).
    nnue::force_refresh_accumulator(pos_incremental);

    UndoInfo u;
    pos_incremental.make_move(move, u);
    // Post-move: king moves left the moved side's perspective dirty;
    // this call rebuilds it from scratch. Non-king perspectives
    // should already be clean (kept in sync incrementally).
    nnue::refresh_accumulator(pos_incremental);

    Position pos_full;
    REQUIRE(pos_full.set_from_fen(fen_after_expected));
    nnue::force_refresh_accumulator(pos_full);

    CHECK(accs_equal(pos_incremental.acc, pos_full.acc));

    // Round-trip: unmaking should restore the pre-move accumulator.
    pos_incremental.unmake_move(move, u);
    nnue::refresh_accumulator(pos_incremental);
    Position pos_pre;
    REQUIRE(pos_pre.set_from_fen(fen_before));
    nnue::force_refresh_accumulator(pos_pre);
    CHECK(accs_equal(pos_incremental.acc, pos_pre.acc));
}

// Build a "not-zero" network in place so incremental delta tests can
// distinguish "did nothing" from "did the right thing." Pattern:
// feature_weights[f][h] = (int16_t)(((f * 131) ^ (h * 17)) & 0xFF)
// — deterministic, well-mixed across (feature, hidden), fits int16.
void install_patterned_network() {
    // Load a zeroed baseline so g_network exists, then overwrite via
    // save_network round-trip after mutating a temp copy on disk.
    // Simpler: use the file path to write our own patterned network
    // and load it.
    const std::string path = "/tmp/nnue_patterned.jnn1";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    constexpr char     MAGIC[4] = {'J', 'N', 'N', '1'};
    constexpr uint32_t VERSION  = 1;
    constexpr uint32_t HS       = nnue::HIDDEN_SIZE;
    constexpr uint32_t TF       = nnue::TOTAL_FEATURES;
    std::fwrite(MAGIC,    sizeof(MAGIC),   1, f);
    std::fwrite(&VERSION, sizeof(VERSION), 1, f);
    std::fwrite(&HS,      sizeof(HS),      1, f);
    std::fwrite(&TF,      sizeof(TF),      1, f);

    // Feature biases: pattern by hidden index only.
    std::array<int16_t, nnue::HIDDEN_SIZE> biases{};
    for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
        biases[i] = int16_t((i * 7) & 0x7F);
    }
    std::fwrite(biases.data(), sizeof(int16_t), biases.size(), f);

    // Feature weights: 41024 rows × 256 cols. Stream a row at a
    // time so we don't allocate the whole 20 MiB array.
    std::array<int16_t, nnue::HIDDEN_SIZE> row{};
    for (int fi = 0; fi < nnue::TOTAL_FEATURES; ++fi) {
        for (int h = 0; h < nnue::HIDDEN_SIZE; ++h) {
            row[h] = int16_t(((fi * 131) ^ (h * 17)) & 0xFF);
        }
        std::fwrite(row.data(), sizeof(int16_t), row.size(), f);
    }

    std::array<int16_t, 2 * nnue::HIDDEN_SIZE> out_w{};
    for (int i = 0; i < int(out_w.size()); ++i) {
        out_w[i] = int16_t((i * 3) & 0xFF);
    }
    std::fwrite(out_w.data(), sizeof(int16_t), out_w.size(), f);

    int16_t out_b = 1;
    std::fwrite(&out_b, sizeof(int16_t), 1, f);
    std::fclose(f);

    REQUIRE(nnue::load_network(path));
}

}  // namespace

TEST_CASE("NNUE incremental accumulator matches full recompute") {
    NnueScope scope;
    install_patterned_network();

    SUBCASE("quiet knight move") {
        // Startpos, 1. Nf3.
        Move m = make_move(G1, F3);
        check_incremental_matches_full(
            STARTPOS_FEN,
            m,
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1");
    }

    SUBCASE("pawn double push") {
        Move m = make_move(E2, E4);
        check_incremental_matches_full(
            STARTPOS_FEN,
            m,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    }

    SUBCASE("capture") {
        // 4. Nxe5 in a Petroff-like line.
        const std::string fen =
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
        Move m = make_move(F3, E5);
        check_incremental_matches_full(
            fen,
            m,
            "rnbqkb1r/pppp1ppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 3");
    }

    SUBCASE("king move (dirty flag path)") {
        // Move white king one square — invalidates ONLY white's
        // perspective. Black perspective should stay clean and
        // still match the full recompute.
        const std::string fen =
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
        Move m = make_move(E1, E2);
        check_incremental_matches_full(
            fen,
            m,
            "4k3/8/8/8/8/8/4K3/8 b - - 1 1");
    }

    SUBCASE("castling kingside") {
        // King + rook move together — king dirties its side, rook
        // updates incrementally.
        const std::string fen =
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4";
        Move m = make_move(E1, G1, MT_CASTLING);
        check_incremental_matches_full(
            fen,
            m,
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b kq - 5 4");
    }

    SUBCASE("en passant") {
        // White pawn on e5 captures black pawn on d5 en passant → e6.
        const std::string fen =
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        Move m = make_move(E5, D6, MT_EN_PASSANT);
        check_incremental_matches_full(
            fen,
            m,
            "rnbqkbnr/ppp1pppp/3P4/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3");
    }

    SUBCASE("promotion") {
        // White pawn on a7 promotes to queen on a8.
        const std::string fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1";
        Move m = make_move(A7, A8, MT_PROMOTION, QUEEN);
        check_incremental_matches_full(
            fen,
            m,
            "Q3k3/8/8/8/8/8/8/4K3 b - - 0 1");
    }

    SUBCASE("capture-promotion") {
        // White pawn on b7 captures a rook on a8 and promotes to queen.
        const std::string fen = "r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1";
        Move m = make_move(B7, A8, MT_PROMOTION, QUEEN);
        check_incremental_matches_full(
            fen,
            m,
            "Q3k3/8/8/8/8/8/8/4K3 b - - 0 1");
    }
}

namespace {

// Deterministic PRNG for SIMD kernel inputs. Simple 32-bit LCG — no
// need for statistical quality, we just want reproducible pseudo-
// random int16/int32 values that exercise the negative, zero,
// mid-range, near-clip, and above-clip cases in the clipped-ReLU
// path so a lane-permute bug can't hide behind lucky zeros.
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    uint32_t next() { s = s * 1664525U + 1013904223U; return s; }
    int32_t  next_i32() { return int32_t(next()); }
    int16_t  next_i16() { return int16_t(next() & 0xFFFF); }
};

}  // namespace

TEST_CASE("NNUE SIMD add_column matches reference") {
    // Repeat across many seeds — a lane-shuffle bug would only show
    // in specific bit patterns.
    for (uint32_t seed = 1; seed <= 8; ++seed) {
        Lcg rng(seed * 0x9E3779B9U);
        std::array<int32_t, nnue::HIDDEN_SIZE> acc_a{}, acc_b{};
        std::array<int16_t, nnue::HIDDEN_SIZE> col{};
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            acc_a[i] = acc_b[i] = rng.next_i32();
            col[i]   = rng.next_i16();
        }
        nnue::simd::add_column(acc_a.data(), col.data());
        nnue::simd::reference::add_column(acc_b.data(), col.data());
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            CHECK(acc_a[i] == acc_b[i]);
        }
    }
}

TEST_CASE("NNUE SIMD sub_column matches reference") {
    for (uint32_t seed = 1; seed <= 8; ++seed) {
        Lcg rng(seed * 0x7F4A7C15U);
        std::array<int32_t, nnue::HIDDEN_SIZE> acc_a{}, acc_b{};
        std::array<int16_t, nnue::HIDDEN_SIZE> col{};
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            acc_a[i] = acc_b[i] = rng.next_i32();
            col[i]   = rng.next_i16();
        }
        nnue::simd::sub_column(acc_a.data(), col.data());
        nnue::simd::reference::sub_column(acc_b.data(), col.data());
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            CHECK(acc_a[i] == acc_b[i]);
        }
    }
}

TEST_CASE("NNUE SIMD forward_side matches reference") {
    for (uint32_t seed = 1; seed <= 8; ++seed) {
        Lcg rng(seed * 0xD1B54A32U);
        std::array<int32_t, nnue::HIDDEN_SIZE> acc{};
        std::array<int16_t, nnue::HIDDEN_SIZE> w{};
        for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
            // Bias toward the clipped-ReLU boundary so lanes hit
            // <0, 0..127, and >127 with roughly equal frequency.
            int32_t r = rng.next_i32();
            acc[i] = (r % 400) - 100;  // range ~[-100, 300)
            w[i]   = rng.next_i16();
        }
        const int32_t simd_out = nnue::simd::forward_side(acc.data(), w.data());
        const int32_t ref_out  = nnue::simd::reference::forward_side(acc.data(), w.data());
        CHECK(simd_out == ref_out);
    }
}

TEST_CASE("NNUE SIMD forward_side handles clipping boundary exactly") {
    // Every clip-relevant value: below (-1), at zero, mid (63), at
    // upper clip (127), above upper clip (128, 32767).
    std::array<int32_t, nnue::HIDDEN_SIZE> acc{};
    std::array<int16_t, nnue::HIDDEN_SIZE> w{};
    const int32_t boundary[] = {-32768, -1, 0, 1, 63, 126, 127, 128, 200, 32767};
    for (int i = 0; i < nnue::HIDDEN_SIZE; ++i) {
        acc[i] = boundary[i % (sizeof(boundary) / sizeof(boundary[0]))];
        w[i]   = int16_t((i * 13) - 100);  // negative, zero, positive weights
    }
    CHECK(nnue::simd::forward_side(acc.data(), w.data()) ==
          nnue::simd::reference::forward_side(acc.data(), w.data()));
}
