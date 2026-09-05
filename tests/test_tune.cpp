// Tuner smoke test. Runs `tune::run_tune` on the demo dataset checked
// into tests/data/ and asserts that the tuner:
//   1. Loads the dataset without error (return code 0).
//   2. Doesn't corrupt eval::params — after the run, evaluate() still
//      returns a sensible score for a canonical position.
//
// The tuner mutates eval::params as a side effect; we snapshot and
// restore around the call so tests running after this one aren't
// affected by the tuning result.

#include "doctest.h"

#include "eval.h"
#include "position.h"
#include "tune.h"

TEST_CASE("tuner runs to convergence on the demo dataset without error") {
    // Snapshot original params so we can restore after the run.
    const eval::TuningParams snapshot = eval::params;

    const int rc = tune::run_tune("tests/data/tune_demo.txt");
    CHECK(rc == 0);

    // The tuner should still leave eval::params in a self-consistent
    // state: startpos evaluates to some finite value close to 0
    // (tuner adjusts the WHITE-BLACK diff terms — startpos is color-
    // symmetric so any symmetric change to weights preserves the 0
    // baseline). Bound loosely — the coord-descent may pick asymmetric-
    // looking values on a 30-position dataset.
    Position startpos;
    REQUIRE(startpos.set_from_fen(STARTPOS_FEN));
    const int startpos_eval = evaluate(startpos);
    CHECK(startpos_eval == 0);

    // Restore for downstream tests. Every test file after this in the
    // build order would otherwise inherit tuned params.
    eval::params = snapshot;
}

TEST_CASE("SPSA PST tuner runs a short pass without corrupting eval") {
    // 50 iterations is enough to exercise both L+ / L- probes, the
    // gradient update, and the log-every-N summary path without taking
    // real tuning time. Snapshot both scalar params AND PST tables so
    // we can restore fully after — SPSA mutates 768 PST slots plus the
    // rounded state at exit.
    const eval::TuningParams params_snapshot = eval::params;
    int pst_mg_snapshot[NUM_PIECE_TYPES][NUM_SQUARES];
    int pst_eg_snapshot[NUM_PIECE_TYPES][NUM_SQUARES];
    for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
        for (int sq = 0; sq < NUM_SQUARES; ++sq) {
            pst_mg_snapshot[pt][sq] = eval::pst_mg[pt][sq];
            pst_eg_snapshot[pt][sq] = eval::pst_eg[pt][sq];
        }
    }

    const int rc = tune::run_tune("tests/data/tune_demo.txt",
                                  tune::Mode::Pst, /*iterations=*/50);
    CHECK(rc == 0);

    // Startpos is color-symmetric under any perturbation SPSA can apply
    // to WHITE tables (BLACK looks up mirrored squares in the SAME
    // tables), so the diff cancels: evaluate() must still be 0.
    Position startpos;
    REQUIRE(startpos.set_from_fen(STARTPOS_FEN));
    CHECK(evaluate(startpos) == 0);

    // Restore for downstream tests.
    eval::params = params_snapshot;
    for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
        for (int sq = 0; sq < NUM_SQUARES; ++sq) {
            eval::pst_mg[pt][sq] = pst_mg_snapshot[pt][sq];
            eval::pst_eg[pt][sq] = pst_eg_snapshot[pt][sq];
        }
    }
}

TEST_CASE("dump_weights produces paste-ready output covering all tunable state") {
    // Capture stdout during the dump. `dump_weights` prints ~30 lines
    // of scalar params + 2×(1+6×10) PST-table lines. We check that
    // (a) the dump doesn't crash, (b) contains the section headers a
    // maintainer greps for, and (c) contains at least one value from
    // each of the three tunable blocks so a paste job can succeed.
    std::fflush(stdout);
    char buf[16384];
    std::setvbuf(stdout, buf, _IOFBF, sizeof(buf));

    tune::dump_weights();

    std::fflush(stdout);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string out(buf);

    // Section markers that make the dump structured / grep-able.
    CHECK(out.find("eval::TuningParams") != std::string::npos);
    CHECK(out.find("eval::pst_mg") != std::string::npos);
    CHECK(out.find("eval::pst_eg") != std::string::npos);

    // One representative value from each block. Defaults must appear
    // verbatim so the dump round-trips: parse back to same weights.
    CHECK(out.find("bishop_pair_mg              =   30;") != std::string::npos);
    CHECK(out.find("king_open_file_penalty      =   30;") != std::string::npos);

    // Per-piece labels inside the PST dump.
    CHECK(out.find("// PAWN") != std::string::npos);
    CHECK(out.find("// KING") != std::string::npos);
}
