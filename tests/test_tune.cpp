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
