#pragma once
#include <string>

// Texel tuner over eval weights, invoked via `./engine tune <dataset>
// [mode] [iterations]`. Reads a labeled dataset (one line per position,
// format `<FEN>;<outcome>` where outcome is 0, 0.5, or 1 from WHITE's
// perspective), fits a sigmoid scale constant K, then optimizes eval
// weights to minimize mean-squared prediction error.
//
// A 30-position demo dataset is checked in at tests/data/tune_demo.txt
// for CI; for real tuning, either fetch a public labeled corpus via
// `scripts/fetch_tuning_dataset.py` (Zurichess quiet-labeled and
// friends) or generate one from self-play via
// `scripts/gen_selfplay_data.py` — the latter is the fallback when
// no public dataset fits, e.g. after a big eval change where the
// engine's play-style has drifted from what published corpora reflect.
//
// Modes:
//   scalar (default) — coordinate descent over eval::params (13 scalar
//                      weights + 5 piece values). Fast; suitable for
//                      small weight sets.
//   pst              — SPSA (Simultaneous Perturbation Stochastic
//                      Approximation) over the 6×64×2 = 768 PST values.
//                      Two dataset evaluations per iteration regardless
//                      of dimensionality, so it scales to the whole
//                      table where coord descent would take hours.
//   all              — scalar first, then pst.
//
// Prints iteration progress and final tuned values to stdout; does NOT
// persist changes across process boundaries. Returns 0 on success,
// non-zero on failure (missing file, empty dataset, malformed line).
//
// End-to-end workflow — Texel loss ≠ Elo, so tuned weights must be
// SPRT-validated before shipping:
//   1. scripts/fetch_tuning_dataset.py            (once)
//   2. make && cp engine engine.baseline          (snapshot pre-tune)
//   3. ./engine tune tests/data/quiet-labeled.txt all 10000
//      → copy printed weights into src/eval.cpp / src/eval.h
//   4. make                                       (rebuild with tuned weights)
//   5. scripts/sprt.py --baseline engine.baseline --tuned engine
//      → wait for SPRT to accept H0 (no gain) or H1 (Elo gain)
namespace tune {

enum class Mode {
    Scalar,
    Pst,
    All,
};

int run_tune(const std::string& dataset_path,
             Mode mode = Mode::Scalar,
             int  iterations = 1000);

// Print the current in-memory eval::params and eval::pst_mg / pst_eg
// as a paste-ready C++ source fragment. Invoked automatically at the
// end of run_tune() so tuned weights can be transferred to
// src/eval.h / src/eval.cpp without hand transcription; also exposed
// as the `dump-weights` subcommand for baseline inspection.
void dump_weights();

}  // namespace tune
