#pragma once
#include <string>

// Texel tuner over eval weights, invoked via `./engine tune <dataset>
// [mode] [iterations]`. Reads a labeled dataset (one line per position,
// format `<FEN>;<outcome>` where outcome is 0, 0.5, or 1 from WHITE's
// perspective), fits a sigmoid scale constant K, then optimizes eval
// weights to minimize mean-squared prediction error.
//
// A 30-position demo dataset is checked in at tests/data/tune_demo.txt
// for CI; for real tuning, fetch Zurichess quiet-labeled (or a similar
// EPD corpus) via `scripts/fetch_tuning_dataset.py`.
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
namespace tune {

enum class Mode {
    Scalar,
    Pst,
    All,
};

int run_tune(const std::string& dataset_path,
             Mode mode = Mode::Scalar,
             int  iterations = 1000);

}  // namespace tune
