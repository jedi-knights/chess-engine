#pragma once
#include <string>

// Texel-style coordinate-descent tuner. Reads a labeled dataset
// (one line per position, format: `<FEN>;<outcome>` where outcome is
// 0, 0.5, or 1 from WHITE's perspective), fits a sigmoid scale
// constant K, then coordinate-descends over eval::params (see eval.h)
// to minimize mean-squared prediction error. Prints iteration progress
// and final proposed weight values to stdout; does NOT persist changes.
//
// Invoked via the `tune` subcommand — `./engine tune <dataset_path>`.
// Returns 0 on success, non-zero on failure (missing file, empty
// dataset, malformed line).
namespace tune {

int run_tune(const std::string& dataset_path);

}  // namespace tune
