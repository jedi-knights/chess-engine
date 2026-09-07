#pragma once
#include <iosfwd>
#include <string>

// Runs the UCI (Universal Chess Interface) command loop. Reads command
// lines from `in`, writes protocol replies to `out`. Blocks until "quit"
// is received or `in` reaches EOF.
//
// Taking streams as parameters (rather than reading std::cin directly)
// keeps the loop testable end-to-end via std::istringstream / ostringstream.
void uci_loop(std::istream& in, std::ostream& out);

// Format a search score for the UCI `info` line's `score ...` field.
// Returns `"score mate <N>"` when `score` is inside the mate range
// (MATE_SCORE ± MATE_RANGE from search.h), converting plies-to-mate to
// full moves (rounded up, matching Stockfish convention). Returns
// `"score cp <N>"` otherwise. Exposed for unit testing — the caller
// splices the result into the wider info line.
std::string format_uci_score(int score);
