#pragma once
#include <iosfwd>

// Runs the UCI (Universal Chess Interface) command loop. Reads command
// lines from `in`, writes protocol replies to `out`. Blocks until "quit"
// is received or `in` reaches EOF.
//
// Taking streams as parameters (rather than reading std::cin directly)
// keeps the loop testable end-to-end via std::istringstream / ostringstream.
void uci_loop(std::istream& in, std::ostream& out);
