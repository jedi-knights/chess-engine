#include "bitboard.h"
#include <sstream>

std::string pretty(Bitboard b) {
    std::ostringstream o;
    o << "  +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        o << (r + 1) << " |";
        for (int f = 0; f < 8; ++f) {
            Square s = make_square(File(f), Rank(r));
            o << ' ' << ((b & square_bb(s)) ? 'X' : '.') << " |";
        }
        o << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    o << "    a   b   c   d   e   f   g   h\n";
    return o.str();
}
