#pragma once
#include "types.h"
#include <bit>
#include <string>

constexpr Bitboard square_bb(Square s) { return Bitboard(1) << s; }

inline int popcount(Bitboard b) { return std::popcount(b); }

// Least-significant bit as a Square. UB if b == 0.
inline Square lsb(Bitboard b) { return Square(std::countr_zero(b)); }

// Pops and returns the least-significant bit.
inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

std::string pretty(Bitboard b);
