#include "zobrist.h"
#include "attacks.h"
#include "bitboard.h"

namespace zobrist {

// True iff an ep capture is pseudo-legally available in `pos` — i.e., a
// pawn of the side to move sits on one of the two squares from which it
// could capture into ep_square. Two positions with identical placement
// but only "phantom" ep flags (double-push with no adjacent capturer)
// should hash the same, otherwise repetition detection and TT collisions
// diverge on positions that a caller would treat as equivalent.
bool ep_is_capturable(const Position& pos) {
    if (pos.ep_square == NO_SQUARE) {
        return false;
    }
    const Color us = pos.side_to_move;
    return (PAWN_ATTACKS[Color(us ^ 1)][pos.ep_square]
            & pos.pieces[us][PAWN]) != 0;
}

uint64_t PIECE_SQ[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES];
uint64_t CASTLING[16];
uint64_t EP_FILE [8];
uint64_t SIDE;

namespace {

// splitmix64: fast, high-quality 64-bit PRNG. Seeded with a fixed value
// so the table is deterministic across runs — tests can hash-compare
// positions without depending on wall-clock or /dev/urandom.
uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void init() {
    uint64_t state = 0xC0FFEE12345678ULL;
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
            for (int s = 0; s < NUM_SQUARES; ++s) {
                PIECE_SQ[c][pt][s] = splitmix64(state);
            }
        }
    }
    for (int i = 0; i < 16; ++i) {
        CASTLING[i] = splitmix64(state);
    }
    for (int i = 0; i < 8;  ++i) {
        EP_FILE[i]  = splitmix64(state);
    }
    SIDE = splitmix64(state);
}

uint64_t compute(const Position& pos) {
    uint64_t k = 0;
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces[c][pt];
            while (bb != 0U) {
                Square s = pop_lsb(bb);
                k ^= PIECE_SQ[c][pt][s];
            }
        }
    }
    k ^= CASTLING[pos.castling & 15];
    if (ep_is_capturable(pos))     {
        k ^= EP_FILE[file_of(pos.ep_square)];
    }
    if (pos.side_to_move == BLACK) {
        k ^= SIDE;
    }
    return k;
}

}  // namespace zobrist
