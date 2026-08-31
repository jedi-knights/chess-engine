#include "zobrist.h"
#include "bitboard.h"

namespace zobrist {

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
    for (int c = 0; c < NUM_COLORS; ++c)
        for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt)
            for (int s = 0; s < NUM_SQUARES; ++s)
                PIECE_SQ[c][pt][s] = splitmix64(state);
    for (int i = 0; i < 16; ++i) CASTLING[i] = splitmix64(state);
    for (int i = 0; i < 8;  ++i) EP_FILE[i]  = splitmix64(state);
    SIDE = splitmix64(state);
}

uint64_t compute(const Position& pos) {
    uint64_t k = 0;
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces[c][pt];
            while (bb) {
                Square s = pop_lsb(bb);
                k ^= PIECE_SQ[c][pt][s];
            }
        }
    }
    k ^= CASTLING[pos.castling & 15];
    if (pos.ep_square != NO_SQUARE) k ^= EP_FILE[file_of(pos.ep_square)];
    if (pos.side_to_move == BLACK)  k ^= SIDE;
    return k;
}

}  // namespace zobrist
