#include "attacks.h"
#include "bitboard.h"

Bitboard KNIGHT_ATTACKS[NUM_SQUARES];
Bitboard KING_ATTACKS[NUM_SQUARES];
Bitboard PAWN_ATTACKS[NUM_COLORS][NUM_SQUARES];

static Bitboard shift_one(Square s, int df, int dr) {
    int f = (s & 7) + df;
    int r = (s >> 3) + dr;
    if (f < 0 || f > 7 || r < 0 || r > 7) {
        return 0;
    }
    return square_bb(Square((r << 3) | f));
}

void init_attacks() {
    static const int KNIGHT_OFFSETS[8][2] = {
        {-1,  2}, { 1,  2}, { 2,  1}, { 2, -1},
        { 1, -2}, {-1, -2}, {-2, -1}, {-2,  1},
    };

    for (int i = 0; i < NUM_SQUARES; ++i) {
        Square s = Square(i);

        Bitboard n = 0;
        for (const auto& o : KNIGHT_OFFSETS) {
            n |= shift_one(s, o[0], o[1]);
        }
        KNIGHT_ATTACKS[i] = n;

        Bitboard k = 0;
        for (int df = -1; df <= 1; ++df) {
            for (int dr = -1; dr <= 1; ++dr) {
                if (df != 0 || dr != 0) {
                    k |= shift_one(s, df, dr);
                }
            }
        }
        KING_ATTACKS[i] = k;

        PAWN_ATTACKS[WHITE][i] = shift_one(s, -1, 1) | shift_one(s, 1, 1);
        PAWN_ATTACKS[BLACK][i] = shift_one(s, -1, -1) | shift_one(s, 1, -1);
    }
}
