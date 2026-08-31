#include "movegen.h"
#include "attacks.h"
#include "bitboard.h"

namespace {

// File/rank masks used only by pawn generation. Kept file-local so
// bitboard.h stays lean.
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_H_BB = 0x8080808080808080ULL;
constexpr Bitboard RANK_3_BB = 0x0000000000FF0000ULL;
constexpr Bitboard RANK_6_BB = 0x0000FF0000000000ULL;
constexpr Bitboard RANK_1_BB = 0x00000000000000FFULL;
constexpr Bitboard RANK_8_BB = 0xFF00000000000000ULL;

// Emit one normal/ep move OR four promotion moves (Q, R, B, N) when the
// destination sits on the promotion rank.
inline void emit_pawn_move(Square from, Square to, Bitboard promo_rank,
                           MoveType mt, std::vector<Move>& moves) {
    if (square_bb(to) & promo_rank) {
        moves.push_back(make_move(from, to, MT_PROMOTION, QUEEN));
        moves.push_back(make_move(from, to, MT_PROMOTION, ROOK));
        moves.push_back(make_move(from, to, MT_PROMOTION, BISHOP));
        moves.push_back(make_move(from, to, MT_PROMOTION, KNIGHT));
    } else {
        moves.push_back(make_move(from, to, mt));
    }
}

void generate_pawn_moves(const Position& pos, std::vector<Move>& moves) {
    const Color    us    = pos.side_to_move;
    const Bitboard empty = ~pos.occupied;
    const Bitboard enemy = pos.colors[Color(us ^ 1)];
    const Bitboard pawns = pos.pieces[us][PAWN];

    if (us == WHITE) {
        Bitboard single = (pawns << 8) & empty;
        // Double push: only pawns whose single push landed on rank 3
        // (i.e., originated on rank 2). Occupancy of rank 3 is already
        // enforced by `single`; occupancy of rank 4 by the second `& empty`.
        Bitboard dbl    = ((single & RANK_3_BB) << 8) & empty;
        // Capture wraparound guard: exclude file-A pawns before NW shift
        // (they'd wrap to file H one rank up), file-H pawns before NE shift.
        Bitboard cap_nw = ((pawns & ~FILE_A_BB) << 7) & enemy;
        Bitboard cap_ne = ((pawns & ~FILE_H_BB) << 9) & enemy;

        while (single) { Square to = pop_lsb(single);
                         emit_pawn_move(Square(to - 8),  to, RANK_8_BB, MT_NORMAL, moves); }
        while (dbl)    { Square to = pop_lsb(dbl);
                         moves.push_back(make_move(Square(to - 16), to)); }
        while (cap_nw) { Square to = pop_lsb(cap_nw);
                         emit_pawn_move(Square(to - 7),  to, RANK_8_BB, MT_NORMAL, moves); }
        while (cap_ne) { Square to = pop_lsb(cap_ne);
                         emit_pawn_move(Square(to - 9),  to, RANK_8_BB, MT_NORMAL, moves); }

        if (pos.ep_square != NO_SQUARE) {
            Bitboard ep_bb = square_bb(pos.ep_square);
            if (((pawns & ~FILE_A_BB) << 7) & ep_bb)
                moves.push_back(make_move(Square(int(pos.ep_square) - 7),
                                          pos.ep_square, MT_EN_PASSANT));
            if (((pawns & ~FILE_H_BB) << 9) & ep_bb)
                moves.push_back(make_move(Square(int(pos.ep_square) - 9),
                                          pos.ep_square, MT_EN_PASSANT));
        }
    } else {
        Bitboard single = (pawns >> 8) & empty;
        Bitboard dbl    = ((single & RANK_6_BB) >> 8) & empty;
        Bitboard cap_se = ((pawns & ~FILE_H_BB) >> 7) & enemy;
        Bitboard cap_sw = ((pawns & ~FILE_A_BB) >> 9) & enemy;

        while (single) { Square to = pop_lsb(single);
                         emit_pawn_move(Square(to + 8),  to, RANK_1_BB, MT_NORMAL, moves); }
        while (dbl)    { Square to = pop_lsb(dbl);
                         moves.push_back(make_move(Square(to + 16), to)); }
        while (cap_se) { Square to = pop_lsb(cap_se);
                         emit_pawn_move(Square(to + 7),  to, RANK_1_BB, MT_NORMAL, moves); }
        while (cap_sw) { Square to = pop_lsb(cap_sw);
                         emit_pawn_move(Square(to + 9),  to, RANK_1_BB, MT_NORMAL, moves); }

        if (pos.ep_square != NO_SQUARE) {
            Bitboard ep_bb = square_bb(pos.ep_square);
            if (((pawns & ~FILE_H_BB) >> 7) & ep_bb)
                moves.push_back(make_move(Square(int(pos.ep_square) + 7),
                                          pos.ep_square, MT_EN_PASSANT));
            if (((pawns & ~FILE_A_BB) >> 9) & ep_bb)
                moves.push_back(make_move(Square(int(pos.ep_square) + 9),
                                          pos.ep_square, MT_EN_PASSANT));
        }
    }
}

}  // namespace

void generate_moves(const Position& pos, std::vector<Move>& moves) {
    const Color    us         = pos.side_to_move;
    const Bitboard our_pieces = pos.colors[us];
    Bitboard       knights    = pos.pieces[us][KNIGHT];

    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard targets = KNIGHT_ATTACKS[from] & ~our_pieces;
        while (targets) {
            Square to = pop_lsb(targets);
            moves.push_back(make_move(from, to));
        }
    }

    // Loop, not `if`, so contrived test positions with 0 or 2+ kings don't
    // crash the generator. Castling is milestone 6; king-adjacency and
    // move-into-check filtering are milestone 7.
    Bitboard kings = pos.pieces[us][KING];
    while (kings) {
        Square from = pop_lsb(kings);
        Bitboard targets = KING_ATTACKS[from] & ~our_pieces;
        while (targets) {
            Square to = pop_lsb(targets);
            moves.push_back(make_move(from, to));
        }
    }

    generate_pawn_moves(pos, moves);

    // TODO: sliders, castling — see movegen.h milestones.
}
