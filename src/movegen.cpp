#include "movegen.h"
#include "attacks.h"
#include "bitboard.h"

#include <algorithm>

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

// Walk one ray from `from` in direction (df, dr) one square at a time.
// Stops at: board edge, own piece (exclude destination), enemy piece
// (include as capture, then stop). Naive per-step iteration — magic
// bitboards can replace this later without touching callers.
void gen_ray(Square from, int df, int dr,
             Bitboard our, Bitboard enemy,
             std::vector<Move>& moves) {
    int f = file_of(from) + df;
    int r = rank_of(from) + dr;
    while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
        Square   to    = make_square(File(f), Rank(r));
        Bitboard to_bb = square_bb(to);
        if (to_bb & our)   return;              // own piece blocks
        moves.push_back(make_move(from, to));
        if (to_bb & enemy) return;              // captured — ray stops
        f += df;
        r += dr;
    }
}

// Return true if the first piece encountered walking (df, dr) from `sq`
// is one of `attackers`. Used by is_square_attacked to detect sliding
// threats without generating a move list.
bool ray_hits_attacker(Square sq, int df, int dr,
                       Bitboard occupied, Bitboard attackers) {
    int f = file_of(sq) + df;
    int r = rank_of(sq) + dr;
    while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
        Bitboard bb = square_bb(make_square(File(f), Rank(r)));
        if (bb & occupied) return (bb & attackers) != 0;
        f += df;
        r += dr;
    }
    return false;
}

// Is `sq` attacked by any piece of color `by` in the current occupancy?
// Symmetric-attack trick for leapers: pieces that attack `sq` sit on the
// same squares that a same-role piece AT `sq` would attack — with the
// pawn direction inverted, since pawns only attack "forward."
bool is_square_attacked(const Position& pos, Square sq, Color by) {
    if (PAWN_ATTACKS[Color(by ^ 1)][sq] & pos.pieces[by][PAWN])   return true;
    if (KNIGHT_ATTACKS[sq]              & pos.pieces[by][KNIGHT]) return true;
    if (KING_ATTACKS[sq]                & pos.pieces[by][KING])   return true;

    const Bitboard occ = pos.occupied;
    const Bitboard bq  = pos.pieces[by][BISHOP] | pos.pieces[by][QUEEN];
    const Bitboard rq  = pos.pieces[by][ROOK]   | pos.pieces[by][QUEEN];
    if (ray_hits_attacker(sq,  1,  1, occ, bq)) return true;
    if (ray_hits_attacker(sq, -1,  1, occ, bq)) return true;
    if (ray_hits_attacker(sq,  1, -1, occ, bq)) return true;
    if (ray_hits_attacker(sq, -1, -1, occ, bq)) return true;
    if (ray_hits_attacker(sq,  1,  0, occ, rq)) return true;
    if (ray_hits_attacker(sq, -1,  0, occ, rq)) return true;
    if (ray_hits_attacker(sq,  0,  1, occ, rq)) return true;
    if (ray_hits_attacker(sq,  0, -1, occ, rq)) return true;
    return false;
}

void generate_castling(const Position& pos, std::vector<Move>& moves) {
    const Color    us   = pos.side_to_move;
    const Color    them = Color(us ^ 1);
    const Bitboard occ  = pos.occupied;

    // Emit one castling move iff: right is present, squares between king
    // and rook are empty, and king does not start, pass through, or land
    // on a square attacked by the opponent. B1/B8 emptiness matters for
    // the rook's transit but NOT for check-safety — the king does not
    // pass through it.
    auto try_castle = [&](int right, Square king_from, Square king_to,
                          Bitboard between_empty, Square transit) {
        if (!(pos.castling & right))                    return;
        if (occ & between_empty)                        return;
        if (is_square_attacked(pos, king_from, them))   return;
        if (is_square_attacked(pos, transit,   them))   return;
        if (is_square_attacked(pos, king_to,   them))   return;
        moves.push_back(make_move(king_from, king_to, MT_CASTLING));
    };

    if (us == WHITE) {
        try_castle(WHITE_OO,  E1, G1, square_bb(F1) | square_bb(G1),                    F1);
        try_castle(WHITE_OOO, E1, C1, square_bb(B1) | square_bb(C1) | square_bb(D1),    D1);
    } else {
        try_castle(BLACK_OO,  E8, G8, square_bb(F8) | square_bb(G8),                    F8);
        try_castle(BLACK_OOO, E8, C8, square_bb(B8) | square_bb(C8) | square_bb(D8),    D8);
    }
}

void generate_slider_moves(const Position& pos, std::vector<Move>& moves) {
    const Color    us    = pos.side_to_move;
    const Bitboard our   = pos.colors[us];
    const Bitboard enemy = pos.colors[Color(us ^ 1)];

    // Bishops and queens share diagonal rays; rooks and queens share
    // orthogonal rays. OR the piece bitboards to iterate each set once.
    Bitboard diag = pos.pieces[us][BISHOP] | pos.pieces[us][QUEEN];
    while (diag) {
        Square from = pop_lsb(diag);
        gen_ray(from,  1,  1, our, enemy, moves);   // NE
        gen_ray(from, -1,  1, our, enemy, moves);   // NW
        gen_ray(from,  1, -1, our, enemy, moves);   // SE
        gen_ray(from, -1, -1, our, enemy, moves);   // SW
    }

    Bitboard orth = pos.pieces[us][ROOK] | pos.pieces[us][QUEEN];
    while (orth) {
        Square from = pop_lsb(orth);
        gen_ray(from,  1,  0, our, enemy, moves);   // E
        gen_ray(from, -1,  0, our, enemy, moves);   // W
        gen_ray(from,  0,  1, our, enemy, moves);   // N
        gen_ray(from,  0, -1, our, enemy, moves);   // S
    }
}

// Legality filter: apply `m`, ask whether the mover's king is now attacked
// by the opponent, unapply. Cheap in the current naive-ray movegen; magic
// bitboards + pin/check awareness can bypass this for most moves later.
bool is_legal(Position& pos, Move m) {
    const Color us = pos.side_to_move;
    UndoInfo u;
    pos.make_move(m, u);
    // Kings can be temporarily absent in contrived perft positions (see the
    // "Position 4 has no white king" note in CLAUDE.md) — treat as legal so
    // the generator still returns something for those artificial cases.
    Bitboard king_bb = pos.pieces[us][KING];
    bool safe = (king_bb == 0) ||
                !is_square_attacked(pos, lsb(king_bb), pos.side_to_move);
    pos.unmake_move(m, u);
    return safe;
}

}  // namespace

void generate_moves(Position& pos, std::vector<Move>& moves) {
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
    generate_slider_moves(pos, moves);
    generate_castling(pos, moves);

    // Post-filter: drop any pseudo-legal move that leaves our king in check.
    // Castling is already legality-filtered inside generate_castling, but
    // is_legal is cheap on those and produces the same answer, so uniform
    // filtering here is simpler than trying to short-circuit them out.
    moves.erase(
        std::remove_if(moves.begin(), moves.end(),
                       [&](Move m) { return !is_legal(pos, m); }),
        moves.end());
}
