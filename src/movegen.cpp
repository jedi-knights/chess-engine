#include "movegen.h"
#include "attacks.h"
#include "bitboard.h"
#include "magic.h"

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
                           MoveType mt, MoveList& moves) {
    if (square_bb(to) & promo_rank) {
        moves.push_back(make_move(from, to, MT_PROMOTION, QUEEN));
        moves.push_back(make_move(from, to, MT_PROMOTION, ROOK));
        moves.push_back(make_move(from, to, MT_PROMOTION, BISHOP));
        moves.push_back(make_move(from, to, MT_PROMOTION, KNIGHT));
    } else {
        moves.push_back(make_move(from, to, mt));
    }
}

// captures_only=true skips pushes/double-pushes but still emits captures,
// en passant, AND non-capture promotions (they're forcing tactical moves
// that a qsearch shouldn't ignore).
void generate_pawn_moves(const Position& pos, MoveList& moves, bool captures_only) {
    const Color    us    = pos.side_to_move;
    const Bitboard empty = ~pos.occupied;
    const Bitboard enemy = pos.colors[Color(us ^ 1)];
    const Bitboard pawns = pos.pieces[us][PAWN];

    if (us == WHITE) {
        Bitboard single = (pawns << 8) & empty;
        Bitboard dbl    = ((single & RANK_3_BB) << 8) & empty;
        Bitboard cap_nw = ((pawns & ~FILE_A_BB) << 7) & enemy;
        Bitboard cap_ne = ((pawns & ~FILE_H_BB) << 9) & enemy;

        if (captures_only) {
            // Retain only pushes that promote (rank 8) — a queen appearing
            // out of thin air is as forcing as any capture.
            single &= RANK_8_BB;
            dbl    = 0;
        }

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

        if (captures_only) {
            single &= RANK_1_BB;
            dbl    = 0;
        }

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

// All squares attacked by color `by` given occupancy `occ`. One-shot
// alternative to per-square is_square_attacked for callers that need a
// full "attack map" (king-move legality does — see the shortcut below).
// Callers can pass a modified `occ` (typically our king removed) so slider
// rays see through blockers that are about to move.
Bitboard attacks_by(const Position& pos, Color by, Bitboard occ) {
    Bitboard atk = 0;
    Bitboard pawns = pos.pieces[by][PAWN];
    if (by == WHITE) {
        atk |= ((pawns & ~FILE_A_BB) << 7) | ((pawns & ~FILE_H_BB) << 9);
    } else {
        atk |= ((pawns & ~FILE_H_BB) >> 7) | ((pawns & ~FILE_A_BB) >> 9);
    }
    Bitboard b = pos.pieces[by][KNIGHT];
    while (b) atk |= KNIGHT_ATTACKS[pop_lsb(b)];
    Bitboard bq = pos.pieces[by][BISHOP] | pos.pieces[by][QUEEN];
    while (bq) atk |= bishop_attacks(pop_lsb(bq), occ);
    Bitboard rq = pos.pieces[by][ROOK]   | pos.pieces[by][QUEEN];
    while (rq) atk |= rook_attacks  (pop_lsb(rq), occ);
    Bitboard k = pos.pieces[by][KING];
    while (k) atk |= KING_ATTACKS[pop_lsb(k)];
    return atk;
}

// Is `sq` attacked by any piece of color `by` in the current occupancy?
// Symmetric-attack trick for leapers: pieces that attack `sq` sit on the
// same squares that a same-role piece AT `sq` would attack — with the
// pawn direction inverted, since pawns only attack "forward". Sliders
// use magic bitboards for O(1) attack-set lookup.
bool is_square_attacked(const Position& pos, Square sq, Color by) {
    if (PAWN_ATTACKS[Color(by ^ 1)][sq] & pos.pieces[by][PAWN])   return true;
    if (KNIGHT_ATTACKS[sq]              & pos.pieces[by][KNIGHT]) return true;
    if (KING_ATTACKS[sq]                & pos.pieces[by][KING])   return true;

    const Bitboard occ = pos.occupied;
    const Bitboard bq  = pos.pieces[by][BISHOP] | pos.pieces[by][QUEEN];
    const Bitboard rq  = pos.pieces[by][ROOK]   | pos.pieces[by][QUEEN];
    if (bishop_attacks(sq, occ) & bq) return true;
    if (rook_attacks  (sq, occ) & rq) return true;
    return false;
}

void generate_castling(const Position& pos, MoveList& moves) {
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

void generate_slider_moves(const Position& pos, MoveList& moves, bool captures_only) {
    const Color    us    = pos.side_to_move;
    const Bitboard our   = pos.colors[us];
    const Bitboard enemy = pos.colors[Color(us ^ 1)];
    const Bitboard occ   = pos.occupied;
    // Full generation walks all reachable squares (minus own pieces);
    // captures-only restricts targets to enemy occupancy.
    const Bitboard target_mask = captures_only ? enemy : ~our;

    Bitboard diag = pos.pieces[us][BISHOP] | pos.pieces[us][QUEEN];
    while (diag) {
        Square from = pop_lsb(diag);
        Bitboard targets = bishop_attacks(from, occ) & target_mask;
        while (targets) moves.push_back(make_move(from, pop_lsb(targets)));
    }

    Bitboard orth = pos.pieces[us][ROOK] | pos.pieces[us][QUEEN];
    while (orth) {
        Square from = pop_lsb(orth);
        Bitboard targets = rook_attacks(from, occ) & target_mask;
        while (targets) moves.push_back(make_move(from, pop_lsb(targets)));
    }
}

// Full-power legality check: apply `m`, ask whether the mover's king is
// attacked, unapply. Used for the "hard" cases where the fast shortcut
// (see filter_illegal) can't rule the move in or out safely.
bool is_legal(Position& pos, Move m) {
    const Color us = pos.side_to_move;
    UndoInfo u;
    pos.make_move(m, u);
    Bitboard king_bb = pos.pieces[us][KING];
    bool safe = (king_bb == 0) ||
                !is_square_attacked(pos, lsb(king_bb), pos.side_to_move);
    pos.unmake_move(m, u);
    return safe;
}

// Squares strictly between two collinear squares (same rank, file, or
// diagonal). Returns 0 if the two squares aren't on any shared ray.
// Used to detect pinning: exactly one blocker between king and a
// potential pinner means that blocker is pinned.
Bitboard squares_between(Square a, Square b) {
    int fa = file_of(a), ra = rank_of(a);
    int fb = file_of(b), rb = rank_of(b);
    int df_raw = fb - fa;
    int dr_raw = rb - ra;
    // Not on a shared ray: not same file, not same rank, not on a diagonal
    // (|df| != |dr|). Return 0 harmlessly.
    if (df_raw == 0 && dr_raw == 0) return 0;
    if (df_raw != 0 && dr_raw != 0 &&
        df_raw != dr_raw && df_raw != -dr_raw) return 0;
    int df = (df_raw > 0) - (df_raw < 0);
    int dr = (dr_raw > 0) - (dr_raw < 0);
    Bitboard result = 0;
    int f = fa + df, r = ra + dr;
    while (f != fb || r != rb) {
        result |= square_bb(make_square(File(f), Rank(r)));
        f += df; r += dr;
    }
    return result;
}

// Bitboard of side-to-move's pieces that are pinned to their king by
// enemy sliders. A piece is pinned when it is the ONLY blocker on a
// king-to-slider ray — moving it off that ray would expose the king.
//
// Fast xray trick: cast slider attacks from the king with own-piece
// blockers REMOVED from the occupancy; enemy sliders reachable via
// that xray are the potential pinners. Then for each, check that
// exactly one of our pieces sits between them and the king.
Bitboard compute_pinned(const Position& pos) {
    const Color    us      = pos.side_to_move;
    const Bitboard king_bb = pos.pieces[us][KING];
    if (!king_bb) return 0;
    const Square   king_sq  = lsb(king_bb);
    const Color    them     = Color(us ^ 1);
    const Bitboard our      = pos.colors[us];
    const Bitboard occ      = pos.occupied;
    const Bitboard occ_xray = occ & ~our;

    const Bitboard enemy_bq = pos.pieces[them][BISHOP] | pos.pieces[them][QUEEN];
    const Bitboard enemy_rq = pos.pieces[them][ROOK]   | pos.pieces[them][QUEEN];
    Bitboard pinners =
        (bishop_attacks(king_sq, occ_xray) & enemy_bq) |
        (rook_attacks  (king_sq, occ_xray) & enemy_rq);

    Bitboard pinned = 0;
    while (pinners) {
        Square   sq       = pop_lsb(pinners);
        Bitboard between  = squares_between(king_sq, sq);
        Bitboard blockers = between & occ;
        // Exactly one blocker AND it's ours → pinned.
        if (popcount(blockers) == 1 && (blockers & our)) {
            pinned |= blockers;
        }
    }
    return pinned;
}

}  // namespace

bool in_check(const Position& pos) {
    Bitboard king_bb = pos.pieces[pos.side_to_move][KING];
    if (!king_bb) return false;   // artificial no-king test positions
    return is_square_attacked(pos, lsb(king_bb),
                              Color(pos.side_to_move ^ 1));
}

// Shared implementation for generate_moves (all legal) and generate_captures
// (captures + non-capture promotions only). Castling is never a capture,
// so it's skipped in captures_only mode. Legality post-filter runs in both.
void generate_moves_impl(Position& pos, MoveList& moves, bool captures_only) {
    const Color    us         = pos.side_to_move;
    const Bitboard our_pieces = pos.colors[us];
    const Bitboard enemy      = pos.colors[Color(us ^ 1)];
    // Full generation → all squares minus own pieces. Captures-only →
    // just enemy squares (leaper attack sets & enemy).
    const Bitboard target_mask = captures_only ? enemy : ~our_pieces;

    Bitboard knights = pos.pieces[us][KNIGHT];
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard targets = KNIGHT_ATTACKS[from] & target_mask;
        while (targets) moves.push_back(make_move(from, pop_lsb(targets)));
    }

    // Loop, not `if`, so contrived test positions with 0 or 2+ kings don't
    // crash the generator.
    Bitboard kings = pos.pieces[us][KING];
    while (kings) {
        Square from = pop_lsb(kings);
        Bitboard targets = KING_ATTACKS[from] & target_mask;
        while (targets) moves.push_back(make_move(from, pop_lsb(targets)));
    }

    generate_pawn_moves  (pos, moves, captures_only);
    generate_slider_moves(pos, moves, captures_only);
    if (!captures_only) generate_castling(pos, moves);

    // Legality filter with fast shortcuts. Classical make/unmake+attack
    // scan is ~100 ns per move; we only need it for moves that could
    // actually leave our king in check:
    //
    //   - In check                        → any response might fail; verify all
    //   - King capture                    → captured piece might be sole attacker
    //   - En passant                      → weird horizontal-pin cases
    //   - Pinned piece                    → moving off the pin line exposes king
    //
    // Two special shortcuts on top of that:
    //
    //   - Non-capture king moves          → precomputed enemy attack map
    //     (own king removed from occupancy so sliders see through where
    //     the king stood) tells us in one AND whether the destination is
    //     safe. No make/unmake needed.
    //   - Castling                        → generate_castling already
    //     verified all three king squares are unattacked; skip the filter.
    const bool     in_check_now = in_check(pos);
    const Bitboard pinned       = compute_pinned(pos);
    const Bitboard king_bb      = pos.pieces[pos.side_to_move][KING];
    const Square   king_sq      = king_bb ? lsb(king_bb) : NO_SQUARE;
    const Bitboard enemy_atk_no_king = king_bb
        ? attacks_by(pos, Color(us ^ 1), pos.occupied ^ king_bb)
        : 0;

    moves.erase(
        std::remove_if(moves.begin(), moves.end(),
                       [&](Move m) {
                           if (move_type(m) == MT_CASTLING) return false;
                           Square from = move_from(m);
                           Square to   = move_to(m);
                           if (from == king_sq && pos.board[to] == NO_PIECE) {
                               // Non-capture king move: illegal iff dest is attacked.
                               return (square_bb(to) & enemy_atk_no_king) != 0;
                           }
                           bool needs_full_check =
                               in_check_now                            ||
                               from == king_sq                         ||
                               move_type(m) == MT_EN_PASSANT           ||
                               (pinned & square_bb(from));
                           return needs_full_check && !is_legal(pos, m);
                       }),
        moves.end());
}

void generate_moves(Position& pos, MoveList& moves) {
    generate_moves_impl(pos, moves, /*captures_only=*/false);
}

void generate_captures(Position& pos, MoveList& moves) {
    generate_moves_impl(pos, moves, /*captures_only=*/true);
}
