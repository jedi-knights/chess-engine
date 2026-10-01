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
// Horde: pawns can start on rank 1; added so the double-push rule can
// detect source-rank-1 pawns (single-push target lands on rank 2).
constexpr Bitboard RANK_2_BB = 0x000000000000FF00ULL;

// Emit one normal/ep move OR four promotion moves (Q, R, B, N) when the
// destination sits on the promotion rank.
inline void emit_pawn_move(Square from, Square to, Bitboard promo_rank,
                           MoveType mt, MoveList& moves) {
    if ((square_bb(to) & promo_rank) != 0U) {
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
// NOLINTNEXTLINE(readability-function-cognitive-complexity) — pawn movegen fuses pushes, double-pushes, captures, en-passant, and 4-piece promotion expansion; splitting doubles the file size for no runtime benefit.
void generate_pawn_moves(const Position& pos, MoveList& moves, bool captures_only) {
    const Color    us    = pos.side_to_move;
    const Bitboard empty = ~pos.occupied;
    const Bitboard enemy = pos.colors[Color(us ^ 1)];
    const Bitboard pawns = pos.pieces[us][PAWN];

    if (us == WHITE) {
        Bitboard single = (pawns << 8) & empty;
        Bitboard dbl    = ((single & RANK_3_BB) << 8) & empty;
        // Horde: white pawns on rank 1 can double-push to rank 3. The
        // standard rule above handles rank-2 pawns via `single & RANK_3_BB`
        // (single-push target on rank 3 => source was rank 2). For
        // rank-1 pawns, the single-push target is on rank 2; the
        // double-push would land on rank 3 if that square is empty.
        if (pos.rules == RV_HORDE) {
            dbl |= ((single & RANK_2_BB) << 8) & empty;
        }
        Bitboard cap_nw = ((pawns & ~FILE_A_BB) << 7) & enemy;
        Bitboard cap_ne = ((pawns & ~FILE_H_BB) << 9) & enemy;

        if (captures_only) {
            // Retain only pushes that promote (rank 8) — a queen appearing
            // out of thin air is as forcing as any capture.
            single &= RANK_8_BB;
            dbl    = 0;
        }

        while (single != 0U) { Square to = pop_lsb(single);
                         emit_pawn_move(Square(to - 8),  to, RANK_8_BB, MT_NORMAL, moves); }
        while (dbl != 0U)    { Square to = pop_lsb(dbl);
                         moves.push_back(make_move(Square(to - 16), to)); }
        while (cap_nw != 0U) { Square to = pop_lsb(cap_nw);
                         emit_pawn_move(Square(to - 7),  to, RANK_8_BB, MT_NORMAL, moves); }
        while (cap_ne != 0U) { Square to = pop_lsb(cap_ne);
                         emit_pawn_move(Square(to - 9),  to, RANK_8_BB, MT_NORMAL, moves); }

        if (pos.ep_square != NO_SQUARE) {
            Bitboard ep_bb = square_bb(pos.ep_square);
            if ((((pawns & ~FILE_A_BB) << 7) & ep_bb) != 0U) {
                moves.push_back(make_move(Square(int(pos.ep_square) - 7),
                                          pos.ep_square, MT_EN_PASSANT));
            }
            if ((((pawns & ~FILE_H_BB) << 9) & ep_bb) != 0U) {
                moves.push_back(make_move(Square(int(pos.ep_square) - 9),
                                          pos.ep_square, MT_EN_PASSANT));
            }
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

        while (single != 0U) { Square to = pop_lsb(single);
                         emit_pawn_move(Square(to + 8),  to, RANK_1_BB, MT_NORMAL, moves); }
        while (dbl != 0U)    { Square to = pop_lsb(dbl);
                         moves.push_back(make_move(Square(to + 16), to)); }
        while (cap_se != 0U) { Square to = pop_lsb(cap_se);
                         emit_pawn_move(Square(to + 7),  to, RANK_1_BB, MT_NORMAL, moves); }
        while (cap_sw != 0U) { Square to = pop_lsb(cap_sw);
                         emit_pawn_move(Square(to + 9),  to, RANK_1_BB, MT_NORMAL, moves); }

        if (pos.ep_square != NO_SQUARE) {
            Bitboard ep_bb = square_bb(pos.ep_square);
            if ((((pawns & ~FILE_H_BB) >> 7) & ep_bb) != 0U) {
                moves.push_back(make_move(Square(int(pos.ep_square) + 7),
                                          pos.ep_square, MT_EN_PASSANT));
            }
            if ((((pawns & ~FILE_A_BB) >> 9) & ep_bb) != 0U) {
                moves.push_back(make_move(Square(int(pos.ep_square) + 9),
                                          pos.ep_square, MT_EN_PASSANT));
            }
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
    while (b != 0U) {
        atk |= KNIGHT_ATTACKS[pop_lsb(b)];
    }
    Bitboard bq = pos.pieces[by][BISHOP] | pos.pieces[by][QUEEN];
    while (bq != 0U) {
        atk |= bishop_attacks(pop_lsb(bq), occ);
    }
    Bitboard rq = pos.pieces[by][ROOK]   | pos.pieces[by][QUEEN];
    while (rq != 0U) {
        atk |= rook_attacks  (pop_lsb(rq), occ);
    }
    Bitboard k = pos.pieces[by][KING];
    while (k != 0U) {
        atk |= KING_ATTACKS[pop_lsb(k)];
    }
    return atk;
}

// Is `sq` attacked by any piece of color `by` in the current occupancy?
// Symmetric-attack trick for leapers: pieces that attack `sq` sit on the
// same squares that a same-role piece AT `sq` would attack — with the
// pawn direction inverted, since pawns only attack "forward". Sliders
// use magic bitboards for O(1) attack-set lookup.
bool is_square_attacked(const Position& pos, Square sq, Color by) {
    if ((PAWN_ATTACKS[Color(by ^ 1)][sq] & pos.pieces[by][PAWN])   != 0U) {
        return true;
    }
    if ((KNIGHT_ATTACKS[sq]              & pos.pieces[by][KNIGHT]) != 0U) {
        return true;
    }
    if ((KING_ATTACKS[sq]                & pos.pieces[by][KING])   != 0U) {
        return true;
    }

    const Bitboard occ = pos.occupied;
    const Bitboard bq  = pos.pieces[by][BISHOP] | pos.pieces[by][QUEEN];
    const Bitboard rq  = pos.pieces[by][ROOK]   | pos.pieces[by][QUEEN];
    if ((bishop_attacks(sq, occ) & bq) != 0U) {
        return true;
    }
    if ((rook_attacks  (sq, occ) & rq) != 0U) {
        return true;
    }
    return false;
}

// Inclusive bitboard of squares between `a` and `b` on the same rank
// (both endpoints included). Used by generate_castling to compute the
// "must be empty" and "must not be attacked" square sets for Chess960
// castling, where king and rook starting files vary.
static Bitboard between_on_rank_inclusive(Square a, Square b) {
    Square lo = std::min(a, b);
    Square hi = std::max(a, b);
    Bitboard bb = 0;
    for (int s = int(lo); s <= int(hi); ++s) bb |= square_bb(Square(s));
    return bb;
}

void generate_castling(const Position& pos, MoveList& moves) {
    const Color us   = pos.side_to_move;
    const Color them = Color(us ^ 1);
    if (pos.pieces[us][KING] == 0U) return;  // test positions may omit king
    const Square king_from = Square(__builtin_ctzll(pos.pieces[us][KING]));
    const Rank   home      = (us == WHITE) ? RANK_1 : RANK_8;
    if (rank_of(king_from) != home) return;

    // Try one side (kingside -> G/F, queenside -> C/D). The rook may
    // start on ANY file on the home rank in Chess960, and in some SPs
    // the king doesn't actually move (its king_to equals its king_from).
    auto try_castle = [&](int right, CastleSide side,
                          File king_to_file, File rook_to_file) {
        if ((pos.castling & right) == 0) return;
        File rook_file = pos.castling_rook_file[us][side];
        Square rook_from = make_square(rook_file, home);
        Square king_to   = make_square(king_to_file, home);
        Square rook_to   = make_square(rook_to_file, home);
        // Pieces that must be absent from the castling corridor: every
        // square between king_from <-> king_to and rook_from <-> rook_to,
        // except for king_from and rook_from themselves (those pieces
        // are the ones moving).
        Bitboard must_be_empty =
            between_on_rank_inclusive(king_from, king_to)
          | between_on_rank_inclusive(rook_from, rook_to);
        must_be_empty &= ~square_bb(king_from);
        must_be_empty &= ~square_bb(rook_from);
        if ((pos.occupied & must_be_empty) != 0U) return;
        // Every square the king traverses (inclusive of king_from and
        // king_to) must not be attacked. Classical chess checks only
        // 3 squares; Chess960 may check 1-4 depending on king travel.
        Bitboard king_path = between_on_rank_inclusive(king_from, king_to);
        while (king_path != 0U) {
            Square s = pop_lsb(king_path);
            if (is_square_attacked(pos, s, them)) return;
        }
        // Internal encoding keeps the classical shape: from = king_from,
        // to = king_to (G or C file). This way move_to_uci's classical
        // output is a pure format of (from, to); the FRC-style
        // "king-captures-own-rook" UCI form is produced only at the
        // uci.cpp emission layer when UCI_Chess960 is enabled.
        moves.push_back(make_move(king_from, king_to, MT_CASTLING));
    };

    int ks_right = (us == WHITE) ? WHITE_OO  : BLACK_OO;
    int qs_right = (us == WHITE) ? WHITE_OOO : BLACK_OOO;
    try_castle(ks_right, KINGSIDE,  FILE_G, FILE_F);
    try_castle(qs_right, QUEENSIDE, FILE_C, FILE_D);
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
    while (diag != 0U) {
        Square from = pop_lsb(diag);
        Bitboard targets = bishop_attacks(from, occ) & target_mask;
        while (targets != 0U) {
            moves.push_back(make_move(from, pop_lsb(targets)));
        }
    }

    Bitboard orth = pos.pieces[us][ROOK] | pos.pieces[us][QUEEN];
    while (orth != 0U) {
        Square from = pop_lsb(orth);
        Bitboard targets = rook_attacks(from, occ) & target_mask;
        while (targets != 0U) {
            moves.push_back(make_move(from, pop_lsb(targets)));
        }
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
    int fa = file_of(a);
    int ra = rank_of(a);
    int fb = file_of(b);
    int rb = rank_of(b);
    int df_raw = fb - fa;
    int dr_raw = rb - ra;
    // Not on a shared ray: not same file, not same rank, not on a diagonal
    // (|df| != |dr|). Return 0 harmlessly.
    if (df_raw == 0 && dr_raw == 0) {
        return 0;
    }
    if (df_raw != 0 && dr_raw != 0 &&
        df_raw != dr_raw && df_raw != -dr_raw) {
        return 0;
    }
    int df = static_cast<int>(df_raw > 0) - static_cast<int>(df_raw < 0);
    int dr = static_cast<int>(dr_raw > 0) - static_cast<int>(dr_raw < 0);
    Bitboard result = 0;
    int f = fa + df;
    int r = ra + dr;
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
    if (king_bb == 0U) {
        return 0;
    }
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
    while (pinners != 0U) {
        Square   sq       = pop_lsb(pinners);
        Bitboard between  = squares_between(king_sq, sq);
        Bitboard blockers = between & occ;
        // Exactly one blocker AND it's ours → pinned.
        if (popcount(blockers) == 1 && ((blockers & our) != 0U)) {
            pinned |= blockers;
        }
    }
    return pinned;
}

}  // namespace

bool in_check(const Position& pos) {
    Bitboard king_bb = pos.pieces[pos.side_to_move][KING];
    if (king_bb == 0U) {
        return false;   // artificial no-king test positions
    }
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
    while (knights != 0U) {
        Square from = pop_lsb(knights);
        Bitboard targets = KNIGHT_ATTACKS[from] & target_mask;
        while (targets != 0U) {
            moves.push_back(make_move(from, pop_lsb(targets)));
        }
    }

    // Loop, not `if`, so contrived test positions with 0 or 2+ kings don't
    // crash the generator.
    Bitboard kings = pos.pieces[us][KING];
    while (kings != 0U) {
        Square from = pop_lsb(kings);
        Bitboard targets = KING_ATTACKS[from] & target_mask;
        while (targets != 0U) {
            moves.push_back(make_move(from, pop_lsb(targets)));
        }
    }

    generate_pawn_moves  (pos, moves, captures_only);
    generate_slider_moves(pos, moves, captures_only);
    if (!captures_only) {
        generate_castling(pos, moves);
    }

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
    const Square   king_sq      = (king_bb != 0U) ? lsb(king_bb) : NO_SQUARE;
    const Bitboard enemy_atk_no_king = (king_bb != 0U)
        ? attacks_by(pos, Color(us ^ 1), pos.occupied ^ king_bb)
        : 0;

    moves.erase(
        std::remove_if(moves.begin(), moves.end(),
                       [&](Move m) {
                           if (move_type(m) == MT_CASTLING) {
                               return false;
                           }
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
                               ((pinned & square_bb(from)) != 0U);
                           return needs_full_check && !is_legal(pos, m);
                       }),
        moves.end());

    // Racing Kings: it's illegal to leave the opponent in check. The
    // standard filter above only enforces our own king-safety; after it
    // runs, every remaining move is pseudo-safe for US but may deliver
    // check to THEM. We filter those out with a make/unmake probe --
    // slow but correctness-first for now; a check-detection shortcut
    // (analogous to the enemy_atk_no_king bitboard) is a tuning task.
    if (pos.rules == RV_RACING_KINGS) {
        moves.erase(
            std::remove_if(moves.begin(), moves.end(),
                           [&](Move m) {
                               UndoInfo u;
                               pos.make_move(m, u);
                               bool gives_check = in_check(pos);  // side-to-move is opponent now
                               pos.unmake_move(m, u);
                               return gives_check;
                           }),
            moves.end());
    }
}

void generate_moves(Position& pos, MoveList& moves) {
    generate_moves_impl(pos, moves, /*captures_only=*/false);
}

void generate_captures(Position& pos, MoveList& moves) {
    generate_moves_impl(pos, moves, /*captures_only=*/true);
}
