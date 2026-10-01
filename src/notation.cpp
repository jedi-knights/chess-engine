#include "notation.h"

#include <cstdlib>

std::string move_to_uci(Move m) {
    if (m == NULL_MOVE) {
        return "0000";
    }
    // Crazyhouse drop: "P@f3". The 32-bit Move encoding packs the
    // piece type in bits 17-19; from field is unused.
    if (move_type(m) == MT_DROP) {
        PieceType pt = move_drop_piece_type(m);
        Square to = move_to(m);
        std::string s;
        s += "?PNBRQK"[pt];
        s += '@';
        s += char('a' + file_of(to));
        s += char('1' + rank_of(to));
        return s;
    }
    Square from = move_from(m);
    Square to   = move_to(m);
    std::string s;
    s += char('a' + file_of(from));
    s += char('1' + rank_of(from));
    s += char('a' + file_of(to));
    s += char('1' + rank_of(to));
    if (move_type(m) == MT_PROMOTION) {
        s += "nbrq"[move_promotion(m) - KNIGHT];
    } else if (move_type(m) == MT_PROMOTION_KING) {
        s += 'k';
    }
    return s;
}

Move parse_uci_move(const Position& pos, const std::string& uci) {
    // Crazyhouse drop notation: "P@f3", "N@e5", etc. Length 4, '@'
    // at index 1. We accept this for ANY variant (if the engine is
    // not Crazyhouse, the move will fail isLegal). The piece is
    // uppercase by Lichess convention and belongs to the side-to-move.
    if (uci.size() == 4 && uci[1] == '@') {
        PieceType pt = NO_PIECE_TYPE;
        switch (uci[0]) {
            case 'P': pt = PAWN;   break;
            case 'N': pt = KNIGHT; break;
            case 'B': pt = BISHOP; break;
            case 'R': pt = ROOK;   break;
            case 'Q': pt = QUEEN;  break;
            default: return NULL_MOVE;
        }
        int tf = uci[2] - 'a';
        int tr = uci[3] - '1';
        if (tf < 0 || tf > 7 || tr < 0 || tr > 7) return NULL_MOVE;
        return make_drop_move(make_square(File(tf), Rank(tr)), pt);
    }

    if (uci.size() < 4 || uci.size() > 5) {
        return NULL_MOVE;
    }

    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file   = uci[2] - 'a';
    int to_rank   = uci[3] - '1';
    if (from_file < 0 || from_file > 7) {
        return NULL_MOVE;
    }
    if (from_rank < 0 || from_rank > 7) {
        return NULL_MOVE;
    }
    if (to_file   < 0 || to_file   > 7) {
        return NULL_MOVE;
    }
    if (to_rank   < 0 || to_rank   > 7) {
        return NULL_MOVE;
    }

    Square from = make_square(File(from_file), Rank(from_rank));
    Square to   = make_square(File(to_file),   Rank(to_rank));

    Piece moving = pos.board[from];
    if (moving == NO_PIECE) {
        return NULL_MOVE;
    }
    PieceType pt = type_of(moving);

    MoveType  mt    = MT_NORMAL;
    PieceType promo = KNIGHT;   // slot value; ignored unless MT_PROMOTION

    // A pawn reaching the back rank must specify a promotion piece;
    // reject any pawn-to-back-rank move that lacks one, because the
    // resulting Move would silently under-promote to knight.
    if (pt == PAWN && (to_rank == 0 || to_rank == 7)) {
        if (uci.size() != 5) {
            return NULL_MOVE;
        }
        switch (uci[4]) {
            case 'n': mt = MT_PROMOTION;      promo = KNIGHT; break;
            case 'b': mt = MT_PROMOTION;      promo = BISHOP; break;
            case 'r': mt = MT_PROMOTION;      promo = ROOK;   break;
            case 'q': mt = MT_PROMOTION;      promo = QUEEN;  break;
            // Antichess only: pawn can promote to king. Standard
            // chess rejects this at the move-legality step (king-
            // count would exceed 1). parse_uci_move just returns the
            // encoded move; it's up to generate_moves/make_move to
            // decide legality.
            case 'k': mt = MT_PROMOTION_KING; promo = KNIGHT; break;
            default: return NULL_MOVE;
        }
    }
    // En passant: pawn moves diagonally to the recorded ep_square. UCI
    // encodes this exactly the same as any other pawn move — the flag
    // is derived from position state alone.
    else if (pt == PAWN && to == pos.ep_square && pos.ep_square != NO_SQUARE) {
        mt = MT_EN_PASSANT;
    }
    // Castling detection has two forms, both reduced to the same internal
    // encoding (`from = king_from, to = king_to on G/C file`):
    //
    //   (a) Classical "e1g1" / "e1c1" -- king moves two files on its home
    //       rank. Lichess sends this when UCI_Chess960 is unset.
    //
    //   (b) Chess960 "king-captures-own-rook" (e.g. "e1h1") -- the king's
    //       target square holds a same-color rook. Lichess's Bot API
    //       sends this form unconditionally. Works for both standard
    //       and 960: in a standard position the kingside rook IS on h1,
    //       so "e1h1" still cleanly means "castle kingside." Case (b)
    //       is checked FIRST so that in SPs where the king also happens
    //       to end two files away, the king-captures-rook form still
    //       routes through this branch.
    //
    // Both forms are rewritten to the classical internal shape; make_move
    // and movegen agree on `to = G/C file`.
    else if (pt == KING) {
        Piece dest = pos.board[to];
        bool is_own_rook = dest != NO_PIECE && color_of(dest) == color_of(moving) && type_of(dest) == ROOK;
        if (is_own_rook) {
            // Determine kingside vs queenside by whether the rook is to
            // the right or left of the king.
            bool kingside = to_file > from_file;
            mt = MT_CASTLING;
            int king_to_file = kingside ? int(FILE_G) : int(FILE_C);
            to = make_square(File(king_to_file), Rank(to_rank));
        } else if (std::abs(to_file - from_file) == 2) {
            mt = MT_CASTLING;
        }
        else if (uci.size() == 5) {
            return NULL_MOVE;
        }
    }
    // Extra promotion char on a non-promoting move is malformed input.
    else if (uci.size() == 5) {
        return NULL_MOVE;
    }

    return make_move(from, to, mt, promo);
}

// FRC-aware emitter: when `pos.is_chess960` is true, castling moves are
// re-encoded as king-captures-own-rook (e.g. "e1h1") because that's what
// Lichess's Bot API expects on the wire. Non-castling moves and
// non-Chess960 positions pass through to the classical formatter.
std::string move_to_uci_output(Move m, const Position& pos) {
    if (!pos.is_chess960 || move_type(m) != MT_CASTLING) {
        return move_to_uci(m);
    }
    Square king_from = move_from(m);
    Square king_to   = move_to(m);
    Color us = color_of(pos.board[king_from]);
    CastleSide side = (file_of(king_to) == FILE_G) ? KINGSIDE : QUEENSIDE;
    File rook_file  = pos.castling_rook_file[us][side];
    Square rook_from = make_square(rook_file, rank_of(king_from));
    std::string s;
    s += char('a' + file_of(king_from));
    s += char('1' + rank_of(king_from));
    s += char('a' + file_of(rook_from));
    s += char('1' + rank_of(rook_from));
    return s;
}
