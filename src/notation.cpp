#include "notation.h"

#include <cstdlib>

std::string move_to_uci(Move m) {
    if (m == NULL_MOVE) return "0000";
    Square from = move_from(m);
    Square to   = move_to(m);
    std::string s;
    s += char('a' + file_of(from));
    s += char('1' + rank_of(from));
    s += char('a' + file_of(to));
    s += char('1' + rank_of(to));
    if (move_type(m) == MT_PROMOTION) {
        s += "nbrq"[move_promotion(m) - KNIGHT];
    }
    return s;
}

Move parse_uci_move(const Position& pos, const std::string& uci) {
    if (uci.size() < 4 || uci.size() > 5) return NULL_MOVE;

    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file   = uci[2] - 'a';
    int to_rank   = uci[3] - '1';
    if (from_file < 0 || from_file > 7) return NULL_MOVE;
    if (from_rank < 0 || from_rank > 7) return NULL_MOVE;
    if (to_file   < 0 || to_file   > 7) return NULL_MOVE;
    if (to_rank   < 0 || to_rank   > 7) return NULL_MOVE;

    Square from = make_square(File(from_file), Rank(from_rank));
    Square to   = make_square(File(to_file),   Rank(to_rank));

    Piece moving = pos.board[from];
    if (moving == NO_PIECE) return NULL_MOVE;
    PieceType pt = type_of(moving);

    MoveType  mt    = MT_NORMAL;
    PieceType promo = KNIGHT;   // slot value; ignored unless MT_PROMOTION

    // A pawn reaching the back rank must specify a promotion piece;
    // reject any pawn-to-back-rank move that lacks one, because the
    // resulting Move would silently under-promote to knight.
    if (pt == PAWN && (to_rank == 0 || to_rank == 7)) {
        if (uci.size() != 5) return NULL_MOVE;
        mt = MT_PROMOTION;
        switch (uci[4]) {
            case 'n': promo = KNIGHT; break;
            case 'b': promo = BISHOP; break;
            case 'r': promo = ROOK;   break;
            case 'q': promo = QUEEN;  break;
            default: return NULL_MOVE;
        }
    }
    // En passant: pawn moves diagonally to the recorded ep_square. UCI
    // encodes this exactly the same as any other pawn move — the flag
    // is derived from position state alone.
    else if (pt == PAWN && to == pos.ep_square && pos.ep_square != NO_SQUARE) {
        mt = MT_EN_PASSANT;
    }
    // Castling: king moves two files. Same encoding rule as ep — GUI
    // sends "e1g1" and we detect it as castling from the file delta.
    else if (pt == KING && std::abs(int(to_file) - int(from_file)) == 2) {
        mt = MT_CASTLING;
    }
    // Extra promotion char on a non-promoting move is malformed input.
    else if (uci.size() == 5) {
        return NULL_MOVE;
    }

    return make_move(from, to, mt, promo);
}
