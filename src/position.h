#pragma once
#include "types.h"
#include <string>

constexpr const char* STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

enum CastlingRights : int {
    NO_CASTLING  = 0,
    WHITE_OO     = 1, WHITE_OOO = 2,
    BLACK_OO     = 4, BLACK_OOO = 8,
    ALL_CASTLING = 15,
};

struct Position {
    Piece    board[NUM_SQUARES]                  = {};
    Bitboard pieces[NUM_COLORS][NUM_PIECE_TYPES] = {};
    Bitboard colors[NUM_COLORS]                  = {};
    Bitboard occupied                            = 0;

    Color   side_to_move    = WHITE;
    int     castling        = NO_CASTLING;
    Square  ep_square       = NO_SQUARE;
    int     halfmove_clock  = 0;
    int     fullmove_number = 1;

    void        clear();
    bool        set_from_fen(const std::string& fen);
    std::string to_fen() const;
    std::string pretty() const;
};
