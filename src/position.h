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

// State that make_move mutates but unmake_move cannot reconstruct from the
// move alone. Caller owns storage (stack-allocate one per ply in search).
struct UndoInfo {
    Piece    captured       = NO_PIECE;   // includes en-passant captures
    int      castling       = NO_CASTLING;
    Square   ep_square      = NO_SQUARE;
    int      halfmove_clock = 0;
    uint64_t key            = 0;          // Zobrist key snapshot for unmake
};

struct Position {
    Piece    board[NUM_SQUARES]                  = {};
    Bitboard pieces[NUM_COLORS][NUM_PIECE_TYPES] = {};
    Bitboard colors[NUM_COLORS]                  = {};
    Bitboard occupied                            = 0;

    Color    side_to_move    = WHITE;
    int      castling        = NO_CASTLING;
    Square   ep_square       = NO_SQUARE;
    int      halfmove_clock  = 0;
    int      fullmove_number = 1;
    uint64_t key             = 0;        // Zobrist hash; kept in sync by set_from_fen and make/unmake

    void        clear();
    bool        set_from_fen(const std::string& fen);
    std::string to_fen() const;
    std::string pretty() const;

    // Apply / revert `m`. `u` must be the same UndoInfo instance for both
    // calls. Supports normal, capture, en-passant, castling, and promotion
    // move types — invariants hold regardless of which movegen milestone
    // generated the move.
    void make_move(Move m, UndoInfo& u);
    void unmake_move(Move m, const UndoInfo& u);

private:
    void put_piece(Square s, Piece p);
    void remove_piece(Square s);
};
