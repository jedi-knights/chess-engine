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

    // Incremental eval accumulators — sum of (material + PST) for every
    // piece on the board, per color. Maintained by put_piece / remove_piece
    // so evaluate() can just subtract and phase-interpolate instead of
    // looping over every piece.
    int      psq_mg[NUM_COLORS] = {0, 0};
    int      psq_eg[NUM_COLORS] = {0, 0};

    // Zobrist-key history for repetition detection. set_from_fen seeds
    // index 0 with the starting key; make_move pushes the post-move key
    // and unmake_move pops. Sized for 1024 halfmoves — comfortably above
    // any legal game (50-move rule caps under 6000 plies but analysis
    // sessions can push further). Overflow asserts in debug, silently
    // caps in release — either way the fault is loud on the ASan tests
    // and quiet in production rather than silently corrupting repetition
    // detection.
    static constexpr int HISTORY_CAPACITY = 1024;
    uint64_t history[HISTORY_CAPACITY] = {};
    int      history_size              = 0;

    void        clear();
    bool        set_from_fen(const std::string& fen);
    std::string to_fen() const;
    std::string pretty() const;

    // True if the current position (identified by `key`) has appeared
    // earlier in `history` — a repetition draw candidate. Only scans
    // back as far as the halfmove_clock (irreversible moves before
    // that reset the pool of reachable positions).
    bool        is_repetition() const;

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
