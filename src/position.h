#pragma once
#include "nnue_types.h"
#include "types.h"
#include <string>

constexpr const char* STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

enum CastlingRights : std::uint8_t {
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
    uint64_t pawn_key       = 0;          // pawn-only Zobrist snapshot (for pawn hash)
    // Three-check only: snapshot of the "checks delivered to <color>"
    // counter the move might increment (we only ever increment the
    // opponent's slot during our turn, but snapshotting both costs 2
    // bytes and avoids a side-dependent restore path).
    std::uint8_t prev_checks[NUM_COLORS] = {0, 0};
    // Atomic: on a capture, up to 8 adjacent squares explode (non-pawn
    // pieces removed). Snapshot the Piece at every 3x3 square (centered
    // on `to`, index 4) so unmake restores. Index layout:
    //   0 1 2
    //   3 4 5
    //   6 7 8
    // where 4 is the explosion center (= `to`). On non-captures or non-
    // Atomic positions, all slots stay NO_PIECE.
    Piece atomic_explode[9] = {NO_PIECE, NO_PIECE, NO_PIECE,
                               NO_PIECE, NO_PIECE, NO_PIECE,
                               NO_PIECE, NO_PIECE, NO_PIECE};
    // Crazyhouse: did this move give its captured piece to our hand
    // (and if so, as what piece type -- promoted-piece captures come
    // back as PAWNs). NO_PIECE_TYPE if no hand transfer happened.
    PieceType ch_captured_as = NO_PIECE_TYPE;
    // Crazyhouse: did `promoted` have the `to` bit set before this
    // move? Needed on unmake to restore it after a non-promotion
    // move of a promoted piece shifts the bit.
    bool prev_promoted_at_to   = false;
    bool prev_promoted_at_from = false;
};

// Side index within castling_rook_file: KINGSIDE first (kingside castle
// target is the G-file), QUEENSIDE second (target is the C-file). The
// index deliberately does NOT match the WHITE_OO/WHITE_OOO bitmask bit
// order -- those are separate concerns (bitmask tracks "right exists",
// castling_rook_file tracks "where does the rook start").
enum CastleSide : int { KINGSIDE = 0, QUEENSIDE = 1 };

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

    // Chess960 castling: for each color and each castle side, the FILE of
    // the rook that owns that castling right. Classical startpos: {H, A}
    // for both colors. For any Chess960 starting position, these are read
    // from the Shredder- or X-FEN castling-rights field. These values are
    // set once (by set_from_fen) and never mutated during a game -- when
    // a rook moves off its starting file, the corresponding bit in
    // `castling` is cleared instead, so the file stored here becomes
    // irrelevant from that point on. Keeping the file value stable means
    // UndoInfo doesn't need to snapshot it.
    File     castling_rook_file[NUM_COLORS][2] = {
        {FILE_H, FILE_A},   // WHITE: {KINGSIDE, QUEENSIDE}
        {FILE_H, FILE_A},   // BLACK
    };
    // True when the position was loaded as a Chess960 game -- either via
    // `UCI_Chess960` option before set_from_fen, or inferred from a FEN
    // whose king is not on E or whose rook is not on A/H. Only affects
    // the UCI output format (move_to_uci_output in uci.cpp): internal
    // move encoding, movegen, and make/unmake are variant-agnostic since
    // they derive from castling_rook_file either way.
    bool     is_chess960     = false;
    // Rule-level variant (KotH, three-check, antichess, atomic, ...).
    // Set by UCI_Variant; preserved across set_from_fen for the same
    // reason is_chess960 is (the UCI option is engine-persistent).
    RuleVariant rules        = RV_STANDARD;
    // Three-check: number of checks delivered to each color so far.
    // When `checks_delivered[c] >= 3`, color c has been checked three
    // times and that color's opponent wins. Only consulted when
    // `rules == RV_THREE_CHECK`; initialized to {0, 0} and incremented
    // in make_move's post-move check-detection step. Snapshotted in
    // UndoInfo so unmake restores. Lichess FEN encoding appends a
    // `+<remaining-white>+<remaining-black>` field (remaining = 3 -
    // delivered); set_from_fen parses it and to_fen emits it when
    // the variant is active.
    std::uint8_t checks_delivered[NUM_COLORS] = {0, 0};
    // Crazyhouse: pieces each side has captured and holds "in hand"
    // for later drop. Index 0 is unused (NO_PIECE_TYPE / sentinel);
    // valid indices 1..5 correspond to PAWN, KNIGHT, BISHOP, ROOK,
    // QUEEN. KING is never in hand (can't be captured). Only consulted
    // when `rules == RV_CRAZYHOUSE`.
    std::uint8_t hand[NUM_COLORS][NUM_PIECE_TYPES] = {{0}, {0}};
    // Crazyhouse: bitboard of squares holding a piece that was once a
    // pawn promoted to its current type. When captured, these come
    // back to the opponent's hand as PAWNs (not as the promoted type).
    // Maintained by make_move (promotion sets the bit on `to`;
    // non-promotion move clears/sets as the piece moves). Only
    // consulted when `rules == RV_CRAZYHOUSE`.
    Bitboard promoted = 0;
    uint64_t key             = 0;        // Zobrist hash; kept in sync by set_from_fen and make/unmake
    // Pawn-only Zobrist: XOR of PIECE_SQ[color][PAWN][sq] over all pawns.
    // Keys the pawn hash table in eval.cpp so pawn-structure terms
    // (passed / isolated / doubled) get cached across positions that
    // share the same pawn skeleton but differ elsewhere. Maintained
    // incrementally in put_piece / remove_piece; snapshot-restored in
    // unmake via UndoInfo. Excludes EP square (pawn structure eval
    // doesn't depend on ep).
    uint64_t pawn_key        = 0;

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

    // NNUE accumulator, updated incrementally by put_piece / remove_piece
    // when NNUE is loaded. `mutable` so nnue::evaluate(const Position&)
    // can lazily refresh dirty sides through pos.acc without threading
    // non-const Position& through eval-only call sites. Default-
    // constructed accumulator is marked dirty (computed[c] = false)
    // so the first evaluate() call triggers a full recompute — no risk
    // of using uninitialized values.
    mutable nnue::Accumulator acc;

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

    // Crazyhouse-only: drop a piece from hand onto an empty square.
    // Does NOT go through the normal Move encoding -- the 16-bit Move
    // doesn't have room for "which piece type to drop" alongside
    // from/to. Callers (uci.cpp's parse of "P@f3") invoke this
    // directly. The engine's own movegen does NOT generate drops yet
    // -- it plays Crazyhouse without using its hand. See CLAUDE.md.
    void make_drop(PieceType pt, Square to, UndoInfo& u);
    void unmake_drop(PieceType pt, Square to, const UndoInfo& u);

    // Rebuild psq_mg[] / psq_eg[] from scratch by summing over every
    // piece on the board. Called by set_from_fen (which sets bitboards
    // directly, not via put_piece); also intended for the Texel tuner
    // to invalidate the incremental accumulators after mutating
    // eval::params.piece_values or a PST table. Does NOT touch the
    // Zobrist key, pawn_key, or history stack — those don't depend on
    // eval weights.
    void recompute_psq();

private:
    void put_piece(Square s, Piece p);
    void remove_piece(Square s);
};
