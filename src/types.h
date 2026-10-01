#pragma once
#include <cstdint>
#include <initializer_list>

using Bitboard = uint64_t;

enum Color : std::uint8_t { WHITE = 0, BLACK = 1, NO_COLOR = 2 };
constexpr int NUM_COLORS = 2;

enum PieceType : std::uint8_t {
    NO_PIECE_TYPE = 0,
    PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
};
constexpr int NUM_PIECE_TYPES = 7;

enum Piece : std::uint8_t {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
    B_PAWN = 9, B_KNIGHT = 10, B_BISHOP = 11, B_ROOK = 12, B_QUEEN = 13, B_KING = 14,
};

// NOLINTNEXTLINE(readability-enum-initial-value) — implicit A1..H8 sequence + NO_SQUARE=64 sentinel is the standard chess-engine idiom.
enum Square : std::uint8_t {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NO_SQUARE = 64,
};
constexpr int NUM_SQUARES = 64;

enum File : std::uint8_t { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };
enum Rank : std::uint8_t { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };

constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
constexpr Square make_square(File f, Rank r) { return Square((r << 3) | f); }

// Piece decomposition: Piece encodes color in the high bit (W_* = 1..6,
// B_* = 9..14, gap at 7-8), so these run without a branch on typical
// compilers. Undefined on NO_PIECE — callers should filter first.
constexpr Color     color_of(Piece p) { return (p < B_PAWN) ? WHITE : BLACK; }
constexpr PieceType type_of (Piece p) { return PieceType(p < B_PAWN ? p : p - 8); }

// Move encoding: 32 bits (was 16; expanded for Crazyhouse drops which
// need to encode "which piece type to drop" alongside the destination,
// and the 16-bit layout had no spare bits).
//   bits 0-5   : from square     (0-63)         — unused for MT_DROP
//   bits 6-11  : to square       (0-63)
//   bits 12-13 : promotion type  (0=N, 1=B, 2=R, 3=Q) — only for MT_PROMOTION
//   bits 14-16 : move type       (3 bits; see MoveType)
//   bits 17-19 : drop piece type (1=PAWN..5=QUEEN) — only for MT_DROP
//   bits 20-31 : unused (reserved)
using Move = uint32_t;

// Rule-level variants (as opposed to Chess960, which is a FEN-level
// variant that doesn't change win conditions). Each entry here adjusts
// terminal detection / eval; movegen is still standard chess. Set by
// the UCI_Variant option; `clear()` deliberately preserves this value
// across set_from_fen (same pattern as is_chess960) so a
// `setoption name UCI_Variant value kingofthehill` persists across
// subsequent `position` commands.
enum RuleVariant : std::uint8_t {
    RV_STANDARD     = 0,
    RV_KOTH         = 1,
    RV_THREE_CHECK  = 2,
    RV_HORDE        = 3,
    RV_RACING_KINGS = 4,
    RV_ATOMIC       = 5,
    RV_ANTICHESS    = 6,
    RV_CRAZYHOUSE   = 7,
};

enum MoveType : std::uint8_t {
    MT_NORMAL     = 0,
    MT_PROMOTION  = 1,
    MT_EN_PASSANT = 2,
    MT_CASTLING   = 3,
    // Crazyhouse: drop a piece from hand onto an empty square.
    // from field is unused; to holds the destination; drop piece
    // type lives in bits 17-19 (see `move_drop_piece_type`).
    MT_DROP       = 4,
};

constexpr Move make_move(Square from, Square to,
                         MoveType mt = MT_NORMAL, PieceType promo = KNIGHT) {
    return Move(from | (to << 6) | ((promo - KNIGHT) << 12) | (mt << 14));
}

// Crazyhouse-specific constructor: builds an MT_DROP Move. `from` is
// deliberately 0 (unused for drops); `pt` is the piece type to drop
// (PAWN..QUEEN). KING cannot be in hand.
constexpr Move make_drop_move(Square to, PieceType pt) {
    return Move((to << 6) | (MT_DROP << 14) | (pt << 17));
}

constexpr Square    move_from(Move m)      { return Square(m & 63); }
constexpr Square    move_to(Move m)        { return Square((m >> 6) & 63); }
constexpr MoveType  move_type(Move m)      { return MoveType((m >> 14) & 7); }
constexpr PieceType move_promotion(Move m) { return PieceType(((m >> 12) & 3) + KNIGHT); }
// Only meaningful when move_type(m) == MT_DROP; returns NO_PIECE_TYPE
// for other move types (the bits happen to be 0 there).
constexpr PieceType move_drop_piece_type(Move m) { return PieceType((m >> 17) & 7); }

constexpr Move NULL_MOVE = 0;

// Fixed-capacity move container — replaces std::vector<Move> in the hot
// path so search nodes don't malloc/free per call. Chess has a proven
// upper bound of 218 legal moves in any position; 256 rounds up comfortably.
struct MoveList {
    static constexpr int MAX_MOVES = 256;
    Move moves[MAX_MOVES];
    int  count = 0;

    MoveList() = default;
    // Enables `MoveList{move1, move2, ...}` construction — convenient
    // for tests that hand-craft short move sequences.
    MoveList(std::initializer_list<Move> il) {
        for (Move m : il) {
            moves[count++] = m;
        }
    }

    void push_back(Move m)                 { moves[count++] = m; }
    int  size()  const                     { return count; }
    bool empty() const                     { return count == 0; }
    void clear()                           { count = 0; }
    Move&       operator[](int i)          { return moves[i]; }
    const Move& operator[](int i) const    { return moves[i]; }
    Move*       begin()                    { return moves; }
    Move*       end()                      { return moves + count; }
    const Move* begin() const              { return moves; }
    const Move* end()   const              { return moves + count; }
    // Erase-tail: used with std::remove_if. `first` must lie within
    // moves; `last` is ignored (interpreted as end()).
    void erase(const Move* first, const Move* /*last*/) { count = int(first - moves); }
};
