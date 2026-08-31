#pragma once
#include <cstdint>

using Bitboard = uint64_t;

enum Color : int { WHITE = 0, BLACK = 1, NO_COLOR = 2 };
constexpr int NUM_COLORS = 2;

enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
};
constexpr int NUM_PIECE_TYPES = 7;

enum Piece : int {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
};

enum Square : int {
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

enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };

constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
constexpr Square make_square(File f, Rank r) { return Square((r << 3) | f); }

// Piece decomposition: Piece encodes color in the high bit (W_* = 1..6,
// B_* = 9..14, gap at 7-8), so these run without a branch on typical
// compilers. Undefined on NO_PIECE — callers should filter first.
constexpr Color     color_of(Piece p) { return (p < B_PAWN) ? WHITE : BLACK; }
constexpr PieceType type_of (Piece p) { return PieceType(p < B_PAWN ? p : p - 8); }

// Move encoding: 16 bits.
//   bits 0-5   : from square    (0-63)
//   bits 6-11  : to square      (0-63)
//   bits 12-13 : promotion type (0=N, 1=B, 2=R, 3=Q)
//   bits 14-15 : move type      (0=normal, 1=promotion, 2=en passant, 3=castling)
using Move = uint16_t;

enum MoveType : int {
    MT_NORMAL     = 0,
    MT_PROMOTION  = 1,
    MT_EN_PASSANT = 2,
    MT_CASTLING   = 3,
};

constexpr Move make_move(Square from, Square to,
                         MoveType mt = MT_NORMAL, PieceType promo = KNIGHT) {
    return Move(from | (to << 6) | ((promo - KNIGHT) << 12) | (mt << 14));
}

constexpr Square    move_from(Move m)      { return Square(m & 63); }
constexpr Square    move_to(Move m)        { return Square((m >> 6) & 63); }
constexpr MoveType  move_type(Move m)      { return MoveType((m >> 14) & 3); }
constexpr PieceType move_promotion(Move m) { return PieceType(((m >> 12) & 3) + KNIGHT); }

constexpr Move NULL_MOVE = 0;
