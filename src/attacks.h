#pragma once
#include "types.h"

// Precomputed attack tables for leaper pieces. Slider attack generation
// (bishop, rook, queen) is a TODO — see movegen.h milestone 4.
extern Bitboard KNIGHT_ATTACKS[NUM_SQUARES];
extern Bitboard KING_ATTACKS[NUM_SQUARES];
extern Bitboard PAWN_ATTACKS[NUM_COLORS][NUM_SQUARES];

void init_attacks();
