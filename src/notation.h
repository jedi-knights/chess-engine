#pragma once
#include "position.h"
#include "types.h"

#include <string>

// UCI long-algebraic move notation: "<from><to>[promo]" — e.g., "e2e4",
// "e7e8q", "e1g1" (castling), "e5d6" (en passant). The move type and
// promotion piece are encoded in the Move bits; UCI notation itself
// doesn't distinguish castling / en passant from normal moves, so
// parse_uci_move infers them from position state.

// Serialize a Move to UCI long-algebraic notation. NULL_MOVE renders as
// "0000" per the UCI convention for "no legal move to report".
std::string move_to_uci(Move m);

// Parse a UCI-formatted move against `pos`. Returns NULL_MOVE for any
// malformed input (wrong length, out-of-range squares, empty from-square,
// promotion without piece char, unknown promotion piece). Move-legality
// is NOT checked here — callers that need it must filter through
// generate_moves themselves.
Move parse_uci_move(const Position& pos, const std::string& uci);
