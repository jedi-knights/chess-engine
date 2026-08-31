#include "movegen.h"
#include "attacks.h"
#include "bitboard.h"

void generate_moves(const Position& pos, std::vector<Move>& moves) {
    const Color    us         = pos.side_to_move;
    const Bitboard our_pieces = pos.colors[us];
    Bitboard       knights    = pos.pieces[us][KNIGHT];

    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard targets = KNIGHT_ATTACKS[from] & ~our_pieces;
        while (targets) {
            Square to = pop_lsb(targets);
            moves.push_back(make_move(from, to));
        }
    }

    // TODO: king, pawns, sliders, castling — see movegen.h milestones.
}
