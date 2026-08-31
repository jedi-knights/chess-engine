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

    // Loop, not `if`, so contrived test positions with 0 or 2+ kings don't
    // crash the generator. Castling is milestone 6; king-adjacency and
    // move-into-check filtering are milestone 7.
    Bitboard kings = pos.pieces[us][KING];
    while (kings) {
        Square from = pop_lsb(kings);
        Bitboard targets = KING_ATTACKS[from] & ~our_pieces;
        while (targets) {
            Square to = pop_lsb(targets);
            moves.push_back(make_move(from, to));
        }
    }

    // TODO: pawns, sliders, castling — see movegen.h milestones.
}
