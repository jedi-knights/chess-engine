#include "position.h"
#include "bitboard.h"
#include "zobrist.h"

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <sstream>

static Piece piece_from_char(char c) {
    switch (c) {
        case 'P': return W_PAWN;   case 'N': return W_KNIGHT;
        case 'B': return W_BISHOP; case 'R': return W_ROOK;
        case 'Q': return W_QUEEN;  case 'K': return W_KING;
        case 'p': return B_PAWN;   case 'n': return B_KNIGHT;
        case 'b': return B_BISHOP; case 'r': return B_ROOK;
        case 'q': return B_QUEEN;  case 'k': return B_KING;
        default:  return NO_PIECE;
    }
}

static char char_from_piece(Piece p) {
    static const char* s = " PNBRQK  pnbrqk";
    return s[p];
}

void Position::clear() {
    for (int i = 0; i < NUM_SQUARES; ++i) board[i] = NO_PIECE;
    for (int c = 0; c < NUM_COLORS; ++c) {
        colors[c] = 0;
        for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt) pieces[c][pt] = 0;
    }
    occupied        = 0;
    side_to_move    = WHITE;
    castling        = NO_CASTLING;
    ep_square       = NO_SQUARE;
    halfmove_clock  = 0;
    fullmove_number = 1;
    key             = 0;
}

bool Position::set_from_fen(const std::string& fen) {
    clear();
    std::istringstream ss(fen);
    std::string placement, active, castle, ep;
    if (!(ss >> placement >> active >> castle >> ep)) return false;
    ss >> halfmove_clock >> fullmove_number;

    int r = 7, f = 0;
    for (char c : placement) {
        if (c == '/') { --r; f = 0; }
        else if (std::isdigit(static_cast<unsigned char>(c))) { f += c - '0'; }
        else {
            Piece p = piece_from_char(c);
            if (p == NO_PIECE || r < 0 || f > 7) return false;
            Square    s   = make_square(File(f), Rank(r));
            Color     col = (p < B_PAWN) ? WHITE : BLACK;
            PieceType pt  = PieceType(p < B_PAWN ? p : p - 8);
            board[s]           = p;
            Bitboard bb        = square_bb(s);
            pieces[col][pt]   |= bb;
            colors[col]       |= bb;
            occupied          |= bb;
            ++f;
        }
    }

    side_to_move = (active == "w") ? WHITE : BLACK;

    for (char c : castle) {
        switch (c) {
            case 'K': castling |= WHITE_OO;  break;
            case 'Q': castling |= WHITE_OOO; break;
            case 'k': castling |= BLACK_OO;  break;
            case 'q': castling |= BLACK_OOO; break;
            default: break;
        }
    }

    if (ep != "-" && ep.size() == 2) {
        ep_square = make_square(File(ep[0] - 'a'), Rank(ep[1] - '1'));
    }
    key = zobrist::compute(*this);
    return true;
}

std::string Position::to_fen() const {
    std::ostringstream o;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(File(f), Rank(r))];
            if (p == NO_PIECE) ++empty;
            else {
                if (empty) { o << empty; empty = 0; }
                o << char_from_piece(p);
            }
        }
        if (empty) o << empty;
        if (r > 0) o << '/';
    }
    o << ' ' << (side_to_move == WHITE ? 'w' : 'b') << ' ';
    if (castling == NO_CASTLING) o << '-';
    else {
        if (castling & WHITE_OO)  o << 'K';
        if (castling & WHITE_OOO) o << 'Q';
        if (castling & BLACK_OO)  o << 'k';
        if (castling & BLACK_OOO) o << 'q';
    }
    o << ' ';
    if (ep_square == NO_SQUARE) o << '-';
    else o << char('a' + file_of(ep_square)) << char('1' + rank_of(ep_square));
    o << ' ' << halfmove_clock << ' ' << fullmove_number;
    return o.str();
}

// Bits cleared from `castling` when a piece leaves OR is captured on that
// square. King-source and rook-source squares are the only entries that
// differ from 15 (ALL_CASTLING). ANDing with CR_MASK[from] & CR_MASK[to]
// handles king moves, rook moves, and rook captures in one step.
static constexpr int CR_MASK[NUM_SQUARES] = {
    13, 15, 15, 15, 12, 15, 15, 14,   // rank 1 (A1=~WHITE_OOO, E1=~(WHITE_OO|OOO), H1=~WHITE_OO)
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11,   // rank 8 mirror
};

void Position::put_piece(Square s, Piece p) {
    assert(board[s] == NO_PIECE);
    assert(p != NO_PIECE);
    board[s]                     = p;
    Bitboard bb                  = square_bb(s);
    pieces[color_of(p)][type_of(p)] |= bb;
    colors[color_of(p)]         |= bb;
    occupied                    |= bb;
    key ^= zobrist::PIECE_SQ[color_of(p)][type_of(p)][s];
}

void Position::remove_piece(Square s) {
    Piece p = board[s];
    assert(p != NO_PIECE);
    board[s]                     = NO_PIECE;
    Bitboard bb                  = square_bb(s);
    pieces[color_of(p)][type_of(p)] &= ~bb;
    colors[color_of(p)]         &= ~bb;
    occupied                    &= ~bb;
    key ^= zobrist::PIECE_SQ[color_of(p)][type_of(p)][s];
}

void Position::make_move(Move m, UndoInfo& u) {
    const Square   from   = move_from(m);
    const Square   to     = move_to(m);
    const MoveType mt     = move_type(m);
    const Piece    moving = board[from];
    const Color    us     = side_to_move;
    const Color    them   = Color(us ^ 1);

    assert(moving != NO_PIECE);
    assert(color_of(moving) == us);

    // Snapshot pre-move state that unmake cannot derive from `m` alone.
    u.castling       = castling;
    u.ep_square      = ep_square;
    u.halfmove_clock = halfmove_clock;
    u.key            = key;
    u.captured       = (mt == MT_EN_PASSANT)
        ? Piece(us == WHITE ? B_PAWN : W_PAWN)
        : board[to];

    // Roll the pre-move castling/ep/side keys OUT now. The corresponding
    // post-move keys are XOR'd back in at the end after those fields are
    // updated. Piece-square keys are handled inside put_piece/remove_piece.
    key ^= zobrist::CASTLING[castling & 15];
    if (ep_square != NO_SQUARE) key ^= zobrist::EP_FILE[file_of(ep_square)];

    // Remove captured piece first (en passant captures off-square).
    if (u.captured != NO_PIECE) {
        Square cap_sq = (mt == MT_EN_PASSANT)
            ? Square(int(to) + (us == WHITE ? -8 : 8))
            : to;
        remove_piece(cap_sq);
    }

    // Move the piece; promotion changes type at the destination.
    remove_piece(from);
    if (mt == MT_PROMOTION) {
        PieceType promo = move_promotion(m);
        put_piece(to, Piece(us == WHITE ? promo : promo + 8));
    } else {
        put_piece(to, moving);
    }

    // Castling: the king move is already applied; also move the rook.
    if (mt == MT_CASTLING) {
        Square rook_from, rook_to;
        if (file_of(to) == FILE_G) {  // kingside
            rook_from = Square(int(to) + 1);
            rook_to   = Square(int(to) - 1);
        } else {                      // queenside (FILE_C)
            rook_from = Square(int(to) - 2);
            rook_to   = Square(int(to) + 1);
        }
        Piece rook = board[rook_from];
        assert(type_of(rook) == ROOK);
        remove_piece(rook_from);
        put_piece(rook_to, rook);
    }

    // Castling rights: single AND handles king move, rook move, and rook capture.
    castling &= CR_MASK[from] & CR_MASK[to];

    // En passant: set only when a pawn double-pushes; cleared otherwise.
    ep_square = NO_SQUARE;
    if (type_of(moving) == PAWN && std::abs(int(to) - int(from)) == 16) {
        ep_square = Square((int(from) + int(to)) / 2);
    }

    // Halfmove clock: reset on pawn move or capture.
    if (type_of(moving) == PAWN || u.captured != NO_PIECE) halfmove_clock = 0;
    else                                                    ++halfmove_clock;

    if (us == BLACK) ++fullmove_number;
    side_to_move = them;

    // Roll the new castling / ep / side keys IN. SIDE toggles on every
    // move (XOR is self-inverse) regardless of which color moved.
    key ^= zobrist::CASTLING[castling & 15];
    if (ep_square != NO_SQUARE) key ^= zobrist::EP_FILE[file_of(ep_square)];
    key ^= zobrist::SIDE;

    // Real games have exactly one king per side, but the standard perft
    // suite includes contrived positions with none — assert only the upper
    // bound to catch actual corruption (double-king) without rejecting them.
    assert(popcount(pieces[WHITE][KING]) <= 1);
    assert(popcount(pieces[BLACK][KING]) <= 1);
}

void Position::unmake_move(Move m, const UndoInfo& u) {
    const Square   from = move_from(m);
    const Square   to   = move_to(m);
    const MoveType mt   = move_type(m);
    const Color    us   = Color(side_to_move ^ 1);   // the mover, before flip

    side_to_move = us;
    if (us == BLACK) --fullmove_number;

    // Undo the piece move. For promotion, restore a pawn at `from` rather
    // than the promoted piece.
    Piece at_to = board[to];
    remove_piece(to);
    if (mt == MT_PROMOTION) {
        put_piece(from, Piece(us == WHITE ? W_PAWN : B_PAWN));
    } else {
        put_piece(from, at_to);
    }

    // Restore captured piece (on the ep-target square for en passant).
    if (u.captured != NO_PIECE) {
        Square cap_sq = (mt == MT_EN_PASSANT)
            ? Square(int(to) + (us == WHITE ? -8 : 8))
            : to;
        put_piece(cap_sq, u.captured);
    }

    // Undo castling rook move.
    if (mt == MT_CASTLING) {
        Square rook_from, rook_to;
        if (file_of(to) == FILE_G) {
            rook_from = Square(int(to) + 1);
            rook_to   = Square(int(to) - 1);
        } else {
            rook_from = Square(int(to) - 2);
            rook_to   = Square(int(to) + 1);
        }
        Piece rook = board[rook_to];
        remove_piece(rook_to);
        put_piece(rook_from, rook);
    }

    ep_square      = u.ep_square;
    castling       = u.castling;
    halfmove_clock = u.halfmove_clock;
    // Snapshot restore beats redoing all the incremental XORs by hand —
    // and it's what tests check against (compute(pos) after unmake must
    // equal the pre-move key).
    key            = u.key;
}

std::string Position::pretty() const {
    std::ostringstream o;
    o << "  +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        o << (r + 1) << " |";
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(File(f), Rank(r))];
            o << ' ' << (p == NO_PIECE ? '.' : char_from_piece(p)) << " |";
        }
        o << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    o << "    a   b   c   d   e   f   g   h\n"
      << "FEN: " << to_fen() << '\n';
    return o.str();
}
