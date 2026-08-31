#include "position.h"
#include "bitboard.h"
#include <cctype>
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
