#include "position.h"
#include "bitboard.h"
#include "eval.h"
#include "nnue.h"
#include "zobrist.h"

#include <algorithm>
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
    for (int i = 0; i < NUM_SQUARES; ++i) {
        board[i] = NO_PIECE;
    }
    for (int c = 0; c < NUM_COLORS; ++c) {
        colors[c] = 0;
        for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
            pieces[c][pt] = 0;
        }
    }
    occupied        = 0;
    side_to_move    = WHITE;
    castling        = NO_CASTLING;
    ep_square       = NO_SQUARE;
    halfmove_clock  = 0;
    fullmove_number = 1;
    key             = 0;
    pawn_key        = 0;
    psq_mg[WHITE] = psq_mg[BLACK] = 0;
    psq_eg[WHITE] = psq_eg[BLACK] = 0;
    history_size    = 0;
    // Classical defaults -- set_from_fen may overwrite these when it
    // encounters Shredder- or X-FEN castling-rights letters.
    castling_rook_file[WHITE][KINGSIDE]  = FILE_H;
    castling_rook_file[WHITE][QUEENSIDE] = FILE_A;
    castling_rook_file[BLACK][KINGSIDE]  = FILE_H;
    castling_rook_file[BLACK][QUEENSIDE] = FILE_A;
    // Deliberately NOT resetting `is_chess960` -- the UCI_Chess960 option
    // is engine-persistent across FEN loads, and set_from_fen only ever
    // upgrades it to true (never back to false). An explicit option
    // handler is the only path to disable it mid-session.
    // NNUE accumulator marked dirty — set_from_fen and the make/unmake
    // hot path bypass put_piece for bitboard reasons, so incremental
    // updates would leave stale weights. Next evaluate() triggers a
    // refresh from the (now correct) bitboards.
    acc.computed[WHITE] = false;
    acc.computed[BLACK] = false;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) — FEN parsing is inherently a multi-field state machine; splitting into helpers would be more lines and no clearer.
bool Position::set_from_fen(const std::string& fen) {
    clear();
    std::istringstream ss(fen);
    std::string placement;
    std::string active;
    std::string castle;
    std::string ep;
    if (!(ss >> placement >> active >> castle >> ep)) {
        return false;
    }
    ss >> halfmove_clock >> fullmove_number;

    int r = 7;
    int f = 0;
    for (char c : placement) {
        if (c == '/') { --r; f = 0; }
        else if (std::isdigit(static_cast<unsigned char>(c)) != 0) { f += c - '0'; }
        else {
            Piece p = piece_from_char(c);
            if (p == NO_PIECE || r < 0 || f > 7) {
                return false;
            }
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

    // Castling rights parsing: accepts classical KQkq, Shredder-FEN file
    // letters (A-H / a-h), and X-FEN hybrid. For K/Q/k/q we resolve to the
    // outermost rook of the matching color on the home rank, on the correct
    // side of the king. For file letters we take the file directly. Any
    // resolution that leaves the king not on E, or any rook not on A/H,
    // trips `is_chess960` so the UCI emitter knows to output king-captures-
    // rook form.
    auto home_rank = [](Color c) -> Rank { return c == WHITE ? RANK_1 : RANK_8; };
    auto king_file = [this](Color c) -> int {
        Bitboard bb = pieces[c][KING];
        if (bb == 0U) return -1;
        return int(file_of(Square(__builtin_ctzll(bb))));
    };
    auto find_outermost_rook = [this, home_rank, king_file](Color c, bool kingside) -> int {
        int kf = king_file(c);
        if (kf < 0) return -1;
        Bitboard rooks = pieces[c][ROOK] & 0xFFULL << (int(home_rank(c)) * 8);
        int best = -1;
        while (rooks != 0U) {
            Square s = Square(__builtin_ctzll(rooks));
            rooks &= rooks - 1;
            int rf = int(file_of(s));
            if (kingside && rf > kf) {
                if (best < 0 || rf > best) best = rf;
            } else if (!kingside && rf < kf) {
                if (best < 0 || rf < best) best = rf;
            }
        }
        return best;
    };
    auto set_right = [this](Color c, CastleSide side, int file) {
        castling_rook_file[c][side] = File(file);
        if (c == WHITE) castling |= (side == KINGSIDE ? WHITE_OO : WHITE_OOO);
        else            castling |= (side == KINGSIDE ? BLACK_OO : BLACK_OOO);
    };
    for (char c : castle) {
        if (c == '-') break;
        if (c == 'K' || c == 'Q' || c == 'k' || c == 'q') {
            Color col = (c == 'K' || c == 'Q') ? WHITE : BLACK;
            bool kingside = (c == 'K' || c == 'k');
            int file = find_outermost_rook(col, kingside);
            if (file < 0) continue;  // malformed FEN: no rook to resolve to
            set_right(col, kingside ? KINGSIDE : QUEENSIDE, file);
        } else if (c >= 'A' && c <= 'H') {
            int file = c - 'A';
            int kf = king_file(WHITE);
            if (kf < 0) continue;
            set_right(WHITE, file > kf ? KINGSIDE : QUEENSIDE, file);
        } else if (c >= 'a' && c <= 'h') {
            int file = c - 'a';
            int kf = king_file(BLACK);
            if (kf < 0) continue;
            set_right(BLACK, file > kf ? KINGSIDE : QUEENSIDE, file);
        }
    }
    // Infer Chess960 from a non-classical king/rook layout. Explicit
    // UCI_Chess960 override comes from the UCI option setter, not this
    // path -- an engine toggle is honored, inference just catches the
    // common case where a Shredder-FEN load didn't go through the option.
    // Only infer Chess960 when a castling right actually exists AND the
    // layout for that right isn't classical. A FEN with a non-E king
    // but no castling rights (e.g. a mid-game position after both sides
    // castled) is still just standard chess.
    bool rook_offset = ((castling & WHITE_OO)  != 0 && castling_rook_file[WHITE][KINGSIDE]  != FILE_H)
                    || ((castling & WHITE_OOO) != 0 && castling_rook_file[WHITE][QUEENSIDE] != FILE_A)
                    || ((castling & BLACK_OO)  != 0 && castling_rook_file[BLACK][KINGSIDE]  != FILE_H)
                    || ((castling & BLACK_OOO) != 0 && castling_rook_file[BLACK][QUEENSIDE] != FILE_A);
    bool king_offset = ((castling & (WHITE_OO | WHITE_OOO)) != 0 && king_file(WHITE) != int(FILE_E))
                    || ((castling & (BLACK_OO | BLACK_OOO)) != 0 && king_file(BLACK) != int(FILE_E));
    if (rook_offset || king_offset) {
        is_chess960 = true;
    }

    if (ep != "-" && ep.size() == 2) {
        ep_square = make_square(File(ep[0] - 'a'), Rank(ep[1] - '1'));
    }
    key = zobrist::compute(*this);

    // set_from_fen writes bitboards directly rather than going through
    // put_piece, so the incremental psq accumulators and pawn_key are
    // still 0. Recompute both from scratch — this only runs on
    // set_from_fen / ucinewgame boundaries so it isn't in the search
    // hot path. PSQ recompute lives in its own method because the
    // Texel tuner reuses it after mutating eval weights.
    recompute_psq();
    pawn_key = 0;
    for (int c = 0; c < NUM_COLORS; ++c) {
        Bitboard bb = pieces[c][PAWN];
        while (bb != 0U) {
            pawn_key ^= zobrist::PIECE_SQ[c][PAWN][pop_lsb(bb)];
        }
    }

    // Seed the repetition-detection history with the initial position.
    // make_move / unmake_move maintain it as a push/pop stack from here.
    history[0]   = key;
    history_size = 1;
    return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) — FEN emission mirrors set_from_fen: a serialization state machine, kept as one function so the round-trip contract is co-located.
std::string Position::to_fen() const {
    std::ostringstream o;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(File(f), Rank(r))];
            if (p == NO_PIECE) {
                ++empty;
            }
            else {
                if (empty != 0) { o << empty; empty = 0; }
                o << char_from_piece(p);
            }
        }
        if (empty != 0) {
            o << empty;
        }
        if (r > 0) {
            o << '/';
        }
    }
    o << ' ' << (side_to_move == WHITE ? 'w' : 'b') << ' ';
    if (castling == NO_CASTLING) {
        o << '-';
    }
    else {
        // X-FEN output: emit K/Q when the rook is the OUTERMOST of its
        // color on the home rank on the correct side of the king;
        // otherwise emit the file letter (Shredder-FEN). Matches chessops'
        // own emitter so round-trips are stable across chess-board <->
        // engine boundaries.
        auto emit_side = [&](Color c, CastleSide side, char outer_letter, char file_base) {
            if (((castling & (c == WHITE ? (side == KINGSIDE ? WHITE_OO : WHITE_OOO)
                                         : (side == KINGSIDE ? BLACK_OO : BLACK_OOO))) == 0)) {
                return;
            }
            File f = castling_rook_file[c][side];
            // Find outermost rook on correct side of king on home rank.
            Rank home = (c == WHITE) ? RANK_1 : RANK_8;
            Bitboard rooks = pieces[c][ROOK] & 0xFFULL << (int(home) * 8);
            Bitboard king  = pieces[c][KING];
            int kf = king != 0U ? int(file_of(Square(__builtin_ctzll(king)))) : -1;
            int outermost = -1;
            Bitboard it = rooks;
            while (it != 0U) {
                Square s = Square(__builtin_ctzll(it));
                it &= it - 1;
                int rf = int(file_of(s));
                if (side == KINGSIDE && rf > kf) {
                    if (outermost < 0 || rf > outermost) outermost = rf;
                } else if (side == QUEENSIDE && rf < kf) {
                    if (outermost < 0 || rf < outermost) outermost = rf;
                }
            }
            if (outermost == int(f) && !is_chess960) {
                o << outer_letter;
            } else {
                o << char(file_base + int(f));
            }
        };
        emit_side(WHITE, KINGSIDE,  'K', 'A');
        emit_side(WHITE, QUEENSIDE, 'Q', 'A');
        emit_side(BLACK, KINGSIDE,  'k', 'a');
        emit_side(BLACK, QUEENSIDE, 'q', 'a');
    }
    o << ' ';
    if (ep_square == NO_SQUARE) {
        o << '-';
    }
    else {
        o << char('a' + file_of(ep_square)) << char('1' + rank_of(ep_square));
    }
    o << ' ' << halfmove_clock << ' ' << fullmove_number;
    return o.str();
}

// Bits cleared from `castling` when a rook moves off its stored
// starting square OR an enemy piece captures on that square. Chess960
// means the rook starting file isn't necessarily A/H, so this must be
// computed from `castling_rook_file[]` rather than a static table.
// Call this for both `from` and `to` of every move; the AND accumulates.
// King moves are handled separately (clear BOTH own-color bits) because
// the king starting square isn't tracked -- any king move clears rights.
static int cr_mask_rook_from(const Position& pos, Square s) {
    int mask = 15;
    for (int c = 0; c < NUM_COLORS; ++c) {
        Rank home = (c == WHITE) ? RANK_1 : RANK_8;
        if (rank_of(s) != home) continue;
        for (int side = 0; side < 2; ++side) {
            int bit = (c == WHITE)
                ? (side == KINGSIDE ? WHITE_OO : WHITE_OOO)
                : (side == KINGSIDE ? BLACK_OO : BLACK_OOO);
            if ((pos.castling & bit) == 0) continue;
            if (file_of(s) == pos.castling_rook_file[c][side]) {
                mask &= ~bit;
            }
        }
    }
    return mask;
}

void Position::recompute_psq() {
    psq_mg[WHITE] = psq_mg[BLACK] = 0;
    psq_eg[WHITE] = psq_eg[BLACK] = 0;
    for (int c = 0; c < NUM_COLORS; ++c) {
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pieces[c][pt];
            while (bb != 0U) {
                Square s = pop_lsb(bb);
                psq_mg[c] += eval::psq_mg(Color(c), PieceType(pt), s);
                psq_eg[c] += eval::psq_eg(Color(c), PieceType(pt), s);
            }
        }
    }
}

void Position::put_piece(Square s, Piece p) {
    assert(board[s] == NO_PIECE);
    assert(p != NO_PIECE);
    Color     c  = color_of(p);
    PieceType pt = type_of(p);
    board[s]                     = p;
    Bitboard bb                  = square_bb(s);
    pieces[c][pt]               |= bb;
    colors[c]                   |= bb;
    occupied                    |= bb;
    key                         ^= zobrist::PIECE_SQ[c][pt][s];
    if (pt == PAWN) {
        pawn_key                ^= zobrist::PIECE_SQ[c][PAWN][s];
    }
    psq_mg[c]                   += eval::psq_mg(c, pt, s);
    psq_eg[c]                   += eval::psq_eg(c, pt, s);
    // NNUE incremental accumulator update. Kings themselves aren't
    // in the HalfKP feature set, but moving them invalidates every
    // feature from that side's perspective (all features are keyed
    // on the friendly king square) — mark dirty and let refresh
    // rebuild. Non-king pieces add their feature column to both
    // perspectives; the columns come from the loaded network so this
    // is a no-op when NNUE isn't loaded.
    if (pt == KING) {
        acc.computed[c] = false;
    } else if (nnue::is_loaded()) {
        nnue::add_piece_to_accumulator(*this, acc, s, pt, c);
    }
}

void Position::remove_piece(Square s) {
    Piece p = board[s];
    assert(p != NO_PIECE);
    Color     c  = color_of(p);
    PieceType pt = type_of(p);
    // NNUE incremental subtract must run BEFORE the bitboard clear
    // so refresh_accumulator's fallback path (if triggered later)
    // still sees the king square. Kings again mark dirty; non-kings
    // subtract their column from both perspectives.
    if (pt == KING) {
        acc.computed[c] = false;
    } else if (nnue::is_loaded()) {
        nnue::sub_piece_from_accumulator(*this, acc, s, pt, c);
    }
    board[s]                     = NO_PIECE;
    Bitboard bb                  = square_bb(s);
    pieces[c][pt]               &= ~bb;
    colors[c]                   &= ~bb;
    occupied                    &= ~bb;
    key                         ^= zobrist::PIECE_SQ[c][pt][s];
    if (pt == PAWN) {
        pawn_key                ^= zobrist::PIECE_SQ[c][PAWN][s];
    }
    psq_mg[c]                   -= eval::psq_mg(c, pt, s);
    psq_eg[c]                   -= eval::psq_eg(c, pt, s);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) — make_move handles normal, capture, en-passant, castling, and promotion in one fused function so the incremental Zobrist + PSQ updates stay in one place; splitting duplicates the piece-swap boilerplate 4x.
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
    u.pawn_key       = pawn_key;
    if (mt == MT_EN_PASSANT) {
        u.captured = Piece(us == WHITE ? B_PAWN : W_PAWN);
    } else if (mt == MT_CASTLING) {
        // Castling never captures. In FRC the generic `board[to]` lookup
        // would see our own rook (king_to can overlap rook_from) and
        // wrongly flag it as a capture.
        u.captured = NO_PIECE;
    } else {
        u.captured = board[to];
    }

    // Roll the pre-move castling/ep/side keys OUT now. The corresponding
    // post-move keys are XOR'd back in at the end after those fields are
    // updated. Piece-square keys are handled inside put_piece/remove_piece.
    // EP is hashed only when the ep capture is pseudo-legal — same rule
    // as zobrist::compute, so incremental key stays in sync.
    key ^= zobrist::CASTLING[castling & 15];
    if (zobrist::ep_is_capturable(*this)) {
        key ^= zobrist::EP_FILE[file_of(ep_square)];
    }

    if (mt == MT_CASTLING) {
        // Internal encoding: from = king_from, to = king_to (G or C file).
        // In FRC any of {king_from, king_to, rook_from, rook_to} may
        // overlap another, so the safe sequence is remove-both-then-
        // put-both: after both removes the four squares are empty and
        // the two puts can land anywhere without tripping put_piece's
        // "destination must be empty" assert.
        CastleSide side = (file_of(to) == FILE_G) ? KINGSIDE : QUEENSIDE;
        File rook_file  = castling_rook_file[us][side];
        Rank rank       = rank_of(to);
        Square king_from = from;
        Square king_to   = to;
        Square rook_from = make_square(rook_file, rank);
        Square rook_to   = make_square(side == KINGSIDE ? FILE_F : FILE_D, rank);
        Piece king_piece = board[king_from];
        Piece rook_piece = board[rook_from];
        assert(type_of(king_piece) == KING);
        assert(type_of(rook_piece) == ROOK);
        assert(color_of(king_piece) == us && color_of(rook_piece) == us);
        remove_piece(king_from);
        remove_piece(rook_from);
        put_piece(king_to, king_piece);
        put_piece(rook_to, rook_piece);
    } else {
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
    }

    // Castling-rights update: king move clears BOTH of own color's bits
    // (MT_CASTLING implicitly covered because moving is a king);
    // rook-square checks catch a rook moving off its stored starting
    // file or being captured there. Unlike classical chess, we can't
    // use a precomputed CR_MASK table because the rook starting files
    // vary per game under Chess960.
    if (type_of(moving) == KING) {
        castling &= ~(us == WHITE ? (WHITE_OO | WHITE_OOO) : (BLACK_OO | BLACK_OOO));
    }
    castling &= cr_mask_rook_from(*this, from) & cr_mask_rook_from(*this, to);

    // En passant: set only when a pawn double-pushes; cleared otherwise.
    ep_square = NO_SQUARE;
    if (type_of(moving) == PAWN && std::abs(int(to) - int(from)) == 16) {
        ep_square = Square((int(from) + int(to)) / 2);
    }

    // Halfmove clock: reset on pawn move or capture.
    if (type_of(moving) == PAWN || u.captured != NO_PIECE) {
        halfmove_clock = 0;
    }
    else {
        ++halfmove_clock;
    }

    if (us == BLACK) {
        ++fullmove_number;
    }
    side_to_move = them;

    // Roll the new castling / ep / side keys IN. SIDE toggles on every
    // move (XOR is self-inverse) regardless of which color moved. EP
    // hash matches compute()'s rule (pseudo-legal only).
    key ^= zobrist::CASTLING[castling & 15];
    if (zobrist::ep_is_capturable(*this)) {
        key ^= zobrist::EP_FILE[file_of(ep_square)];
    }
    key ^= zobrist::SIDE;

    // Push the post-move key so repetition detection sees this state.
    // Bounded by HISTORY_CAPACITY; overflow would be a search-depth bug
    // (real games can't approach it), so assert loud in debug and cap
    // silently in release rather than corrupting the stack.
    assert(history_size < HISTORY_CAPACITY);
    if (history_size < HISTORY_CAPACITY) {
        history[history_size++] = key;
    }

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
    if (us == BLACK) {
        --fullmove_number;
    }

    if (mt == MT_CASTLING) {
        // Mirror make_move's remove-both-then-put-both sequence. The
        // rook starting file is still in castling_rook_file (it never
        // changes during a game -- only the castling bitmask does).
        CastleSide side = (file_of(to) == FILE_G) ? KINGSIDE : QUEENSIDE;
        File rook_file  = castling_rook_file[us][side];
        Rank rank       = rank_of(to);
        Square king_from = from;
        Square king_to   = to;
        Square rook_from = make_square(rook_file, rank);
        Square rook_to   = make_square(side == KINGSIDE ? FILE_F : FILE_D, rank);
        Piece king_piece = board[king_to];
        Piece rook_piece = board[rook_to];
        remove_piece(king_to);
        remove_piece(rook_to);
        put_piece(king_from, king_piece);
        put_piece(rook_from, rook_piece);
    } else {
        // Undo the piece move. For promotion, restore a pawn at `from`
        // rather than the promoted piece.
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
    }

    ep_square      = u.ep_square;
    castling       = u.castling;
    halfmove_clock = u.halfmove_clock;
    // Snapshot restore beats redoing all the incremental XORs by hand —
    // and it's what tests check against (compute(pos) after unmake must
    // equal the pre-move key). Pawn key is fully rebuildable from the
    // incremental XORs in put_piece / remove_piece, but the snapshot
    // form is cheaper and matches how `key` is restored.
    key            = u.key;
    pawn_key       = u.pawn_key;
    // Pop the repetition-history entry pushed by the matching make_move.
    if (history_size > 0) {
        --history_size;
    }
}

bool Position::is_repetition() const {
    // Scan back for a matching key. Steps by 2 (only same-side-to-move
    // positions can share a Zobrist key) and stops at the halfmove_clock
    // boundary — pawn moves and captures are irreversible, so any
    // position before the last such move can't be reached again.
    int stop = history_size - 1 - halfmove_clock;
    stop = std::max(stop, 0);
    // history[history_size - 1] is the CURRENT position — skip it.
    for (int i = history_size - 3; i >= stop; i -= 2) {
        if (history[i] == key) {
            return true;
        }
    }
    return false;
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
