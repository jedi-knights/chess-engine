#include "uci.h"
#include "position.h"
#include "search.h"
#include "types.h"

#include <iostream>
#include <sstream>
#include <string>

namespace {

// Serialize a Move to UCI long-algebraic notation ("e2e4", "e7e8q").
// NULL_MOVE → "0000" per the UCI convention for "no legal move."
std::string move_to_uci(Move m) {
    if (m == NULL_MOVE) return "0000";
    Square from = move_from(m);
    Square to   = move_to(m);
    std::string s;
    s += char('a' + file_of(from));
    s += char('1' + rank_of(from));
    s += char('a' + file_of(to));
    s += char('1' + rank_of(to));
    if (move_type(m) == MT_PROMOTION) {
        s += "nbrq"[move_promotion(m) - KNIGHT];
    }
    return s;
}

void cmd_uci(std::ostream& out) {
    out << "id name jedi-engine 0.0.1\n"
        << "id author omar\n"
        << "uciok\n" << std::flush;
}

void cmd_isready(std::ostream& out) {
    out << "readyok\n" << std::flush;
}

void cmd_position(std::istringstream& is, Position& pos) {
    std::string token;
    is >> token;
    if (token == "startpos") {
        pos.set_from_fen(STARTPOS_FEN);
        is >> token;  // possibly "moves"
    } else if (token == "fen") {
        std::string fen;
        while (is >> token && token != "moves") {
            if (!fen.empty()) fen += ' ';
            fen += token;
        }
        pos.set_from_fen(fen);
    }
    // TODO: consume trailing "moves e2e4 e7e5 ..." tokens and apply them
    //       via pos.make_move() once move parsing exists.
    while (is >> token) { /* drain */ }
}

// Default depth ceiling when the GUI specifies neither `depth` nor
// `movetime`. Small enough to finish in well under a second even in a
// debug build.
constexpr int DEFAULT_DEPTH = 4;
// When the GUI specifies `movetime` but no explicit depth, iterative
// deepening runs until time is up — cap at a depth that would take
// well beyond any practical think-time on this engine's speed.
constexpr int MOVETIME_MAX_DEPTH = 64;

void cmd_go(std::istringstream& is, Position& pos, std::ostream& out) {
    SearchLimits limits;
    bool depth_set = false, movetime_set = false;
    std::string token;
    while (is >> token) {
        if (token == "depth" && (is >> token)) {
            try { limits.max_depth = std::stoi(token); depth_set = true; }
            catch (...) { /* ignore malformed value, keep default */ }
        } else if (token == "movetime" && (is >> token)) {
            try { limits.movetime_ms = std::stoi(token); movetime_set = true; }
            catch (...) { /* ignore malformed value */ }
        }
        // TODO: wtime/btime, infinite — later.
    }
    if (!depth_set) {
        limits.max_depth = movetime_set ? MOVETIME_MAX_DEPTH : DEFAULT_DEPTH;
    }

    SearchResult r = search_iterative(pos, limits,
        [&](const SearchResult& iter) {
            out << "info depth " << iter.depth
                << " score cp "  << iter.score
                << " nodes "     << iter.nodes
                << " pv "        << move_to_uci(iter.best_move)
                << "\n" << std::flush;
        });
    out << "bestmove " << move_to_uci(r.best_move) << "\n" << std::flush;
}

}  // namespace

void uci_loop(std::istream& in, std::ostream& out) {
    Position pos;
    pos.set_from_fen(STARTPOS_FEN);

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if      (cmd == "uci")        cmd_uci(out);
        else if (cmd == "isready")    cmd_isready(out);
        else if (cmd == "ucinewgame") { pos.set_from_fen(STARTPOS_FEN);
                                        clear_transposition_table(); }
        else if (cmd == "position")   cmd_position(is, pos);
        else if (cmd == "go")         cmd_go(is, pos, out);
        else if (cmd == "d")          out << pos.pretty() << std::flush;
        else if (cmd == "quit")       break;
    }
}
