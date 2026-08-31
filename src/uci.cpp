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

// Search depth used when the GUI doesn't specify one. Small enough that
// every response comes back in under a second even in debug builds; the
// engine currently has no time management or iterative deepening.
constexpr int DEFAULT_DEPTH = 4;

void cmd_go(std::istringstream& is, Position& pos, std::ostream& out) {
    int depth = DEFAULT_DEPTH;
    std::string token;
    while (is >> token) {
        if (token == "depth" && (is >> token)) {
            try { depth = std::stoi(token); }
            catch (...) { depth = DEFAULT_DEPTH; }
        }
        // TODO: movetime, wtime/btime, infinite — later.
    }

    SearchResult r = search_best(pos, depth);
    // UCI info line — the GUI shows this in its search-info panel. `pv`
    // is only the root move for now (no PV extraction yet); UCI permits
    // that, and it's honest about what the engine actually knows.
    out << "info depth " << r.depth
        << " score cp "  << r.score
        << " nodes "     << r.nodes
        << " pv "        << move_to_uci(r.best_move)
        << "\n";
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
        else if (cmd == "ucinewgame") pos.set_from_fen(STARTPOS_FEN);
        else if (cmd == "position")   cmd_position(is, pos);
        else if (cmd == "go")         cmd_go(is, pos, out);
        else if (cmd == "d")          out << pos.pretty() << std::flush;
        else if (cmd == "quit")       break;
    }
}
