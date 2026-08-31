#include "uci.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "search.h"
#include "types.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

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
    // Trailing `moves e2e4 e7e5 ...`: parse each token, verify it's in
    // the legal-move list, apply. Any parse failure or illegal move
    // stops processing — safer than trusting the GUI blindly, since a
    // malformed token could otherwise trigger a make_move assertion.
    if (token != "moves") return;
    while (is >> token) {
        Move m = parse_uci_move(pos, token);
        if (m == NULL_MOVE) return;
        std::vector<Move> legal;
        generate_moves(pos, legal);
        if (std::find(legal.begin(), legal.end(), m) == legal.end()) return;
        UndoInfo u;
        pos.make_move(m, u);  // UndoInfo discarded — moves are cumulative
    }
}

// Default depth ceiling when the GUI specifies neither `depth` nor
// `movetime`. Small enough to finish in well under a second even in a
// debug build.
constexpr int DEFAULT_DEPTH = 4;
// When the GUI specifies `movetime` (or clock args) but no explicit
// depth, iterative deepening runs until time is up — cap at a depth
// that would take well beyond any practical think-time on this engine.
constexpr int MOVETIME_MAX_DEPTH = 64;

// Compute how long to spend on this move given the clock. Uses the
// classical divisor approach: budget ≈ remaining / (movestogo + slack),
// plus most of the increment (which is "free" — it refills the clock).
// Under sudden death (movestogo == 0), guess ~30 more moves ahead.
//
// Safety margins are conservative — losing on time is a categorical
// failure and the search's own polling granularity (1024 nodes) plus
// scheduler latency can easily overshoot the deadline by 10-50 ms.
int compute_movetime(int remaining_ms, int increment_ms, int movestogo) {
    constexpr int SAFETY_BUFFER_MS = 100;
    int budget = remaining_ms - SAFETY_BUFFER_MS;
    if (budget <= 0) return 1;                     // out of time — play instantly

    int moves_left = (movestogo > 0) ? movestogo : 30;
    int base       = budget / (moves_left + 2);   // +2 keeps some in reserve

    // Increments come back after the move, so spending them doesn't
    // shrink the clock — take most of the increment on every move.
    int inc        = (increment_ms * 3) / 4;

    int movetime   = base + inc;

    // Hard cap: never spend more than a third of remaining time on one
    // move, even under aggressive time controls or slow-search bugs.
    int hard_cap   = budget / 3;
    if (movetime > hard_cap) movetime = hard_cap;

    // Floor at a few ms so search always runs at least depth 1 (which
    // guarantees a legal bestmove for the UCI contract).
    if (movetime < 5)        movetime = 5;
    return movetime;
}

void cmd_go(std::istringstream& is, Position& pos, std::ostream& out) {
    SearchLimits limits;
    bool depth_set = false, movetime_set = false;
    int  wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 0;
    bool clock_given = false;

    std::string token;
    auto next_int = [&](int& out_value) {
        if (!(is >> token)) return false;
        try { out_value = std::stoi(token); return true; }
        catch (...) { return false; }
    };

    while (is >> token) {
        if      (token == "depth")     { int v; if (next_int(v)) { limits.max_depth = v; depth_set = true; } }
        else if (token == "movetime")  { int v; if (next_int(v)) { limits.movetime_ms = v; movetime_set = true; } }
        else if (token == "wtime")     { if (next_int(wtime)) clock_given = true; }
        else if (token == "btime")     { if (next_int(btime)) clock_given = true; }
        else if (token == "winc")      { next_int(winc); }
        else if (token == "binc")      { next_int(binc); }
        else if (token == "movestogo") { next_int(movestogo); }
        // TODO: `infinite`, `nodes`, `mate` — later.
    }

    // Explicit movetime wins over clock-derived time (both may be set;
    // GUIs sometimes send both for redundancy). Only fall back to clock
    // computation when the GUI didn't specify a fixed movetime.
    if (!movetime_set && clock_given) {
        int remaining = (pos.side_to_move == WHITE) ? wtime : btime;
        int increment = (pos.side_to_move == WHITE) ? winc  : binc;
        limits.movetime_ms = compute_movetime(remaining, increment, movestogo);
    }

    if (!depth_set) {
        limits.max_depth = (limits.movetime_ms > 0) ? MOVETIME_MAX_DEPTH : DEFAULT_DEPTH;
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
