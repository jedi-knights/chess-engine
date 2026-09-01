#include "uci.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "search.h"
#include "types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// Search thread state. `g_stop` is polled by search on the same 1024-
// node cadence as the time-up check; `g_search_thread` is the worker
// running the current search (may or may not be joinable); `g_out_mutex`
// serializes writes to `out` since the search thread emits info lines
// while the main loop might also print.
std::atomic<bool> g_stop{false};
std::thread       g_search_thread;
std::mutex        g_out_mutex;

// Send a chunk of output atomically. All writes to `out` (main-thread
// commands AND background info/bestmove lines) go through this so an
// info line never gets interleaved with an isready reply.
void emit(std::ostream& out, const std::string& msg) {
    std::lock_guard<std::mutex> lg(g_out_mutex);
    out << msg << std::flush;
}

// Cancel any active search, wait for it to finish. Called before starting
// a new search, on `stop`, and during `quit` / EOF cleanup so no thread
// outlives its `out` reference.
void wait_for_search() {
    if (g_search_thread.joinable()) {
        g_stop.store(true, std::memory_order_relaxed);
        g_search_thread.join();
    }
    g_stop.store(false, std::memory_order_relaxed);
}

void cmd_uci(std::ostream& out) {
    emit(out,
         "id name jedi-engine 0.0.1\n"
         "id author omar\n"
         "uciok\n");
}

void cmd_isready(std::ostream& out) {
    emit(out, "readyok\n");
}

void cmd_position(std::istringstream& is, Position& pos, std::ostream& out) {
    // A `position` command mid-search would race with the running search's
    // read of `pos` (the search thread copies pos, so it's safe on the
    // search's side, but the semantics of "search this old position, then
    // report on the new position" are useless to any real GUI). Cancel
    // and drop any pending bestmove — but surface it, since sending
    // `position` during `go infinite` without a `stop` first is a GUI bug
    // that would otherwise fail silently.
    if (g_search_thread.joinable()) {
        emit(out, "info string position received during active search; "
                  "canceling and applying new position\n");
    }
    wait_for_search();

    std::string token;
    is >> token;
    if (token == "startpos") {
        pos.set_from_fen(STARTPOS_FEN);
        is >> token;  // possibly "moves"
    } else if (token == "fen") {
        std::string fen;
        while (is >> token && token != "moves") {
            if (!fen.empty()) {
                fen += ' ';
            }
            fen += token;
        }
        pos.set_from_fen(fen);
    }
    if (token != "moves") {
        return;
    }
    while (is >> token) {
        Move m = parse_uci_move(pos, token);
        if (m == NULL_MOVE) {
            return;
        }
        MoveList legal;
        generate_moves(pos, legal);
        if (std::find(legal.begin(), legal.end(), m) == legal.end()) {
            return;
        }
        UndoInfo u;
        pos.make_move(m, u);
    }
}

// Default depth ceiling when the GUI specifies neither `depth` nor
// `movetime`. Small enough to finish in well under a second even in a
// debug build.
constexpr int DEFAULT_DEPTH = 4;
// When the GUI specifies `movetime`, clock args, or `infinite` but no
// explicit depth, iterative deepening runs until stopped externally —
// cap at a depth that would take well beyond any practical think-time.
constexpr int MOVETIME_MAX_DEPTH = 64;

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — (remaining, increment, movestogo) mirrors the UCI wtime/winc/movestogo token order the parser feeds in.
int compute_movetime(int remaining_ms, int increment_ms, int movestogo) {
    constexpr int SAFETY_BUFFER_MS = 100;
    int budget = remaining_ms - SAFETY_BUFFER_MS;
    if (budget <= 0) {
        return 1;
    }

    int moves_left = (movestogo > 0) ? movestogo : 30;
    int base       = budget / (moves_left + 2);
    int inc        = (increment_ms * 3) / 4;
    int movetime   = base + inc;
    int hard_cap   = budget / 3;
    movetime = std::min(movetime, hard_cap);
    movetime = std::max(movetime, 5);
    return movetime;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) — UCI `go` parses N optional keyword-value pairs (depth, movetime, wtime/btime/winc/binc/movestogo, infinite) and dispatches to sync or async search; the shape follows the UCI spec, not our decomposition.
void cmd_go(std::istringstream& is, const Position& pos, std::ostream& out) {
    // Cancel any prior search before starting a new one. Its bestmove
    // (if it managed to emit one before the cancel) has already gone out.
    wait_for_search();

    SearchLimits limits;
    bool depth_set = false;
    bool movetime_set = false;
    bool infinite = false;
    int  wtime = 0;
    int  btime = 0;
    int  winc = 0;
    int  binc = 0;
    int  movestogo = 0;
    bool clock_given = false;

    std::string token;
    auto next_int = [&](int& out_value) {
        if (!(is >> token)) {
            return false;
        }
        try { out_value = std::stoi(token); return true; }
        catch (...) { return false; }
    };

    while (is >> token) {
        if      (token == "depth")     { int v; if (next_int(v)) { limits.max_depth = v; depth_set = true; } }
        else if (token == "movetime")  { int v; if (next_int(v)) { limits.movetime_ms = v; movetime_set = true; } }
        else if (token == "wtime")     { if (next_int(wtime)) { clock_given = true; } }
        else if (token == "btime")     { if (next_int(btime)) { clock_given = true; } }
        else if (token == "winc")      { next_int(winc); }
        else if (token == "binc")      { next_int(binc); }
        else if (token == "movestogo") { next_int(movestogo); }
        else if (token == "infinite")  { infinite = true; }
    }

    if (!movetime_set && clock_given) {
        int remaining = (pos.side_to_move == WHITE) ? wtime : btime;
        int increment = (pos.side_to_move == WHITE) ? winc  : binc;
        limits.movetime_ms = compute_movetime(remaining, increment, movestogo);
    }

    if (infinite) {
        // `infinite` disables all natural stop conditions — search runs
        // until `stop` fires the external cancellation.
        if (!depth_set) {
            limits.max_depth = MOVETIME_MAX_DEPTH;
        }
        limits.movetime_ms = 0;
    } else if (!depth_set) {
        limits.max_depth = (limits.movetime_ms > 0) ? MOVETIME_MAX_DEPTH : DEFAULT_DEPTH;
    }

    g_stop.store(false, std::memory_order_relaxed);
    limits.external_stop = &g_stop;

    // Copy pos into the thread — the main loop keeps its own reference
    // for future commands, and detaching decouples the two.
    Position pos_copy = pos;
    std::ostream* out_ptr = &out;

    auto search_start = std::chrono::steady_clock::now();
    g_search_thread = std::thread(
        [pos_copy, limits, out_ptr, search_start]() mutable {
            SearchResult r = search_iterative(pos_copy, limits,
                [&](const SearchResult& iter) {
                    auto now = std::chrono::steady_clock::now();
                    int elapsed_ms = static_cast<int>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - search_start).count());
                    // NPS: guard against divide-by-zero on sub-ms iterations.
                    uint64_t nps = elapsed_ms > 0
                        ? (iter.nodes * 1000ULL) / static_cast<uint64_t>(elapsed_ms)
                        : iter.nodes * 1000ULL;

                    std::ostringstream line;
                    line << "info depth " << iter.depth
                         << " score cp "  << iter.score
                         << " nodes "     << iter.nodes
                         << " nps "       << nps
                         << " time "      << elapsed_ms
                         << " pv";
                    // Full PV walked from the TT. Fall back to just the
                    // bestmove if the walk came up empty (shouldn't
                    // happen post-iteration but the check is cheap).
                    if (iter.pv.empty()) {
                        line << ' ' << move_to_uci(iter.best_move);
                    } else {
                        for (Move m : iter.pv) {
                            line << ' ' << move_to_uci(m);
                        }
                    }
                    line << "\n";
                    emit(*out_ptr, line.str());
                });
            emit(*out_ptr, "bestmove " + move_to_uci(r.best_move) + "\n");
        });

    // Non-infinite `go` waits synchronously — the search has a natural
    // stop condition (depth or movetime) so blocking here matches
    // pre-async behavior and gives GUIs a bestmove-before-return
    // guarantee. `go infinite` doesn't have a natural terminator: we
    // return immediately and rely on a subsequent `stop` to end the
    // search.
    if (!infinite && g_search_thread.joinable()) {
        g_search_thread.join();
    }
}

void cmd_stop() {
    // Signals the search to stop; join guarantees the bestmove has been
    // emitted before we return control to the loop.
    wait_for_search();
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
        if      (cmd == "uci")        { cmd_uci(out); }
        else if (cmd == "isready")    { cmd_isready(out); }
        else if (cmd == "ucinewgame") { wait_for_search();
                                        pos.set_from_fen(STARTPOS_FEN);
                                        clear_transposition_table(); }
        else if (cmd == "position")   { cmd_position(is, pos, out); }
        else if (cmd == "go")         { cmd_go(is, pos, out); }
        else if (cmd == "stop")       { cmd_stop(); }
        else if (cmd == "d")          { wait_for_search();
                                        emit(out, pos.pretty()); }
        else if (cmd == "quit")       { break; }
    }
    // EOF or `quit`: drain any pending search so its writes to `out`
    // complete before `out` (typically a caller-owned stream) is torn down.
    wait_for_search();
}
