// UCI protocol tests. uci_loop's public contract is "read command lines
// from `in`, write protocol replies to `out`." Feed input via a stringstream
// and assert on captured output — same shape a real UCI GUI experiences.

#include "doctest.h"

#include "uci.h"

#include <sstream>
#include <string>

// Drive one uci_loop session with the given script; return everything the
// engine printed. Every script must end with "quit\n" so the loop returns.
static std::string run_session(const std::string& input) {
    std::istringstream in(input);
    std::ostringstream out;
    uci_loop(in, out);
    return out.str();
}

// doctest doesn't ship a "contains" matcher; a helper reads cleaner than
// find != npos scattered across every CHECK.
static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

TEST_CASE("uci handshake identifies the engine and terminates with uciok") {
    std::string out = run_session("uci\nquit\n");
    CHECK(contains(out, "id name jedi-engine"));
    CHECK(contains(out, "id author"));
    CHECK(contains(out, "uciok"));
}

TEST_CASE("isready responds with readyok") {
    CHECK(contains(run_session("isready\nquit\n"), "readyok"));
}

TEST_CASE("position startpos followed by d shows the initial FEN") {
    std::string out = run_session("position startpos\nd\nquit\n");
    CHECK(contains(out, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
}

TEST_CASE("position fen sets the requested FEN verbatim") {
    // Uses a distinctive FEN so partial-match false positives are unlikely.
    const std::string fen = "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1";
    std::string out = run_session("position fen " + fen + "\nd\nquit\n");
    CHECK(contains(out, fen));
}

TEST_CASE("ucinewgame resets to startpos after loading a different position") {
    std::string out = run_session(
        "position fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1\n"
        "ucinewgame\n"
        "d\n"
        "quit\n");
    CHECK(contains(out, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
}

// Count how many lines in `haystack` start with `prefix`. Used to verify
// iterative deepening emits one info line per completed depth rather than
// one summary line.
static int count_lines_starting(const std::string& haystack, const std::string& prefix) {
    int n = 0;
    size_t pos = 0;
    while (pos < haystack.size()) {
        if (haystack.compare(pos, prefix.size(), prefix) == 0) ++n;
        size_t nl = haystack.find('\n', pos);
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    return n;
}

TEST_CASE("go emits one info line per iteration plus a bestmove") {
    // Iterative deepening: `go depth 3` runs depth 1, 2, 3 and emits an
    // info line after each — so a GUI can display live progress.
    std::string out = run_session("position startpos\ngo depth 3\nquit\n");
    CHECK(contains(out, "info depth 1"));
    CHECK(contains(out, "info depth 2"));
    CHECK(contains(out, "info depth 3"));
    CHECK(count_lines_starting(out, "info depth ") == 3);
    CHECK(contains(out, "score cp "));
    CHECK(contains(out, "nodes "));
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("go on a mated position emits bestmove 0000") {
    // No legal moves → NULL_MOVE → serialized as "0000" per UCI convention.
    std::string out = run_session(
        "position fen 4k3/8/8/8/8/8/6PP/5r1K w - - 0 1\n"
        "go depth 1\n"
        "quit\n");
    CHECK(contains(out, "bestmove 0000"));
}

TEST_CASE("go depth 1 reports depth 1 in info line") {
    std::string out = run_session("position startpos\ngo depth 1\nquit\n");
    CHECK(contains(out, "info depth 1"));
    // No deeper iterations — depth 1 is the ceiling.
    CHECK_FALSE(contains(out, "info depth 2"));
}

TEST_CASE("go infinite runs asynchronously; stop returns bestmove") {
    // `go infinite` has no natural terminator — cmd_go must return
    // immediately so the loop can process the subsequent `stop`. The
    // stop handler then joins the search thread and its bestmove has
    // already been emitted from the thread's exit path.
    std::string out = run_session(
        "position startpos\n"
        "go infinite\n"
        "stop\n"
        "quit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
    // At least one iteration should have completed before stop landed
    // (depth 1 on startpos is ~microseconds); if the info line is
    // missing, either the search didn't start or the stop check is
    // firing on the wrong condition.
    CHECK(contains(out, "info depth "));
}

TEST_CASE("stop with no active search is a no-op") {
    // A GUI can send stop after bestmove has already been emitted — the
    // engine should just ignore it, not crash on a joined thread.
    std::string out = run_session("stop\nquit\n");
    CHECK(out.empty());
}

TEST_CASE("multiple go infinite + stop cycles work in one session") {
    // Second `go infinite` after the first has been stopped must start
    // a fresh search, not inherit stopped=true from the previous cycle.
    std::string out = run_session(
        "position startpos\n"
        "go infinite\n"
        "stop\n"
        "go infinite\n"
        "stop\n"
        "quit\n");
    // Two bestmoves — one per go/stop pair.
    int bestmove_count = 0;
    size_t p = 0;
    while ((p = out.find("bestmove ", p)) != std::string::npos) {
        ++bestmove_count;
        p += 9;
    }
    CHECK(bestmove_count == 2);
}

TEST_CASE("go movetime completes within the deadline with a legal move") {
    // Movetime alone (no explicit depth) → iterative deepening searches
    // deeper and deeper until the deadline. Must always return a legal move.
    std::string out = run_session("position startpos\ngo movetime 100\nquit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("go with wtime/btime derives a movetime and returns a bestmove") {
    // 60-second clock, sudden-death. Engine should compute ~2s (60000/30)
    // and return within that. We only assert on the observable UCI output:
    // bestmove exists and isn't the null move.
    std::string out = run_session(
        "position startpos\n"
        "go wtime 60000 btime 60000\n"
        "quit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("go with clock args + increment returns a legal move") {
    // Fischer time control: 30 seconds base + 1 second per move.
    std::string out = run_session(
        "position startpos\n"
        "go wtime 30000 btime 30000 winc 1000 binc 1000\n"
        "quit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("go with movestogo (classical time control) returns a legal move") {
    // 40 moves in 5 minutes. Engine should use ~7-8s per move.
    std::string out = run_session(
        "position startpos\n"
        "go wtime 300000 btime 300000 movestogo 40\n"
        "quit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("go with critically low time still returns a bestmove") {
    // 50 ms remaining — the safety buffer eats most of it. Engine must
    // still return something rather than losing on time by hanging.
    std::string out = run_session(
        "position startpos\n"
        "go wtime 50 btime 50\n"
        "quit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("explicit movetime overrides clock-derived movetime") {
    // Both movetime and wtime given. The explicit movetime wins so a
    // GUI can force short thinking during analysis mode without lying
    // about the clock.
    std::string out = run_session(
        "position startpos\n"
        "go wtime 60000 movetime 50\n"
        "quit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("position startpos moves e2e4 → board reflects the applied move") {
    // After 1. e4, black should be to move and the e2 pawn should be on e4.
    // The `d` command's FEN output is our observable — it round-trips the
    // full position state including side-to-move and ep_square.
    std::string out = run_session(
        "position startpos moves e2e4\n"
        "d\n"
        "quit\n");
    CHECK(contains(out, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"));
}

TEST_CASE("position startpos moves ... applies a multi-move sequence") {
    // Ruy Lopez opening moves. Each move must land at its expected square
    // for the final FEN to match — a bug in any intermediate move breaks
    // the string comparison.
    std::string out = run_session(
        "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5\n"
        "d\n"
        "quit\n");
    CHECK(contains(out,
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"));
}

TEST_CASE("position moves handles promotion syntax (a7a8q)") {
    // Pawn promotes to queen on the last move.
    std::string out = run_session(
        "position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1 moves a7a8q\n"
        "d\n"
        "quit\n");
    CHECK(contains(out, "Q3k3/8/8/8/8/8/8/4K3 b - - 0 1"));
}

TEST_CASE("position moves handles castling syntax (e1g1)") {
    std::string out = run_session(
        "position fen 4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1 moves e1g1\n"
        "d\n"
        "quit\n");
    CHECK(contains(out, "4k3/8/8/8/8/8/8/R4RK1 b - - 1 1"));
}

TEST_CASE("position moves handles en passant (d5c6 with ep set)") {
    std::string out = run_session(
        "position fen 4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1 moves d5c6\n"
        "d\n"
        "quit\n");
    CHECK(contains(out, "4k3/8/2P5/8/8/8/8/4K3 b - - 0 1"));
}

TEST_CASE("position moves silently stops on a bogus move token") {
    // The engine should apply moves up to the first bad token and then
    // stop — dropping the tail is safer than either crashing or picking
    // some fake substitute. After "e2e4 nonsense f2f4", position should
    // reflect only e2e4.
    std::string out = run_session(
        "position startpos moves e2e4 nonsense f2f4\n"
        "d\n"
        "quit\n");
    CHECK(contains(out, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"));
}

TEST_CASE("go after position moves searches from the resulting position") {
    // Full integration: load a position via moves, then search. The
    // engine must consider the post-move position, not startpos.
    std::string out = run_session(
        "position startpos moves e2e4 e7e5\n"
        "go depth 2\n"
        "quit\n");
    CHECK(contains(out, "bestmove "));
    CHECK_FALSE(contains(out, "bestmove 0000"));
}

TEST_CASE("wtime is used when white to move; btime when black") {
    // Both sides get near-zero on their own clock but plenty on the
    // opponent's. If we correctly pick the mover's clock, both cases
    // return quickly; if we picked the wrong side, one case would
    // search for seconds (slow to detect in a doctest).
    // Assertion is behavioral: both return a legal move.
    std::string white_side = run_session(
        "position fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1\n"
        "go wtime 50 btime 60000\n"
        "quit\n");
    CHECK(contains(white_side, "bestmove "));
    CHECK_FALSE(contains(white_side, "bestmove 0000"));

    std::string black_side = run_session(
        "position fen 4k3/8/8/8/8/8/8/4K3 b - - 0 1\n"
        "go wtime 60000 btime 50\n"
        "quit\n");
    CHECK(contains(black_side, "bestmove "));
    CHECK_FALSE(contains(black_side, "bestmove 0000"));
}

TEST_CASE("quit alone terminates the loop without output") {
    CHECK(run_session("quit\n").empty());
}

TEST_CASE("unknown commands are ignored, not errored") {
    // UCI spec: unknown lines should be silently ignored so future extensions
    // don't break older engines. Verify by interleaving with a known command
    // and checking the known command still worked.
    std::string out = run_session("wibble\nblargh\nisready\nquit\n");
    CHECK(contains(out, "readyok"));
}

TEST_CASE("EOF on input terminates the loop cleanly (no explicit quit)") {
    // A GUI that closes the pipe should not leave the engine hung. Same as
    // any other loop-exit — we just want run_session to return.
    std::string out = run_session("isready\n");   // no "quit" — EOF closes it
    CHECK(contains(out, "readyok"));
}
