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

TEST_CASE("go emits a bestmove line (currently the null move)") {
    // Contract with any UCI GUI: `go` must always produce `bestmove <move>`
    // so the GUI doesn't hang waiting. Search milestone will replace 0000
    // with a real move — test asserts the line exists, not its content.
    std::string out = run_session("position startpos\ngo\nquit\n");
    CHECK(contains(out, "bestmove "));
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
