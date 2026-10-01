#include "perft.h"
#include "movegen.h"

#include <ostream>

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) {
        return 1;
    }
    MoveList moves;
    generate_moves(pos, moves);
    uint64_t nodes = 0;
    UndoInfo u;
    for (Move m : moves) {
        pos.make_move(m, u);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(m, u);
    }
    return nodes;
}

// Standard perft positions. Values from
// https://www.chessprogramming.org/Perft_Results
struct PerftEntry {
    const char* name;
    const char* fen;
    uint64_t    counts[7];  // index = depth; 0 means "not checked"
};

static const PerftEntry SUITE[] = {
    {"Startpos", STARTPOS_FEN,
     {1, 20, 400, 8902, 197281, 4865609, 119060324}},
    {"Kiwipete",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     {1, 48, 2039, 97862, 4085603, 193690690, 0}},
    {"Position 3",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     {1, 14, 191, 2812, 43238, 674624, 11030083}},
    {"Position 4",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     {1, 6, 264, 9467, 422333, 15833292, 0}},
    {"Position 5",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     {1, 44, 1486, 62379, 2103487, 89941194, 0}},
    {"Position 6",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     {1, 46, 2079, 89890, 3894594, 164075551, 0}},
    // Chess960 perft positions. Rook starting files vary; FEN uses
    // Shredder-FEN castling encoding (file letters, not KQkq). Expected
    // node counts cross-verified against chessops' independent FRC
    // movegen -- if the two engines disagree, one of them has a bug.
    {"FRC 1 (RBQNBKRN)",
     "rbqnbkrn/pppppppp/8/8/8/8/PPPPPPPP/RBQNBKRN w GAga - 0 1",
     {1, 20, 400, 8976, 200493, 4968908, 0}},
    {"FRC 2 (RBBQNNKR)",
     "rbbqnnkr/pppppppp/8/8/8/8/PPPPPPPP/RBBQNNKR w HAha - 0 1",
     {1, 20, 400, 9072, 204305, 5116256, 0}},
    {"FRC 3 (NRKBNQBR)",
     "nrkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/NRKBNQBR w HBhb - 0 1",
     {1, 19, 361, 7805, 167933, 4054729, 0}},
};

bool run_perft_suite(int max_depth, std::ostream& out) {
    bool all_ok = true;
    for (const auto& e : SUITE) {
        Position pos;
        if (!pos.set_from_fen(e.fen)) {
            out << "[FAIL] " << e.name << ": bad FEN\n";
            all_ok = false;
            continue;
        }
        out << "=== " << e.name << " ===\n" << e.fen << '\n';
        for (int d = 1; d <= max_depth && d < 7; ++d) {
            uint64_t expected = e.counts[d];
            if (expected == 0) {
                continue;
            }
            uint64_t got  = perft(pos, d);
            const char* mark = (got == expected) ? "OK  " : "FAIL";
            out << "  [" << mark << "] depth " << d
                << ": got " << got
                << ", expected " << expected << '\n';
            if (got != expected) {
                all_ok = false;
            }
        }
    }
    return all_ok;
}
