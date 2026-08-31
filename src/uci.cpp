#include "uci.h"
#include "position.h"
#include "movegen.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void cmd_uci() {
    std::cout << "id name jedi-engine 0.0.1\n"
              << "id author omar\n"
              << "uciok\n" << std::flush;
}

static void cmd_isready() {
    std::cout << "readyok\n" << std::flush;
}

static void cmd_position(std::istringstream& is, Position& pos) {
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

static void cmd_go(const Position& pos) {
    // TODO: real search. For now, always emit a null bestmove so a GUI
    //       will not hang waiting for a response.
    std::vector<Move> moves;
    generate_moves(pos, moves);
    std::cout << "bestmove 0000\n" << std::flush;
}

void uci_loop() {
    Position pos;
    pos.set_from_fen(STARTPOS_FEN);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if      (cmd == "uci")        cmd_uci();
        else if (cmd == "isready")    cmd_isready();
        else if (cmd == "ucinewgame") pos.set_from_fen(STARTPOS_FEN);
        else if (cmd == "position")   cmd_position(is, pos);
        else if (cmd == "go")         cmd_go(pos);
        else if (cmd == "d")          std::cout << pos.pretty() << std::flush;
        else if (cmd == "quit")       break;
    }
}
