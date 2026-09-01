#include "attacks.h"
#include "eval.h"
#include "magic.h"
#include "perft.h"
#include "uci.h"
#include "zobrist.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
    init_attacks();
    init_magic();
    zobrist::init();
    eval::init();

    if (argc >= 2 && std::strcmp(argv[1], "perft") == 0) {
        int depth = (argc >= 3) ? std::atoi(argv[2]) : 4;
        std::printf("Running perft suite up to depth %d\n", depth);
        return run_perft_suite(depth, std::cout) ? 0 : 1;
    }

    uci_loop(std::cin, std::cout);
    return 0;
}
