#include "attacks.h"
#include "perft.h"
#include "uci.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    init_attacks();

    if (argc >= 2 && std::strcmp(argv[1], "perft") == 0) {
        int depth = (argc >= 3) ? std::atoi(argv[2]) : 4;
        std::printf("Running perft suite up to depth %d\n", depth);
        return run_perft_suite(depth) ? 0 : 1;
    }

    uci_loop();
    return 0;
}
