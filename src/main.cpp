#include "attacks.h"
#include "eval.h"
#include "magic.h"
#include "perft.h"
#include "tune.h"
#include "uci.h"
#include "zobrist.h"
#include <cerrno>
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
        int depth = 4;
        if (argc >= 3) {
            char* end = nullptr;
            errno = 0;
            long parsed = std::strtol(argv[2], &end, 10);
            if (errno != 0 || end == argv[2] || *end != '\0' || parsed <= 0 || parsed > 20) {
                std::fprintf(stderr, "perft: invalid depth '%s' (expected 1..20)\n", argv[2]);
                return 1;
            }
            depth = static_cast<int>(parsed);
        }
        std::printf("Running perft suite up to depth %d\n", depth);
        return run_perft_suite(depth, std::cout) ? 0 : 1;
    }

    if (argc >= 2 && std::strcmp(argv[1], "tune") == 0) {
        if (argc < 3) {
            std::fprintf(stderr,
                "tune: missing dataset path\n"
                "usage: %s tune <dataset.txt>\n"
                "  dataset format: one line per position, `<FEN>;<outcome>`\n"
                "  outcome in {0, 0.5, 1} from WHITE's perspective\n",
                argv[0]);
            return 1;
        }
        return tune::run_tune(argv[2]);
    }

    uci_loop(std::cin, std::cout);
    return 0;
}
