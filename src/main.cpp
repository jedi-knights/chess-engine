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
                "usage: %s tune <dataset.txt> [mode] [iterations]\n"
                "  dataset format: one line per position, `<FEN>;<outcome>`\n"
                "  outcome in {0, 0.5, 1} from WHITE's perspective\n"
                "  mode: scalar (default), pst, all\n"
                "  iterations: SPSA iteration count (default 1000; unused in scalar mode)\n",
                argv[0]);
            return 1;
        }
        tune::Mode mode = tune::Mode::Scalar;
        if (argc >= 4) {
            if      (std::strcmp(argv[3], "scalar") == 0) { mode = tune::Mode::Scalar; }
            else if (std::strcmp(argv[3], "pst")    == 0) { mode = tune::Mode::Pst; }
            else if (std::strcmp(argv[3], "all")    == 0) { mode = tune::Mode::All; }
            else {
                std::fprintf(stderr, "tune: unknown mode '%s' (expected scalar|pst|all)\n",
                             argv[3]);
                return 1;
            }
        }
        int iterations = 1000;
        if (argc >= 5) {
            char* end = nullptr;
            errno = 0;
            long parsed = std::strtol(argv[4], &end, 10);
            if (errno != 0 || end == argv[4] || *end != '\0' || parsed <= 0 || parsed > 1'000'000) {
                std::fprintf(stderr, "tune: invalid iterations '%s' (expected 1..1000000)\n",
                             argv[4]);
                return 1;
            }
            iterations = int(parsed);
        }
        return tune::run_tune(argv[2], mode, iterations);
    }

    uci_loop(std::cin, std::cout);
    return 0;
}
