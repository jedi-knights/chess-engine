#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "attacks.h"
#include "zobrist.h"

// Own main so attack tables + Zobrist keys are initialized exactly once
// before any TEST_CASE runs — avoids scattering static-init hacks across TUs.
int main(int argc, char** argv) {
    init_attacks();
    zobrist::init();
    doctest::Context ctx(argc, argv);
    return ctx.run();
}
