#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "attacks.h"
#include "magic.h"
#include "zobrist.h"

// Own main so attack tables, magic bitboards, and Zobrist keys are
// initialized exactly once before any TEST_CASE runs.
int main(int argc, char** argv) {
    init_attacks();
    init_magic();
    zobrist::init();
    doctest::Context ctx(argc, argv);
    return ctx.run();
}
