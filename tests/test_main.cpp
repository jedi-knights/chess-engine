#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "attacks.h"

// Own main so attack tables are initialized exactly once before any
// TEST_CASE runs — avoids scattering static-init hacks across TUs.
int main(int argc, char** argv) {
    init_attacks();
    doctest::Context ctx(argc, argv);
    return ctx.run();
}
