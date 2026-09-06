#include "nnue.h"
#include "bitboard.h"
#include "position.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>

namespace nnue {

namespace {

// Module-scope state. Simple globals — the runtime is single-threaded
// like the rest of the engine, and NNUE state is legitimately global
// (one loaded network per process).
bool                     g_loaded = false;
bool                     g_use_nnue = false;
std::unique_ptr<Network> g_network;

// Piece → 0..9 slot per HalfKP:
//   { own pawn, own knight, own bishop, own rook, own queen,
//     enemy pawn, enemy knight, enemy bishop, enemy rook, enemy queen }
// Indexed by (perspective_color, piece_color, piece_type). King
// intentionally excluded — the perspective IS the king reference.
int piece_slot(Color perspective, Color piece_color, PieceType pt) {
    assert(pt != KING && pt != NO_PIECE_TYPE);
    int base = (perspective == piece_color) ? 0 : PIECES_PER_SIDE;
    return base + (pt - PAWN);  // PAWN=1 in enum → slot 0
}

// Black perspective needs vertical mirror of square indices so both
// kings see "in front of me = larger rank." Standard NNUE convention.
Square oriented(Color perspective, Square sq) {
    return (perspective == WHITE) ? sq : Square(int(sq) ^ 56);
}

}  // namespace

int feature_index(Color perspective, Square king_sq, Square piece_sq,
                  PieceType pt, Color piece_color) {
    assert(pt != KING && pt != NO_PIECE_TYPE);
    const Square kk = oriented(perspective, king_sq);
    const Square ps = oriented(perspective, piece_sq);
    const int slot = piece_slot(perspective, piece_color, pt);
    return int(kk) * FEATURES_PER_KING + int(ps) * (2 * PIECES_PER_SIDE) + slot;
}

void refresh_accumulator(const ::Position& pos, Accumulator& acc) {
    if (!g_loaded) {
        // Without a loaded network, refresh is a no-op that leaves
        // the accumulator in its default (all-zero) state. Callers
        // that reach this path with `use_nnue()` enabled shouldn't —
        // eval.cpp checks is_loaded() before calling.
        acc.computed = true;
        return;
    }
    for (int c = 0; c < NUM_COLORS; ++c) {
        const Color persp = Color(c);
        // Widen int16 biases → int32 accumulator. Element-wise;
        // std::array of different value_type doesn't cross-assign.
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            acc.values[c][i] = g_network->feature_biases[i];
        }
        const Bitboard king_bb = pos.pieces[persp][KING];
        if (king_bb == 0U) {
            continue;  // artificial no-king position; leave biases only
        }
        const Square king_sq = lsb(king_bb);
        for (int pc = 0; pc < NUM_COLORS; ++pc) {
            for (int pt = PAWN; pt <= QUEEN; ++pt) {
                Bitboard b = pos.pieces[pc][pt];
                while (b != 0U) {
                    const Square s = pop_lsb(b);
                    const int idx = feature_index(
                        persp, king_sq, s, PieceType(pt), Color(pc));
                    const auto& col = g_network->feature_weights[idx];
                    for (int i = 0; i < HIDDEN_SIZE; ++i) {
                        acc.values[c][i] += col[i];
                    }
                }
            }
        }
    }
    acc.computed = true;
}

bool is_loaded() { return g_loaded; }
bool use_nnue()  { return g_use_nnue && g_loaded; }
void set_use_nnue(bool on) { g_use_nnue = on; }

// Custom binary format for this engine's HalfKP-256-1 architecture.
// SF-compatible parsing was rejected because (a) SF uses a different
// architecture (41024→256×2→32→32→1 in SF12; even bigger in modern
// SF), (b) SF quantization scales don't apply to our int32-accumulator
// runtime, (c) even loading SF12 weights, our missing hidden layers
// would produce nonsense output. When the training pipeline lands
// (future PR), it emits this format.
//
// Layout (all little-endian):
//   0x00: 4 bytes  ASCII "JNN1" — magic, so `file` / `hexdump` are legible
//   0x04: 4 bytes  uint32 format_version (currently 1)
//   0x08: 4 bytes  uint32 hidden_size — must equal HIDDEN_SIZE
//   0x0C: 4 bytes  uint32 total_features — must equal TOTAL_FEATURES
//   0x10: int16 feature_biases[HIDDEN_SIZE]
//   ....: int16 feature_weights[TOTAL_FEATURES][HIDDEN_SIZE]
//   ....: int16 output_weights[2 * HIDDEN_SIZE]
//   ....: int16 output_bias
//
// Total on disk: 16 + 2*HIDDEN_SIZE + 2*TOTAL_FEATURES*HIDDEN_SIZE
//              + 2*2*HIDDEN_SIZE + 2
//              = 16 + 512 + 21,004,288 + 1024 + 2 ≈ 20 MiB.
//
// All header fields are validated. Architecture-mismatch is rejected
// rather than silently reinterpreted.
constexpr char     FILE_MAGIC[4]      = {'J', 'N', 'N', '1'};
constexpr uint32_t FORMAT_VERSION     = 1;

bool load_network(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "nnue: cannot open '%s'\n", path.c_str());
        return false;
    }
    // Header.
    char magic[4] = {};
    uint32_t version = 0, hs = 0, tf = 0;
    f.read(magic, sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    f.read(reinterpret_cast<char*>(&hs), sizeof(hs));
    f.read(reinterpret_cast<char*>(&tf), sizeof(tf));
    if (!f) {
        std::fprintf(stderr, "nnue: '%s' truncated header\n", path.c_str());
        return false;
    }
    if (std::memcmp(magic, FILE_MAGIC, 4) != 0) {
        std::fprintf(stderr,
            "nnue: '%s' magic \"%c%c%c%c\" != expected \"JNN1\"\n",
            path.c_str(), magic[0], magic[1], magic[2], magic[3]);
        return false;
    }
    if (version != FORMAT_VERSION) {
        std::fprintf(stderr,
            "nnue: '%s' format version %u != expected %u\n",
            path.c_str(), version, FORMAT_VERSION);
        return false;
    }
    if (hs != HIDDEN_SIZE || tf != TOTAL_FEATURES) {
        std::fprintf(stderr,
            "nnue: '%s' architecture mismatch (file: hidden=%u feats=%u; "
            "engine: hidden=%d feats=%d)\n",
            path.c_str(), hs, tf, HIDDEN_SIZE, TOTAL_FEATURES);
        return false;
    }
    // Weights. Reading directly into the arrays leverages the struct's
    // contiguous layout — std::array of trivially-copyable T is
    // guaranteed to be contiguous in memory.
    auto net = std::make_unique<Network>();
    f.read(reinterpret_cast<char*>(net->feature_biases.data()),
           sizeof(net->feature_biases));
    f.read(reinterpret_cast<char*>(net->feature_weights.data()),
           sizeof(net->feature_weights));
    f.read(reinterpret_cast<char*>(net->output_weights.data()),
           sizeof(net->output_weights));
    f.read(reinterpret_cast<char*>(&net->output_bias),
           sizeof(net->output_bias));
    if (!f) {
        std::fprintf(stderr,
            "nnue: '%s' truncated — expected ~%zu bytes past header\n",
            path.c_str(),
            sizeof(net->feature_biases) + sizeof(net->feature_weights)
              + sizeof(net->output_weights) + sizeof(net->output_bias));
        return false;
    }
    // Extra trailing bytes are a mismatch — the format is fixed-size,
    // any tail is either a different architecture or corruption.
    f.peek();
    if (!f.eof()) {
        std::fprintf(stderr,
            "nnue: '%s' has trailing bytes past network end — refusing\n",
            path.c_str());
        return false;
    }

    g_network = std::move(net);
    g_loaded = true;
    std::fprintf(stderr, "nnue: loaded '%s' (%d hidden units, %d features)\n",
                 path.c_str(), HIDDEN_SIZE, TOTAL_FEATURES);
    return true;
}

bool save_network(const std::string& path) {
    if (!g_loaded) {
        std::fprintf(stderr,
            "nnue: save_network called with no network loaded\n");
        return false;
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "nnue: cannot open '%s' for write\n",
                     path.c_str());
        return false;
    }
    constexpr uint32_t hs = HIDDEN_SIZE;
    constexpr uint32_t tf = TOTAL_FEATURES;
    f.write(FILE_MAGIC, sizeof(FILE_MAGIC));
    f.write(reinterpret_cast<const char*>(&FORMAT_VERSION),
            sizeof(FORMAT_VERSION));
    f.write(reinterpret_cast<const char*>(&hs), sizeof(hs));
    f.write(reinterpret_cast<const char*>(&tf), sizeof(tf));
    f.write(reinterpret_cast<const char*>(g_network->feature_biases.data()),
            sizeof(g_network->feature_biases));
    f.write(reinterpret_cast<const char*>(g_network->feature_weights.data()),
            sizeof(g_network->feature_weights));
    f.write(reinterpret_cast<const char*>(g_network->output_weights.data()),
            sizeof(g_network->output_weights));
    f.write(reinterpret_cast<const char*>(&g_network->output_bias),
            sizeof(g_network->output_bias));
    if (!f) {
        std::fprintf(stderr, "nnue: write to '%s' failed\n", path.c_str());
        return false;
    }
    return true;
}

int evaluate(const ::Position& pos) {
    assert(is_loaded());
    // Full-recompute forward pass. Incremental accumulator updates
    // in make/unmake come next; for now we pay the ~30 popcount
    // cost per eval. The Accumulator struct isn't yet a Position
    // member — scaffolding computes into a stack local.
    Accumulator acc;
    refresh_accumulator(pos, acc);

    // Clipped ReLU (Stockfish-compat activation): clamp to [0, 127].
    // Then dot with the output weight vector. Concatenation order is
    // [side_to_move, opponent] so the network sees "me first."
    const int stm = pos.side_to_move;
    const int opp = stm ^ 1;
    int64_t sum = int64_t(g_network->output_bias);
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        int32_t v = acc.values[stm][i];
        if (v < 0)   { v = 0; }
        if (v > 127) { v = 127; }
        sum += int64_t(g_network->output_weights[i]) * v;
    }
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        int32_t v = acc.values[opp][i];
        if (v < 0)   { v = 0; }
        if (v > 127) { v = 127; }
        sum += int64_t(g_network->output_weights[HIDDEN_SIZE + i]) * v;
    }
    // Standard NNUE output scale: divide by 16 * 512 = 8192 to bring
    // integer accumulator back to centipawn range. Zeroed network
    // means sum = bias = 0, so scale factor is a placeholder.
    return int(sum / 8192);
}

}  // namespace nnue
