#include "nnue.h"
#include "bitboard.h"
#include "position.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
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

// v1 loader: accept any file whose first 4 bytes are the Stockfish
// magic. Real weight parsing is deferred — for now we allocate a
// zero-initialized network and mark loaded, so the runtime path is
// exercised (returns 0 for every position, which is honest given the
// stub). Full binary parse comes in the next commit.
bool load_network(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "nnue: cannot open '%s'\n", path.c_str());
        return false;
    }
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!f) {
        std::fprintf(stderr, "nnue: '%s' too short to read magic\n", path.c_str());
        return false;
    }
    // Stockfish 12+ NNUE version magic (little-endian). Anything else
    // is rejected — better a clear "unsupported" than random weights.
    constexpr uint32_t SF_MAGIC = 0x7AF32F16;
    if (magic != SF_MAGIC) {
        std::fprintf(stderr,
            "nnue: '%s' magic 0x%08X != expected 0x%08X — scaffolding "
            "accepts Stockfish HalfKP header only; real weight parse "
            "is a follow-up commit\n",
            path.c_str(), magic, SF_MAGIC);
        return false;
    }
    // Real parse skipped — allocate a zeroed network so the eval
    // path returns a defined value (0 cp for every position) rather
    // than garbage. This makes SPRT vs classical measure "runtime
    // works" not "runtime produces meaningful scores."
    g_network = std::make_unique<Network>();
    g_loaded = true;
    std::fprintf(stderr,
        "nnue: loaded '%s' as ZEROED scaffolding network. Every "
        "position evaluates to 0 cp. Wire up real weight parsing "
        "before shipping.\n", path.c_str());
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
