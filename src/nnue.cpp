#include "nnue.h"
#include "bitboard.h"
#include "nnue_simd.h"
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

// Finny table (accumulator refresh cache). One entry per
// (perspective, king_square). Each entry caches an accumulator
// value AND the per-(color, piece-type) bitboards that produced it.
// On a dirty-side refresh, we hit the entry for the current king
// square, bitboard-diff against the cached layout, and apply only
// the pieces that changed — typically a handful rather than ~30.
//
// Cold entries (first visit to a king square) do a full populate,
// identical cost to the old refresh_side. The win is in search,
// where king squares get revisited constantly and the warm-entry
// delta is usually 0-4 pieces.
//
// Invariant: when `initialized`, `pieces` is the exact piece layout
// used to compute `values` for the entry's (perspective, king_sq).
// Cache is a pure function of (perspective, king_sq, piece layout,
// network weights) — invalidated only when the network is reloaded.
constexpr int MAX_FINNY_DELTAS = 64;  // 32 pieces × 2 (adds+subs) bound
struct FinnyEntry {
    std::array<int32_t, HIDDEN_SIZE> values{};
    Bitboard pieces[NUM_COLORS][NUM_PIECE_TYPES] = {};
    bool initialized = false;
};
// Heap-allocated so the ~276 KiB table doesn't bloat the BSS of any
// TU that transitively pulls in nnue_types.h. unique_ptr with the
// network itself — paired lifetime, cleared together.
std::unique_ptr<FinnyEntry[]> g_finny;  // sized NUM_COLORS * NUM_SQUARES

FinnyEntry& finny_entry(Color persp, Square king_sq) {
    return g_finny[int(persp) * NUM_SQUARES + int(king_sq)];
}

void clear_finny() {
    if (!g_finny) {
        g_finny = std::make_unique<FinnyEntry[]>(NUM_COLORS * NUM_SQUARES);
        return;
    }
    for (int i = 0; i < NUM_COLORS * NUM_SQUARES; ++i) {
        g_finny[i].initialized = false;
        for (int c = 0; c < NUM_COLORS; ++c) {
            for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
                g_finny[i].pieces[c][pt] = 0;
            }
        }
    }
}

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

namespace {

// Rebuild one side's accumulator through the Finny table.
//
// Cold entry (first time this king square is seen since the network
// was loaded): initialize entry.values from biases, sum every piece's
// feature column, and snapshot the piece bitboards. Cost identical to
// the old full recompute.
//
// Warm entry: bitboard-diff cached vs current per (color, piece-type);
// collect add-columns for new pieces and sub-columns for departed
// pieces; apply them in one fused pass via apply_deltas. In practice
// the delta is usually 0-4 pieces when a king returns to a visited
// square during search, so this is near-free vs the full recompute.
//
// After either path, memcpy the entry's values into acc.values and
// mark the side computed. No-op when the friendly king isn't on the
// board (artificial test positions): accumulator initialized to biases.
void refresh_side(const ::Position& pos, Accumulator& acc, Color persp) {
    const Bitboard king_bb = pos.pieces[persp][KING];
    if (king_bb == 0U) {
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            acc.values[persp][i] = g_network->feature_biases[i];
        }
        acc.computed[persp] = true;
        return;
    }
    const Square king_sq = lsb(king_bb);
    FinnyEntry& entry = finny_entry(persp, king_sq);

    if (!entry.initialized) {
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            entry.values[i] = g_network->feature_biases[i];
        }
        for (int pc = 0; pc < NUM_COLORS; ++pc) {
            for (int pt = PAWN; pt <= QUEEN; ++pt) {
                entry.pieces[pc][pt] = pos.pieces[pc][pt];
                Bitboard b = pos.pieces[pc][pt];
                while (b != 0U) {
                    const Square s = pop_lsb(b);
                    const int idx = feature_index(
                        persp, king_sq, s, PieceType(pt), Color(pc));
                    simd::add_column(entry.values.data(),
                                     g_network->feature_weights[idx].data());
                }
            }
        }
        entry.initialized = true;
    } else {
        const int16_t* adds[MAX_FINNY_DELTAS];
        const int16_t* subs[MAX_FINNY_DELTAS];
        int n_add = 0;
        int n_sub = 0;
        for (int pc = 0; pc < NUM_COLORS; ++pc) {
            for (int pt = PAWN; pt <= QUEEN; ++pt) {
                const Bitboard cur     = pos.pieces[pc][pt];
                const Bitboard cached  = entry.pieces[pc][pt];
                Bitboard       added   = cur    & ~cached;
                Bitboard       removed = cached & ~cur;
                while (added != 0U) {
                    const Square s = pop_lsb(added);
                    const int idx = feature_index(
                        persp, king_sq, s, PieceType(pt), Color(pc));
                    assert(n_add < MAX_FINNY_DELTAS);
                    adds[n_add++] = g_network->feature_weights[idx].data();
                }
                while (removed != 0U) {
                    const Square s = pop_lsb(removed);
                    const int idx = feature_index(
                        persp, king_sq, s, PieceType(pt), Color(pc));
                    assert(n_sub < MAX_FINNY_DELTAS);
                    subs[n_sub++] = g_network->feature_weights[idx].data();
                }
                entry.pieces[pc][pt] = cur;
            }
        }
        if (n_add + n_sub > 0) {
            simd::apply_deltas(entry.values.data(), adds, n_add, subs, n_sub);
        }
    }

    std::memcpy(acc.values[persp].data(), entry.values.data(),
                sizeof(entry.values));
    acc.computed[persp] = true;
}

}  // namespace

void refresh_accumulator(const ::Position& pos) {
    if (!g_loaded) {
        // Without a loaded network the accumulator is meaningless
        // anyway; callers under use_nnue() gate first, so this only
        // fires in tests. Mark both sides "computed" (== all-zero
        // biases) so the check-and-refresh logic doesn't loop.
        pos.acc.computed[WHITE] = true;
        pos.acc.computed[BLACK] = true;
        return;
    }
    if (!pos.acc.computed[WHITE]) { refresh_side(pos, pos.acc, WHITE); }
    if (!pos.acc.computed[BLACK]) { refresh_side(pos, pos.acc, BLACK); }
}

void force_refresh_accumulator(const ::Position& pos) {
    if (!g_loaded) { return; }
    refresh_side(pos, pos.acc, WHITE);
    refresh_side(pos, pos.acc, BLACK);
}

// Add / subtract a piece's feature column from both perspectives'
// accumulators. Called from Position::put_piece / remove_piece
// exclusively — bypasses the dirty check because the caller has
// already handled the king case. A dirty perspective is skipped
// because refresh_side will overwrite the entire array on next
// evaluate() — spending SIMD cycles on a value about to be
// discarded is pure waste in the search hot path.
void add_piece_to_accumulator(const ::Position& pos, Accumulator& acc,
                              Square sq, PieceType pt, Color piece_color) {
    if (!g_loaded) { return; }
    if (acc.batch.active) {
        assert(acc.batch.n < AccumulatorBatch::MAX_DELTAS);
        acc.batch.deltas[acc.batch.n++] = {
            +1, std::uint8_t(sq), std::uint8_t(pt), std::uint8_t(piece_color)};
        return;
    }
    for (int c = 0; c < NUM_COLORS; ++c) {
        if (!acc.computed[c]) { continue; }
        const Color persp   = Color(c);
        const Bitboard kbb  = pos.pieces[persp][KING];
        if (kbb == 0U) { continue; }
        const Square king_sq = lsb(kbb);
        const int    idx     = feature_index(persp, king_sq, sq, pt, piece_color);
        simd::add_column(acc.values[c].data(),
                         g_network->feature_weights[idx].data());
    }
}

void sub_piece_from_accumulator(const ::Position& pos, Accumulator& acc,
                                Square sq, PieceType pt, Color piece_color) {
    if (!g_loaded) { return; }
    if (acc.batch.active) {
        assert(acc.batch.n < AccumulatorBatch::MAX_DELTAS);
        acc.batch.deltas[acc.batch.n++] = {
            -1, std::uint8_t(sq), std::uint8_t(pt), std::uint8_t(piece_color)};
        return;
    }
    for (int c = 0; c < NUM_COLORS; ++c) {
        if (!acc.computed[c]) { continue; }
        const Color persp   = Color(c);
        const Bitboard kbb  = pos.pieces[persp][KING];
        if (kbb == 0U) { continue; }
        const Square king_sq = lsb(kbb);
        const int    idx     = feature_index(persp, king_sq, sq, pt, piece_color);
        simd::sub_column(acc.values[c].data(),
                         g_network->feature_weights[idx].data());
    }
}

void begin_batch(Accumulator& acc) {
    assert(!acc.batch.active);
    acc.batch.n      = 0;
    acc.batch.active = true;
}

void end_batch(const ::Position& pos, Accumulator& acc) {
    assert(acc.batch.active);
    acc.batch.active = false;
    if (!g_loaded || acc.batch.n == 0) {
        acc.batch.n = 0;
        return;
    }
    const int16_t* adds[AccumulatorBatch::MAX_DELTAS];
    const int16_t* subs[AccumulatorBatch::MAX_DELTAS];
    for (int c = 0; c < NUM_COLORS; ++c) {
        if (!acc.computed[c]) { continue; }
        const Color    persp   = Color(c);
        const Bitboard kbb     = pos.pieces[persp][KING];
        if (kbb == 0U) { continue; }
        const Square king_sq = lsb(kbb);
        int n_add = 0;
        int n_sub = 0;
        for (int i = 0; i < acc.batch.n; ++i) {
            const auto& d = acc.batch.deltas[i];
            const int idx = feature_index(persp, king_sq,
                                          Square(d.sq), PieceType(d.pt),
                                          Color(d.pc));
            const int16_t* col = g_network->feature_weights[idx].data();
            if (d.sign > 0) { adds[n_add++] = col; }
            else            { subs[n_sub++] = col; }
        }
        simd::apply_deltas(acc.values[c].data(), adds, n_add, subs, n_sub);
    }
    acc.batch.n = 0;
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
    // Finny cache is a function of the loaded weights; a new load
    // means every entry is stale. clear_finny lazily allocates the
    // 276 KiB table on first load so engines that never enable NNUE
    // don't pay the memory cost.
    clear_finny();
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
    // Lazily refresh any dirty sides of the per-Position accumulator.
    // Non-king piece movement already updated `pos.acc` incrementally
    // via put_piece / remove_piece; king moves and set_from_fen mark
    // the affected side(s) dirty so this call rebuilds them from
    // scratch. `pos.acc` is `mutable`, so this stays const from the
    // caller's perspective (logically-const cache refresh).
    refresh_accumulator(pos);

    // Clipped ReLU (clamp to [0, 127]) + int16 dot product per side,
    // dispatched to the SIMD kernel. Concatenation order is
    // [side_to_move, opponent] so the network sees "me first."
    // int64 sum guards against theoretical overflow when a real
    // trained network fills 512 hidden units — max |product| =
    // 127 * 32767 = 4.16M, worst-case sum ≈ 2.13B, which is inside
    // int32 but uncomfortably close.
    const int stm = pos.side_to_move;
    const int opp = stm ^ 1;
    int64_t sum = int64_t(g_network->output_bias);
    sum += simd::forward_side(pos.acc.values[stm].data(),
                              g_network->output_weights.data());
    sum += simd::forward_side(pos.acc.values[opp].data(),
                              g_network->output_weights.data() + HIDDEN_SIZE);
    // Dequantize: training-side scales SCALE_FEATURES=128 (float
    // feature weight → int16) and SCALE_OUTPUT=64 (float output
    // weight → int16); their product is the integer-to-centipawn
    // divisor. See training/README.md's "Quantization" section —
    // these three constants must stay in sync. Changing any of them
    // means retraining or an on-disk format bump.
    return int(sum / 8192);
}

}  // namespace nnue
