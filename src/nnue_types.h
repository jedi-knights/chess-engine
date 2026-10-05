#pragma once
#include "types.h"

#include <array>
#include <cstdint>

// Types shared between nnue.h and position.h. Split out so `Position`
// can hold an `Accumulator` by value without pulling in the full NNUE
// API (which itself depends on Position). Keep this header lightweight
// — only the struct definitions and shape constants.

namespace nnue {

constexpr int HIDDEN_SIZE       = 512;
constexpr int PIECES_PER_SIDE   = 5;    // pawn, knight, bishop, rook, queen (king excluded)
constexpr int FEATURES_PER_KING = 641;  // 64 squares × 10 piece slots + 1 padding
constexpr int TOTAL_FEATURES    = 64 * FEATURES_PER_KING;

// Deferred-update buffer for the accumulator. During a single
// make_move / unmake_move, piece operations (remove captured, remove
// moving-from, put moving-to) each generate a feature-column delta.
// Applying them one at a time means N sequential sweeps over the
// hidden array (each ~2 KiB); batching them lets the flush do a
// single sweep that applies all N columns per hidden chunk, keeping
// the accumulator slice in a register across all deltas and the
// columns in L1 across all chunks.
//
// MAX_DELTAS = 16 covers castling (4), capture + promotion (3), and
// atomic 3x3 explosions (up to 10). An overflow is a bug (we'd
// silently drop updates) — the batch asserts in debug and the
// non-batched caller path flushes before overflowing in release.
struct AccumulatorBatch {
    static constexpr int MAX_DELTAS = 16;
    struct Delta {
        std::int8_t sign;         // +1 add, -1 sub
        std::uint8_t sq;          // Square 0..63
        std::uint8_t pt;          // PieceType (never KING here — kings dirty-flag)
        std::uint8_t pc;          // Color 0..1
    };
    Delta deltas[MAX_DELTAS] = {};
    int   n      = 0;
    bool  active = false;
};

// Per-side accumulator carried by `Position`. Values are the sum of
// the network's feature-weight columns for all currently-active
// features from that side's perspective, plus the feature biases.
//
// `computed[c]` is the dirty flag for perspective `c` — false means
// the values are stale (must be recomputed via refresh_accumulator
// before use). Dirty flags flip false in two situations:
//   1. Position::set_from_fen / clear: no state to update from.
//   2. King moves — since HalfKP features are keyed on the friendly
//      king's square, moving the king invalidates every feature for
//      THAT SIDE'S perspective (the other side is untouched: it
//      references its own king's square).
struct Accumulator {
    std::array<int32_t, HIDDEN_SIZE> values[NUM_COLORS] = {};
    bool                             computed[NUM_COLORS] = {false, false};
    AccumulatorBatch                 batch;
};

}  // namespace nnue
