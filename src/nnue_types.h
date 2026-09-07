#pragma once
#include "types.h"

#include <array>
#include <cstdint>

// Types shared between nnue.h and position.h. Split out so `Position`
// can hold an `Accumulator` by value without pulling in the full NNUE
// API (which itself depends on Position). Keep this header lightweight
// — only the struct definitions and shape constants.

namespace nnue {

constexpr int HIDDEN_SIZE       = 256;
constexpr int PIECES_PER_SIDE   = 5;    // pawn, knight, bishop, rook, queen (king excluded)
constexpr int FEATURES_PER_KING = 641;  // 64 squares × 10 piece slots + 1 padding
constexpr int TOTAL_FEATURES    = 64 * FEATURES_PER_KING;

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
};

}  // namespace nnue
