#pragma once
#include "position.h"
#include "types.h"

#include <array>
#include <cstdint>
#include <string>

// NNUE (Efficiently Updatable Neural Network) evaluation runtime.
//
// Architecture (v1 scaffolding): HalfKP-style features → 256 hidden
// units → 1 output. Each side gets its own 256-unit accumulator so
// forward pass is:
//
//   raw   = concat(us.acc, them.acc)              // 512 values
//   hidden = clipped_relu(raw)                    // 512 values
//   score  = dot(hidden, w_output) + b_output     // 1 value
//
// The word "efficiently updatable" refers to the accumulator: when a
// piece moves, only the features involving that piece change, so the
// accumulator can be updated by subtracting the departing feature's
// weight column and adding the arriving one — O(features_per_move)
// instead of O(all_features). Incremental update is deferred to a
// follow-up commit; the scaffolding recomputes from scratch.
//
// HalfKP feature layout (Stockfish-compat):
//   feature_index(us_king_sq, piece_sq, piece_type, piece_color) =
//       us_king_sq * 641 + piece_sq * 10 + piece_index(pt, color)
//
// where piece_index maps (pt, color) to 0..9 excluding kings. Kings
// are the "reference" — every feature is relative to the friendly
// king's square — so kings themselves don't appear as features.
// (10 = 5 non-king types × 2 colors.) Total feature space per side:
// 64 king squares × 641 slots = 41,024 features. Only ~30 of those
// are active for any given position.
//
// The runtime lives behind two UCI-facing knobs:
//   UseNNUE (bool):   enable NNUE eval. Falls back to classical when
//                     disabled OR when no network file is loaded.
//   EvalFile (str):   path to a .nnue file. When set to a valid file,
//                     the network is loaded and enables NNUE mode.
//                     Missing / malformed file keeps classical eval.
//
// This first commit ships:
//   - Types (Accumulator, Network, feature-index helpers).
//   - Forward-pass skeleton wired into evaluate().
//   - UCI option plumbing.
//   - Fallback to classical when disabled/unloaded (default).
// It does NOT ship: real network loading, SIMD, incremental updates,
// or any pretense of a trained network. See tests/test_nnue.cpp for
// what's verified.

namespace nnue {

constexpr int HIDDEN_SIZE      = 256;
constexpr int PIECES_PER_SIDE  = 5;   // pawn, knight, bishop, rook, queen (king excluded)
constexpr int FEATURES_PER_KING = 641; // 64 squares × 10 piece slots + 1 padding
constexpr int TOTAL_FEATURES   = 64 * FEATURES_PER_KING;

// Per-side accumulator — one int32 vector per color per position.
// Post-transform 8-bit representation is a later optimization; scaffolding
// uses int32 for numerical clarity.
struct Accumulator {
    std::array<int32_t, HIDDEN_SIZE> values[NUM_COLORS] = {};
    // Marked false at Position construction / set_from_fen; set true
    // when the accumulator has been (re-)computed for the current
    // board state. Incremental update in make/unmake keeps it true;
    // any change that we can't yet handle incrementally falls back to
    // full recompute.
    bool computed = false;
};

// Network parameters. All values are int32 in the scaffolding for
// simplicity; a real production runtime uses int8/int16 with
// quantization scales matching the training pipeline.
struct Network {
    // Feature transformer: one weight column per (feature_index, hidden_unit).
    // Sized as [TOTAL_FEATURES][HIDDEN_SIZE].
    std::array<std::array<int16_t, HIDDEN_SIZE>, TOTAL_FEATURES> feature_weights;
    std::array<int16_t, HIDDEN_SIZE>                             feature_biases;

    // Output layer: dot product of 512 (both sides concatenated) with weights.
    std::array<int16_t, 2 * HIDDEN_SIZE> output_weights;
    int16_t                              output_bias;
};

// True iff a network has been loaded successfully. When false,
// `evaluate()` in eval.cpp falls through to the classical path even
// if `use_nnue` is true — safety net so a mis-typed `EvalFile` UCI
// option doesn't silently break the engine.
bool is_loaded();

// Toggle NNUE mode via UCI. `set_use_nnue(false)` forces classical
// eval regardless of network load state.
void set_use_nnue(bool on);
bool use_nnue();

// Attempt to load a network from disk. Returns true on success.
// Failure leaves the loaded state unchanged (still classical). Format
// is Stockfish-compatible HalfKP header + weights blob — full parse
// is a follow-up; scaffolding accepts a magic-header-only file for
// smoke testing.
bool load_network(const std::string& path);

// Evaluate a position via NNUE. Assumes is_loaded() && use_nnue().
// Callers (eval.cpp) must fall back to the classical path otherwise.
// Returns centipawn score from side-to-move perspective, same
// convention as `evaluate()`.
int evaluate(const Position& pos);

// Compute the accumulator for a position from scratch. Called by the
// forward pass when the position's accumulator is stale (or on any
// path that hasn't wired up incremental updates yet). Public for
// tests that verify determinism.
void refresh_accumulator(const Position& pos, Accumulator& acc);

// Map (king_sq, piece_sq, piece_type, piece_color) → feature index
// for a given "perspective" (which side's king the features are
// relative to). Follows the HalfKP layout above. Kings are excluded
// — passing PieceType::KING is a caller bug and asserts in debug.
int feature_index(Color perspective, Square king_sq, Square piece_sq,
                  PieceType pt, Color piece_color);

}  // namespace nnue
