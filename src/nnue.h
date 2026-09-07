#pragma once
#include "nnue_types.h"
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

// Constants + Accumulator struct live in nnue_types.h — see there for
// their definitions and rationale. Split out because `Position`
// holds an Accumulator by value; that dependency has to break the
// nnue.h ↔ position.h cycle.

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
// Failure leaves the loaded state unchanged (still classical).
//
// File format is our own — NOT Stockfish-compatible. SF nets use a
// different architecture (41024→256×2→32→32→1 in SF12; even bigger
// in newer SF) and different quantization, so cross-loading would
// require an architecture conversion pass that doesn't exist. See
// nnue.cpp for the exact layout; magic bytes are "JNN1" so `file`
// and `hexdump` show something legible.
//
// Real trained weights come from a training pipeline (future PR).
// Until then, use save_network() to dump the current in-memory
// (zero-initialized) network for round-trip testing.
bool load_network(const std::string& path);

// Write the current in-memory network to disk in our binary format.
// Returns true on success. Intended for training-pipeline consumers
// and the round-trip test; also useful for reproducing a specific
// weight state across process boundaries.
bool save_network(const std::string& path);

// Evaluate a position via NNUE. Assumes is_loaded() && use_nnue().
// Callers (eval.cpp) must fall back to the classical path otherwise.
// Returns centipawn score from side-to-move perspective, same
// convention as `evaluate()`.
int evaluate(const Position& pos);

// Rebuild any dirty sides of `pos.acc` from scratch. No-op for sides
// already marked computed. Called lazily by evaluate() before the
// forward pass; also invokable directly from tests. `Position&` is
// const because acc is marked `mutable` — refresh is logically-const
// (populates a cache).
void refresh_accumulator(const Position& pos);

// Force-refresh both sides regardless of dirty flags. Used by tests
// that need a known-clean starting state before comparing incremental
// vs. full-recompute.
void force_refresh_accumulator(const Position& pos);

// Incremental accumulator updates called from Position::put_piece /
// remove_piece when a non-king piece enters / leaves the board.
// Both sides' accumulators are touched (each perspective sees the
// piece); kings are handled via the dirty-flag path instead. No-op
// when the network isn't loaded.
void add_piece_to_accumulator(const Position& pos, Accumulator& acc,
                              Square sq, PieceType pt, Color piece_color);
void sub_piece_from_accumulator(const Position& pos, Accumulator& acc,
                                Square sq, PieceType pt, Color piece_color);

// Map (king_sq, piece_sq, piece_type, piece_color) → feature index
// for a given "perspective" (which side's king the features are
// relative to). Follows the HalfKP layout above. Kings are excluded
// — passing PieceType::KING is a caller bug and asserts in debug.
int feature_index(Color perspective, Square king_sq, Square piece_sq,
                  PieceType pt, Color piece_color);

}  // namespace nnue
