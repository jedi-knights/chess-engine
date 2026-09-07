#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-chess>=1.9",
# ]
# ///
"""HalfKP feature-encoding tests for the training pipeline.

Deliberately depends only on python-chess (no torch, no numpy) so
the Python↔C++ feature-index agreement — the pipeline's foundational
correctness invariant — can be verified without downloading ~800 MB
of PyTorch.

Run:
    training/test_encoding.py

Pinned values match `tests/test_nnue.cpp` (`NNUE feature_index
follows HalfKP layout`). Divergence here means the Python trainer
would write columns the C++ runtime looks up at different indices —
a network trained with this pipeline would produce noise at inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent))
from encoding import TOTAL_FEATURES, encode_position, feature_index  # noqa: E402


def test_feature_index_matches_cpp() -> None:
    """Encoding must match `nnue::feature_index` in the C++ runtime.

    Values pinned from `tests/test_nnue.cpp` (`NNUE feature_index
    follows HalfKP layout`). A trained network is unusable if Python
    writes column `i` for a feature the runtime reads at column `j`.
    """
    assert (
        feature_index(chess.WHITE, chess.E1, chess.E2, chess.PAWN, chess.WHITE) == 2684
    )
    assert (
        feature_index(chess.BLACK, chess.E1, chess.E2, chess.PAWN, chess.WHITE) == 38985
    )
    assert (
        feature_index(chess.WHITE, chess.E1, chess.B1, chess.KNIGHT, chess.WHITE)
        == 2575
    )


def test_encode_startpos_has_30_features() -> None:
    """Starting position: 32 pieces − 2 kings = 30 features per perspective.

    Each perspective encodes ALL non-king pieces (own + enemy), so
    both `stm` and `opp` lists contain 30 unique indices. Every index
    must fall inside `[0, TOTAL_FEATURES)`.
    """
    board = chess.Board()
    stm, opp = encode_position(board)
    assert len(stm) == 30, (
        f"expected 30 stm features (32 pieces - 2 kings), got {len(stm)}"
    )
    assert len(opp) == 30, f"expected 30 opp features, got {len(opp)}"
    assert len(set(stm)) == 30, "stm features must be unique"
    assert len(set(opp)) == 30, "opp features must be unique"
    for idx in stm + opp:
        assert 0 <= idx < TOTAL_FEATURES, f"feature {idx} out of range"


def test_encode_symmetry_after_null_move() -> None:
    """Rotating side-to-move must swap the (stm, opp) accumulators.

    In a symmetric starting position, whichever side is to move sees
    the same feature set, just with own↔enemy piece slots flipped.
    Concretely: encoding startpos-as-WHITE gives (S, O); encoding
    startpos-as-BLACK gives (O', S') where O' equals what WHITE saw
    as opp, and S' equals what WHITE saw as stm.
    """
    white_to_move = chess.Board()
    black_to_move = chess.Board()
    black_to_move.turn = chess.BLACK

    w_stm, w_opp = encode_position(white_to_move)
    b_stm, b_opp = encode_position(black_to_move)

    assert sorted(w_stm) == sorted(b_opp), "WHITE's stm view == BLACK's opp view"
    assert sorted(w_opp) == sorted(b_stm), "WHITE's opp view == BLACK's stm view"


def test_encode_excludes_kings() -> None:
    """Only-kings position must produce empty feature lists on both sides."""
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    stm, opp = encode_position(board)
    assert stm == []
    assert opp == []


TESTS = [
    test_feature_index_matches_cpp,
    test_encode_startpos_has_30_features,
    test_encode_symmetry_after_null_move,
    test_encode_excludes_kings,
]


def main() -> int:
    """Run every test in `TESTS` and print a per-case status.

    Returns:
        0 if every test passed, 1 otherwise.
    """
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL ] {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 — top-level test harness reports all errors
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
        else:
            print(f"[OK   ] {t.__name__}")
    if failed:
        print(f"\n{failed} test(s) failed")
        return 1
    print(f"\nall {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
