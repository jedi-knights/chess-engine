#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Unit tests for the Stockfish labeler's UCI output parser.

The labeler itself (`stockfish_label.py`) spawns Stockfish via
subprocess and pipes it UCI commands. The *parsing* of Stockfish's
`info ... score cp N` / `score mate K` output is pure, deterministic,
and easy to unit-test — this file pins that behavior.

The subprocess integration is exercised when the labeler is actually
run against the corpus; this file catches regressions in the parser
without needing Stockfish on the test runner.

Run:
    scripts/test_stockfish_labeler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stockfish_label import parse_score_from_info_lines  # noqa: E402


def test_parse_score_from_cp_line() -> None:
    """Standard centipawn score is extracted from the last info line.

    Covers the common case: Stockfish prints several `info` lines at
    increasing depth; we take the deepest one's `score cp N` value.
    """
    # Arrange
    lines = [
        "info depth 1 seldepth 1 score cp 25 nodes 20 time 0 pv e2e4",
        "info depth 8 seldepth 12 score cp 42 nodes 1200 time 15 pv e2e4 e7e5",
    ]

    # Act
    score = parse_score_from_info_lines(lines)

    # Assert
    assert score == 42, f"expected 42, got {score}"


def test_parse_score_negative_cp() -> None:
    """Negative centipawn (side-to-move losing) must parse correctly.

    Stockfish's `score cp` is always from the perspective of the
    side-to-move. A regression that strips the sign would silently
    flip every losing position.
    """
    # Arrange
    lines = ["info depth 8 score cp -175 nodes 500 pv a2a3"]

    # Act
    score = parse_score_from_info_lines(lines)

    # Assert
    assert score == -175, f"expected -175, got {score}"


def test_parse_mate_score_clamps_to_mate_value() -> None:
    """`score mate K` must be clamped to a large sentinel (±10000).

    cp labels in training should not include genuine mate scores
    (`+/- infinity`); clamping to ±10000 cp keeps the sigmoid loss
    well-defined without biasing the training signal.
    """
    # Arrange
    mate_in_3 = ["info depth 8 score mate 3 nodes 200 pv e5e6"]
    mated_in_5 = ["info depth 8 score mate -5 nodes 200 pv e5e6"]

    # Act / Assert
    assert parse_score_from_info_lines(mate_in_3) == 10000
    assert parse_score_from_info_lines(mated_in_5) == -10000


def test_parse_takes_last_depth_line() -> None:
    """Later info lines override earlier ones.

    Catches a regression where the parser returns the first depth's
    score instead of the deepest (and most accurate) one.
    """
    # Arrange
    lines = [
        "info depth 1 score cp 10 pv a",
        "info depth 2 score cp 20 pv b",
        "info depth 8 score cp 50 pv c",
    ]

    # Act
    score = parse_score_from_info_lines(lines)

    # Assert
    assert score == 50, f"expected deepest score 50, got {score}"


def test_parse_returns_none_on_no_score() -> None:
    """Output without any `score` token returns `None`.

    Allows the caller to skip positions where Stockfish failed to
    search (unlikely but possible — malformed FEN, truncated output).
    """
    # Arrange
    lines = ["info string Hash cleared", "readyok"]

    # Act
    score = parse_score_from_info_lines(lines)

    # Assert
    assert score is None, f"expected None, got {score}"


TESTS = [
    test_parse_score_from_cp_line,
    test_parse_score_negative_cp,
    test_parse_mate_score_clamps_to_mate_value,
    test_parse_takes_last_depth_line,
    test_parse_returns_none_on_no_score,
]


def main() -> int:
    """Run every test in `TESTS` and print a per-case status line.

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
