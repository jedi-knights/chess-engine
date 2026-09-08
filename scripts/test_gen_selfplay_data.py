#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chess>=1.10",
# ]
# ///
"""Unit tests for scripts/gen_selfplay_data.py — currently pins the
`format_score_label` helper used by the score-labeled data mode.

Deliberately assertion-based with no external test framework — matches
the `training/test_*.py` and `scripts/test_sprt.py` convention so the
whole script runs via `scripts/test_gen_selfplay_data.py` without
pulling in pytest.

Run:
    scripts/test_gen_selfplay_data.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import chess.engine


def _load_gen():
    """Import gen_selfplay_data.py as a module."""
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "gen_selfplay_data", here / "gen_selfplay_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_gen()


def _cp_score(cp: int) -> chess.engine.PovScore:
    """Build a PovScore representing `cp` centipawns from STM POV."""
    return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)


def _mate_score(distance: int) -> chess.engine.PovScore:
    """Build a PovScore representing mate in `distance` plies from STM POV."""
    return chess.engine.PovScore(chess.engine.Mate(distance), chess.WHITE)


def test_format_score_label_returns_positive_cp_verbatim():
    assert gen.format_score_label(_cp_score(45)) == "45"


def test_format_score_label_returns_negative_cp_with_sign():
    assert gen.format_score_label(_cp_score(-127)) == "-127"


def test_format_score_label_returns_zero_for_dead_equal():
    assert gen.format_score_label(_cp_score(0)) == "0"


def test_format_score_label_clamps_positive_mate_to_sentinel():
    """A mate-in-N-for-us score maps to +MATE_SCORE_CP (bounded label
    for the trainer — raw mate distance is not a comparable cp value)."""
    label = gen.format_score_label(_mate_score(3))
    assert label == str(gen.MATE_SCORE_CP), (
        f"expected +{gen.MATE_SCORE_CP} for mate-in-3-for-us, got {label!r}"
    )


def test_format_score_label_clamps_negative_mate_to_sentinel():
    label = gen.format_score_label(_mate_score(-5))
    assert label == str(-gen.MATE_SCORE_CP), (
        f"expected -{gen.MATE_SCORE_CP} for mate-in-5-against-us, got {label!r}"
    )


def test_mate_score_cp_is_bounded_and_positive():
    """Sanity: MATE_SCORE_CP is a positive int large enough to be
    unambiguous (> any plausible material eval) but not so large it
    breaks downstream sigmoid math."""
    assert isinstance(gen.MATE_SCORE_CP, int)
    assert 1000 < gen.MATE_SCORE_CP < 100_000


def test_format_score_label_clamps_engine_cp_mate_signal_positive():
    """Regression: this engine emits `score cp <mate_score - ply>`
    instead of the UCI-standard `score mate <plies>` (src/uci.cpp:250).
    A large positive Cp value arriving from the engine's mate range
    must clamp to +MATE_SCORE_CP, not pass through as +99998."""
    label = gen.format_score_label(_cp_score(99998))
    assert label == str(gen.MATE_SCORE_CP), (
        f"large positive Cp (engine mate signal) not clamped: got {label!r}"
    )


def test_format_score_label_clamps_engine_cp_mate_signal_negative():
    label = gen.format_score_label(_cp_score(-99998))
    assert label == str(-gen.MATE_SCORE_CP), (
        f"large negative Cp (engine mate signal) not clamped: got {label!r}"
    )


def test_format_score_label_preserves_large_but_reasonable_cp():
    """A cp value below the engine's mate range (~99000) is a real
    eval and must pass through — do not over-clamp."""
    assert gen.format_score_label(_cp_score(2500)) == "2500"
    assert gen.format_score_label(_cp_score(-2500)) == "-2500"


# --- parse_engine_options: --engine-option KEY=VALUE handling ---------------


def test_parse_engine_options_empty_list_returns_empty_dict():
    assert gen.parse_engine_options([]) == {}


def test_parse_engine_options_single_pair():
    assert gen.parse_engine_options(["UseNNUE=true"]) == {"UseNNUE": "true"}


def test_parse_engine_options_multiple_pairs():
    got = gen.parse_engine_options(["UseNNUE=true", "EvalFile=/tmp/net.jnn1"])
    assert got == {"UseNNUE": "true", "EvalFile": "/tmp/net.jnn1"}


def test_parse_engine_options_preserves_equals_in_value():
    """A value containing `=` (like a URL with query params or an EPD
    line with `c9 "..."` operations) must keep everything after the
    first `=` intact."""
    got = gen.parse_engine_options(["EvalFile=/path/with=equals/net.jnn1"])
    assert got == {"EvalFile": "/path/with=equals/net.jnn1"}


def test_parse_engine_options_rejects_missing_equals():
    """A token with no `=` is a user error — surface it, don't silently
    ignore. Silent-ignore is the failure mode where the user thinks
    NNUE is loaded but it never was."""
    try:
        gen.parse_engine_options(["justakey"])
    except ValueError as e:
        assert "justakey" in str(e), f"error should name the bad token: {e}"
    else:
        raise AssertionError("expected ValueError for token without `=`")


# ----------------------------------------------------------------------------

TESTS = [
    test_format_score_label_returns_positive_cp_verbatim,
    test_format_score_label_returns_negative_cp_with_sign,
    test_format_score_label_returns_zero_for_dead_equal,
    test_format_score_label_clamps_positive_mate_to_sentinel,
    test_format_score_label_clamps_negative_mate_to_sentinel,
    test_mate_score_cp_is_bounded_and_positive,
    test_format_score_label_clamps_engine_cp_mate_signal_positive,
    test_format_score_label_clamps_engine_cp_mate_signal_negative,
    test_format_score_label_preserves_large_but_reasonable_cp,
    test_parse_engine_options_empty_list_returns_empty_dict,
    test_parse_engine_options_single_pair,
    test_parse_engine_options_multiple_pairs,
    test_parse_engine_options_preserves_equals_in_value,
    test_parse_engine_options_rejects_missing_equals,
]


def main() -> int:
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
