#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Unit tests for scripts/sprt.py — `build_cmd` option threading and
`required_fds` preflight math.

Deliberately assertion-based with no external test framework — matches
the `training/test_*.py` convention so the whole script runs via
`scripts/test_sprt.py` without pulling in pytest.

Run:
    scripts/test_sprt.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_sprt():
    """Import sprt.py as a module without executing its PEP 723 shebang."""
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location("sprt", here / "sprt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sprt = _load_sprt()


def _args(**overrides):
    """Build a Namespace matching sprt.py's argparse output, with overrides."""
    defaults = dict(
        baseline=Path("engine.baseline"),
        tuned=Path("./engine"),
        book=Path("tests/data/openings_demo.pgn"),
        tc="10+0.1",
        elo0=0.0,
        elo1=5.0,
        alpha=0.05,
        beta=0.05,
        concurrency=1,
        max_games=5000,
        pgn_out=None,
        baseline_option=[],
        tuned_option=[],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- build_cmd: option threading ---------------------------------------------


def test_build_cmd_no_options_matches_legacy_shape():
    """Regression: build_cmd with empty option lists produces the pre-change command."""
    cmd = sprt.build_cmd("fastchess", _args())
    # Sanity: no `option.` tokens when nothing is passed.
    assert not any(t.startswith("option.") for t in cmd), (
        f"unexpected option token in default cmd: {cmd}"
    )
    # Both engine blocks still present.
    assert cmd.count("-engine") == 2, f"expected exactly two -engine blocks, got: {cmd}"


def test_build_cmd_appends_single_tuned_option():
    """A single --tuned-option lands as `option.KEY=VALUE` inside the tuned engine block."""
    cmd = sprt.build_cmd("fastchess", _args(tuned_option=["UseNNUE=true"]))
    assert "option.UseNNUE=true" in cmd, f"tuned option missing: {cmd}"
    # Must appear AFTER the tuned block's proto=uci, not the baseline's.
    tuned_idx = cmd.index("name=tuned")
    opt_idx = cmd.index("option.UseNNUE=true")
    assert opt_idx > tuned_idx, f"tuned option ordered before name=tuned: {cmd}"


def test_build_cmd_appends_single_baseline_option():
    """A single --baseline-option lands in the baseline engine block only."""
    cmd = sprt.build_cmd("fastchess", _args(baseline_option=["Hash=32"]))
    assert "option.Hash=32" in cmd, f"baseline option missing: {cmd}"
    baseline_idx = cmd.index("name=baseline")
    tuned_idx = cmd.index("name=tuned")
    opt_idx = cmd.index("option.Hash=32")
    assert baseline_idx < opt_idx < tuned_idx, (
        f"baseline option not between baseline block and tuned block: {cmd}"
    )


def test_build_cmd_supports_multiple_options_per_side():
    """Repeated --tuned-option flags all appear in the tuned block."""
    cmd = sprt.build_cmd(
        "fastchess",
        _args(tuned_option=["UseNNUE=true", "EvalFile=/tmp/net.jnn1"]),
    )
    assert "option.UseNNUE=true" in cmd
    assert "option.EvalFile=/tmp/net.jnn1" in cmd


def test_build_cmd_isolates_options_per_side():
    """Baseline options do NOT leak into the tuned block and vice versa."""
    cmd = sprt.build_cmd(
        "fastchess",
        _args(
            baseline_option=["Hash=16"],
            tuned_option=["UseNNUE=true"],
        ),
    )
    baseline_idx = cmd.index("name=baseline")
    tuned_idx = cmd.index("name=tuned")
    hash_idx = cmd.index("option.Hash=16")
    nnue_idx = cmd.index("option.UseNNUE=true")
    # Baseline option lives between the two `name=` markers.
    assert baseline_idx < hash_idx < tuned_idx
    # Tuned option lives after the second `name=` marker.
    assert nnue_idx > tuned_idx


# --- required_fds: preflight math --------------------------------------------


def test_required_fds_scales_with_concurrency():
    """More concurrent games → more FDs. Monotonically non-decreasing."""
    assert sprt.required_fds(1) < sprt.required_fds(4) < sprt.required_fds(16)


def test_required_fds_covers_observed_fastchess_need():
    """fastchess reported needing 62 FDs at concurrency=4 in a real run;
    our estimate must not underprovision."""
    assert sprt.required_fds(4) >= 62, (
        f"required_fds(4)={sprt.required_fds(4)} underprovisions "
        f"vs fastchess's observed need of 62"
    )


def test_required_fds_leaves_headroom_for_base_process():
    """Even at concurrency=1 the process needs FDs for stdout, stderr,
    the PGN file, logs, etc. — reject a formula that assumes zero base."""
    assert sprt.required_fds(1) >= 16


# ----------------------------------------------------------------------------

TESTS = [
    test_build_cmd_no_options_matches_legacy_shape,
    test_build_cmd_appends_single_tuned_option,
    test_build_cmd_appends_single_baseline_option,
    test_build_cmd_supports_multiple_options_per_side,
    test_build_cmd_isolates_options_per_side,
    test_required_fds_scales_with_concurrency,
    test_required_fds_covers_observed_fastchess_need,
    test_required_fds_leaves_headroom_for_base_process,
]


def main() -> int:
    """Run every test in `TESTS` and print a per-case status."""
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
