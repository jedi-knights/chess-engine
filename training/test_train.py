#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "numpy>=1.24",
#     "python-chess>=1.9",
# ]
# ///
"""Unit tests for the training loop's optimizer and scheduler wiring.

These tests exercise the pure-function helpers in `train.py` through
the public construction surface — no actual training runs, no data
files, no multi-epoch loops. The purpose is to pin behavior that is
otherwise invisible until a full training run has executed:

1. Optimizer is `AdamW` with the configured `weight_decay`
2. Scheduler is `CosineAnnealingLR` after a linear-warmup phase
3. Gradient clipping actually clips when the norm exceeds the cap

Heavier end-to-end training smoke lives in `test_export.py` (one SGD
step on hand-built examples). This file is deliberately narrower.

Run:
    training/test_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import NNUE  # noqa: E402
from train import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    clip_gradients,
)


def test_build_optimizer_returns_adamw_with_weight_decay() -> None:
    """The optimizer must be `AdamW` with the configured `weight_decay`.

    Catches a regression to plain `Adam` (no decoupled weight decay) —
    a silent change that would show up only as slower Elo gain in a
    full training run, i.e. not for days.
    """
    # Arrange
    model = NNUE()

    # Act
    opt = build_optimizer(model, lr=1e-3, weight_decay=5e-4)

    # Assert
    assert isinstance(opt, torch.optim.AdamW), (
        f"expected AdamW, got {type(opt).__name__}"
    )
    assert opt.param_groups[0]["weight_decay"] == 5e-4


def test_build_scheduler_warms_up_then_decays() -> None:
    """Scheduler must linearly warm up for `warmup_frac` of total steps.

    Verifies the LR trajectory crosses the peak LR at the end of the
    warmup phase and decays below it afterward. If the scheduler is
    swapped for a flat LR or the warmup is wired to the wrong step
    count, this test catches it.
    """
    # Arrange
    model = NNUE()
    opt = build_optimizer(model, lr=1.0, weight_decay=0.0)
    total_steps = 100
    warmup_frac = 0.1
    sched = build_scheduler(opt, total_steps=total_steps, warmup_frac=warmup_frac)
    warmup_steps = int(total_steps * warmup_frac)

    # Act
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    # Assert
    assert lrs[0] < 0.5, f"warmup start LR {lrs[0]} should be well below peak 1.0"
    assert abs(lrs[warmup_steps] - 1.0) < 1e-6, (
        f"LR at end of warmup should be peak 1.0, got {lrs[warmup_steps]}"
    )
    assert lrs[-1] < 0.1, f"final LR {lrs[-1]} should have decayed well below peak 1.0"


def test_clip_gradients_caps_large_norm() -> None:
    """`clip_gradients` must scale gradients so their L2 norm ≤ `max_norm`.

    Constructs a model, injects an artificially large gradient on one
    parameter, applies the clip, and verifies the resulting L2 norm
    across all parameters is at or below the cap.
    """
    # Arrange
    model = NNUE()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    # Inject a large gradient — total norm will be far above 1.0.
    model.output.weight.grad.fill_(10.0)

    # Act
    clip_gradients(model, max_norm=1.0)

    # Assert
    total_sq = sum((p.grad.detach() ** 2).sum().item() for p in model.parameters())
    total_norm = total_sq**0.5
    assert total_norm <= 1.0 + 1e-5, f"clipped grad norm {total_norm} exceeds cap 1.0"


def test_clip_gradients_zero_disables_clip() -> None:
    """Passing `max_norm=0` must leave gradients untouched.

    Allows users to disable clipping via `--grad-clip 0` without having
    to add branching at every call site.
    """
    # Arrange
    model = NNUE()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    model.output.weight.grad.fill_(10.0)
    before = model.output.weight.grad.detach().clone()

    # Act
    clip_gradients(model, max_norm=0.0)

    # Assert
    after = model.output.weight.grad.detach()
    assert torch.equal(before, after), (
        "max_norm=0 should disable clipping, but grads were modified"
    )


TESTS = [
    test_build_optimizer_returns_adamw_with_weight_decay,
    test_build_scheduler_warms_up_then_decays,
    test_clip_gradients_caps_large_norm,
    test_clip_gradients_zero_disables_clip,
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
