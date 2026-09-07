#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "numpy>=1.24",
#     "python-chess>=1.9",
# ]
# ///
"""End-to-end tests for the trainer's export + training loop.

Verifies:

1. `export_network` on a zero-weight model produces a file with the
   exact byte size the C++ loader expects and a header that parses
   as `JNN1` / version=1 / hidden=256 / total_features=41024.
2. `export_network` writes int16 weights (round-trips a small
   patterned model through `_quantize` and reads the bytes back).
3. One training step actually updates weights (SGD with lr=1.0 on
   two hand-built examples must produce a nonzero output-weight delta).

Heavier than `test_encoding.py` — pulls in torch — but no C++ engine
run is required. The round-trip against the C++ loader is exercised
by the existing `NNUE load: succeeds on well-formed zeroed network`
doctest in `tests/test_nnue.cpp`, which loads a JNN1 file written
with the same layout constants used here.

Run:
    training/test_export.py
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import collate_batch  # noqa: E402
from encoding import encode_position  # noqa: E402
from export import (  # noqa: E402
    FORMAT_VERSION,
    HIDDEN_SIZE,
    MAGIC,
    SCALE_FEATURES,
    SCALE_OUTPUT,
    TOTAL_FEATURES,
    expected_file_size,
    export_network,
)
from model import NNUE  # noqa: E402


def test_scale_product_matches_cpp_divisor() -> None:
    """`SCALE_FEATURES * SCALE_OUTPUT` must equal the C++ /8192 divisor.

    A mismatch here would silently rescale every eval — a trained
    network would produce plausible but wrong centipawn output.
    """
    # Arrange / Act
    product = SCALE_FEATURES * SCALE_OUTPUT

    # Assert
    assert product == 8192, f"scale product {product} != C++ evaluate() divisor 8192"


def test_export_zeroed_model_has_correct_header_and_size() -> None:
    """A zeroed model must round-trip through `export_network` byte-exactly.

    Verifies the header layout (magic, version, hidden_size,
    total_features) and the total file size against the closed-form
    expected size. If either drifts, the C++ loader would reject
    the file at header parse time.
    """
    # Arrange
    model = NNUE()
    with torch.no_grad():
        model.feature_weights.weight.zero_()
        model.feature_bias.zero_()
        model.output.weight.zero_()
        model.output.bias.zero_()
    tmp = tempfile.NamedTemporaryFile(suffix=".jnn1", delete=False)
    tmp.close()
    path = tmp.name

    try:
        # Act
        export_network(model, path)

        # Assert
        size = os.path.getsize(path)
        assert size == expected_file_size(), (
            f"file size {size} != expected {expected_file_size()}"
        )
        with open(path, "rb") as f:
            magic = f.read(4)
            version, hs, tf = struct.unpack("<III", f.read(12))
        assert magic == MAGIC
        assert version == FORMAT_VERSION
        assert hs == HIDDEN_SIZE
        assert tf == TOTAL_FEATURES
    finally:
        os.unlink(path)


def test_export_preserves_bias_values() -> None:
    """Feature biases must be quantized as `round(value * SCALE_FEATURES)`.

    Constructs a tiny model with a known-nonzero feature bias vector,
    exports, then reads the first HIDDEN_SIZE int16 values past the
    header and confirms they match `round(bias * SCALE_FEATURES)`.
    Catches endianness bugs, layout drift, and missing rounding.
    """
    # Arrange
    model = NNUE()
    with torch.no_grad():
        model.feature_weights.weight.zero_()
        model.output.weight.zero_()
        model.output.bias.zero_()
        # Known values that survive quantization losslessly at scale 128:
        # every 8th slot is set to i/128 so int16 value equals i.
        for i in range(HIDDEN_SIZE):
            model.feature_bias[i] = (i % 32) / SCALE_FEATURES
    tmp = tempfile.NamedTemporaryFile(suffix=".jnn1", delete=False)
    tmp.close()
    path = tmp.name

    try:
        # Act
        export_network(model, path)
        with open(path, "rb") as f:
            f.read(16)  # skip header
            raw = f.read(HIDDEN_SIZE * 2)
        stored = struct.unpack(f"<{HIDDEN_SIZE}h", raw)

        # Assert
        for i in range(HIDDEN_SIZE):
            expected = i % 32  # (i%32)/128 * 128 = i%32
            assert stored[i] == expected, (
                f"bias slot {i}: stored {stored[i]}, expected {expected}"
            )
    finally:
        os.unlink(path)


def test_training_step_updates_output_weights() -> None:
    """One SGD step on hand-built examples must move the output weights.

    Non-training-quality — just proves the loss / backward / step
    pipeline is wired up and the model receives gradient. Uses lr=1.0
    to make any change easy to detect.
    """
    # Arrange
    model = NNUE()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)

    boards = [chess.Board(), chess.Board()]
    boards[1].push_uci("e2e4")
    batch = []
    for board, target in zip(boards, [0.7, 0.3], strict=True):
        stm, opp = encode_position(board)
        batch.append((stm, opp, target, True))  # is_wdl=True
    stm_i, stm_o, opp_i, opp_o, targets, _ = collate_batch(batch)

    before = model.output.weight.detach().clone()

    # Act
    out = model(stm_i, stm_o, opp_i, opp_o)
    loss = torch.nn.functional.mse_loss(torch.sigmoid(out / 400.0), targets)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Assert
    after = model.output.weight.detach()
    assert not torch.allclose(before, after), (
        "output weights unchanged after backprop — gradient flow broken"
    )


TESTS = [
    test_scale_product_matches_cpp_divisor,
    test_export_zeroed_model_has_correct_header_and_size,
    test_export_preserves_bias_values,
    test_training_step_updates_output_weights,
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
