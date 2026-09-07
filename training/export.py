"""Quantize float NNUE weights → int16 and write JNN1 binary format.

Layout matches `src/nnue.cpp` (`nnue::load_network`):

    0x00: 4 bytes  ASCII "JNN1"
    0x04: uint32   format_version (=1)
    0x08: uint32   hidden_size          (must equal HIDDEN_SIZE = 256)
    0x0C: uint32   total_features       (must equal TOTAL_FEATURES = 41024)
    0x10+: int16   feature_biases[HIDDEN_SIZE]
           int16   feature_weights[TOTAL_FEATURES][HIDDEN_SIZE]
           int16   output_weights[2 * HIDDEN_SIZE]
           int16   output_bias

Quantization scales are chosen so `SCALE_FEATURES * SCALE_OUTPUT`
matches the `/ 8192` divisor in `nnue::evaluate`. A different pair
would silently rescale every eval — the runtime would still parse
the file, but returned centipawns would be off by a constant factor.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

MAGIC = b"JNN1"
FORMAT_VERSION = 1
HIDDEN_SIZE = 256
TOTAL_FEATURES = 41024

# Quantization scales. Product must equal 8192 to match the C++
# `evaluate()` output divisor.
SCALE_FEATURES = 128
SCALE_OUTPUT = 64
assert SCALE_FEATURES * SCALE_OUTPUT == 8192, "must match nnue.cpp divisor"

_INT16_MIN = -32768
_INT16_MAX = 32767


def _quantize(arr: np.ndarray, scale: int, name: str) -> np.ndarray:
    """Convert a float array to int16 with saturating clip.

    Prints a warning to stderr if any value saturates — a saturating
    weight is a bug in training (weights should have been kept small
    by initialization + regularization).

    Args:
        arr: Float weights.
        scale: Multiplier applied before rounding.
        name: Human-readable tensor name for the saturation warning.

    Returns:
        `int16` array with the same shape as `arr`.
    """
    q = np.round(arr * scale)
    saturated_mask = (q < _INT16_MIN) | (q > _INT16_MAX)
    if saturated_mask.any():
        import sys

        n = int(saturated_mask.sum())
        peak = float(np.abs(arr).max())
        print(
            f"warning: {name}: {n} weights saturated at int16 boundary "
            f"(peak float |weight|={peak:.4f}, scale={scale})",
            file=sys.stderr,
        )
    q = np.clip(q, _INT16_MIN, _INT16_MAX)
    return q.astype(np.int16)


def export_network(model: Any, path: str) -> None:
    """Serialize a `training.model.NNUE` instance to `path` in JNN1 format.

    Args:
        model: Trained NNUE. Duck-typed against
            `feature_weights.weight`, `feature_bias`, `output.weight`,
            `output.bias` — either the concrete class from
            `training.model` or any compatible shape.
        path: Destination file path. Overwritten if it exists.

    Raises:
        AssertionError: If any tensor has an unexpected shape (guards
            against silent layout drift between the trainer and the
            C++ loader).
    """
    fw = model.feature_weights.weight.detach().cpu().numpy()  # (TF, HS)
    fb = model.feature_bias.detach().cpu().numpy()  # (HS,)
    ow = model.output.weight.detach().cpu().numpy().reshape(-1)  # (2*HS,)
    # nn.Linear(bias=True) makes output.bias a 1-D tensor of shape (1,).
    ob = float(model.output.bias.detach().cpu().numpy().item())  # scalar

    assert fw.shape == (TOTAL_FEATURES, HIDDEN_SIZE), (
        f"feature_weights shape {fw.shape} != ({TOTAL_FEATURES}, {HIDDEN_SIZE})"
    )
    assert fb.shape == (HIDDEN_SIZE,), (
        f"feature_bias shape {fb.shape} != ({HIDDEN_SIZE},)"
    )
    assert ow.shape == (2 * HIDDEN_SIZE,), (
        f"output_weights shape {ow.shape} != ({2 * HIDDEN_SIZE},)"
    )

    fw_q = _quantize(fw, SCALE_FEATURES, "feature_weights")
    fb_q = _quantize(fb, SCALE_FEATURES, "feature_biases")
    ow_q = _quantize(ow, SCALE_OUTPUT, "output_weights")

    # Output bias contributes to the raw int64 sum before /8192 in
    # `evaluate()`, so scale it by the full divisor.
    ob_raw = round(ob * SCALE_FEATURES * SCALE_OUTPUT)
    if not _INT16_MIN <= ob_raw <= _INT16_MAX:
        import sys

        print(
            f"warning: output_bias saturated (ob={ob:.4f}, scaled={ob_raw})",
            file=sys.stderr,
        )
    ob_q = max(_INT16_MIN, min(_INT16_MAX, ob_raw))

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", HIDDEN_SIZE))
        f.write(struct.pack("<I", TOTAL_FEATURES))
        f.write(fb_q.tobytes())
        f.write(fw_q.tobytes())
        f.write(ow_q.tobytes())
        f.write(struct.pack("<h", ob_q))


def expected_file_size() -> int:
    """Return the exact byte size of a well-formed JNN1 file.

    Closed form for the layout above — useful for the round-trip
    size assertion in `test_export.py`. Any drift here means the C++
    loader's "trailing bytes" check will reject the file.

    Returns:
        Byte count of a valid JNN1 file.
    """
    return (
        16  # header (4 magic + 3×4 uint32)
        + HIDDEN_SIZE * 2  # feature_biases
        + TOTAL_FEATURES * HIDDEN_SIZE * 2  # feature_weights
        + 2 * HIDDEN_SIZE * 2  # output_weights
        + 2  # output_bias
    )
