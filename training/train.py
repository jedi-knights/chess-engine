#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "numpy>=1.24",
#     "python-chess>=1.9",
# ]
# ///
"""Train an NNUE network from self-play (`fen;wdl` or `fen|cp`) data
and export it in the engine's JNN1 binary format.

Usage:
    training/train.py --data path/to/selfplay.txt --out net.jnn1 \\
        [--epochs 20] [--batch 1024] [--lr 1e-3]

Self-play data is expected from `scripts/gen_selfplay_data.py` — see
`training/README.md` for the full workflow.

Loss: MSE on `sigmoid(pred / score_scale)` vs. target in WDL space.
WDL targets pass through; centipawn targets are converted to WDL via
`sigmoid(cp / score_scale)` first — same K-factor for both, so a mixed
file trains coherently.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))
from dataset import SelfplayDataset, collate_batch  # noqa: E402
from export import SCALE_FEATURES, SCALE_OUTPUT, export_network  # noqa: E402
from model import NNUE  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the trainer.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="Train NNUE for chess-engine")
    p.add_argument("--data", required=True, help="Training data (fen;wdl or fen|cp)")
    p.add_argument("--out", required=True, help="Output .jnn1 path")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW decoupled weight decay (0 disables)",
    )
    p.add_argument(
        "--warmup-frac",
        type=float,
        default=0.05,
        help="Fraction of total epochs spent in linear LR warmup",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max L2 norm for gradient clipping (0 disables)",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument(
        "--score-scale",
        type=float,
        default=400.0,
        help="Divisor for centipawns → sigmoid space (Stockfish uses 400)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Construct the trainer's optimizer.

    Uses `AdamW` so weight decay is decoupled from the gradient
    update — plain `Adam` with a nonzero `weight_decay` applies the
    penalty through the running moment estimates, which interacts
    poorly with the sparse `EmbeddingBag` feature table.

    Args:
        model: The NNUE module whose parameters should be optimized.
        lr: Peak learning rate (before any scheduler modulation).
        weight_decay: Decoupled weight decay coefficient.

    Returns:
        An `AdamW` optimizer bound to every trainable parameter.
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    opt: torch.optim.Optimizer, total_steps: int, warmup_frac: float
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a linear-warmup → cosine-decay LR schedule.

    The first `floor(total_steps * warmup_frac)` steps ramp the LR
    linearly from near-zero to the optimizer's base LR; the remaining
    steps follow a half-cosine down to near-zero.

    Args:
        opt: Optimizer whose LR will be modulated.
        total_steps: Total scheduler steps across the whole training run.
            Pass the epoch count when stepping once per epoch.
        warmup_frac: Fraction of `total_steps` to spend in warmup.
            Must be in `[0, 1)`.

    Returns:
        A `LambdaLR` scheduler. Call `.step()` once per epoch.
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))
    cos_denom = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        """Return the LR multiplier at the given scheduler step.

        Args:
            step: Zero-indexed step count (scheduler-internal).

        Returns:
            Multiplier applied to the optimizer's base LR.
        """
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / cos_denom
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(opt, lr_lambda=lr_lambda)


def clip_quantization_weights(model: torch.nn.Module) -> None:
    """Clamp weights in-place to each layer's int16 quantization bound.

    The exporter multiplies float weights by per-layer scales
    (`SCALE_FEATURES=128` for feature weights/biases, `SCALE_OUTPUT=64`
    for output weights, their product `8192` for output bias) and
    truncates to int16. A weight outside the resulting float range
    saturates on export — the on-disk net then no longer matches the
    trained one. Calling this after every optimizer step turns the
    "won't saturate if weights stay small" hope into a construction-
    time invariant.

    Bounds use a 1-ULP margin against the signed-int16 max (32767):
    ``round(w * scale)`` with a float exactly at ``32767.5 / scale``
    may round up to 32768 and overflow. Shrinking by `-1.0 / scale`
    guarantees ``round`` lands on 32767 or below.

    Args:
        model: NNUE module whose weights will be clamped.
    """
    # Bounds live in export.py so trainer + exporter can't drift.
    fw_bound = (32767.0 - 1.0) / SCALE_FEATURES
    ow_bound = (32767.0 - 1.0) / SCALE_OUTPUT
    ob_bound = (32767.0 - 1.0) / (SCALE_FEATURES * SCALE_OUTPUT)
    with torch.no_grad():
        model.feature_weights.weight.clamp_(-fw_bound, fw_bound)
        model.feature_bias.clamp_(-fw_bound, fw_bound)
        model.output.weight.clamp_(-ow_bound, ow_bound)
        model.output.bias.clamp_(-ob_bound, ob_bound)


def clip_gradients(model: torch.nn.Module, max_norm: float) -> None:
    """Clip gradients in-place so their total L2 norm is at most `max_norm`.

    A single catastrophic batch can produce gradients large enough to
    push accumulator weights outside the int16 quantization range —
    the resulting network loads but evaluates nonsensically. Clipping
    at 1.0 is a cheap guardrail against that drift.

    Args:
        model: Module whose parameters' `.grad` fields will be clipped.
        max_norm: L2 norm cap. Pass `0` to disable (no-op).
    """
    if max_norm <= 0.0:
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _to_wdl(
    pred: torch.Tensor, target: torch.Tensor, is_wdl: torch.Tensor, k: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring model output and target into a common WDL sigmoid space.

    Args:
        pred: Raw model output.
        target: Raw target (either WDL in [0, 1] or centipawn score).
        is_wdl: 1 where the target is WDL, 0 where it is centipawns.
        k: Score scale for the centipawn → sigmoid conversion.

    Returns:
        `(pred_wdl, target_wdl)`, both in `[0, 1]`.
    """
    pred_wdl = torch.sigmoid(pred / k)
    cp_wdl = torch.sigmoid(target / k)
    target_wdl = torch.where(is_wdl.bool(), target, cp_wdl)
    return pred_wdl, target_wdl


def _run_epoch(
    model: NNUE,
    loader: DataLoader,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
    k: float,
    grad_clip: float = 0.0,
) -> float:
    """Run one training or eval epoch.

    Args:
        model: The NNUE module.
        loader: Data loader over the current split.
        opt: Optimizer for training; pass `None` for eval mode.
        device: Torch device.
        k: Score scale for the sigmoid conversion.
        grad_clip: Max L2 norm for gradient clipping. Pass `0` to
            disable (default for backward compatibility with the
            eval-mode call site, where no gradients are produced).

    Returns:
        Mean per-example loss over the pass.
    """
    training = opt is not None
    model.train(mode=training)
    total = 0.0
    n_seen = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for stm_i, stm_o, opp_i, opp_o, targets, is_wdl in loader:
            stm_i, stm_o = stm_i.to(device), stm_o.to(device)
            opp_i, opp_o = opp_i.to(device), opp_o.to(device)
            targets = targets.to(device)
            is_wdl = is_wdl.to(device)

            out = model(stm_i, stm_o, opp_i, opp_o)
            pred_wdl, tgt_wdl = _to_wdl(out, targets, is_wdl, k)
            loss = torch.nn.functional.mse_loss(pred_wdl, tgt_wdl)

            if opt is not None:
                opt.zero_grad()
                loss.backward()
                clip_gradients(model, grad_clip)
                opt.step()
                # Enforce the int16 quantization bound on every weight
                # as part of the step itself. Doing this after the
                # optimizer runs means AdamW's moment estimates see
                # the pre-clamp gradients; the clamp only trims
                # weights that would otherwise saturate on export.
                clip_quantization_weights(model)

            total += loss.item() * targets.size(0)
            n_seen += targets.size(0)
    return total / max(1, n_seen)


def main() -> int:
    """Run the full training loop end-to-end.

    Returns:
        Process exit code — 0 on success.
    """
    args = _parse_args()
    torch.manual_seed(args.seed)

    ds = SelfplayDataset(args.data)
    print(f"loaded {len(ds)} positions from {args.data}")
    if len(ds) < 100:
        print(
            "warning: <100 positions — model will not learn anything useful",
            file=sys.stderr,
        )

    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_batch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = NNUE().to(device)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, total_steps=args.epochs, warmup_frac=args.warmup_frac)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(
            model,
            train_loader,
            opt,
            device,
            args.score_scale,
            grad_clip=args.grad_clip,
        )
        val_loss = _run_epoch(model, val_loader, None, device, args.score_scale)
        sched.step()
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}  train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  lr={lr_now:.2e}  ({dt:.1f}s)"
        )

    export_network(model, args.out)
    print(f"exported → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
