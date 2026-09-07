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
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))
from dataset import SelfplayDataset, collate_batch  # noqa: E402
from export import export_network  # noqa: E402
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
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument(
        "--score-scale",
        type=float,
        default=400.0,
        help="Divisor for centipawns → sigmoid space (Stockfish uses 400)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
) -> float:
    """Run one training or eval epoch.

    Args:
        model: The NNUE module.
        loader: Data loader over the current split.
        opt: Optimizer for training; pass `None` for eval mode.
        device: Torch device.
        k: Score scale for the sigmoid conversion.

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
                opt.step()

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
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, opt, device, args.score_scale)
        val_loss = _run_epoch(model, val_loader, None, device, args.score_scale)
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}  train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  ({dt:.1f}s)"
        )

    export_network(model, args.out)
    print(f"exported → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
