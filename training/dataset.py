"""Torch `Dataset` and `EmbeddingBag`-ready collator over self-play files.

The line format expected by `SelfplayDataset` is the one produced by
`scripts/gen_selfplay_data.py`:

    <fen>;<outcome_wdl>

where `outcome_wdl` is the game result from WHITE's perspective —
`1.0` white win, `0.5` draw, `0.0` black win. `__getitem__` rotates
the target into side-to-move perspective so training sees "STM
winning = 1.0" regardless of color.

Also accepts `<fen>|<centipawn_score>` for future score-labeled data
(centipawns are already STM-perspective; no rotation needed).

Feature encoding lives in `encoding.py` — this module only wires it
into the torch `Dataset` API.
"""

from __future__ import annotations

import chess
import torch
from torch.utils.data import Dataset

from encoding import encode_position


class SelfplayDataset(Dataset):
    """Read `(fen, target)` pairs from a text file.

    Recognized line formats:
        `<fen>;<outcome_wdl>` — from `scripts/gen_selfplay_data.py`;
            outcome is WHITE-perspective WDL (0.0 / 0.5 / 1.0), rotated
            to STM perspective inside `__getitem__`.
        `<fen>|<centipawn>` — centipawn score (already STM perspective;
            not rotated).

    Comment (`#`) and blank lines are skipped.
    """

    def __init__(self, path: str) -> None:
        """Read `path` line-by-line, keeping every valid entry in memory.

        Args:
            path: Path to a text file with one training example per line.
        """
        self.entries: list[tuple[str, float, bool]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ";" in line:
                    parts = line.split(";", 1)
                    is_wdl = True
                elif "|" in line:
                    parts = line.split("|", 1)
                    is_wdl = False
                else:
                    continue
                if len(parts) != 2:
                    continue
                fen = parts[0].strip()
                try:
                    target = float(parts[1].strip())
                except ValueError:
                    continue
                self.entries.append((fen, target, is_wdl))

    def __len__(self) -> int:
        """Number of usable training examples."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], float, bool]:
        """Encode one example.

        Args:
            idx: Row index in the parsed file.

        Returns:
            `(stm_indices, opp_indices, target, is_wdl)`. Target is
            already in STM perspective — WDL entries are rotated when
            side-to-move is BLACK (`1 - target`); centipawn entries
            pass through.
        """
        fen, target, is_wdl = self.entries[idx]
        board = chess.Board(fen)
        stm_idx, opp_idx = encode_position(board)
        if is_wdl and board.turn == chess.BLACK:
            target = 1.0 - target
        return stm_idx, opp_idx, target, is_wdl


def collate_batch(
    batch: list[tuple[list[int], list[int], float, bool]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pack per-example encodings into `EmbeddingBag`-ready tensors.

    `torch.nn.EmbeddingBag` consumes a flat index tensor plus a
    per-example offset tensor pointing at each bag's start; that's what
    this collator produces.

    Args:
        batch: Output of `SelfplayDataset.__getitem__` for each example.

    Returns:
        `(stm_indices, stm_offsets, opp_indices, opp_offsets, targets, is_wdl)`.
        `targets` is float32; `is_wdl` is uint8 (0 = cp, 1 = wdl).
    """
    stm_indices: list[int] = []
    opp_indices: list[int] = []
    stm_offsets: list[int] = [0]
    opp_offsets: list[int] = [0]
    targets: list[float] = []
    is_wdl_flags: list[int] = []
    for stm_idx, opp_idx, target, is_wdl in batch:
        stm_indices.extend(stm_idx)
        opp_indices.extend(opp_idx)
        stm_offsets.append(stm_offsets[-1] + len(stm_idx))
        opp_offsets.append(opp_offsets[-1] + len(opp_idx))
        targets.append(target)
        is_wdl_flags.append(1 if is_wdl else 0)
    return (
        torch.tensor(stm_indices, dtype=torch.long),
        torch.tensor(stm_offsets[:-1], dtype=torch.long),
        torch.tensor(opp_indices, dtype=torch.long),
        torch.tensor(opp_offsets[:-1], dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(is_wdl_flags, dtype=torch.uint8),
    )
