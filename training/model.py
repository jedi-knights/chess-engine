"""HalfKP → 256 → 1 NNUE model in PyTorch.

Mirrors the C++ runtime layout in src/nnue.cpp:

    features (41,024 sparse, per perspective)
        → feature_transformer (shared, → 256)
        → clipped ReLU
        → concat(stm_acc, opp_acc)  # 512 units
        → output (→ 1)

Both perspectives share the same feature-transformer weights; the
network sees the side-to-move accumulator first, then the opponent's.
Kings are excluded from features — they are the reference point that
every other piece's feature index is relative to.

Feature encoding lives in `dataset.py` — this file is only the
`nn.Module`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN_SIZE = 256
FEATURES_PER_KING = 641  # 64 squares × 10 slots + 1 padding
TOTAL_FEATURES = 64 * FEATURES_PER_KING  # 41,024


class NNUE(nn.Module):
    """HalfKP → 256 → 1 network with a shared feature transformer.

    Both perspectives share the same `feature_weights` embedding table
    — WHITE and BLACK just index it with perspective-oriented feature
    indices (see `dataset.feature_index`).
    """

    def __init__(self) -> None:
        """Initialize the network with small weights.

        Weights use `std=0.05` (features) and `std=0.10` (output) so
        int16 quantization at scale 128 does not saturate. Default
        Xavier init would blow the scale budget on the first pass.
        """
        super().__init__()
        # EmbeddingBag(mode='sum') is the sparse-input analog of a
        # matrix multiply: sum the rows of an embedding table
        # selected by the active-feature indices. O(active_features)
        # per position, ~30 for typical middlegames — exactly what
        # C++ refresh_side does when it rebuilds an accumulator.
        self.feature_weights = nn.EmbeddingBag(TOTAL_FEATURES, HIDDEN_SIZE, mode="sum")
        self.feature_bias = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.output = nn.Linear(2 * HIDDEN_SIZE, 1, bias=True)

        nn.init.normal_(self.feature_weights.weight, std=0.05)
        nn.init.normal_(self.output.weight, std=0.10)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        stm_indices: torch.Tensor,
        stm_offsets: torch.Tensor,
        opp_indices: torch.Tensor,
        opp_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Run the forward pass for a batch of positions.

        Args:
            stm_indices: Flat concatenation of active-feature indices
                across the batch, side-to-move perspective.
            stm_offsets: Start index of each example inside
                `stm_indices` (`EmbeddingBag` layout).
            opp_indices: Same for the opponent perspective.
            opp_offsets: Same for the opponent perspective.

        Returns:
            One scalar per example. In WDL space when trained with
            sigmoid-scaled MSE loss.
        """
        stm_acc = self.feature_weights(stm_indices, stm_offsets) + self.feature_bias
        opp_acc = self.feature_weights(opp_indices, opp_offsets) + self.feature_bias
        combined = torch.cat([stm_acc, opp_acc], dim=1)
        # Float-space clipped ReLU: clamp to [0, 1]. Corresponds to the
        # C++ runtime's [0, 127] int32 clamp after scaling by
        # SCALE_FEATURES=128 (see export.py).
        clipped = torch.clamp(combined, 0.0, 1.0)
        return self.output(clipped).squeeze(-1)
