"""HalfKP feature encoding — the Python↔C++ parity layer.

Pure-Python, pure `python-chess`. No torch, no numpy. This module is
imported by the lightweight `test_encoding.py` so the pipeline's
foundational correctness invariant (Python-encoded feature indices
must match the C++ runtime bit-for-bit) can be verified without
downloading a full ML stack.

Anything that needs torch — the `Dataset` subclass, the collator —
lives in `dataset.py`.
"""

from __future__ import annotations

import chess

PIECES_PER_SIDE = 5
FEATURES_PER_KING = 641
TOTAL_FEATURES = 64 * FEATURES_PER_KING  # 41,024


def piece_slot(perspective: bool, piece_color: bool, piece_type: int) -> int:
    """Compute the 0..9 piece-slot inside a per-king feature block.

    Matches `nnue.cpp` `piece_slot`. Slots 0..4 are own P/N/B/R/Q;
    slots 5..9 are enemy P/N/B/R/Q. Kings are excluded from the
    feature set entirely — they are the reference point.

    Args:
        perspective: The side whose accumulator is being built.
        piece_color: The color of the piece on the board.
        piece_type: python-chess piece type (`chess.PAWN`..`chess.QUEEN`).

    Returns:
        Slot index in `[0, 10)`.
    """
    base = 0 if perspective == piece_color else PIECES_PER_SIDE
    return base + (piece_type - chess.PAWN)


def oriented(perspective: bool, sq: int) -> int:
    """Mirror a square vertically when perspective is BLACK.

    Both perspectives see "in front of me = larger rank" after this
    flip, matching `nnue.cpp` `oriented`.

    Args:
        perspective: Perspective color.
        sq: Square index in `[0, 64)`.

    Returns:
        The oriented square index.
    """
    return sq ^ 56 if perspective == chess.BLACK else sq


def feature_index(
    perspective: bool,
    king_sq: int,
    piece_sq: int,
    piece_type: int,
    piece_color: bool,
) -> int:
    """Map `(perspective, king_sq, piece_sq, piece_type, piece_color)` → feature index.

    Layout matches `nnue.cpp` `feature_index`:
        `idx = oriented_king_sq * 641 + oriented_piece_sq * 10 + slot`

    Args:
        perspective: Whose accumulator this feature contributes to.
        king_sq: The friendly king's square (raw, not oriented).
        piece_sq: The piece's square (raw, not oriented).
        piece_type: `chess.PAWN`..`chess.QUEEN`. Kings are not valid.
        piece_color: The color of the piece.

    Returns:
        Feature index in `[0, TOTAL_FEATURES)`.
    """
    kk = oriented(perspective, king_sq)
    ps = oriented(perspective, piece_sq)
    slot = piece_slot(perspective, piece_color, piece_type)
    return kk * FEATURES_PER_KING + ps * (2 * PIECES_PER_SIDE) + slot


def encode_position(board: chess.Board) -> tuple[list[int], list[int]]:
    """Return the active feature indices for both perspectives.

    Args:
        board: Position to encode. `board.turn` determines side-to-move.
            If either king is missing (constructed test position), that
            perspective's list comes back empty — orientation depends
            on the friendly king square.

    Returns:
        `(stm_indices, opp_indices)` — one integer per non-king piece
        on the board, per perspective.
    """
    stm_color = board.turn
    opp_color = not stm_color
    stm_king = board.king(stm_color)
    opp_king = board.king(opp_color)

    stm_idx: list[int] = []
    opp_idx: list[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        if stm_king is not None:
            stm_idx.append(
                feature_index(
                    stm_color, stm_king, square, piece.piece_type, piece.color
                )
            )
        if opp_king is not None:
            opp_idx.append(
                feature_index(
                    opp_color, opp_king, square, piece.piece_type, piece.color
                )
            )
    return stm_idx, opp_idx
