#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chess>=1.10",
# ]
# ///
"""Unit tests for scripts/pgn_to_fens.py.

Covers the three decisions that make the ingester correct:

1. Parsing the game outcome from the PGN `Result` header
2. Filtering out opening plies + non-quiet positions
3. Idempotency: a processed-files log prevents double-counting

Everything else (file I/O glue) is exercised by a round-trip test
through a tempdir.

Run:
    scripts/test_pgn_to_fens.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parent))
from pgn_to_fens import (  # noqa: E402
    extract_positions,
    ingest_directory,
    outcome_from_result,
)


def _game_from_moves(result: str, moves: list[str]) -> chess.pgn.Game:
    """Build a PGN game object from a result header and SAN move list.

    Args:
        result: PGN result header value (e.g. "1-0").
        moves: SAN moves in game order (e.g. ["e4", "e5", "Nf3"]).

    Returns:
        A ``chess.pgn.Game`` with the moves applied and header set.
    """
    game = chess.pgn.Game()
    game.headers["Result"] = result
    node = game
    board = chess.Board()
    for san in moves:
        move = board.parse_san(san)
        board.push(move)
        node = node.add_variation(move)
    return game


def test_outcome_from_result_header() -> None:
    """`Result` header must map to the WDL float the trainer expects.

    White-win → 1.0, Black-win → 0.0, Draw → 0.5. Ongoing/unknown
    games (`*`) return None so the caller can skip them — a `*` game
    labeled 0.5 would quietly bias the corpus toward draws.
    """
    # Arrange / Act / Assert
    assert outcome_from_result("1-0") == "1.0"
    assert outcome_from_result("0-1") == "0.0"
    assert outcome_from_result("1/2-1/2") == "0.5"
    assert outcome_from_result("*") is None
    assert outcome_from_result("") is None


def test_extract_positions_skips_opening_plies() -> None:
    """Positions in the first `min_ply` plies must be dropped.

    Early positions still look like the opening book; labeling them
    with the game outcome overweights opening theory and biases the
    net toward the specific repertoire of whichever side was played.
    """
    # Arrange
    game = _game_from_moves(
        "1-0", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
    )

    # Act
    positions = extract_positions(game, min_ply=8)

    # Assert — 10 plies, min_ply=8 drops the first 8 → 2 positions remain
    #         (both are also quiet; Be7 and 0-0 then Be7 are non-captures)
    assert len(positions) == 2, f"expected 2, got {len(positions)}: {positions}"


def test_extract_positions_filters_captures_and_checks() -> None:
    """Non-quiet positions (capture-aftermath or in-check) must be dropped.

    Texel/NNUE training compares a static value against a label; both
    are noisy on tactical positions where the eval is dominated by the
    forced reply. Filtering non-quiet positions is the industry
    standard (Zurichess quiet-labeled et al do the same).
    """
    # Arrange — a short line with a capture at ply 9 and a check at
    #           ply 10. With min_ply=0, both must be filtered out
    #           regardless of ply number.
    game = _game_from_moves(
        "1-0",
        ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "Bxc6", "dxc6"],
    )

    # Act
    positions = extract_positions(game, min_ply=0)
    # The capture at ply 9 (Bxc6) and the recapture at ply 10 (dxc6)
    # produce non-quiet post-move positions that must be excluded.
    non_quiet_fens = {
        # After 5. Bxc6 — Black to move, last move was a capture
        "r1bqkb1r/1ppp1ppp/p1B2n2/4p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 5",
        # After 5...dxc6 — White to move, last move was a capture
        "r1bqkb1r/1pp2ppp/p1p2n2/4p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 6",
    }
    kept_fens = {fen for fen, _ in positions}

    # Assert
    assert non_quiet_fens.isdisjoint(kept_fens), (
        f"non-quiet positions leaked through: {non_quiet_fens & kept_fens}"
    )


def test_extract_positions_tags_every_position_with_game_outcome() -> None:
    """Every extracted FEN carries the same WDL label from the result.

    PGN gives one outcome per game; every quiet position in that game
    gets the same label. A regression that flips signs mid-game would
    corrupt the corpus silently.
    """
    # Arrange — a Black-win game
    game = _game_from_moves(
        "0-1", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
    )

    # Act
    positions = extract_positions(game, min_ply=8)

    # Assert
    labels = {label for _, label in positions}
    assert labels == {"0.0"}, f"expected only '0.0', got {labels}"


def test_ingest_directory_is_idempotent() -> None:
    """Re-running ingest on the same PGN files must not duplicate output.

    The processed-files log is the primary defense against double-
    counting when the ingester is run repeatedly (e.g. nightly) against
    an append-only PGN archive.
    """
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        pgn_dir = Path(tmpdir) / "pgns"
        pgn_dir.mkdir()
        corpus = Path(tmpdir) / "corpus.txt"
        processed_log = Path(tmpdir) / "corpus.processed"
        game = _game_from_moves(
            "1-0",
            ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
        )
        pgn_text = str(game) + "\n"
        (pgn_dir / "a.pgn").write_text(pgn_text)

        # Act — first ingest writes positions, second ingest sees the
        # file in the processed log and skips it
        first = ingest_directory(pgn_dir, corpus, processed_log, min_ply=8)
        second = ingest_directory(pgn_dir, corpus, processed_log, min_ply=8)

        # Assert
        assert first > 0, "first ingest should write at least one position"
        assert second == 0, f"second ingest should write 0 (idempotent); got {second}"
        # Corpus file size matches the first run — nothing appended on second
        lines = corpus.read_text().splitlines()
        data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
        assert len(data_lines) == first


TESTS = [
    test_outcome_from_result_header,
    test_extract_positions_skips_opening_plies,
    test_extract_positions_filters_captures_and_checks,
    test_extract_positions_tags_every_position_with_game_outcome,
    test_ingest_directory_is_idempotent,
]


def main() -> int:
    """Run every test in `TESTS` and print a per-case status line.

    Returns:
        0 if every test passed, 1 otherwise.
    """
    _ = io  # keep the import for subtests that may add raw-PGN cases later
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
