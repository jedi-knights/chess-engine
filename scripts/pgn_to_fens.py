#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chess>=1.10",
# ]
# ///
"""Ingest a directory of PGN files into a `<FEN>;<outcome>` training corpus.

The last mile of the "learn from games you actually played" loop (plan
Phase 5): take PGNs produced by your chess GUI / lichess bot / cutechess
matches, filter to quiet positions, and label each one with the final
game outcome from WHITE's perspective. The output format is what
`training/train.py` already consumes.

Idempotency: every .pgn file processed is recorded in a `.processed`
sidecar log. Re-running the ingester on the same directory is cheap and
safe — files already seen are skipped. New files are picked up on the
next run, which makes this safe to run on a cron / GitHub Action.

Quiet filter matches `scripts/gen_selfplay_data.py`: side-to-move not in
check AND the last move was not a capture. Standard practice for Texel
and NNUE training alike — the eval is unreliable mid-exchange.

Usage:
    scripts/pgn_to_fens.py --input ~/pgns/ \\
        --corpus tests/data/played_games.txt \\
        --processed-log tests/data/played_games.processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn

OUTCOME_MAP = {
    "1-0": "1.0",
    "0-1": "0.0",
    "1/2-1/2": "0.5",
}


def outcome_from_result(result: str) -> str | None:
    """Map a PGN `Result` header to the WDL float the trainer expects.

    Args:
        result: Value of the `[Result "..."]` PGN header.

    Returns:
        One of ``"1.0"``, ``"0.5"``, ``"0.0"`` for completed games, or
        ``None`` for ongoing/unknown results (`*` or missing). Callers
        skip positions when ``None`` is returned so unfinished games
        do not get silently labeled as draws.
    """
    return OUTCOME_MAP.get(result)


def _is_quiet(board: chess.Board) -> bool:
    """Return True if the position is quiet (not in check, last move not a capture).

    Args:
        board: Position to classify. Mutated transiently (last move
            popped then re-pushed) and restored before return.

    Returns:
        True when the position is suitable for Texel/NNUE labeling.
    """
    if board.is_check():
        return False
    if board.move_stack:
        last = board.pop()
        was_capture = board.is_capture(last)
        board.push(last)
        if was_capture:
            return False
    return True


def extract_positions(game: chess.pgn.Game, min_ply: int = 8) -> list[tuple[str, str]]:
    """Walk a PGN game and yield quiet labeled positions past the opening.

    The game's outcome is read from the `Result` header once; every
    quiet position in the game is labeled with that same outcome.
    Positions at or before `min_ply` are skipped to reduce opening
    bias (early positions are overweighted by any book).

    Args:
        game: Parsed `chess.pgn.Game`. Must have a `Result` header
            with a decided outcome; ongoing games (`*`) produce an
            empty list.
        min_ply: Skip positions whose ply count is less than or equal
            to this value. The engine's self-play generator uses 8.

    Returns:
        List of `(fen, outcome_str)` tuples, one per quiet position
        past the opening.
    """
    outcome = outcome_from_result(game.headers.get("Result", ""))
    if outcome is None:
        return []

    positions: list[tuple[str, str]] = []
    board = game.board()
    ply = 0
    for move in game.mainline_moves():
        board.push(move)
        ply += 1
        if ply <= min_ply:
            continue
        if not _is_quiet(board):
            continue
        positions.append((board.fen(), outcome))
    return positions


def _load_processed(processed_log: Path) -> set[str]:
    """Read the sidecar log listing PGN filenames already ingested.

    Args:
        processed_log: Path to the newline-separated log. Missing file
            is treated as "nothing processed yet".

    Returns:
        Set of filenames that have already been ingested.
    """
    if not processed_log.exists():
        return set()
    return {ln.strip() for ln in processed_log.read_text().splitlines() if ln.strip()}


def _append_processed(processed_log: Path, names: list[str]) -> None:
    """Append newly-ingested filenames to the processed-files log.

    Args:
        processed_log: Path to the newline-separated log. Parent
            directory is created if missing.
        names: Filenames to append, one per line.
    """
    processed_log.parent.mkdir(parents=True, exist_ok=True)
    with processed_log.open("a") as f:
        for n in names:
            f.write(n + "\n")


def ingest_directory(
    pgn_dir: Path, corpus: Path, processed_log: Path, min_ply: int = 8
) -> int:
    """Walk `pgn_dir`, append new quiet positions to `corpus`.

    Only `.pgn` files not already listed in `processed_log` are read.
    Each new file is appended to the processed-files log as a single
    atomic step after its positions have been written, so a mid-run
    crash leaves the corpus and the log in sync (worst case, one
    file's positions appear twice on the next run — never missed).

    Args:
        pgn_dir: Directory containing one or more `.pgn` files.
        corpus: Output `<FEN>;<outcome>` file. Created if missing,
            appended to otherwise.
        processed_log: Sidecar log tracking which `.pgn` files have
            been ingested. Created if missing.
        min_ply: Ply threshold passed to `extract_positions`.

    Returns:
        Total positions appended to the corpus on this call.
    """
    already = _load_processed(processed_log)
    newly_processed: list[str] = []
    total_appended = 0

    corpus.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not corpus.exists()
    with corpus.open("a") as out:
        if needs_header:
            out.write(f"# Ingested from PGNs by {Path(__file__).name}\n")
        for pgn_path in sorted(pgn_dir.glob("*.pgn")):
            name = pgn_path.name
            if name in already:
                continue
            with pgn_path.open() as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    for fen, outcome in extract_positions(game, min_ply=min_ply):
                        out.write(f"{fen};{outcome}\n")
                        total_appended += 1
            newly_processed.append(name)

    if newly_processed:
        _append_processed(processed_log, newly_processed)
    return total_appended


def _parse_args() -> argparse.Namespace:
    """Parse the CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Directory of .pgn files")
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("tests/data/played_games.txt"),
        help="Output corpus (appended to; created if missing)",
    )
    p.add_argument(
        "--processed-log",
        type=Path,
        default=None,
        help="Sidecar log of ingested filenames (defaults to <corpus>.processed)",
    )
    p.add_argument(
        "--min-ply",
        type=int,
        default=8,
        help="Skip positions this many plies in or shallower",
    )
    return p.parse_args()


def main() -> int:
    """Entry point: walk input, append positions, update processed log.

    Returns:
        Process exit code — 0 on success.
    """
    args = _parse_args()
    if not args.input.is_dir():
        print(f"ERROR: --input is not a directory: {args.input}", file=sys.stderr)
        return 1
    processed_log = args.processed_log or args.corpus.with_suffix(
        args.corpus.suffix + ".processed"
    )
    appended = ingest_directory(args.input, args.corpus, processed_log, args.min_ply)
    print(f"appended {appended:,} positions to {args.corpus}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
