#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Orchestrate the end-to-end "learn from played games" cycle.

Chains the shipped tools in order so the user can run one command
instead of five:

    1. scripts/pgn_to_fens.py       --input $PGN_DIR
    2. scripts/stockfish_label.py   (re-score the new FENs with Stockfish)
    3. training/train.py            (train a candidate net)
    4. scripts/sprt.py              (optional: SPRT the candidate vs the champion)
    5. cp /tmp/candidate.jnn1 nets/default.jnn1   (optional: promote on H1)

Everything this script does is composable from the component tools —
no new logic. It just wires them together with sane defaults and skips
cleanly when a step has no work (e.g. no new PGNs since last run).

The script is deliberately dumb glue: no `os.chdir`, no in-process
imports, no retries. If a subcommand fails, exit non-zero so cron /
GitHub Actions surface the failure. Rerun idempotency comes from
pgn_to_fens.py's processed-files log — the loop itself holds no
state across runs.

**This script never overwrites `nets/default.jnn1` unless `--promote`
is passed AND the SPRT accepts H1.** Default behavior is "produce a
candidate; promote manually after review."

Typical usage (manual, no promote):

    scripts/learning_loop.py --pgn-dir ~/games/ \\
        --corpus tests/data/played_games.txt \\
        --candidate /tmp/candidate.jnn1

Typical usage (nightly, auto-promote on SPRT H1):

    scripts/learning_loop.py --pgn-dir /mnt/games/ \\
        --corpus tests/data/played_games.txt \\
        --candidate /tmp/candidate.jnn1 \\
        --baseline nets/default.jnn1 \\
        --sprt --promote
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _run(cmd: list[str]) -> int:
    """Run a subprocess, streaming its output; return the exit code.

    Args:
        cmd: argv list to execute. First element is the executable.

    Returns:
        The subprocess exit code. Nonzero means the step failed and
        the caller should abort the loop.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def _count_data_lines(path: Path) -> int:
    """Count non-comment, non-blank lines in a FEN corpus file.

    Args:
        path: Corpus path. Missing file returns 0.

    Returns:
        Number of real data lines.
    """
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for ln in f if ln.strip() and not ln.startswith("#"))


def _parse_args() -> argparse.Namespace:
    """Parse the CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pgn-dir",
        type=Path,
        required=True,
        help="Directory of .pgn files to ingest",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "played_games.txt",
        help="Running FEN;outcome corpus (appended to, never rewritten)",
    )
    p.add_argument(
        "--labeled",
        type=Path,
        default=Path("/tmp/played_games_labeled.txt"),
        help="Stockfish-labeled output (fen|cp format) — overwritten each run",
    )
    p.add_argument(
        "--candidate",
        type=Path,
        default=Path("/tmp/candidate.jnn1"),
        help="Candidate network path",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "nets" / "default.jnn1",
        help="Shipped network path (used for SPRT and for the engine during self-play)",
    )
    p.add_argument(
        "--min-new-positions",
        type=int,
        default=1000,
        help="Skip train/SPRT when fewer than N new positions were ingested",
    )
    p.add_argument(
        "--sf-depth",
        type=int,
        default=10,
        help="Stockfish depth for re-labeling",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument(
        "--sprt",
        action="store_true",
        help="Run SPRT between candidate and baseline after training",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="If SPRT accepts H1, copy candidate over the baseline (default: never)",
    )
    p.add_argument(
        "--sprt-max-games",
        type=int,
        default=400,
        help="Game cap for SPRT (passed through to scripts/sprt.py)",
    )
    return p.parse_args()


def main() -> int:
    """Entry point: ingest → label → train → optional SPRT → optional promote.

    Returns:
        Process exit code — 0 on success, nonzero if any step failed.
    """
    args = _parse_args()
    if not args.pgn_dir.is_dir():
        print(f"ERROR: --pgn-dir is not a directory: {args.pgn_dir}", file=sys.stderr)
        return 1

    before = _count_data_lines(args.corpus)

    # Step 1 — ingest new PGNs.
    rc = _run(
        [
            "scripts/pgn_to_fens.py",
            "--input",
            str(args.pgn_dir),
            "--corpus",
            str(args.corpus),
        ]
    )
    if rc != 0:
        return rc

    after = _count_data_lines(args.corpus)
    new = after - before
    print(f"\n[loop] {new:,} new positions ({after:,} total in corpus)")

    if new < args.min_new_positions:
        print(
            f"[loop] fewer than --min-new-positions={args.min_new_positions:,} "
            "new positions — stopping before retrain"
        )
        return 0

    # Step 2 — re-label with Stockfish (stronger teacher than game outcome alone).
    rc = _run(
        [
            "scripts/stockfish_label.py",
            "--input",
            str(args.corpus),
            "--output",
            str(args.labeled),
            "--depth",
            str(args.sf_depth),
        ]
    )
    if rc != 0:
        return rc

    # Step 3 — train.
    rc = _run(
        [
            "training/train.py",
            "--data",
            str(args.labeled),
            "--out",
            str(args.candidate),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
        ]
    )
    if rc != 0:
        return rc

    print(f"\n[loop] trained candidate: {args.candidate}")

    # Step 4 — optional SPRT.
    if not args.sprt:
        print("[loop] --sprt not passed; candidate is NOT promoted")
        return 0

    rc = _run(
        [
            "scripts/sprt.py",
            "--baseline",
            str(args.baseline),
            "--tuned",
            str(args.candidate),
            "--max-games",
            str(args.sprt_max_games),
        ]
    )
    # SPRT convention: 0 = H1 (gain), 1 = H0 (no gain / regression), >1 = error.
    if rc > 1:
        return rc

    # Step 5 — optional promote.
    if not args.promote:
        print(f"[loop] --promote not passed; leaving {args.baseline} unchanged")
        return 0
    if rc != 0:
        print(
            f"[loop] SPRT did not accept H1 (exit {rc}); NOT promoting candidate",
            file=sys.stderr,
        )
        return 0

    shutil.copy2(args.candidate, args.baseline)
    print(f"[loop] promoted: {args.candidate} -> {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
