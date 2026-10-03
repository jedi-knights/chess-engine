#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Re-label a FEN corpus with Stockfish centipawn scores.

Reads a `<FEN>;<outcome>` or plain-FEN file, spawns Stockfish as a
UCI subprocess, scores each position at a configured depth, and
writes a `<FEN>|<cp>` file suitable for `training/train.py`.

Why: public quiet-labeled EPD mirrors have rotted. Local Stockfish is
a materially stronger teacher than the engine's own classical eval
(the current self-distillation ceiling noted in nets/README.md), so
re-scoring positions we already have gets us the Phase 1 "break the
teacher ceiling" win without needing an external dataset.

Mate scores are clamped to ±10000 cp — training's sigmoid-space loss
is undefined at ±infinity and genuine mates shouldn't bias cp labels
near 0.

Usage:
    scripts/stockfish_label.py --input tests/data/selfplay_500.txt \\
        --output tests/data/sf_labeled.txt --depth 8
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

MATE_CP_CLAMP = 10000
SCORE_PATTERN = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)")


def parse_score_from_info_lines(lines: list[str]) -> int | None:
    """Extract the deepest-reported score from a list of UCI `info` lines.

    Walks the lines in order and keeps the last `score cp N` or
    `score mate K` value seen. Mate scores are clamped to
    `±MATE_CP_CLAMP` so downstream sigmoid-space training stays
    well-defined.

    Args:
        lines: UCI output lines from a single `go` command (anything
            emitted between `position ...` and the next `bestmove`).

    Returns:
        The deepest centipawn score (side-to-move perspective), or
        `None` if no `score` token appears in any line.
    """
    last_score: int | None = None
    for line in lines:
        m = SCORE_PATTERN.search(line)
        if m is None:
            continue
        kind, value = m.group(1), int(m.group(2))
        if kind == "mate":
            last_score = MATE_CP_CLAMP if value > 0 else -MATE_CP_CLAMP
        else:
            last_score = value
    return last_score


def iter_fens(input_path: Path) -> list[str]:
    """Load FENs from an input file, dropping any trailing label.

    Accepts both `<fen>;<wdl>` and `<fen>|<cp>` lines. Comment lines
    (starting with `#`) and blank lines are skipped.

    Args:
        input_path: Path to a FEN corpus — labeled or unlabeled.

    Returns:
        A list of FEN strings, one per non-comment line.
    """
    fens: list[str] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in (";", "|"):
                if sep in line:
                    line = line.split(sep, 1)[0]
                    break
            fens.append(line)
    return fens


def spawn_stockfish(path: str, threads: int, hash_mb: int) -> subprocess.Popen:
    """Launch Stockfish and complete the UCI handshake.

    Args:
        path: Executable name or absolute path.
        threads: Threads for the search.
        hash_mb: Transposition table size in MB.

    Returns:
        The `Popen` with stdin/stdout wired for text-mode UCI.
    """
    proc = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def send(cmd: str) -> None:
        """Send a UCI command and newline to Stockfish.

        Args:
            cmd: UCI command text without a trailing newline.
        """
        proc.stdin.write(cmd + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    def wait_for(token: str) -> None:
        """Block until Stockfish prints a line containing `token`.

        Args:
            token: Substring the function waits to see on any line.
        """
        for line in proc.stdout:  # type: ignore[union-attr]
            if token in line:
                return
        raise RuntimeError(f"stockfish exited before emitting {token!r}")

    send("uci")
    wait_for("uciok")
    send(f"setoption name Threads value {threads}")
    send(f"setoption name Hash value {hash_mb}")
    send("isready")
    wait_for("readyok")
    return proc


def score_fen(proc: subprocess.Popen, fen: str, depth: int) -> int | None:
    """Score a single FEN with the running Stockfish process.

    Args:
        proc: An already-handshook Stockfish subprocess.
        fen: FEN string to score. Must be well-formed.
        depth: Fixed search depth.

    Returns:
        Centipawn score (side-to-move perspective) or `None` if
        Stockfish produced no score line (e.g., malformed FEN).
    """
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go depth {depth}\n")
    proc.stdin.flush()

    info_lines: list[str] = []
    for line in proc.stdout:
        if line.startswith("bestmove"):
            break
        info_lines.append(line.rstrip())
    return parse_score_from_info_lines(info_lines)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--hash", type=int, default=64, dest="hash_mb")
    p.add_argument(
        "--limit", type=int, default=0, help="Score at most N positions (0 = all)"
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress line every N scored positions",
    )
    return p.parse_args()


def main() -> int:
    """Entry point: fetch FENs, score with Stockfish, write labeled file.

    Returns:
        Process exit code — 0 on success.
    """
    args = _parse_args()
    fens = iter_fens(args.input)
    if args.limit > 0:
        fens = fens[: args.limit]
    print(f"loaded {len(fens):,} fens from {args.input}", file=sys.stderr)

    proc = spawn_stockfish(args.stockfish, args.threads, args.hash_mb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    t0 = time.time()
    with args.output.open("w") as out:
        out.write(f"# Labeled with Stockfish depth={args.depth}\n")
        out.write(f"# Source: {args.input}\n")
        for i, fen in enumerate(fens, 1):
            score = score_fen(proc, fen, args.depth)
            if score is None:
                skipped += 1
                continue
            out.write(f"{fen}|{score}\n")
            written += 1
            if i % args.progress_every == 0:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0.0
                eta = (len(fens) - i) / rate if rate > 0 else 0.0
                print(
                    f"  [{i:>7,}/{len(fens):,}] {rate:5.1f} pos/s  "
                    f"eta {eta / 60:5.1f} min",
                    file=sys.stderr,
                )

    proc.stdin.write("quit\n")  # type: ignore[union-attr]
    proc.wait(timeout=5)

    dt = time.time() - t0
    print(
        f"wrote {written:,} labeled positions (skipped {skipped:,}) in {dt / 60:.1f} min",
        file=sys.stderr,
    )
    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
