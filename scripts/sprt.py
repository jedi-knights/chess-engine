#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""
SPRT (Sequential Probability Ratio Test) validation harness for eval
changes. Wraps cutechess-cli (or fastchess) to play games between a
BASELINE and a TUNED engine binary until enough evidence accumulates
to accept either "the change adds Elo" (H1) or "the change is neutral
/ regresses" (H0) with the specified error rates. This is the gate
before committing tuned eval weights to the shipped engine.

Prerequisites: cutechess-cli or fastchess installed and on PATH.
  Debian/Ubuntu: sudo apt install cutechess-cli
  Release binaries: https://github.com/cutechess/cutechess/releases
                    https://github.com/Disservin/fastchess/releases
  From source: cutechess needs Qt 6.8+ / cmake; fastchess is a
               single-Makefile build (much simpler).

Typical workflow:

  # 1. Save baseline binary before applying tuned weights:
  make && cp engine engine.baseline

  # 2. Apply tuned eval weights, rebuild:
  # (edit src/eval.cpp / src/eval.h with tuner output, then)
  make

  # 3. Run SPRT until conclusive:
  scripts/sprt.py --baseline engine.baseline --tuned engine \
                  --tc 10+0.1 --elo0 0 --elo1 5

  # SPRT stops early — a positive result on a genuine +5 Elo change
  # typically converges in 1000-3000 games; a null result may take
  # 5000-10000. Time-control × games determines wall time.

Cutechess handles the SPRT statistic natively (Wald-based, LLR test).
This script's role is preflight (binaries exist, book exists), building
the correct command line, and passing results back through exit codes.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_BASELINE = Path("engine.baseline")
DEFAULT_TUNED = Path("./engine")
DEFAULT_BOOK = Path("tests/data/openings_demo.pgn")
DEFAULT_TC = "10+0.1"
DEFAULT_ELO0 = 0.0
DEFAULT_ELO1 = 5.0
DEFAULT_ALPHA = 0.05
DEFAULT_BETA = 0.05
DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_GAMES = 5000


def find_gamemanager() -> str | None:
    """Prefer fastchess (faster, drop-in replacement); fall back to cutechess-cli."""
    for name in ("fastchess", "cutechess-cli"):
        found = shutil.which(name)
        if found:
            return found
    return None


def check_prereqs(baseline: Path, tuned: Path, book: Path) -> str | None:
    gm = find_gamemanager()
    if gm is None:
        print(
            "ERROR: neither cutechess-cli nor fastchess found on PATH.\n"
            "  Debian/Ubuntu:    sudo apt install cutechess-cli\n"
            "  Release binaries: https://github.com/cutechess/cutechess/releases\n"
            "                    https://github.com/Disservin/fastchess/releases\n"
            "  From source:      cutechess needs Qt 6.8+ / cmake;\n"
            "                    fastchess is a single-Makefile build.",
            file=sys.stderr,
        )
        return None
    for label, path in [("baseline", baseline), ("tuned", tuned), ("book", book)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            if label != "book":
                print(
                    "  Build with `make` and copy the binary before "
                    "applying tuned weights:\n"
                    "    make && cp engine engine.baseline",
                    file=sys.stderr,
                )
            return None
    return gm


def build_cmd(gm: str, args: argparse.Namespace) -> list[str]:
    # Cutechess/fastchess flags — shared surface between the two.
    # Book format inferred from file suffix; PGN is what the demo ships.
    book_format = "epd" if args.book.suffix.lower() == ".epd" else "pgn"

    # `-repeat` plays each opening twice (once with each color) — halves
    # opening-choice bias per pair of games.
    cmd: list[str] = [
        gm,
        "-engine",
        f"cmd={args.baseline}",
        "name=baseline",
        "proto=uci",
        "-engine",
        f"cmd={args.tuned}",
        "name=tuned",
        "proto=uci",
        "-each",
        f"tc={args.tc}",
        "-openings",
        f"file={args.book}",
        f"format={book_format}",
        "order=random",
        "-sprt",
        f"elo0={args.elo0}",
        f"elo1={args.elo1}",
        f"alpha={args.alpha}",
        f"beta={args.beta}",
        "-games",
        str(args.max_games),
        "-concurrency",
        str(args.concurrency),
        "-repeat",
    ]
    # PGN output — useful for debugging pathological positions after a run.
    if args.pgn_out:
        cmd += ["-pgnout", str(args.pgn_out)]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Baseline engine binary (default: %(default)s)",
    )
    ap.add_argument(
        "--tuned",
        type=Path,
        default=DEFAULT_TUNED,
        help="Tuned engine binary (default: %(default)s)",
    )
    ap.add_argument(
        "--book",
        type=Path,
        default=DEFAULT_BOOK,
        help="Opening book (.pgn or .epd) (default: %(default)s)",
    )
    ap.add_argument(
        "--tc",
        default=DEFAULT_TC,
        help="Time control in cutechess format "
        "(e.g. '10+0.1' = 10s + 100ms/move) "
        "(default: %(default)s)",
    )
    ap.add_argument(
        "--elo0",
        type=float,
        default=DEFAULT_ELO0,
        help="H0 Elo bound — 'no gain' (default: %(default)s)",
    )
    ap.add_argument(
        "--elo1",
        type=float,
        default=DEFAULT_ELO1,
        help="H1 Elo bound — 'meaningful gain' (default: %(default)s)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Type I error rate (default: %(default)s)",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Type II error rate (default: %(default)s)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Parallel games (default: %(default)s)",
    )
    ap.add_argument(
        "--max-games",
        type=int,
        default=DEFAULT_MAX_GAMES,
        help="Upper bound on games; SPRT usually stops sooner (default: %(default)s)",
    )
    ap.add_argument(
        "--pgn-out",
        type=Path,
        default=None,
        help="Write played games as PGN to this path",
    )
    args = ap.parse_args()

    gm = check_prereqs(args.baseline, args.tuned, args.book)
    if gm is None:
        return 1

    cmd = build_cmd(gm, args)
    print(f"Using: {gm}", file=sys.stderr)
    print(f"Command: {' '.join(cmd)}", file=sys.stderr)
    print("", file=sys.stderr)

    # Stream output live — SPRT progress is the whole point of running
    # this, and letting it silently accumulate for hours is a UX disaster.
    try:
        result = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nInterrupted — no verdict.", file=sys.stderr)
        return 130
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
