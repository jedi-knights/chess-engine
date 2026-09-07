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
  scripts/sprt.py --baseline engine.baseline --tuned engine \\
                  --tc 10+0.1 --elo0 0 --elo1 5

  # SPRT stops early — a positive result on a genuine +5 Elo change
  # typically converges in 1000-3000 games; a null result may take
  # 5000-10000. Time-control × games determines wall time.

Per-engine UCI options (--baseline-option / --tuned-option, repeatable)
support A/B'ing UCI-toggled features like NNUE against the classical
eval from the same binary:

  scripts/sprt.py --baseline ./engine --tuned ./engine \\
                  --tuned-option UseNNUE=true \\
                  --tuned-option EvalFile=/path/to/net.jnn1 \\
                  --tc 10+0.1 --elo0 -5 --elo1 5

Cutechess handles the SPRT statistic natively (Wald-based, LLR test).
This script's role is preflight (binaries exist, book exists, FD limit
high enough), building the correct command line, and passing results
back through exit codes.
"""

import argparse
import resource
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

# fastchess opens ~14 file descriptors per concurrent game pair (stdin,
# stdout, stderr for each of the two engines, plus PGN, log, and pipe
# bookkeeping). The base process itself needs headroom for stdout,
# stderr, opening book, PGN output file, and internal logging.
# Empirically at concurrency=4 fastchess reports needing 62 FDs; the
# formula below covers that with a safety margin so nothing surprises
# a run at high concurrency.
FD_BASE = 32
FD_PER_CONCURRENT_GAME = 16


def find_gamemanager() -> str | None:
    """Prefer fastchess (faster, drop-in replacement); fall back to cutechess-cli."""
    for name in ("fastchess", "cutechess-cli"):
        found = shutil.which(name)
        if found:
            return found
    return None


def required_fds(concurrency: int) -> int:
    """File descriptors fastchess needs for the given concurrency.

    Formula: FD_BASE (process-wide overhead) + FD_PER_CONCURRENT_GAME per
    parallel game pair. Pins against the observed fastchess need of 62
    at concurrency=4 with headroom, so bumping concurrency doesn't hit
    the default macOS soft limit (256) mid-run.
    """
    return FD_BASE + FD_PER_CONCURRENT_GAME * max(1, concurrency)


FD_TARGET_MIN = 65536


def ensure_fd_limit(needed: int) -> str | None:
    """Set RLIMIT_NOFILE soft limit to a finite value >= `needed`.

    macOS quirk: `ulimit -n unlimited` (the default in many shells) exposes
    RLIM_INFINITY to processes, and fastchess's own FD preflight fails
    against that value even though the kernel would allow the operation.
    Always set a specific finite cap (min FD_TARGET_MIN, or `needed` if
    higher) so fastchess sees a concrete number it can compare against.

    Returns None on success, or an error string on failure so the caller
    can print + exit.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = max(needed, FD_TARGET_MIN)
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    # Skip only if soft is already a specific value at or above target.
    if soft != resource.RLIM_INFINITY and soft >= target:
        return None
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError) as e:
        return (
            f"cannot set RLIMIT_NOFILE to {target} "
            f"(soft={soft}, hard={hard}): {e}. "
            f"Run with lower --concurrency, or `ulimit -n {target}` manually."
        )
    if target < needed:
        return (
            f"RLIMIT_NOFILE hard limit is {hard}; need {needed} for "
            f"concurrency={concurrency_from_needed(needed)}. "
            f"Lower --concurrency or raise the hard limit via "
            f"/etc/security/limits.conf (Linux) or launchctl (macOS)."
        )
    return None


def concurrency_from_needed(needed: int) -> int:
    """Inverse of `required_fds` — used in error messages only."""
    return max(1, (needed - FD_BASE) // FD_PER_CONCURRENT_GAME)


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


def _engine_block(binary: Path, name: str, options: list[str]) -> list[str]:
    """One `-engine ...` block, with any per-engine `option.KEY=VALUE` tokens.

    fastchess and cutechess-cli both parse `option.KEY=VALUE` tokens inside
    an engine block as UCI options sent to that engine only. Options must
    live between this engine's block and the next `-engine` / `-each`
    boundary — hence the block is built as a contiguous list.
    """
    block = ["-engine", f"cmd={binary}", f"name={name}", "proto=uci"]
    for opt in options:
        block.append(f"option.{opt}")
    return block


def build_cmd(gm: str, args: argparse.Namespace) -> list[str]:
    # Cutechess/fastchess flags — shared surface between the two.
    # Book format inferred from file suffix; PGN is what the demo ships.
    book_format = "epd" if args.book.suffix.lower() == ".epd" else "pgn"

    # `-repeat` plays each opening twice (once with each color) — halves
    # opening-choice bias per pair of games.
    cmd: list[str] = [gm]
    cmd += _engine_block(args.baseline, "baseline", args.baseline_option or [])
    cmd += _engine_block(args.tuned, "tuned", args.tuned_option or [])
    cmd += [
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
        # fastchess semantics: `-rounds N -repeat` = N rounds × 2
        # games/round = 2N total games with colors swapped per opening.
        # Cutechess-cli accepts the same form. Previously used
        # `-games N` which fastchess interprets as "games per round"
        # (default 2) — capped total games at 4 regardless of N.
        "-rounds",
        str(max(1, args.max_games // 2)),
        "-concurrency",
        str(args.concurrency),
        "-repeat",
    ]
    # PGN output — useful for debugging pathological positions after a
    # run. fastchess requires the `file=` key-value form; cutechess-cli
    # accepts it too (bare `<path>` is a cutechess-only shorthand).
    if args.pgn_out:
        cmd += ["-pgnout", f"file={args.pgn_out}"]
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
    ap.add_argument(
        "--baseline-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="UCI option to send to the baseline engine only "
        "(repeatable, e.g. --baseline-option Hash=32)",
    )
    ap.add_argument(
        "--tuned-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="UCI option to send to the tuned engine only "
        "(repeatable, e.g. --tuned-option UseNNUE=true "
        "--tuned-option EvalFile=/path/to/net.jnn1)",
    )
    args = ap.parse_args()

    gm = check_prereqs(args.baseline, args.tuned, args.book)
    if gm is None:
        return 1

    fd_err = ensure_fd_limit(required_fds(args.concurrency))
    if fd_err is not None:
        print(f"ERROR: {fd_err}", file=sys.stderr)
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
