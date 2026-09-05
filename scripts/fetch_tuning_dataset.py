#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""
Fetch and convert a Texel-style labeled tuning dataset into the format
`./engine tune` expects.

The engine reads one line per position: `<FEN>;<outcome>` where outcome
is `0`, `0.5`, or `1` from WHITE's perspective. The standard public
datasets (Zurichess quiet-labeled, Leorik's mirror, Ethereal noob) all
ship as EPD with a `c9` operation code carrying the outcome as a
"1-0" / "0-1" / "1/2-1/2" string. This script converts either a locally-
provided EPD file (--input) or a downloaded one (--url) into the tuner
format.

Usage examples:

  # Convert a local EPD:
  scripts/fetch_tuning_dataset.py --input quiet-labeled.epd

  # Download from a URL, convert, and cache the raw EPD:
  scripts/fetch_tuning_dataset.py --url https://.../quiet-labeled.epd

  # Then feed the result to the tuner:
  ./engine tune tests/data/quiet-labeled.txt pst 5000
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Best-effort default source. URLs rot; if this stops working, pass
# --url or --input pointing at any Texel-style labeled EPD. The
# lithander/Leorik repo has been a stable public mirror; the Zurichess
# original (bitbucket.org/zurichess/tuner) is archived and its Mercurial
# hosting has been down since 2020.
DEFAULT_URL = "https://raw.githubusercontent.com/lithander/Leorik/master/Leorik.Test/quiet-labeled.epd"

# EPD outcome-string → WHITE-perspective float, as the engine parses it.
OUTCOME_MAP = {
    "1-0": "1.0",
    "0-1": "0.0",
    "1/2-1/2": "0.5",
}


def download(url: str, target: Path) -> None:
    print(f"Fetching {url}\n     -> {target}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as resp:
        target.write_bytes(resp.read())
    print(f"     ({target.stat().st_size:,} bytes)", file=sys.stderr)


def parse_epd_line(line: str) -> tuple[str, str] | None:
    """Parse one EPD line → (fen, outcome_str). None if malformed / no c9.

    EPD layout:
        <placement> <side> <castling> <ep-square> [operations...];
    Standard EPD omits the halfmove/fullmove counters (unlike FEN);
    the engine's set_from_fen accepts them missing but we append
    "0 1" defaults so the FEN column is a well-formed standalone
    position for downstream tooling that expects the full form.

    Operations are semicolon-separated key/value pairs; we look for
    the `c9 "<outcome>"` opcode. Outcome strings must be one of
    "1-0", "0-1", or "1/2-1/2".
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) < 5:
        return None

    # Find the c9 opcode; capture its quoted value up to the trailing
    # semicolon. The whole operation set can wrap arbitrary tokens, so
    # scan linearly rather than assuming a fixed offset.
    outcome_raw: str | None = None
    for i, tok in enumerate(parts):
        if tok == "c9" and i + 1 < len(parts):
            # Strip surrounding quotes, semicolons, whitespace.
            outcome_raw = parts[i + 1].strip('";')
            break
    if outcome_raw is None:
        return None

    outcome = OUTCOME_MAP.get(outcome_raw)
    if outcome is None:
        return None

    # First 4 tokens are always the FEN placement fields in EPD.
    fen = " ".join(parts[:4]) + " 0 1"
    return fen, outcome


def convert(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Convert EPD file to tuner format. Returns (converted, skipped)."""
    converted = 0
    skipped = 0
    with input_path.open() as inp, output_path.open("w") as out:
        out.write(
            "# Converted from Texel EPD format via "
            "scripts/fetch_tuning_dataset.py\n"
            f"# Source: {input_path.name}\n"
        )
        for line in inp:
            result = parse_epd_line(line)
            if result is None:
                skipped += 1
                continue
            fen, outcome = result
            out.write(f"{fen};{outcome}\n")
            converted += 1
    return converted, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--url", default=DEFAULT_URL, help="Source URL (default: %(default)s)"
    )
    ap.add_argument(
        "--input", type=Path, help="Local EPD file; skips download when supplied"
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/quiet-labeled.txt"),
        help="Output path in FEN;outcome format (default: %(default)s)",
    )
    ap.add_argument(
        "--epd-cache",
        type=Path,
        default=Path("tests/data/.cache/quiet-labeled.epd"),
        help="Cache path for the downloaded raw EPD (default: %(default)s)",
    )
    args = ap.parse_args()

    # Source selection: explicit --input wins; otherwise use the URL
    # cache path (download if absent, reuse if present).
    if args.input:
        epd_source = args.input
        if not epd_source.exists():
            print(f"ERROR: --input file does not exist: {epd_source}", file=sys.stderr)
            return 1
    else:
        args.epd_cache.parent.mkdir(parents=True, exist_ok=True)
        if args.epd_cache.exists():
            print(
                f"Reusing cached EPD: {args.epd_cache} "
                f"({args.epd_cache.stat().st_size:,} bytes)",
                file=sys.stderr,
            )
        else:
            try:
                download(args.url, args.epd_cache)
            except urllib.error.HTTPError as e:
                print(f"ERROR: HTTP {e.code} from {args.url}", file=sys.stderr)
                print("URLs rot; try:", file=sys.stderr)
                print(
                    "  --input <local.epd> to convert a file you already have",
                    file=sys.stderr,
                )
                print(
                    "  --url <alt-url>     to point at a different mirror",
                    file=sys.stderr,
                )
                return 1
            except (urllib.error.URLError, TimeoutError) as e:
                print(
                    f"ERROR: network failure fetching {args.url}: {e}", file=sys.stderr
                )
                return 1
        epd_source = args.epd_cache

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted, skipped = convert(epd_source, args.output)
    print(
        f"Wrote {converted:,} positions to {args.output} (skipped {skipped:,})",
        file=sys.stderr,
    )

    if converted == 0:
        print(
            "ERROR: no positions converted — is the input EPD-format "
            'with `c9 "<outcome>"` opcodes?',
            file=sys.stderr,
        )
        return 1

    print("\nRun the tuner with:", file=sys.stderr)
    print(f"  ./engine tune {args.output} scalar", file=sys.stderr)
    print(f"  ./engine tune {args.output} pst 10000", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
