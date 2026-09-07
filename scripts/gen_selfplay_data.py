#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chess>=1.10",
# ]
# ///
"""
Self-play data generator for Texel/SPSA tuning and NNUE training.
Plays N games against the engine itself, extracts quiet positions
from each game, and writes labeled training data.

Two label modes:

  --label wdl    (default) Label each position with the eventual game
                 outcome from WHITE's perspective. Output format:
                 `FEN;outcome` where outcome ∈ {1.0, 0.5, 0.0}. Feeds
                 `./engine tune` for Texel/SPSA weight tuning.

  --label score  Label each position with the engine's classical eval
                 at --score-depth from the SIDE-TO-MOVE perspective.
                 Output format: `FEN|centipawns`. Feeds
                 `training/train.py` for NNUE training. Higher signal
                 density than WDL (per-position, not per-game) but
                 costs one engine.analyse per quiet position.

Pipeline per game:
  1. Pick a random opening from --book (PGN or EPD).
  2. Play both sides via UCI with the SAME engine (deterministic
     self-play; different openings drive the variety).
  3. Terminate on checkmate / stalemate / 50-move / repetition /
     insufficient material. Cap at --max-ply plies as a runaway
     guard (long games ruled a draw).
  4. Label every recorded position — WDL from the game outcome, or
     score from a per-position engine.analyse call.
  5. Filter: drop the first --min-ply plies (opening bias) and any
     non-quiet position (side-to-move in check, or the last move
     was a capture). Quiet filtering matters — Texel loss compares
     against a static eval which is noisy on tactical positions;
     the same holds for NNUE training.

Prerequisites: Python 3.11+, python-chess >= 1.10 (auto-installed
via uv PEP 723 header). Cutechess NOT required — this script drives
the engine directly.

Typical usage:

  # Build the engine first.
  make

  # WDL data for Texel tuning (~5000 quiet positions, ~5-10 min):
  scripts/gen_selfplay_data.py --games 50
  ./engine tune tests/data/selfplay.txt all 5000

  # Score data for NNUE training (per-position eval, ~2x slower):
  scripts/gen_selfplay_data.py --games 50 --label score --score-depth 8 \\
                               --output tests/data/selfplay_score.txt
  training/train.py --data tests/data/selfplay_score.txt --out net.jnn1

Diversity note: with a small opening book (like the checked-in
tests/data/openings_demo.pgn — 16 openings) and a deterministic
engine, expect the same book opening to produce nearly-identical
games across --games invocations. TT state carries across games so
some drift accumulates, but for a diverse dataset use a bigger book
(noob_3moves.epd and friends ship 5000+ openings) or vary --seed.
"""

import argparse
import random
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn


DEFAULT_ENGINE = Path("./engine")
DEFAULT_BOOK = Path("tests/data/openings_demo.pgn")
DEFAULT_OUTPUT = Path("tests/data/selfplay.txt")
DEFAULT_GAMES = 100
DEFAULT_MOVETIME_MS = 100
DEFAULT_MIN_PLY = 8  # skip opening plies (bias)
DEFAULT_MAX_PLY = 200  # runaway guard
DEFAULT_SEED = 42
DEFAULT_LABEL = "wdl"
DEFAULT_SCORE_DEPTH = 8

OUTCOME_MAP = {
    "1-0": 1.0,
    "0-1": 0.0,
    "1/2-1/2": 0.5,
}

# Sentinel centipawn value for mate positions. Bounded so downstream
# sigmoid math (WDL loss on sigmoid(cp/400)) stays numerically stable;
# large enough to be unambiguous vs any plausible material eval.
MATE_SCORE_CP = 30000

# Threshold above which a raw Cp value is interpreted as an engine
# mate signal rather than a genuine evaluation. This engine emits
# `score cp <MATE_SCORE - ply>` instead of the UCI-standard
# `score mate <plies>` (see src/uci.cpp:250), so python-chess reads
# mate scores as very large Cp values. Any |cp| ≥ this threshold is
# clamped to ±MATE_SCORE_CP; the threshold sits well above any
# plausible non-mate eval and just below the engine's mate range
# (which starts at MATE_SCORE - 1000 = 99000).
ENGINE_MATE_CP_THRESHOLD = 30000


def format_score_label(pov_score: chess.engine.PovScore) -> str:
    """Format a python-chess PovScore as a centipawn label string.

    The score is taken from the side-to-move perspective (``.relative``)
    to match the trainer's expected input format. Mate scores are
    clamped to ``±MATE_SCORE_CP`` — raw mate distance is not a
    comparable centipawn value and would break the sigmoid loss.

    Handles both UCI mate representations: proper ``score mate N`` and
    the engine's current ``score cp (MATE_SCORE - N)`` (see
    src/uci.cpp:250; if the engine is fixed to emit standard mate
    scores, the ``is_mate`` branch takes over and the Cp-clamp branch
    becomes a no-op).
    """
    score = pov_score.relative
    if score.is_mate():
        mate_dist = score.mate()
        return str(MATE_SCORE_CP if mate_dist > 0 else -MATE_SCORE_CP)
    cp = score.score()
    if cp >= ENGINE_MATE_CP_THRESHOLD:
        return str(MATE_SCORE_CP)
    if cp <= -ENGINE_MATE_CP_THRESHOLD:
        return str(-MATE_SCORE_CP)
    return str(cp)


def load_openings(book_path: Path) -> list[chess.Board]:
    """Load starting positions from a PGN or EPD opening book.

    PGN files are played through to the game's final position (typically
    a few opening moves); EPD files are read one FEN per line, each an
    independent starting position. Malformed lines are silently skipped
    so a single bad entry doesn't kill a large book.

    Args:
        book_path: Path to the opening book. Format inferred from the
            file suffix (``.pgn`` or ``.epd``).

    Returns:
        A list of Board objects — each ready to be played from.
    """
    openings: list[chess.Board] = []
    if book_path.suffix.lower() == ".epd":
        for line in book_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                # EPD lines may have trailing operations; the first
                # semicolon-separated token is the FEN prefix.
                fen = line.split(";", 1)[0]
                openings.append(chess.Board(fen))
            except (ValueError, IndexError):
                continue
    else:  # PGN
        with book_path.open() as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                openings.append(game.end().board())
    return openings


def is_quiet(board: chess.Board) -> bool:
    """Report whether a position is quiet enough for Texel labeling.

    Quiet = side to move NOT in check AND the last move (if any) was
    not a capture. Texel loss compares an outcome label against the
    engine's static eval, which is unreliable mid-exchange or when
    forced responses distort the eval. Filtering out non-quiet
    positions is standard practice — Zurichess quiet-labeled and
    Ethereal noob-quiet both do the same thing.

    Args:
        board: Position to check. Mutated transiently (last move popped
            and re-pushed) but restored before return.

    Returns:
        True if the position is quiet, False otherwise.
    """
    if board.is_check():
        return False
    if board.move_stack:
        last = board.pop()
        was_capture = board.is_capture(last)
        board.push(last)  # restore
        if was_capture:
            return False
    return True


def play_game(
    engine: chess.engine.SimpleEngine,
    starting_board: chess.Board,
    movetime_ms: int,
    max_ply: int,
) -> tuple[list[chess.Board], float]:
    """Play a single self-play game from the given starting position.

    Both sides are the same engine binary — diversity comes from
    different openings. Games terminate on any standard game-over
    condition (mate, stalemate, 50-move, repetition, insufficient
    material) or on hitting ``max_ply`` (ruled a draw).

    Args:
        engine: An open UCI engine subprocess.
        starting_board: Board to play from. Not mutated.
        movetime_ms: Per-move time budget passed to the engine.
        max_ply: Cap on plies played this game.

    Returns:
        A tuple ``(positions, outcome)`` where ``positions`` is one
        Board snapshot per ply played (the position the OPPONENT sees
        after our move — matches Texel labeling convention) and
        ``outcome`` is 1.0/0.5/0.0 from WHITE's perspective.
    """
    board = starting_board.copy()
    positions: list[chess.Board] = []
    start_ply = board.ply()

    limit = chess.engine.Limit(time=movetime_ms / 1000.0)
    while not board.is_game_over(claim_draw=True):
        if board.ply() - start_ply >= max_ply:
            break
        try:
            result = engine.play(board, limit)
        except chess.engine.EngineError as e:
            print(f"  engine error mid-game: {e}", file=sys.stderr)
            break
        if result.move is None:
            break
        board.push(result.move)
        positions.append(board.copy())

    outcome_str = board.result(claim_draw=True)
    # A game that hit --max-ply produces "*" from board.result(); rule
    # it a draw so no game's positions get discarded.
    outcome = OUTCOME_MAP.get(outcome_str, 0.5)
    return positions, outcome


def main() -> int:
    """Parse CLI args, drive self-play, write labeled positions.

    Returns:
        0 on success, 1 on preflight failure or empty output.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--engine",
        type=Path,
        default=DEFAULT_ENGINE,
        help="Engine binary (default: %(default)s)",
    )
    ap.add_argument(
        "--book",
        type=Path,
        default=DEFAULT_BOOK,
        help="Opening book, .pgn or .epd (default: %(default)s)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path (default: %(default)s)",
    )
    ap.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
        help="Number of games to play (default: %(default)s)",
    )
    ap.add_argument(
        "--movetime-ms",
        type=int,
        default=DEFAULT_MOVETIME_MS,
        help="Per-move time budget in ms (default: %(default)s)",
    )
    ap.add_argument(
        "--min-ply",
        type=int,
        default=DEFAULT_MIN_PLY,
        help="Skip positions this many plies into the game "
        "or shallower — reduces opening bias "
        "(default: %(default)s)",
    )
    ap.add_argument(
        "--max-ply",
        type=int,
        default=DEFAULT_MAX_PLY,
        help="Cap plies per game; games hitting this are "
        "labeled as draws (default: %(default)s)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for opening selection (default: %(default)s)",
    )
    ap.add_argument(
        "--label",
        choices=["wdl", "score"],
        default=DEFAULT_LABEL,
        help="Label kind: 'wdl' writes 'FEN;outcome' for Texel tuning "
        "(game outcome from WHITE's perspective); 'score' writes "
        "'FEN|centipawns' for NNUE training (classical eval from the "
        "side-to-move perspective, at --score-depth) "
        "(default: %(default)s)",
    )
    ap.add_argument(
        "--score-depth",
        type=int,
        default=DEFAULT_SCORE_DEPTH,
        help="Search depth for classical-eval labels when --label=score. "
        "Higher = better quality per position but linearly slower. "
        "Ignored when --label=wdl (default: %(default)s)",
    )
    args = ap.parse_args()

    if not args.engine.exists():
        print(
            f"ERROR: engine binary not found: {args.engine}\n"
            f"  Build with `make` first.",
            file=sys.stderr,
        )
        return 1
    if not args.book.exists():
        print(f"ERROR: opening book not found: {args.book}", file=sys.stderr)
        return 1

    openings = load_openings(args.book)
    if not openings:
        print(f"ERROR: no openings loaded from {args.book}", file=sys.stderr)
        return 1
    print(
        f"Loaded {len(openings)} opening position(s) from {args.book}", file=sys.stderr
    )

    # Resolve to absolute path — `str(Path("./engine"))` normalizes to
    # `"engine"`, which subprocess then looks for on PATH instead of cwd.
    engine_abs = args.engine.resolve()
    print(f"Spawning engine: {engine_abs}", file=sys.stderr)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_abs))
    except (chess.engine.EngineError, OSError) as e:
        print(f"ERROR: failed to start engine: {e}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    outcome_counts = {1.0: 0, 0.5: 0, 0.0: 0}

    score_limit = chess.engine.Limit(depth=args.score_depth)

    try:
        with args.output.open("w") as out:
            out.write(
                f"# Self-play data — {args.games} games, {args.movetime_ms}ms/move\n"
            )
            out.write(f"# Engine: {args.engine}\n")
            out.write(
                f"# Opening book: {args.book} "
                f"({len(openings)} positions, seed={args.seed})\n"
            )
            if args.label == "score":
                out.write(
                    f"# Label: score (classical eval, depth={args.score_depth}) "
                    f"— format: FEN|centipawns (STM perspective)\n"
                )
            else:
                out.write("# Label: wdl (game outcome) — format: FEN;outcome\n")

            for i in range(args.games):
                # Reset TT between games so cross-game state doesn't
                # bias which side wins.
                try:
                    engine.protocol.send_line("ucinewgame")  # type: ignore[attr-defined]
                except Exception:
                    pass  # non-critical; best-effort

                opening = rng.choice(openings)
                positions, outcome = play_game(
                    engine,
                    opening,
                    args.movetime_ms,
                    args.max_ply,
                )
                outcome_counts[outcome] += 1

                # Absolute-ply cutoff (min_ply is measured from THIS
                # game's opening depth, not from move 1, so books
                # with deeper openings don't over-filter).
                cutoff = opening.ply() + args.min_ply
                for pos in positions:
                    if pos.ply() <= cutoff or not is_quiet(pos):
                        skipped += 1
                        continue
                    if args.label == "score":
                        try:
                            info = engine.analyse(pos, score_limit)
                        except chess.engine.EngineError as e:
                            print(
                                f"  engine analyse failed at {pos.fen()}: {e}",
                                file=sys.stderr,
                            )
                            skipped += 1
                            continue
                        label = format_score_label(info["score"])
                        out.write(f"{pos.fen()}|{label}\n")
                    else:
                        out.write(f"{pos.fen()};{outcome}\n")
                    written += 1

                if (i + 1) % 10 == 0 or i + 1 == args.games:
                    print(
                        f"  game {i + 1:4d}/{args.games}: "
                        f"{written:,} positions written "
                        f"(W={outcome_counts[1.0]} "
                        f"D={outcome_counts[0.5]} "
                        f"L={outcome_counts[0.0]})",
                        file=sys.stderr,
                    )
    finally:
        engine.quit()

    print(
        f"\nWrote {written:,} labeled quiet positions to {args.output}", file=sys.stderr
    )
    print(f"Skipped {skipped:,} (opening or non-quiet)", file=sys.stderr)
    if written == 0:
        print(
            "ERROR: no positions written — check --min-ply or increase --games",
            file=sys.stderr,
        )
        return 1

    if args.label == "score":
        print("\nFeed to the NNUE trainer with:", file=sys.stderr)
        print(
            f"  training/train.py --data {args.output} --out net.jnn1 --epochs 20",
            file=sys.stderr,
        )
    else:
        print("\nFeed to the tuner with:", file=sys.stderr)
        print(f"  ./engine tune {args.output} scalar", file=sys.stderr)
        print(f"  ./engine tune {args.output} pst 10000", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
