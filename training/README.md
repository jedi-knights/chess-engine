# NNUE training pipeline

Python trainer for the engine's HalfKP → 256 → 1 network. Produces
`.jnn1` files that the engine loads via the `EvalFile` UCI option.

## Requirements

Everything runs through `uv` — every script is a PEP 723 self-installing
script:

    uv --version   # 0.4+

If you don't have `uv`:

    curl -LsSf https://astral.sh/uv/install.sh | sh

The two test drivers have separate dependency sets so `test_encoding.py`
runs without downloading PyTorch:

- `test_encoding.py` → `python-chess` only (~2 MB, seconds)
- `test_export.py`   → `torch`, `numpy`, `python-chess` (~900 MB first run)
- `train.py`         → `torch`, `numpy`, `python-chess`

`uv` caches the environment, so second and subsequent runs are fast.

## Workflow

1. **Generate self-play data.**

   Use `scripts/gen_selfplay_data.py` to produce lines of the form:

       <fen>;<outcome_wdl>

   where `outcome_wdl` is the game result from WHITE's perspective
   (1.0 white win, 0.5 draw, 0.0 black win). The trainer rotates to
   side-to-move perspective internally. `<fen>|<centipawn>` is also
   accepted for score-labeled data (already STM perspective).

2. **Train.**

       training/train.py --data data/selfplay.txt --out net.jnn1

   Flags:

   - `--epochs 20` — training epochs
   - `--batch 1024` — batch size
   - `--lr 1e-3` — AdamW peak learning rate
   - `--weight-decay 1e-4` — AdamW decoupled weight decay (0 disables)
   - `--warmup-frac 0.05` — fraction of total epochs spent in linear warmup; cosine decay follows
   - `--grad-clip 1.0` — max L2 norm for gradient clipping (0 disables)
   - `--score-scale 400` — WDL sigmoid divisor (Stockfish uses 400)
   - `--val-frac 0.05` — fraction held out for validation
   - `--seed 42` — RNG seed for split + init

3. **Point the engine at the trained network via UCI.**

       ./engine
       setoption name UseNNUE value true
       setoption name EvalFile value /abs/path/to/net.jnn1
       ucinewgame
       position startpos
       go depth 8

## Architecture

- **HalfKP** feature layout matching `src/nnue.cpp` — 41,024 features
  per perspective. Feature encoding lives in `encoding.py` and is
  pinned against the C++ runtime by `test_encoding.py`.
- **EmbeddingBag(mode='sum')** for the feature transformer —
  O(active features) per position (~30 for typical middlegames)
  rather than O(41024) that a dense linear layer would cost.
- **Clipped ReLU** activation in `[0, 1]` (float), corresponding to
  the C++ runtime's `[0, 127]` int32 clamp after quantization.
- **MSE loss** on `sigmoid(pred / 400)` vs. target in WDL space —
  standard Stockfish loss.

## Quantization

- `SCALE_FEATURES = 128` (float weight × 128 → int16)
- `SCALE_OUTPUT   = 64`  (float weight × 64  → int16)
- Product = 8192 — matches the divisor in `nnue::evaluate()`.

The exporter warns on int16 saturation (stderr). If it fires,
either lower the initialization magnitude, regularize weights, or
add per-layer weight clipping to `training/model.py`. Saturation is
information-lossy — the model as loaded will differ from the model
as trained.

## Learning from played games

If you have `.pgn` files of actual games (lichess downloads, cutechess
match PGNs, your own GUI exports), you can feed them back into the
corpus and retrain. Pipeline:

    # 1. Ingest new PGNs into a growing FEN;outcome corpus (idempotent
    #    via a .processed sidecar log; safe to re-run on the same dir).
    scripts/pgn_to_fens.py --input ~/games/ \
        --corpus data/played_games.txt

    # 2. Re-label with Stockfish — stronger teacher than the game
    #    outcome alone.
    scripts/stockfish_label.py --input data/played_games.txt \
        --output data/played_games_sf.txt --depth 10

    # 3. Train on the labeled corpus.
    training/train.py --data data/played_games_sf.txt \
        --out /tmp/candidate.jnn1 --epochs 40

`scripts/learning_loop.py` wires those three steps together with
optional SPRT validation and auto-promote (see the engine's root
`README.md` section "Teaching the net from your own games"). It is
deliberately conservative — never overwrites `nets/default.jnn1`
unless both `--sprt` and `--promote` are passed *and* the SPRT accepts
H1.

**The engine never updates weights during gameplay.** All learning is
offline, from captured PGNs through this pipeline, between release
builds.

## Testing

    training/test_encoding.py    # fast, no PyTorch
    training/test_export.py      # full pipeline, needs PyTorch
    training/test_train.py       # optimizer/scheduler/clip wiring, needs PyTorch

Both are self-contained runners — no `pytest` required, though
`pytest training/` also discovers and runs them since they follow
`test_*.py` / `test_*` naming.

The C++ side has its own coverage of the binary format in
`tests/test_nnue.cpp` (`NNUE load: succeeds on well-formed zeroed
network`, `NNUE save/load round-trip preserves the network`). A file
written by the Python exporter and loaded by the C++ engine has been
verified end-to-end during the training-pipeline PR (#43).

## Files

    training/
    ├── README.md            -- this file
    ├── encoding.py          -- HalfKP feature encoding (chess only)
    ├── dataset.py           -- torch Dataset + collate_fn
    ├── model.py             -- NNUE nn.Module
    ├── export.py            -- float → int16 + JNN1 writer
    ├── train.py             -- training loop entry point (AdamW + cosine LR + grad clip)
    ├── test_encoding.py     -- encoding tests (lightweight)
    ├── test_export.py       -- export + training tests
    └── test_train.py        -- optimizer / scheduler / grad clip tests

## Limitations

- No GPU-specific optimizations (works on CPU or CUDA if available).
- No incremental training / checkpointing — every run trains from
  scratch. Add if you plan multi-day training runs.
- Fixed HalfKP → 256 → 1 architecture. Changing hidden size means
  changing the C++ runtime constants too (`nnue_types.h`).
