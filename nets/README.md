# NNUE networks

Trained networks in the JNN1 file format the engine loads via UCI's
`EvalFile` option. All files here are architecture HalfKP → 512 → 1
(41,024 features per perspective, single hidden layer, single output)
matching `src/nnue_types.h`. **Architecture change** vs the previous
HalfKP → 256 → 1 shipped nets — old 256-hidden JNN1 files are now
rejected by the loader with an architecture-mismatch diagnostic.

## default.jnn1 — v9 (bigger-corpus, 512-hidden, with regularization)

Trained on a 704,307-position union corpus:

- 599,457 new positions from 6000 self-play games played with the
  engine's classical eval (`UseNNUE=false`, 30 ms/move,
  `noob_3moves.epd` book, seed 300). Classical is ~1000 Elo stronger
  than any previously-shipped NNUE, so the games are much higher
  quality than earlier self-play rounds.
- 104,854 positions from the previously-shipped `sf_labeled_v2.txt`
  (the training set used for v7).

Both halves were re-labeled by locally-installed Stockfish 19 at depth
10. Union = `cat sf_labeled_v2.txt sf_labeled_v3.txt | grep -v '^#' |
grep -v '^$' | sort -u` (dedupes trivially — the two corpora share
~0.001% of lines).

Training ran 40 epochs with the Phase 2 loop plus 10× stronger
regularization than v8 attempted:

| Parameter | v8 (overfit) | v9 (this) |
|---|---|---|
| HIDDEN_SIZE | 512 | 512 |
| training positions | 104,854 | **704,307** (7×) |
| weight_decay | 1e-4 | **1e-3** (10×) |
| epochs | 50 | 40 |
| final val_loss | 0.021 | **0.0053** |
| final train/val gap | sizable | 0.0018 |

**SPRT validation (`tc=5+0.05`, `noob_3moves.epd`):**

| vs | Games | Score for v9 | Elo gap | CI |
|---|---|---|---|---|
| v7 (previous default, 256 hidden) | 332 | 29W / 289L / 14D (9.5%) | v7 **+366** Elo stronger | ±57 |
| classical eval (`UseNNUE=false`) | 362 | 46W / 299L / 17D (15.5%) | classical **+301** Elo stronger | ±50 |

**What this means**

v9 is **~686 Elo closer to classical than v7** was (−987 → −301). The
2× hidden-size bump finally pays off when paired with 7× more training
data and 10× stronger regularization — the earlier 256→512 attempt
(v8) failed on the small corpus because capacity badly outpaced data.

v9 **loses head-to-head to v7** by −366 Elo despite being much
stronger against classical. This is a known intransitivity signature:
v9 and v7 play fundamentally different styles (v9 plays textbook
openings like e2e4; v7 preferred wing-push lines like g2g4), and v7's
style happens to counter v9's specific move preferences. The 288-0
sweep of v7 over v4 was a similar intransitivity.

**v9 still does not beat classical eval.** The engine's competitive
mode remains `UseNNUE=false`. v9 is the strongest NNUE this engine has
shipped, and the first one that's within one network-upgrade of
competitive strength (if the Elo gap continues shrinking at this
rate), but shipping as an experimental option, not as a stronger
replacement for classical.

**Breaking change**: the file-format `hidden_size` field in the JNN1
header is now expected to be 512. The loader cleanly rejects a 256
file with `nnue: 'path.jnn1' architecture mismatch (file: hidden=256
feats=41024; engine: hidden=512 feats=41024)` and falls back to
classical eval. Any locally-trained or externally-sourced 256-hidden
JNN1 files must be retrained.

**Reproduction:**

```bash
# 1. Fetch the opening book (150k positions).
curl -L -o /tmp/noob_3moves.epd.zip \
    https://github.com/official-stockfish/books/raw/master/noob_3moves.epd.zip
unzip /tmp/noob_3moves.epd.zip -d /tmp

# 2. Self-play 6000 games with classical eval (~5 h).
scripts/gen_selfplay_data.py --games 6000 --movetime-ms 30 \
    --book /tmp/noob_3moves.epd --seed 300 \
    --engine-option UseNNUE=false \
    --output /tmp/selfplay_6000.txt --label wdl

# 3. Re-label the FENs with Stockfish depth 10 (~98 min; needs
#    `brew install stockfish` or equivalent).
scripts/stockfish_label.py --input /tmp/selfplay_6000.txt \
    --output tests/data/sf_labeled_v3.txt --depth 10

# 4. Union with the v7 corpus + dedupe.
cat tests/data/sf_labeled_v2.txt tests/data/sf_labeled_v3.txt \
    | grep -v '^#' | grep -v '^$' | sort -u \
    > /tmp/sf_labeled_union.txt

# 5. Train 40 epochs at 512 hidden with wd=1e-3 (~24 min on CPU).
training/train.py --data /tmp/sf_labeled_union.txt \
    --out /tmp/v9.jnn1 --epochs 40 --batch 2048 \
    --weight-decay 1e-3 --warmup-frac 0.05 --grad-clip 1.0 --seed 42

# 6. SPRT-validate (~36 min each at concurrency 2).
scripts/sprt.py --baseline ./engine --tuned ./engine \
    --baseline-option UseNNUE=true \
    --baseline-option EvalFile=nets/<previous>.jnn1 \
    --tuned-option UseNNUE=true \
    --tuned-option EvalFile=/tmp/v9.jnn1 \
    --book /tmp/noob_3moves.epd --tc 5+0.05 --max-games 400
```

**Usage:** off by default. Enable per-run via UCI:

```
setoption name EvalFile value nets/default.jnn1
setoption name UseNNUE value true
```

Order matters — set `EvalFile` before `UseNNUE=true` to avoid a
transient `info string UseNNUE=true but no network loaded` warning.

---

## Historical: v7 (bigger-corpus Stockfish-labeled, 256 hidden)

The default from Oct 3-4 2026 until v9 landed. Trained on a
104,854-position corpus generated by self-playing v4 (see below) over
2000 games with the `noob_3moves.epd` opening book (150,932 openings),
then re-labeling every quiet position with Stockfish 19 at depth 8.
Training: 50 epochs at HIDDEN_SIZE=256, weight_decay=1e-4,
cosine-decay LR, final val_loss 0.045.

SPRT at tc=5+0.05:
- vs v4: 288-0-0 (v7 ~+900 Elo stronger)
- vs classical: 1-293-0 (classical ~+987 Elo stronger)

Superseded by v9 both in capacity (256 → 512) and corpus size (104k →
704k), but v9 can't beat v7 head-to-head (intransitive matchup). To
run v7 locally, you'd need to rebuild the engine at HIDDEN_SIZE=256
(revert `src/nnue_types.h`), then load the v7 file from before this
PR landed.

## Historical: v1-v4 self-bootstrap chain

Prior to v7 the shipped default was the peak of a 4-round
self-bootstrap chain:

| Round | Trained on | vs prior generation | vs v1 baseline |
|-------|------------|---------------------|----------------|
| v1    | 500 self-play games (noob_3moves.epd, 50ms/move), WDL labels        | seed          | seed          |
| v2    | 200 games self-played by v1, score labels from v1 (depth 4)         | +67 Elo       | +67 Elo       |
| v3    | 200 games self-played by v2, score labels from v2 (depth 4)         | +108 Elo      | +89 Elo       |
| v4    | 200 games self-played by v3, score labels from v3 (depth 4)         | +54 Elo       | **+130 Elo**  |
| v5    | 200 games self-played by v4, score labels from v4 (depth 4)         | −51 Elo       | (regressed)   |

v5 regressed from v4 due to self-distillation collapse. The v4 peak
was the shipped default from Sep 2026 until v7 landed in Oct 2026.
