# NNUE networks

Trained networks in the JNN1 file format the engine loads via UCI's
`EvalFile` option. All files here are architecture HalfKP → 256 → 1
(41,024 features per perspective, single hidden layer, single output)
matching `src/nnue_types.h`.

## default.jnn1

Trained as the peak of a 4-round self-bootstrap chain:

| Round | Trained on | vs prior generation | vs v1 baseline |
|-------|------------|---------------------|----------------|
| v1    | 500 self-play games (noob_3moves.epd, 50ms/move), WDL labels        | seed          | seed          |
| v2    | 200 games self-played by v1, score labels from v1 (depth 4)         | +67 Elo       | +67 Elo       |
| v3    | 200 games self-played by v2, score labels from v2 (depth 4)         | +108 Elo      | +89 Elo       |
| v4    | 200 games self-played by v3, score labels from v3 (depth 4)         | +54 Elo       | **+130 Elo**  |
| v5    | 200 games self-played by v4, score labels from v4 (depth 4)         | −51 Elo       | (regressed)   |

V4 is the peak — v5 measurably regresses (self-distillation collapse:
compounded label bias eventually overwhelms smoothing benefit).

**Elo picture** (0.5+0.05 TC, `noob_3moves.epd` test book, SPRT-decided):

- +130 Elo vs the WDL-trained seed (v1)
- Still loses ~200-0 to the classical eval — the NNUE is a learned
  approximation of a weak teacher (classical eval at depth 4 during
  bootstrap), so it can't cross the classical ceiling in this pipeline

**Reproduction:**

```bash
# 1. Fetch a large opening book (~150k openings).
curl -L -o /tmp/noob_3moves.epd.zip \
    https://github.com/official-stockfish/books/raw/master/noob_3moves.epd.zip
unzip /tmp/noob_3moves.epd.zip -d /tmp

# 2. Seed WDL run (~25 minutes).
scripts/gen_selfplay_data.py --games 500 --movetime-ms 50 --label wdl \
    --book /tmp/noob_3moves.epd --seed 200 \
    --output /tmp/wdl_500.txt
training/train.py --data /tmp/wdl_500.txt --out /tmp/v1.jnn1 \
    --epochs 50 --batch 512 --seed 42

# 3. Bootstrap rounds v2..v4 (~15 minutes each — stop when SPRT vs
#    the prior round shows regression).
for i in 2 3 4; do
    prev=$((i-1))
    scripts/gen_selfplay_data.py --games 200 --movetime-ms 50 \
        --label score --score-depth 4 \
        --book /tmp/noob_3moves.epd --seed $((300+i)) \
        --engine-option UseNNUE=true \
        --engine-option EvalFile=/tmp/v${prev}.jnn1 \
        --output /tmp/bootstrap_v${prev}.txt
    training/train.py --data /tmp/bootstrap_v${prev}.txt \
        --out /tmp/v${i}.jnn1 --epochs 50 --batch 512 --seed 42
done
```

**Usage:** off by default. Enable per-run via UCI:

```
setoption name EvalFile value nets/default.jnn1
setoption name UseNNUE value true
```

Order matters — set `EvalFile` before `UseNNUE=true` to avoid a
transient `info string UseNNUE=true but no network loaded` warning.
