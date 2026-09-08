# chess-engine

A C++20 chess engine (bitboard movegen + UCI). Author: single dev, hobby project.

## Build & test

```
make            # release build (-O3 -march=native)         → ./engine
make debug      # -O0 -g + ASan/UBSan                        → ./engine
make test       # compile + run doctest suite under ASan     → ./tests/run
make perft      # ./engine perft 5 (correctness gate)
make perft-cert # ./engine perft 6 (~8B nodes, ~10 min — offline certification)
make run        # ./engine  (UCI stdin loop)
make clean
```

Every change that touches movegen or `Position` must pass **both** `make test` and `make perft` at the highest depth that currently works. Add tests when adding behavior; extend the perft target depth as movegen milestones complete.

## Layout

```
src/                engine sources
  types.h           Bitboard/Move/Piece/Square + encoding helpers
  bitboard.[h|cpp]  popcount, lsb, pretty
  attacks.[h|cpp]   precomputed leaper attack tables (knight/king/pawn)
  position.[h|cpp]  Position, FEN, make_move/unmake_move + UndoInfo; incremental
                    Zobrist + pawn-only Zobrist + PSQ + NNUE accumulators;
                    repetition-key stack
  movegen.[h|cpp]   generate_moves (legal moves) with pin-aware + enemy-attack
                    shortcuts + in_check helper
  perft.[h|cpp]     perft driver + 6-position standard suite
  magic.[h|cpp]     magic bitboards — init-time search + O(1) slider attacks
  zobrist.[h|cpp]   Zobrist keys + init + full-recompute reference; ep is
                    hashed only when the capture is pseudo-legal
  tt.[h|cpp]        transposition table (fixed-size direct-mapped, always-replace)
  eval.[h|cpp]      material + PST + tapered eval + mobility + passed /
                    isolated / doubled pawns (cached in a 16k-entry pawn
                    hash by Position::pawn_key) + bishop pair
                    (side-to-move perspective, incremental PSQ); routes
                    to nnue::evaluate when UseNNUE + a loaded network
  nnue_types.h      Accumulator + shape constants (HIDDEN_SIZE, FEATURES);
                    split from nnue.h so position.h can declare an
                    accumulator value without cycling nnue.h ↔ position.h
  nnue.[h|cpp]      NNUE runtime — HalfKP → 256 → 1 forward pass; per-side
                    accumulator with king-move dirty flags; JNN1 binary
                    loader/saver (magic + header validated; not
                    Stockfish-compatible); off by default
  nnue_simd.h       NEON kernels (AArch64) + AVX2 kernels (x86-64); scalar
                    reference always compiled and pinned by
                    SIMD-vs-reference equivalence tests
  search.[h|cpp]    iterative-deepening negamax with alpha-beta, TT, qsearch,
                    SEE-scored captures/promotions, PVS + null-window LMR,
                    null-move pruning, check extensions, reverse futility +
                    razoring, killers + capped history, aspiration windows,
                    repetition + 50-move draw detection, TT-walked PV
  notation.[h|cpp]  UCI move ↔ Move (move_to_uci, parse_uci_move)
  uci.[h|cpp]       UCI protocol loop on a background std::thread; per-iter
                    info lines with full PV + nps + time (score cp / score
                    mate N per UCI standard); clock time management for
                    wtime/btime; info string on mid-search position command
  tune.[h|cpp]      Texel coordinate-descent + SPSA tuner over eval
                    weights; reads `<FEN>;<outcome>` datasets, fits
                    sigmoid K, minimizes MSE; auto-prints tuned weights
                    as paste-ready C++ for eval.[h|cpp]
  main.cpp          entry (dispatches `perft` / `tune` / `dump-weights`
                    subcommands or falls into UCI)
tests/              doctest suite; one file per src unit under test
third_party/
  doctest.h         v2.4.11 (pinned single-header). Update via curl from the release tag.
scripts/            Python helpers for the tuning workflow
  fetch_tuning_dataset.py  Pull public labeled corpora (Zurichess quiet-labeled, etc.)
                           → `<FEN>;<outcome>` lines for `./engine tune`.
  gen_selfplay_data.py     Generate self-play datasets — fallback when public
                           corpora don't reflect this engine's play-style.
  sprt.py                  Drive `fastchess` SPRT matches vs a baseline binary
                           (`-rounds/-repeat`, `-pgnout file=…`) until H0/H1.
training/           Python NNUE trainer (uv-scripts) — turns self-play data into
                    loadable `.jnn1` networks. See training/README.md for details;
                    `feature_index()` pinned against C++ in tests/test_nnue.cpp.
nets/               Shipped NNUE weight files (JNN1 format, HalfKP → 256 → 1).
                    `default.jnn1` is the peak of a 4-round self-bootstrap chain
                    from a 500-game WDL seed; see nets/README.md for provenance
                    + reproduction. Off by default — load via UCI's EvalFile.
```

## Movegen milestone roadmap

Tracked in `src/movegen.h`. Each milestone is committed separately and validated by extending the perft depth that passes.

1. ✅ `Position::make_move` / `unmake_move` (supports all move types including ep, castling, promotion)
2. ✅ Knight moves (no legality filter yet)
3. ✅ King moves (no legality filter, no castling — those are 7 and 6)
4. ✅ Pawn moves (pushes, double, captures, ep, promotions incl. underpromotion + capture-promotion)
5. ✅ Sliding pieces — bishop, rook, queen (naive per-step rays; magic bitboards deferred)
6. ✅ Castling (rights + emptiness + king start/transit/land squares not attacked)
7. ✅ Legality filter — make/unmake round-trip + `is_square_attacked` on own king. **All 6 standard perft positions match through depth 5** (~200M node checks); `make perft-cert` extends to depth 6 (~8B nodes) as an offline gate — move generation is provably correct.
8. ✅ Search — material eval + negamax with alpha-beta. `cmd_go` parses `depth N`, emits UCI `info depth/score/nodes/pv` and `bestmove`. Default search depth 4. Distinguishes checkmate (`-MATE_SCORE + ply`, so shorter mates score higher) from stalemate (0).

## Post-roadmap next steps (each is an independent PR)

- ✅ Iterative deepening + `go movetime N` (any-time property, per-iteration `info` output). `search_best(pos, depth)` is preserved as a primitive; `search_iterative(pos, limits, callback)` is the production entry point.
- ✅ Quiescence search — extends leaves with captures only until quiet; standpat gives a lower bound; in-check nodes skip standpat and consider all moves (evasion). Startpos scores now stay at 0 across depths (was ±100), and total node counts DROP at deeper depths because stable evals give tighter alpha-beta cutoffs.
- ✅ Move ordering — MVV-LVA at every search node (negamax, qsearch, root). Score = `100000 + victim_value * 10 - attacker_value` for captures, promotion piece value added on top, 0 for other quiet moves. Startpos depth-6 nodes dropped 316k → 184k (~42% cut on top of qsearch). Killers/history not yet.
- ✅ Zobrist hashing + transposition table. Position.key updated incrementally in make/unmake (snapshot restored via UndoInfo on unmake). Fixed-size direct-mapped TT (2^20 entries, always-replace). Bounds handled (EXACT/LOWER/UPPER), mate scores ply-adjusted on store/probe. TT move hint fed into `order_moves` — hash hits from prior iterations become the very first move tried at the next depth. Startpos depth-6 nodes: 184k → 89k (~51% additional cut, 9× total speedup from baseline). `ucinewgame` clears the table.
- ✅ Killer moves + history heuristic. `killers[MAX_PLY][2]` records the two most recent quiet-move beta cutoffs per ply; `history[color][piece][to]` accumulates depth² on every quiet-move cutoff. Ordering bands rescaled so TT (10M) > captures (1M+MVV-LVA) > killer 1 (900k) > killer 2 (800k) > history-ranked quiets. Startpos depth-6: 89k → 69k nodes (**11.6× total from baseline**). Both live in SearchContext so they're fresh per `go` call.
- ✅ `go wtime W btime B [winc W] [binc B] [movestogo N]` — derive per-move budget from the clock. Divisor formula: `budget = (remaining - safety) / (movestogo + 2)` plus 3/4 of increment, capped at 1/3 of remaining. Sudden death assumes 30 more moves. Explicit `movetime` overrides. Under-safety-buffer clocks fall back to a 1ms budget so a bestmove still ships.
- ✅ Piece-square tables — Simplified Evaluation Function (Michniewski) values for all six piece types. White-perspective; black looks up `sq ^ 56` (rank flip). Startpos opening move is now Nb1-c3 (real chess) rather than Nb1-a3 (bad — "knight on the rim").
- ✅ Tapered eval — phase = 4·Q + 2·R + 1·B + 1·N per side (capped at 24 = starting non-pawn material). King PST interpolates linearly between MG (safety, favors castled positions) and EG (activity, favors center) by phase weight. In pure K+K endgames the engine now walks the king toward the center (Ka1-b2, Ke1-d2) instead of the corner. Other pieces still use a single table since MG/EG differences are small.
- ✅ Magic bitboards for slider attacks. `bishop_attacks(sq, occ)` / `rook_attacks(sq, occ)` are O(1) (3 loads + 1 multiply + 1 shift). Magic numbers discovered at init time via seeded random search (~0.25s startup for 128 magics — no hardcoded magic constant tables). Replaces the per-step gen_ray in slider move generation AND ray_hits_attacker in is_square_attacked. Perft 5 (~200M nodes total across all 6 standard positions) runs in ~16s = ~13M nodes/sec.
- ✅ `go infinite` + `stop`. cmd_go runs on a background `std::thread`; for infinite mode it returns immediately, for depth/movetime/clock modes it joins synchronously (preserves the bestmove-before-return contract). `should_stop` polls both the movetime deadline and an atomic external-stop pointer (`SearchLimits::external_stop`) on the same 1024-node cadence. `cmd_stop` signals + joins. All writes to `out` go through a mutex-guarded `emit` helper so info lines don't interleave with concurrent isready replies. Search falls back to the first legal move if a stop lands before any move completes at d=1 — bestmove is always legal when any legal move exists.
- ✅ UCI `position ... moves e2e4 e7e5 ...` — parse_uci_move infers MT_EN_PASSANT (pawn to ep_square) and MT_CASTLING (king ±2 files) from position state. Every move token is verified against the legal-move list before apply, so a malformed or illegal token stops processing rather than triggering a make_move assertion. Notation helpers live in `src/notation.[h|cpp]` with a full move_to_uci ↔ parse_uci_move round-trip test.
- ✅ PVS + null-window LMR + check extensions. First move at each node gets full-window; subsequent moves get null-window probes (with a depth reduction for late quiet non-check moves) and only re-search on fail-high. In-check nodes extend depth by 1 so mate lines don't fall off the horizon.
- ✅ Null-move pruning (R=3). At depth ≥ 3 non-check nodes with at least one non-pawn non-king piece (zugzwang guard) and a non-mate window, pass the turn and search at reduced depth; if the position still beats beta, prune. Board is unchanged — we just flip side_to_move, clear ep, and XOR the Zobrist deltas.
- ✅ Repetition detection + 50-move rule. `Position::history[1024]` is a Zobrist-key stack pushed by make_move / popped by unmake_move; `is_repetition` scans back by 2 (same-side-to-move) stopping at halfmove_clock (irreversible-move boundary). Draw detection at any non-root node returns 0 on repetition or `halfmove_clock >= 100`.
- ✅ Static Exchange Evaluation. Classical `gain[]` array with minimax backup, xray attackers via magic bitboards. Captures and promotions are ordered by SEE (winning above killers, losing below killers but above quiets). Qsearch skips losing captures. Non-capture promotions go through SEE too so a queen promo that gets recaptured is correctly ordered as a losing move.
- ✅ Mobility + passed pawn scoring. Piece-type-weighted count of attack squares (safe mobility — excludes enemy pawn attacks), plus per-rank passed-pawn bonuses in separate MG/EG tables. Passed-pawn masks precomputed at init via `eval::init()`.
- ✅ Bishop pair bonus. +30 MG / +50 EG when a side has 2+ bishops — the classical bonus for covering both color complexes.
- ✅ Reverse futility (aka static null pruning) + razoring + root PVS. At non-check nodes with non-mate windows: if `eval - 80·depth >= beta` at depth ≤ 6, prune; if `eval + 200 <= alpha` at depth ≤ 2, drop to qsearch. `search_root` also PVS's — later root moves get a null-window probe first.
- ✅ Conditional EP hashing. `zobrist::EP_FILE` is XOR'd only when a pawn of the side to move actually attacks the ep square. Fixes a hole where a "phantom" ep flag (double-push landing next to no enemy pawn) hashed differently from the same position without the flag, causing false-negative repetition detection.
- ✅ Precomputed enemy attacks for king-move legality. `movegen::attacks_by` computes the full enemy-attack bitboard once per node (with own king removed from occupancy so sliders see through); non-capture king moves check legality with one AND instead of make/unmake+is_square_attacked. Captures + castling still fall back to full is_legal.
- ✅ Full PV extraction. After each iteration, `build_pv` walks the TT from the post-bestmove position (bounded by depth, cycle-guarded, verified against generate_moves at each hop) and stores the line in `SearchResult::pv`. UCI info lines now emit the full PV plus `nps` and `time`.
- ✅ Cmd_position mid-search warning. Sending `position` during an active `go infinite` without a preceding `stop` used to silently cancel the search; now emits an `info string` first so the GUI bug is visible.
- ✅ Late Move Pruning (LMP). At non-check nodes with `depth <= 3` and a non-mate best-so-far, skip quiet moves once `quiets_searched >= LMP_LIMIT[depth]` (`{4, 8, 12}`). Captures, promotions, killers, and TT moves are unaffected — the pruning fires only on low-history quiets past the threshold. Startpos depth 10 drops from 301,533 → 116,327 nodes (~61% cut); Kiwipete depth 10 drops from 717,590 → 586,283 nodes (~18% cut).
- ✅ Pawn hash table + isolated / doubled pawn penalties. `Position::pawn_key` is an incremental XOR of `PIECE_SQ[color][PAWN][sq]` over all pawns, maintained by put_piece / remove_piece and snapshot-restored in unmake. The pawn hash (16k entries × 16 bytes = 256KB) caches the (mg, eg) diff of `pawn_structure_side` (passed + isolated + doubled combined). No clearing between games since pawn eval is a pure function of pawn placement. Isolated pawn penalty: -15 MG / -20 EG; doubled: -10 MG / -20 EG per extra pawn on file. Perf is roughly neutral on synthetic benchmarks (small savings on a cheap eval, offset by hash probe) — the real value is the new eval terms and the amortization path for future pawn terms (backward pawns, pawn chains, king shelter, storm/shelter).
- ✅ Countermove heuristic. `SearchContext::countermove[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES]` records our quiet-move beta cutoff response indexed by the opponent's last move (their color, piece type, destination square). On the next node, if the current move matches the countermove for the opponent's last move, it gets an ordering score of 700,000 — below killers (800k/900k), above losing captures (100k+SEE). `negamax` gains a `prev_move` parameter threaded through score_moves and the beta-cutoff update; null-move recursion passes NULL_MOVE (no move made). Kiwipete depth 9 drops 374,598 → 324,451 nodes (-13%); startpos depth 10 drops 107,355 → 104,034 nodes (-3%).
- ✅ Internal Iterative Reduction (IIR). At non-check nodes with `depth >= 4` and no TT move, cut this node's depth by 1 rather than search full depth with weak ordering. Preferred over classical IID (shallow search + re-probe) because it avoids the duplicated move-generation cost. Fires after null-move so null-move sees the original depth. Kiwipete depth 12 drops 3,055,071 → 2,655,077 nodes (-13%, -83 ms); startpos depth 10 drops 104,034 → 100,152 nodes (-4%). The gain scales with depth: the deeper the search, the more the reduction pays off.
- ✅ TT prefetch. `TranspositionTable::prefetch(key)` issues `__builtin_prefetch(&entries_[key & mask_], 0, 3)` — a non-blocking cache-line load hint. Called right after every `make_move` in the search hot path (negamax, qsearch, search_root, and after the null-move key XOR) so the child's TT slot starts loading in parallel with the child's setup work (node counter, stop check, in_check computation) before the actual probe. On Apple Silicon most of the effect is captured by the hardware prefetcher already; the software hint adds ~1-2% additional wall-time speedup on tactical middlegame positions. Nodes unchanged (prefetch is speed-only).
- ✅ Log-based LMR formula. Precomputed 64x256 table indexed by (depth, 1-indexed move number), formula `int(0.75 + log(d) * log(mn) / 2.25)`. Replaces the old `(i >= 12) ? 2 : 1` step which was uniformly conservative at deep nodes. New formula scales with both depth AND move number — d=10, mn=12 reduces by 3 (old: 2); d=15, mn=30 reduces by 4 (old: 2). Reduction clamped to `depth - 1` so we never over-reduce past qsearch. Startpos d12: 337,966 → 192,961 nodes (-43%, 52 → 30 ms). Kiwipete d12: 2,655,077 → 1,754,272 nodes (-34%, 579 → 378 ms). The PVS re-search on fail-high protects tactical moves the aggressive reduction would miss.
- ✅ Lazy eval. `evaluate(pos, alpha, beta)` computes the cheap material + PST + phase-blend "lazy score" first; if it's already outside the window by more than `EVAL_LAZY_MARGIN` (500 cp), returns early before touching mobility / pawn structure / bishop pair. Margin sized so `EVAL_LAZY_MARGIN >= RFP_MARGIN * MAX_RFP_DEPTH` (500 >= 80 × 6), guaranteeing the caller's RFP fail-high check triggers on any lazy-returned value that itself passed the lazy fail-high test. Wrapper `evaluate(pos)` defaults to `(-EVAL_UNBOUNDED, EVAL_UNBOUNDED)` for tests that pin exact values. Kiwipete d12/d13: ~1-2% wall-time speedup; effect is small because the mobility term is the only genuinely expensive per-call cost and log-LMR already cut total eval calls sharply.
- ✅ Link-time optimization (`-flto`). Enables cross-file inlining and dead-code elimination — the linker inlines hot leaf functions (magic-bitboard lookups, `tt().prefetch`, `is_capture`, small position accessors) across translation-unit boundaries that per-file compilation couldn't see. Kiwipete d12: 397 → 367 ms (**-7.5%**, 6-run average). Node counts differ by 1-2 (LTO inlined `should_stop` slightly differently, changing when the 1024-node wall-clock poll fires by one iteration — semantic no-op).
- ✅ Improving flag + LMR discount. `SearchContext::eval_stack[MAX_PLY]` records static eval at each ply (from the existing RFP `evaluate()` call — no extra cost). At each non-check non-mate node we compute `improving = eval_stack[ply] > eval_stack[ply-2]` (same-color plies, directly comparable). When NOT improving, LMR reduces one extra ply — a bad eval trend means late quiet moves are even less likely to change the outcome. Startpos d12: 27.4 → 18.0 ms (**-34%**, -29% nodes) — quiet positions where improving stays stable benefit most. Kiwipete d13: essentially flat (tactical positions where improving oscillates cancel the win, but nps compensates for the extra re-searches).
- ✅ PV-node LMR discount. At genuine PV nodes (non-null window, `beta - alpha > 1`), give back one ply of LMR reduction — moves on the principal variation matter more, and the re-search cost we save more than pays for the deeper reduced probe. Reduction is now clamped both ways: `std::clamp(reduction, 0, depth - 1)` so the discount can't push us into a negative reduction (which would extend, not reduce). Startpos d12: 18 → 17 ms (-5.5%, -7% nodes). Kiwipete d13: +1% time (within noise).
- ✅ King safety attack-count penalty. Rebel/CPW-style. Per side: `attack_units = Σ WEIGHT[pt] * popcount(piece_attacks & king_ring)` across enemy N/B/R/Q (weights N=2, B=2, R=3, Q=5), indexed into a saturating quadratic table (peak 500 cp at 61+ units). MG-only — the tapered king PST already rewards active endgame kings; layering safety on top would double-count. Per-side queen gate skips the piece scans when the opposing queen is off (arithmetic caps around ~40 cp without a queen; not worth the cycles). `EVAL_LAZY_MARGIN` bumped 500 → 1000 to cover the new ±500 cp swing. Kiwipete d13: +5.5% nodes (fixed piece-scan cost); startpos d13: -11% nodes; middlegame d13: -40% nodes (better move ordering once eval understands king exposure).
- ✅ Pawn shield bonus. Three-file scan in front of a castled king (king's file ± 1) — pawn on rank 2 = full bonus, rank 3 = partial, rank 4 = trace, none = missing-shield penalty. Gated on castled kings only (wing file, home rank); central-file kings score neutral because "shield" doesn't describe them. Scored independently of the king-safety attack table — a positional feature, not part of the attack arithmetic. MG-only.
- ✅ Pawn storm penalty. Same three-file scan as shield but on enemy pawns advancing toward the king. Shield asks "am I sheltered now?"; storm asks "how quickly will that shelter dissolve?" — both matter and aren't redundant (castled + intact shield + fast storm = shelter dissolving within moves). Same castling / pawnless gates. MG-only.
- ✅ King on open / half-open file penalty. Distinct from shield/storm — those measure the shelter's *thickness*, this measures whether a rook or queen can invade the king's file. Wing + home-rank gated, matching the other king-safety terms.
- ✅ NNUE evaluation (opt-in). HalfKP → 256 → 1 architecture. Per-`Position` accumulator with per-side king-move dirty flags — only the moving side's perspective invalidates on a king move; non-king pieces update in O(features-per-piece) via `put_piece`/`remove_piece` hooks. SIMD hot-loop kernels: NEON on AArch64 (Apple Silicon), AVX2 on x86-64; scalar reference always compiled and pinned by SIMD-vs-reference equivalence tests. Custom **JNN1** binary format (`"JNN1"` magic + `uint32 {version, hidden, features}` header — header-validated on load; not Stockfish-compatible). UCI-gated: off unless `UseNNUE=true` **and** `EvalFile` points to a loadable `.jnn1` — falls back to classical eval on any failure. Companion Python training pipeline (`training/`, uv-scripts) turns self-play data into loadable networks; Python `feature_index()` is pinned against C++ `nnue::feature_index()` in `tests/test_nnue.cpp` so trained networks and inference can't silently drift.
- ✅ Texel coordinate-descent tuner + PSQ recompute. `./engine tune <dataset>` reads `<FEN>;<outcome>` lines (outcome ∈ {0, 0.5, 1} from WHITE's perspective), fits sigmoid scale K, then coordinate-descends over `eval::params` (13 scalar weights + 5 piece values) to minimize MSE prediction error against the dataset labels. Piece values become independently tunable because PSTs no longer bake the piece value in — `Position::psq_mg/eg` is recomputed from live `eval::params` at eval time instead of being pre-baked at position construction. Suitable for small weight sets; 30-position `tests/data/tune_demo.txt` checked in as a CI smoke test.
- ✅ SPSA for PST tuning. Simultaneous Perturbation Stochastic Approximation over the 6×64×2 = 768 PST values — two dataset evaluations per iteration regardless of dimensionality, unlike coord descent's per-parameter probe. Invoked via `./engine tune <dataset> pst`; `all` mode runs scalar first, then PST.
- ✅ Tuning data pipeline. `scripts/fetch_tuning_dataset.py` pulls public labeled corpora (Zurichess quiet-labeled and friends) into `<FEN>;<outcome>` lines. `scripts/gen_selfplay_data.py` generates self-play datasets — the fallback for when public corpora don't reflect this engine's play-style after a big eval change.
- ✅ SPRT validation harness + nightly CI. Texel loss ≠ Elo, so tuned weights must be validated by real games before shipping. `scripts/sprt.py` drives `fastchess` (`-rounds/-repeat`, `-pgnout file=…`) with a Sequential Probability Ratio Test vs a baseline binary snapshot until H0 (no gain) or H1 (Elo gain) is accepted. `.github/workflows/nightly-sprt.yml` runs the harness against a baseline tag every night; regressions auto-open a GitHub issue, resolved regressions auto-close when the baseline tag bumps (`.github/workflows/baseline-bump.yml`). End-to-end tuning workflow lives in `src/tune.h`'s header comment.
- ✅ `dump-weights` subcommand. `./engine dump-weights` prints current in-memory `eval::params` + `pst_mg` + `pst_eg` as paste-ready C++ source, so tuned weights transfer to `src/eval.[h|cpp]` without hand transcription. Also invoked automatically at the end of `tune::run_tune`, so a tuning run's output already contains the source fragment.

## Search / eval performance stack

Startpos depth 8 (release, -O3 -march=native), each row adds on top of the previous:

| Layer                           | Time   | Nodes      |
|---------------------------------|-------:|-----------:|
| Baseline (post-milestone-8)     | 0.85 s | 1,424,033  |
| MoveList (stack, no malloc)     | 0.78 s | 1,424,033  |
| Compute-once + lazy pick sort   | 0.58 s | 1,326,211  |
| Captures-only qsearch generator | 0.43 s | 1,415,621  |
| Aspiration windows (±75 cp)     | 0.41 s | 1,356,981  |
| Late Move Reductions (LMR)      | 0.28 s |   198,183  |
| Incremental eval (psq_mg/eg)    | 0.28 s |   198,183  |
| Pin-aware legality shortcut     | 0.28 s |   198,183  |

Total: **3× search speedup, ~7× node reduction** on the depth-8 startpos benchmark. The node collapse comes almost entirely from LMR — reducing depth on late quiet moves prunes huge subtrees. Later phases (incremental eval, pin-aware) show as neutral on this benchmark because LMR + TT already dominated the runtime; both pay off on other workloads (pure perft shows ~26 % gain from pin-aware alone; eval gains from incremental will compound when more terms are added).

**Grill round 2 additions** (post-milestone-8 stack above → after PVS + null-move + SEE + reverse futility + razoring + root PVS + king-move enemy-attack shortcut): startpos depth 8 drops from 198,183 nodes to **~54,700 nodes** — another ~3.6× cut on top of the earlier work. Depth 10 completes in ~27 ms / ~301k nodes with a full 10-ply PV.

**LMP addition**: startpos depth 8 drops from 54,732 → 36,113 nodes (34%); depth 10 drops from 301,533 → 116,327 nodes (61%) in ~18 ms. Kiwipete depth 10 drops from 717,590 → 586,283 nodes (18%). The cut compounds with LMR — LMR reduces depth on late quiets, LMP skips them entirely once enough have been searched.

Perft 5 (~200 M nodes across the 6 standard positions): 16.5 s → 12.2 s = ~26 % faster on pure movegen throughput (~16 M nodes/sec).

Do not skip a milestone. Perft numbers stay artificially low until every piece type generates, but each milestone's *round-trip* invariants (see `tests/test_position.cpp`) must hold before advancing.

## Testing conventions

- Framework: **doctest** (single header, no CMake). `TEST_CASE` / `SUBCASE` / `CHECK` / `REQUIRE`. See any existing `tests/test_*.cpp` for style.
- Every new pure-logic behavior gets a test. Round-trip invariants (make → unmake, encode → decode, FEN parse → emit) are the highest-value shape.
- **One test file per src unit under test**, mirroring `src/`:
    - `tests/test_bitboard.cpp` ↔ `src/bitboard.[h|cpp]`
    - `tests/test_attacks.cpp`  ↔ `src/attacks.[h|cpp]`
    - `tests/test_magic.cpp`    ↔ `src/magic.[h|cpp]`
    - `tests/test_position.cpp` ↔ `src/position.[h|cpp]` (FEN + make/unmake forward-correctness + repetition + pawn_key round-trip)
    - `tests/test_movegen.cpp`  ↔ `src/movegen.[h|cpp]` (generator shape + movegen-driven make/unmake walks)
    - `tests/test_perft.cpp`    ↔ `src/perft.[h|cpp]`
    - `tests/test_zobrist.cpp`  ↔ `src/zobrist.[h|cpp]` (key round-trips + phantom-ep-hash regression)
    - `tests/test_tt.cpp`       ↔ `src/tt.[h|cpp]`
    - `tests/test_eval.cpp`     ↔ `src/eval.[h|cpp]` (material + PST + mobility + passed / isolated / doubled pawns + bishop pair + pawn-hash consistency)
    - `tests/test_search.cpp`   ↔ `src/search.[h|cpp]` (negamax + qsearch + TT + SEE-promo regression)
    - `tests/test_notation.cpp` ↔ `src/notation.[h|cpp]` (UCI move round-trip)
    - `tests/test_uci.cpp`      ↔ `src/uci.[h|cpp]` (protocol via stringstream — `uci_loop` takes `std::istream&/std::ostream&` for exactly this reason; do not reintroduce `std::cin`/`std::cout` inside the loop)
    - `tests/test_nnue.cpp`     ↔ `src/nnue.[h|cpp]` + `src/nnue_simd.h` (HalfKP feature-index round-trip, accumulator incremental-vs-full equivalence, SIMD-vs-scalar kernel equivalence, JNN1 loader/saver round-trip, C++↔Python feature-index pin)
    - `tests/test_tune.cpp`     ↔ `src/tune.[h|cpp]` (Texel loss monotonicity, sigmoid K fit, coord-descent + SPSA convergence on the checked-in `tune_demo.txt`; dump-weights round-trip)
  New src units require a matching `tests/test_<unit>.cpp`. Shared fixtures / helpers live in `tests/support.h`. Under ASan + UBSan the doctest suite passes cleanly — see `make test` for the current test/assertion counts.
- `tests/test_main.cpp` uses `DOCTEST_CONFIG_IMPLEMENT` and provides `main()` — this is the single place where `init_attacks()`, `init_magic()`, `zobrist::init()`, and `eval::init()` are called, so per-TU static-init hacks are unnecessary.
- Tests link the whole `src/` tree (except `main.cpp`) — see Makefile `$(TEST_SRCS)`.
- Do not mock `Position` internals. Verify through `to_fen()` / public accessors.

## Perft gotcha

`make_move`'s king-count assertion is `<= 1`, not `== 1`. Real games and every current perft position have exactly one king per side, but the loose form defends against hand-constructed test positions that omit a king (bishop-only slider tests, etc.). Do not tighten it without auditing every test FEN.

## Move encoding (types.h)

16 bits: `from(6) | to(6) | promo(2) | movetype(2)`. `promo` is offset from KNIGHT, so it always fits in 2 bits. `NULL_MOVE == 0` is safe because a legal move never has `from == to == A1`.

## References

- [Chess Programming Wiki — Getting Started](https://chessprogramming.org/Getting_Started) is the canonical reference for every technique in this repo (bitboards, move generation, perft, search, evaluation). Consult it before designing a new subsystem.
- [Perft Results](https://www.chessprogramming.org/Perft_Results) — source of the six-position suite in `src/perft.cpp`. Any change to move generation must be validated against these.
- [UCI protocol](https://backscattering.de/chess/uci/) — spec for the stdin/stdout interface `src/uci.cpp` implements.

## Non-goals (for now)

- Opening book / endgame tablebases
- Multi-threading (Lazy SMP or similar — search stays single-threaded)
- Pondering (`go ponder`)
- MultiPV output

Do not add these speculatively — they each add substantial surface area
and only pay off once the underlying search + eval is much stronger.
