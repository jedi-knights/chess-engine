#include "tune.h"
#include "eval.h"
#include "position.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace tune {

namespace {

struct Sample {
    Position pos;
    double   outcome;   // 0, 0.5, or 1 — from WHITE's perspective
};

// Load dataset. Skip blank lines and comment lines starting with '#'.
// Each valid line is `<FEN>;<outcome>`. Reports and discards malformed
// entries rather than aborting — a corrupted line in a 1M-position
// dataset shouldn't kill a multi-hour tuning run.
std::vector<Sample> load_dataset(const std::string& path) {
    std::vector<Sample> out;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "tune: cannot open dataset '%s'\n", path.c_str());
        return out;
    }
    std::string line;
    int line_no = 0;
    int bad     = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto sep = line.find(';');
        if (sep == std::string::npos) {
            ++bad;
            continue;
        }
        std::string fen     = line.substr(0, sep);
        std::string outcome = line.substr(sep + 1);
        Sample s;
        if (!s.pos.set_from_fen(fen)) {
            std::fprintf(stderr, "tune: bad FEN at line %d\n", line_no);
            ++bad;
            continue;
        }
        try {
            s.outcome = std::stod(outcome);
        } catch (...) {
            ++bad;
            continue;
        }
        if (s.outcome < 0.0 || s.outcome > 1.0) {
            ++bad;
            continue;
        }
        out.push_back(std::move(s));
    }
    if (bad > 0) {
        std::fprintf(stderr, "tune: skipped %d malformed line(s)\n", bad);
    }
    return out;
}

// Sigmoid mapping centipawn eval → win probability. K is the scaling
// constant fitted per dataset; 400 is the classical Elo-scale divisor
// (paired with the base-e sigmoid this puts K in the same range other
// Texel tuners report, typically 0.7-1.5).
double sigmoid(int cp, double K) {
    return 1.0 / (1.0 + std::exp(-K * double(cp) / 400.0));
}

// Mean squared error across the dataset, from WHITE's perspective.
// evaluate() returns from side-to-move perspective — negate for BLACK.
double compute_mse(const std::vector<Sample>& data, double K) {
    double sum = 0;
    for (const auto& s : data) {
        int stm = evaluate(s.pos);
        int white_cp = (s.pos.side_to_move == WHITE) ? stm : -stm;
        double p = sigmoid(white_cp, K);
        double d = p - s.outcome;
        sum += d * d;
    }
    return sum / double(data.size());
}

// Fit K by golden-section-style trisection search. 40 iterations
// converges from [0.5, 3.0] to within ~1e-9. Cheap — each iteration
// costs 2 dataset evals.
double fit_k(const std::vector<Sample>& data) {
    double lo = 0.5;
    double hi = 3.0;
    for (int i = 0; i < 40; ++i) {
        double m1 = lo + (hi - lo) / 3.0;
        double m2 = hi - (hi - lo) / 3.0;
        if (compute_mse(data, m1) < compute_mse(data, m2)) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    return (lo + hi) / 2.0;
}

struct Weight {
    const char* name;
    int*        value;
    int         lo;
    int         hi;
    // PSQ-affecting weights (piece_values) feed the incremental
    // psq_mg / psq_eg accumulators on Position. Mutating one means
    // every position's accumulator is now stale relative to the new
    // weight; the tuner must call recompute_psq() on the whole
    // dataset before the next MSE evaluation.
    bool        affects_psq = false;
};

// Refresh psq accumulators on every position. Only needed after a
// PSQ-affecting weight mutation. Cost is ~32 iterations per position
// (average non-pawn material) — negligible for tiny demo datasets,
// ~3M ops per pass for a 100k-position tuning run.
void refresh_psq(std::vector<Sample>& data) {
    for (auto& s : data) {
        s.pos.recompute_psq();
    }
}

// Coordinate descent: for each weight, walk in the direction that
// improves MSE until it stops improving, then move to the next weight.
// Repeats full sweeps over all weights until a sweep produces no
// improvement. Simple and monotonically-improving; O(iterations *
// weights * dataset_size) evaluations per sweep.
void coord_descent(std::vector<Weight>& weights,
                   std::vector<Sample>& data, double K) {
    double best = compute_mse(data, K);
    std::printf("Initial MSE: %.7f\n", best);

    // Small helper: set the weight and refresh PSQ if the weight
    // feeds the incremental accumulators. Grouping the two together
    // means MSE always sees fresh state.
    auto set_weight = [&](Weight& w, int v) {
        *w.value = v;
        if (w.affects_psq) {
            refresh_psq(data);
        }
    };

    bool improved = true;
    int  sweep    = 0;
    while (improved) {
        improved = false;
        ++sweep;
        for (auto& w : weights) {
            const int orig = *w.value;
            int best_val   = orig;

            // Search up.
            for (int v = orig + 1; v <= w.hi; ++v) {
                set_weight(w, v);
                double m = compute_mse(data, K);
                if (m + 1e-9 < best) {
                    best     = m;
                    best_val = v;
                    improved = true;
                } else {
                    break;
                }
            }
            // Search down only if up didn't help — they're mutually
            // exclusive by monotonicity of the loss along one axis.
            if (best_val == orig) {
                for (int v = orig - 1; v >= w.lo; --v) {
                    set_weight(w, v);
                    double m = compute_mse(data, K);
                    if (m + 1e-9 < best) {
                        best     = m;
                        best_val = v;
                        improved = true;
                    } else {
                        break;
                    }
                }
            }
            // Always restore through set_weight so PSQ state matches
            // best_val even after a backed-out probe.
            set_weight(w, best_val);
        }
        std::printf("Sweep %d: MSE = %.7f\n", sweep, best);
    }
    std::printf("Converged after %d sweep(s).\n", sweep);
}

// Coordinate descent over eval::params (scalar weights + piece values).
void run_scalar(std::vector<Sample>& data, double K) {
    // Weights exposed for tuning. `lo` / `hi` are search bounds — set
    // wide but not unbounded so a divergent sweep can't run forever.
    // `affects_psq` = true means mutating the weight invalidates the
    // incremental Position::psq_mg / psq_eg accumulators; the tuner
    // rebuilds them via Position::recompute_psq on every candidate.
    // King's piece_values[KING] intentionally omitted — the king has
    // no material value (losing it means the game is already lost).
    std::vector<Weight> weights = {
        // Piece values (PSQ-affecting).
        {"piece_pawn",   &eval::params.piece_values[PAWN],   50, 200, true},
        {"piece_knight", &eval::params.piece_values[KNIGHT], 200, 500, true},
        {"piece_bishop", &eval::params.piece_values[BISHOP], 200, 500, true},
        {"piece_rook",   &eval::params.piece_values[ROOK],   350, 700, true},
        {"piece_queen",  &eval::params.piece_values[QUEEN],  700, 1300, true},

        // Non-PSQ terms.
        {"isolated_mg",                 &eval::params.isolated_mg,                 -100, 100},
        {"isolated_eg",                 &eval::params.isolated_eg,                 -100, 100},
        {"doubled_mg",                  &eval::params.doubled_mg,                  -100, 100},
        {"doubled_eg",                  &eval::params.doubled_eg,                  -100, 100},
        {"bishop_pair_mg",              &eval::params.bishop_pair_mg,               -50, 200},
        {"bishop_pair_eg",              &eval::params.bishop_pair_eg,               -50, 200},
        {"mob_knight",                  &eval::params.mob_knight,                     0,  20},
        {"mob_bishop",                  &eval::params.mob_bishop,                     0,  20},
        {"mob_rook",                    &eval::params.mob_rook,                       0,  20},
        {"mob_queen",                   &eval::params.mob_queen,                      0,  20},
        {"shield_missing_penalty",      &eval::params.shield_missing_penalty,         0, 100},
        {"king_open_file_penalty",      &eval::params.king_open_file_penalty,         0, 100},
        {"king_semi_open_file_penalty", &eval::params.king_semi_open_file_penalty,    0, 100},
    };

    coord_descent(weights, data, K);

    std::printf("\nFinal tuned scalar weights (baseline defaults in parens):\n");
    eval::TuningParams defaults;
    for (const auto& w : weights) {
        // Fetch the default by name so the diff is obvious at review.
        int def = 0;
        const std::string name = w.name;
        if      (name == "piece_pawn")                   { def = defaults.piece_values[PAWN]; }
        else if (name == "piece_knight")                 { def = defaults.piece_values[KNIGHT]; }
        else if (name == "piece_bishop")                 { def = defaults.piece_values[BISHOP]; }
        else if (name == "piece_rook")                   { def = defaults.piece_values[ROOK]; }
        else if (name == "piece_queen")                  { def = defaults.piece_values[QUEEN]; }
        else if (name == "isolated_mg")                  { def = defaults.isolated_mg; }
        else if (name == "isolated_eg")                  { def = defaults.isolated_eg; }
        else if (name == "doubled_mg")                   { def = defaults.doubled_mg; }
        else if (name == "doubled_eg")                   { def = defaults.doubled_eg; }
        else if (name == "bishop_pair_mg")               { def = defaults.bishop_pair_mg; }
        else if (name == "bishop_pair_eg")               { def = defaults.bishop_pair_eg; }
        else if (name == "mob_knight")                   { def = defaults.mob_knight; }
        else if (name == "mob_bishop")                   { def = defaults.mob_bishop; }
        else if (name == "mob_rook")                     { def = defaults.mob_rook; }
        else if (name == "mob_queen")                    { def = defaults.mob_queen; }
        else if (name == "shield_missing_penalty")       { def = defaults.shield_missing_penalty; }
        else if (name == "king_open_file_penalty")       { def = defaults.king_open_file_penalty; }
        else if (name == "king_semi_open_file_penalty")  { def = defaults.king_semi_open_file_penalty; }
        std::printf("  %-30s = %4d   (was %d)\n", w.name, *w.value, def);
    }
}

// --- SPSA tuning of the piece-square tables -----------------------------
// Simultaneous Perturbation Stochastic Approximation (Spall, 1992). For
// a high-dimensional weight vector θ ∈ ℝ^n, SPSA estimates the gradient
// from just TWO loss evaluations per iteration regardless of n — a
// random Bernoulli sign vector Δ perturbs all weights simultaneously,
// and the finite-difference (L(θ+cΔ) - L(θ-cΔ)) / (2c·Δᵢ) gives an
// unbiased noisy gradient estimate for each component.
//
// PST tuning dimensionality:
//   6 piece types × 64 squares × 2 phases (MG, EG) = 768 weights.
// Coordinate descent would need 768×N evaluations per full sweep; SPSA
// needs 2 evaluations per iteration, converging in 1000-10000 iterations
// on a real dataset. On our 30-position demo dataset that's ~60k evals
// vs coord's 46k, so SPSA is comparable in cost but scales to real
// datasets where coord descent doesn't.

constexpr int PST_DIM = 6 * NUM_SQUARES * 2;   // 6 pieces (PAWN..KING), 64 squares, 2 phases.

// Map an SPSA weight index in [0, PST_DIM) → (phase, pt, sq). Piece
// types [1..6] map to [0..5] here so index 0 is meaningful.
void pst_index_to_slot(int i, int& phase, int& pt, int& sq) {
    phase = i / (6 * NUM_SQUARES);            // 0 = MG, 1 = EG
    int rem = i % (6 * NUM_SQUARES);
    pt = 1 + (rem / NUM_SQUARES);             // shift to PAWN..KING
    sq = rem % NUM_SQUARES;
}

int  read_pst_slot (int i) {
    int phase, pt, sq;
    pst_index_to_slot(i, phase, pt, sq);
    return (phase == 0) ? eval::pst_mg[pt][sq] : eval::pst_eg[pt][sq];
}

void write_pst_slot(int i, int value) {
    int phase, pt, sq;
    pst_index_to_slot(i, phase, pt, sq);
    if (phase == 0) { eval::pst_mg[pt][sq] = value; }
    else            { eval::pst_eg[pt][sq] = value; }
}

// SPSA hyperparameters, from Spall's "Introduction to Stochastic
// Search and Optimization" (2003), tuned for this problem size:
//   α, γ  — decay exponents for learning rate and perturbation.
//   a, c  — initial magnitudes.
//   A     — learning-rate stability constant (typically ~10% of N).
// c chosen large enough that a rounded ±c step actually changes the
// int-valued weight; a chosen so a * gradient stays in the ~1 cp per
// iteration range at typical PST scales.
void spsa_pst(std::vector<Sample>& data, double K, int n_iter) {
    constexpr double alpha = 0.602;
    constexpr double gamma = 0.101;
    // a0 tuned to produce visible theta drift within 1000 iterations
    // on the demo dataset. Typical MSE-scale gradients are ~1e-3, so
    // per-iteration update magnitude a_k * g_i ends up ~1-2 units at
    // k=1 and decays under Spall's schedule. On real 100k-position
    // datasets the gradient is noisier per iteration but averages more
    // cleanly; if you see slots stuck at their starting values, lower
    // a0 and raise n_iter (Spall recommends a0 ~ desired step / typical
    // gradient magnitude).
    constexpr double a0    = 5000.0;
    constexpr double c0    = 6.0;
    const double A         = std::max(1.0, n_iter * 0.1);

    // Continuous weight vector — keeps sub-integer state between
    // iterations even though the eval reads int weights via rounding.
    std::vector<double> theta(PST_DIM);
    for (int i = 0; i < PST_DIM; ++i) {
        theta[i] = double(read_pst_slot(i));
    }

    std::mt19937 rng(1729);   // Ramanujan; deterministic across runs.

    const double initial_mse = compute_mse(data, K);
    std::printf("SPSA start: %d iterations over %d PST weights, "
                "initial MSE = %.7f\n",
                n_iter, PST_DIM, initial_mse);

    // Log every ~10% of the run, plus a final pass.
    const int log_every = std::max(1, n_iter / 10);

    for (int k = 1; k <= n_iter; ++k) {
        const double a_k = a0 / std::pow(double(k) + A, alpha);
        const double c_k = c0 / std::pow(double(k),     gamma);

        // Bernoulli ±1 perturbation vector.
        std::vector<int> delta(PST_DIM);
        for (int i = 0; i < PST_DIM; ++i) {
            delta[i] = (rng() & 1U) ? 1 : -1;
        }

        // L(θ + cΔ). Round after adding to keep int weights.
        for (int i = 0; i < PST_DIM; ++i) {
            write_pst_slot(i, int(std::lround(theta[i] + c_k * delta[i])));
        }
        refresh_psq(data);
        const double L_plus = compute_mse(data, K);

        // L(θ - cΔ).
        for (int i = 0; i < PST_DIM; ++i) {
            write_pst_slot(i, int(std::lround(theta[i] - c_k * delta[i])));
        }
        refresh_psq(data);
        const double L_minus = compute_mse(data, K);

        // Gradient estimate and update.
        // g_i = (L_plus - L_minus) / (2 * c_k * delta[i]);
        // Since delta[i] = ±1, dividing by delta[i] is multiplying by it.
        const double diff = L_plus - L_minus;
        for (int i = 0; i < PST_DIM; ++i) {
            const double g_i = diff * double(delta[i]) / (2.0 * c_k);
            theta[i] -= a_k * g_i;
        }

        // Periodic progress: write theta back for a clean MSE reading.
        if (k % log_every == 0 || k == n_iter) {
            for (int i = 0; i < PST_DIM; ++i) {
                write_pst_slot(i, int(std::lround(theta[i])));
            }
            refresh_psq(data);
            const double cur = compute_mse(data, K);
            std::printf("SPSA iter %5d/%d: a_k=%.4f c_k=%.4f MSE=%.7f\n",
                        k, n_iter, a_k, c_k, cur);
        }
    }

    // Ensure final state is the rounded theta, not the last probe.
    for (int i = 0; i < PST_DIM; ++i) {
        write_pst_slot(i, int(std::lround(theta[i])));
    }
    refresh_psq(data);
    std::printf("SPSA done: final MSE = %.7f (start was %.7f)\n",
                compute_mse(data, K), initial_mse);
}

void run_pst(std::vector<Sample>& data, double K, int iterations) {
    // Snapshot starting PSTs so the summary can report which slots
    // moved the most. 768 ints is negligible memory.
    std::vector<int> before(PST_DIM);
    for (int i = 0; i < PST_DIM; ++i) {
        before[i] = read_pst_slot(i);
    }

    spsa_pst(data, K, iterations);

    // Summary: top 10 largest |delta| slots. Real tuning workflows
    // dump the full pst_mg / pst_eg tables externally and diff —
    // this is the "did anything meaningful move?" quick-look.
    struct SlotDelta { int idx; int delta; };
    std::vector<SlotDelta> deltas;
    deltas.reserve(PST_DIM);
    for (int i = 0; i < PST_DIM; ++i) {
        int d = read_pst_slot(i) - before[i];
        if (d != 0) {
            deltas.push_back({i, d});
        }
    }
    std::sort(deltas.begin(), deltas.end(),
              [](const SlotDelta& a, const SlotDelta& b) {
                  return std::abs(a.delta) > std::abs(b.delta);
              });

    std::printf("\nPST tuning summary — %zu slots changed, top 10 by |delta|:\n",
                deltas.size());
    static const char* PT_NAME[NUM_PIECE_TYPES] = {
        "?", "pawn", "knight", "bishop", "rook", "queen", "king",
    };
    const int limit = int(std::min<size_t>(deltas.size(), 10));
    for (int r = 0; r < limit; ++r) {
        int phase, pt, sq;
        pst_index_to_slot(deltas[r].idx, phase, pt, sq);
        const char* phase_name = (phase == 0) ? "MG" : "EG";
        // sq → file/rank letters for readability.
        char file_char = char('a' + (sq & 7));
        char rank_char = char('1' + (sq >> 3));
        std::printf("  %s %-6s %c%c : %+d\n",
                    phase_name, PT_NAME[pt], file_char, rank_char,
                    deltas[r].delta);
    }
    std::printf("  (call dump_weights() for the full tables)\n");
}

// Pretty-print an 8×8 PST table using the same layout as the source
// arrays in eval.cpp — 8 values per row, 4-column formatted, so a
// direct paste preserves the file/rank grid a maintainer expects.
void dump_pst_table(const int table[NUM_PIECE_TYPES][NUM_SQUARES]) {
    // Comment header per piece type. PAWN..KING = indices 1..6.
    static const char* PT_NAME[NUM_PIECE_TYPES] = {
        nullptr, "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING",
    };
    // Skip index 0 (NO_PIECE_TYPE) — it's a sentinel that stays 0.
    std::printf("    { 0 },\n");
    for (int pt = PAWN; pt <= KING; ++pt) {
        std::printf("    // %s\n", PT_NAME[pt]);
        std::printf("    {\n");
        // Source layout goes rank 1 → rank 8 (bottom → top from
        // white's perspective), matching the existing eval.cpp arrays.
        for (int r = 0; r < 8; ++r) {
            std::printf("       ");
            for (int f = 0; f < 8; ++f) {
                std::printf(" %4d,", table[pt][(r * 8) + f]);
            }
            std::printf("\n");
        }
        std::printf("    },\n");
    }
}

}  // namespace

void dump_weights() {
    std::printf("// === Tuned eval weights ===\n"
                "// Paste this over the corresponding blocks in src/eval.h\n"
                "// (TuningParams initializers) and src/eval.cpp (pst_mg / pst_eg).\n"
                "// Generated by `./engine dump-weights` or the end of `./engine tune`.\n\n");

    // --- Scalar params (paste into src/eval.h TuningParams) ---
    std::printf("// --- eval::TuningParams member initializers (src/eval.h) ---\n");
    std::printf("int piece_values[NUM_PIECE_TYPES] = { 0, %d, %d, %d, %d, %d, 0 };\n",
                eval::params.piece_values[PAWN],
                eval::params.piece_values[KNIGHT],
                eval::params.piece_values[BISHOP],
                eval::params.piece_values[ROOK],
                eval::params.piece_values[QUEEN]);
    std::printf("int isolated_mg                 = %4d;\n", eval::params.isolated_mg);
    std::printf("int isolated_eg                 = %4d;\n", eval::params.isolated_eg);
    std::printf("int doubled_mg                  = %4d;\n", eval::params.doubled_mg);
    std::printf("int doubled_eg                  = %4d;\n", eval::params.doubled_eg);
    std::printf("int bishop_pair_mg              = %4d;\n", eval::params.bishop_pair_mg);
    std::printf("int bishop_pair_eg              = %4d;\n", eval::params.bishop_pair_eg);
    std::printf("int mob_knight                  = %4d;\n", eval::params.mob_knight);
    std::printf("int mob_bishop                  = %4d;\n", eval::params.mob_bishop);
    std::printf("int mob_rook                    = %4d;\n", eval::params.mob_rook);
    std::printf("int mob_queen                   = %4d;\n", eval::params.mob_queen);
    std::printf("int shield_missing_penalty      = %4d;\n", eval::params.shield_missing_penalty);
    std::printf("int king_open_file_penalty      = %4d;\n", eval::params.king_open_file_penalty);
    std::printf("int king_semi_open_file_penalty = %4d;\n", eval::params.king_semi_open_file_penalty);

    // --- PSTs (paste into src/eval.cpp) ---
    std::printf("\n// --- eval::pst_mg (src/eval.cpp) ---\n");
    std::printf("int pst_mg[NUM_PIECE_TYPES][NUM_SQUARES] = {\n");
    dump_pst_table(eval::pst_mg);
    std::printf("};\n");

    std::printf("\n// --- eval::pst_eg (src/eval.cpp) ---\n");
    std::printf("int pst_eg[NUM_PIECE_TYPES][NUM_SQUARES] = {\n");
    dump_pst_table(eval::pst_eg);
    std::printf("};\n");
}

int run_tune(const std::string& dataset_path, Mode mode, int iterations) {
    std::vector<Sample> data = load_dataset(dataset_path);
    if (data.empty()) {
        std::fprintf(stderr, "tune: empty dataset — nothing to do\n");
        return 1;
    }
    std::printf("Loaded %zu positions from '%s'\n",
                data.size(), dataset_path.c_str());

    const double K = fit_k(data);
    std::printf("Fitted K = %.6f\n", K);

    if (mode == Mode::Scalar || mode == Mode::All) {
        run_scalar(data, K);
    }
    if (mode == Mode::Pst || mode == Mode::All) {
        run_pst(data, K, iterations);
    }

    // Emit paste-ready source at the end so the tuned state doesn't
    // die with the process. Marked so it's grep-able out of a longer
    // tune log.
    std::printf("\n=== BEGIN DUMPED WEIGHTS ===\n");
    dump_weights();
    std::printf("=== END DUMPED WEIGHTS ===\n");
    return 0;
}

}  // namespace tune
