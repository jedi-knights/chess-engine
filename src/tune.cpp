#include "tune.h"
#include "eval.h"
#include "position.h"

#include <cmath>
#include <cstdio>
#include <fstream>
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

}  // namespace

int run_tune(const std::string& dataset_path) {
    std::vector<Sample> data = load_dataset(dataset_path);
    if (data.empty()) {
        std::fprintf(stderr, "tune: empty dataset — nothing to do\n");
        return 1;
    }
    std::printf("Loaded %zu positions from '%s'\n",
                data.size(), dataset_path.c_str());

    double K = fit_k(data);
    std::printf("Fitted K = %.6f\n", K);

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

    std::printf("\nFinal tuned weights (baseline defaults in parens):\n");
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
    return 0;
}

}  // namespace tune
