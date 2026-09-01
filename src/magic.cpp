#include "magic.h"
#include "bitboard.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Per-square precomputed data set up by init_magic().
Bitboard BISHOP_MASK [NUM_SQUARES];
Bitboard ROOK_MASK   [NUM_SQUARES];
uint64_t BISHOP_MAGIC[NUM_SQUARES];
uint64_t ROOK_MAGIC  [NUM_SQUARES];
int      BISHOP_SHIFT[NUM_SQUARES];
int      ROOK_SHIFT  [NUM_SQUARES];

// Fixed-size attack tables. Bishop needs at most 2^9 = 512 entries per
// square (max relevant bits is 9 for center squares); rook needs 2^12
// = 4096 (corner rook has 12 relevant bits). Uniform sizing simplifies
// the lookup at a modest memory cost (~2.4 MB total, .bss so no impact
// on binary size).
Bitboard BISHOP_ATTACKS[NUM_SQUARES][512];
Bitboard ROOK_ATTACKS  [NUM_SQUARES][4096];

// Reference slow attack generation — walks rays including edges. Used
// to populate the lookup tables at init and to verify magic hashes
// during the search. Never called on the hot path.
Bitboard slow_rook_attacks(Square s, Bitboard occ) {
    Bitboard atk = 0;
    int f = file_of(s);
    int r = rank_of(s);
    auto walk = [&](int df, int dr) {
        int ff = f + df;
        int rr = r + dr;
        while (ff >= 0 && ff <= 7 && rr >= 0 && rr <= 7) {
            Bitboard bb = square_bb(make_square(File(ff), Rank(rr)));
            atk |= bb;
            if ((occ & bb) != 0U) {
                break;
            }
            ff += df; rr += dr;
        }
    };
    walk( 0,  1); walk( 0, -1); walk( 1,  0); walk(-1,  0);
    return atk;
}

Bitboard slow_bishop_attacks(Square s, Bitboard occ) {
    Bitboard atk = 0;
    int f = file_of(s);
    int r = rank_of(s);
    auto walk = [&](int df, int dr) {
        int ff = f + df;
        int rr = r + dr;
        while (ff >= 0 && ff <= 7 && rr >= 0 && rr <= 7) {
            Bitboard bb = square_bb(make_square(File(ff), Rank(rr)));
            atk |= bb;
            if ((occ & bb) != 0U) {
                break;
            }
            ff += df; rr += dr;
        }
    };
    walk( 1,  1); walk( 1, -1); walk(-1,  1); walk(-1, -1);
    return atk;
}

// Relevant-blocker mask: subset of the slider's attack rays EXCLUDING
// the edge squares. Blockers on the edge don't affect what lies beyond
// (nothing does), so leaving them out of the mask lets us pack the
// hash into fewer bits.
Bitboard rook_mask(Square s) {
    Bitboard m = 0;
    int f = file_of(s);
    int r = rank_of(s);
    for (int rr = r + 1; rr <= 6; ++rr) {
        m |= square_bb(make_square(File(f),  Rank(rr)));
    }
    for (int rr = r - 1; rr >= 1; --rr) {
        m |= square_bb(make_square(File(f),  Rank(rr)));
    }
    for (int ff = f + 1; ff <= 6; ++ff) {
        m |= square_bb(make_square(File(ff), Rank(r)));
    }
    for (int ff = f - 1; ff >= 1; --ff) {
        m |= square_bb(make_square(File(ff), Rank(r)));
    }
    return m;
}

Bitboard bishop_mask(Square s) {
    Bitboard m = 0;
    int f = file_of(s);
    int r = rank_of(s);
    for (int ff = f+1, rr = r+1; ff <= 6 && rr <= 6; ++ff, ++rr) {
        m |= square_bb(make_square(File(ff), Rank(rr)));
    }
    for (int ff = f+1, rr = r-1; ff <= 6 && rr >= 1; ++ff, --rr) {
        m |= square_bb(make_square(File(ff), Rank(rr)));
    }
    for (int ff = f-1, rr = r+1; ff >= 1 && rr <= 6; --ff, ++rr) {
        m |= square_bb(make_square(File(ff), Rank(rr)));
    }
    for (int ff = f-1, rr = r-1; ff >= 1 && rr >= 1; --ff, --rr) {
        m |= square_bb(make_square(File(ff), Rank(rr)));
    }
    return m;
}

// splitmix64 PRNG — deterministic, seedable, high-quality. Same seed at
// every startup means the same magic numbers get found in the same
// order, so builds are reproducible without shipping the magics.
uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Sparse random: AND three PRNG outputs so the result has few set bits
// (typical: 8-12 bits set). Magic candidates with few bits distribute
// the multiplied product more evenly across the hash's high bits, and
// are empirically far more likely to be collision-free.
uint64_t sparse_random(uint64_t& state) {
    return splitmix64(state) & splitmix64(state) & splitmix64(state);
}

// Enumerate every subset of `mask`'s bits using the Carry-Rippler trick.
// Yields subsets in a Gray-code order, visiting each of 2^popcount(mask)
// exactly once and ending at 0.
class SubsetIter {
public:
    explicit SubsetIter(Bitboard mask) : mask_(mask), subset_(0), first_(true) {}
    bool next(Bitboard& out) {
        if (first_) { first_ = false; out = 0; return true; }
        subset_ = (subset_ - mask_) & mask_;
        if (subset_ == 0) {
            return false;
        }
        out = subset_;
        return true;
    }
private:
    Bitboard mask_, subset_;
    bool first_;
};

// Randomized search for a magic multiplier: try candidates until one
// produces no destructive collisions across all subsets of `mask`.
// Constructive collisions (two blocker patterns hashing to the same
// index but producing the same attack set) are allowed and common.
uint64_t find_magic(Square s, Bitboard mask, bool is_rook, uint64_t& seed,
                    Bitboard* out_table) {
    const int n_bits     = popcount(mask);
    const int table_size = 1 << n_bits;
    const int shift      = 64 - n_bits;

    std::vector<Bitboard> occupancies(table_size);
    std::vector<Bitboard> attacks    (table_size);
    {
        SubsetIter it(mask);
        Bitboard sub;
        for (int i = 0; it.next(sub); ++i) {
            occupancies[i] = sub;
            attacks    [i] = is_rook ? slow_rook_attacks(s, sub)
                                     : slow_bishop_attacks(s, sub);
        }
    }

    std::vector<Bitboard> used(table_size);
    for (int attempt = 0; attempt < 100'000'000; ++attempt) {
        uint64_t magic = sparse_random(seed);
        // Heuristic filter: magics whose product with the mask has few
        // high bits set are almost always losers — skip early.
        if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6) {
            continue;
        }

        std::fill(used.begin(), used.end(), 0ULL);
        bool ok = true;
        for (int j = 0; j < table_size; ++j) {
            size_t idx = static_cast<size_t>((occupancies[j] * magic) >> shift);
            if (used[idx] == 0) {
                used[idx] = attacks[j];
            } else if (used[idx] != attacks[j]) {
                ok = false;
                break;
            }
        }
        if (ok) {
            for (int j = 0; j < table_size; ++j) {
                out_table[j] = used[j];
            }
            return magic;
        }
    }
    // No magic found in the attempt budget — should never happen with
    // sparse randoms on standard chess. Report loudly rather than
    // silently returning a bogus magic.
    std::fprintf(stderr, "magic: no magic found for square %d\n", int(s));
    std::abort();
}

}  // namespace

void init_magic() {
    uint64_t seed = 0x1F63A9BC12345678ULL;
    for (int s = 0; s < NUM_SQUARES; ++s) {
        BISHOP_MASK [s] = bishop_mask(Square(s));
        ROOK_MASK   [s] = rook_mask(Square(s));
        BISHOP_SHIFT[s] = 64 - popcount(BISHOP_MASK[s]);
        ROOK_SHIFT  [s] = 64 - popcount(ROOK_MASK  [s]);

        BISHOP_MAGIC[s] = find_magic(Square(s), BISHOP_MASK[s], /*is_rook=*/false,
                                     seed, BISHOP_ATTACKS[s]);
        ROOK_MAGIC  [s] = find_magic(Square(s), ROOK_MASK[s],   /*is_rook=*/true,
                                     seed, ROOK_ATTACKS  [s]);
    }
}

Bitboard bishop_attacks(Square s, Bitboard occ) {
    Bitboard blockers = occ & BISHOP_MASK[s];
    size_t   idx      = static_cast<size_t>((blockers * BISHOP_MAGIC[s]) >> BISHOP_SHIFT[s]);
    return BISHOP_ATTACKS[s][idx];
}

Bitboard rook_attacks(Square s, Bitboard occ) {
    Bitboard blockers = occ & ROOK_MASK[s];
    size_t   idx      = static_cast<size_t>((blockers * ROOK_MAGIC[s]) >> ROOK_SHIFT[s]);
    return ROOK_ATTACKS[s][idx];
}
