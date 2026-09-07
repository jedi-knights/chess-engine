#pragma once
#include "nnue_types.h"

#include <cstdint>

// SIMD kernels for the three NNUE hot loops:
//   - add_column:  acc[i] += col[i]           (feature enters)
//   - sub_column:  acc[i] -= col[i]           (feature leaves)
//   - forward_side: sum_i clip(acc[i]) * w[i] (output layer, one side)
//
// Each kernel has a NEON build (AArch64), an AVX2 build (x86-64 with
// AVX2 available), and a scalar reference. The reference lives in
// `nnue::simd::reference::` so tests can compare it against the
// SIMD-active path element-for-element and catch any lane-ordering
// or saturation regression before it ships.
//
// Kernel selection is compile-time via macros the compiler defines
// under `-march=native`:
//   - __AVX2__      : AVX2 kernel
//   - __ARM_NEON    : NEON kernel
//   - otherwise     : scalar reference
//
// HIDDEN_SIZE (=256) is a compile-time constant divisible by 16, so
// both NEON (8-lane int16) and AVX2 (16-lane int16) trip counts are
// exact — no scalar-tail cleanup path needed.

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace nnue::simd {

namespace reference {

inline void add_column(int32_t* acc, const int16_t* col) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        acc[i] += col[i];
    }
}

inline void sub_column(int32_t* acc, const int16_t* col) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        acc[i] -= col[i];
    }
}

// Clipped-ReLU (clamp to [0, 127]) + int16 dot product. Returns the
// int32 partial sum for one side; caller adds contributions from both
// sides plus the output bias.
inline int32_t forward_side(const int32_t* acc, const int16_t* weights) {
    int32_t sum = 0;
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        int32_t v = acc[i];
        if (v < 0)   { v = 0; }
        if (v > 127) { v = 127; }
        sum += int32_t(weights[i]) * v;
    }
    return sum;
}

}  // namespace reference

#if defined(__AVX2__)

// 8 int32 lanes per iteration, 32 iterations for HIDDEN_SIZE=256.
inline void add_column(int32_t* acc, const int16_t* col) {
    static_assert(HIDDEN_SIZE % 8 == 0, "AVX2 add_column needs HIDDEN_SIZE % 8 == 0");
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        __m128i w16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(col + i));
        __m256i w32 = _mm256_cvtepi16_epi32(w16);
        __m256i a   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i sum = _mm256_add_epi32(a, w32);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), sum);
    }
}

inline void sub_column(int32_t* acc, const int16_t* col) {
    static_assert(HIDDEN_SIZE % 8 == 0, "AVX2 sub_column needs HIDDEN_SIZE % 8 == 0");
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        __m128i w16  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(col + i));
        __m256i w32  = _mm256_cvtepi16_epi32(w16);
        __m256i a    = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i diff = _mm256_sub_epi32(a, w32);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), diff);
    }
}

// Process 16 hidden units per iteration. Clip int32 accumulator to
// [0, 127] (fits in int16 without loss), pack down to int16, then
// use vpmaddwd for int16 × int16 → int32 pair-add.
inline int32_t forward_side(const int32_t* acc, const int16_t* weights) {
    static_assert(HIDDEN_SIZE % 16 == 0, "AVX2 forward_side needs HIDDEN_SIZE % 16 == 0");
    const __m256i lo_clip = _mm256_setzero_si256();
    const __m256i hi_clip = _mm256_set1_epi32(127);
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i + 8));
        a0 = _mm256_max_epi32(a0, lo_clip);
        a0 = _mm256_min_epi32(a0, hi_clip);
        a1 = _mm256_max_epi32(a1, lo_clip);
        a1 = _mm256_min_epi32(a1, hi_clip);
        // packs_epi32 interleaves per-128-bit lane; permute4x64
        // (0xD8 = 0b11_01_10_00 → [0,2,1,3]) restores sequential
        // int16 order so lanes line up with weights[i..i+16).
        __m256i packed = _mm256_packs_epi32(a0, a1);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        __m256i w      = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i));
        __m256i prod   = _mm256_madd_epi16(packed, w);
        sum = _mm256_add_epi32(sum, prod);
    }
    // Horizontal sum of 8 int32 lanes → scalar.
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s4 = _mm_add_epi32(lo, hi);
    __m128i s2 = _mm_hadd_epi32(s4, s4);
    __m128i s1 = _mm_hadd_epi32(s2, s2);
    return _mm_cvtsi128_si32(s1);
}

#elif defined(__ARM_NEON)

// 8 int16 → 2 × 4 int32 lanes per iteration, 32 iterations for
// HIDDEN_SIZE=256. `vaddw_s16` widens the int16 addend to int32
// as part of the add, so no separate widen instruction is needed.
inline void add_column(int32_t* acc, const int16_t* col) {
    static_assert(HIDDEN_SIZE % 8 == 0, "NEON add_column needs HIDDEN_SIZE % 8 == 0");
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        int16x8_t w  = vld1q_s16(col + i);
        int32x4_t lo = vld1q_s32(acc + i);
        int32x4_t hi = vld1q_s32(acc + i + 4);
        lo = vaddw_s16(lo, vget_low_s16(w));
        hi = vaddw_s16(hi, vget_high_s16(w));
        vst1q_s32(acc + i,     lo);
        vst1q_s32(acc + i + 4, hi);
    }
}

inline void sub_column(int32_t* acc, const int16_t* col) {
    static_assert(HIDDEN_SIZE % 8 == 0, "NEON sub_column needs HIDDEN_SIZE % 8 == 0");
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        int16x8_t w  = vld1q_s16(col + i);
        int32x4_t lo = vld1q_s32(acc + i);
        int32x4_t hi = vld1q_s32(acc + i + 4);
        lo = vsubw_s16(lo, vget_low_s16(w));
        hi = vsubw_s16(hi, vget_high_s16(w));
        vst1q_s32(acc + i,     lo);
        vst1q_s32(acc + i + 4, hi);
    }
}

// Process 8 hidden units per iteration. Clip int32 accumulator to
// [0, 127], saturating-narrow to int16, multiply-widen against int16
// weights into int32 partial sums, horizontally reduce at the end.
inline int32_t forward_side(const int32_t* acc, const int16_t* weights) {
    static_assert(HIDDEN_SIZE % 8 == 0, "NEON forward_side needs HIDDEN_SIZE % 8 == 0");
    const int32x4_t lo_clip = vdupq_n_s32(0);
    const int32x4_t hi_clip = vdupq_n_s32(127);
    int32x4_t sum = vdupq_n_s32(0);
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        int32x4_t a0 = vld1q_s32(acc + i);
        int32x4_t a1 = vld1q_s32(acc + i + 4);
        a0 = vminq_s32(vmaxq_s32(a0, lo_clip), hi_clip);
        a1 = vminq_s32(vmaxq_s32(a1, lo_clip), hi_clip);
        // vqmovn saturates on narrow; [0,127] fits without loss.
        int16x4_t p0 = vqmovn_s32(a0);
        int16x4_t p1 = vqmovn_s32(a1);
        int16x8_t packed = vcombine_s16(p0, p1);
        int16x8_t w = vld1q_s16(weights + i);
        // Multiply-widen: two int16×int16→int32 halves.
        int32x4_t p_lo = vmull_s16(vget_low_s16(packed), vget_low_s16(w));
        int32x4_t p_hi = vmull_high_s16(packed, w);
        sum = vaddq_s32(sum, p_lo);
        sum = vaddq_s32(sum, p_hi);
    }
    return vaddvq_s32(sum);
}

#else  // No SIMD available — delegate to reference.

inline void add_column(int32_t* acc, const int16_t* col) {
    reference::add_column(acc, col);
}
inline void sub_column(int32_t* acc, const int16_t* col) {
    reference::sub_column(acc, col);
}
inline int32_t forward_side(const int32_t* acc, const int16_t* weights) {
    return reference::forward_side(acc, weights);
}

#endif

}  // namespace nnue::simd
