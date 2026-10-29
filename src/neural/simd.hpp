#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace nmo::neural {

inline constexpr std::size_t simd_alignment = 64;

// Zero-initialised heap array aligned for SIMD loads.
template <class T>
class AlignedArray {
public:
    AlignedArray() = default;
    explicit AlignedArray(std::size_t size)
        : data_(static_cast<T*>(::operator new[](size * sizeof(T), std::align_val_t(simd_alignment)))), size_(size) {
        for (std::size_t index = 0; index < size; ++index) data_.get()[index] = T{};
    }
    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    T& operator[](std::size_t index) noexcept { return data_.get()[index]; }
    const T& operator[](std::size_t index) const noexcept { return data_.get()[index]; }

private:
    struct Delete {
        void operator()(T* pointer) const noexcept { ::operator delete[](pointer, std::align_val_t(simd_alignment)); }
    };
    std::unique_ptr<T[], Delete> data_;
    std::size_t size_ = 0;
};

using AlignedFloats = AlignedArray<float>;

// Dot product of two arrays whose length is a multiple of 8.
[[nodiscard]] inline float dot(const float* a, const float* b, int size) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int index = 0;
    for (; index + 16 <= size; index += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + index), _mm256_loadu_ps(b + index), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + index + 8), _mm256_loadu_ps(b + index + 8), sum1);
    }
    for (; index < size; index += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + index), _mm256_loadu_ps(b + index), sum0);
    }
    const __m256 sum = _mm256_add_ps(sum0, sum1);
    __m128 half = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    half = _mm_add_ps(half, _mm_movehl_ps(half, half));
    half = _mm_add_ss(half, _mm_shuffle_ps(half, half, 1));
    return _mm_cvtss_f32(half);
#else
    float sums[8]{};
    for (int index = 0; index < size; index += 8) {
        for (int lane = 0; lane < 8; ++lane) sums[lane] += a[index + lane] * b[index + lane];
    }
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
#endif
}

// Dot product of unsigned 7-bit activations with signed 8-bit weights; the
// length must be a multiple of 32. Activations below 128 keep the paired
// 16-bit sums of _mm256_maddubs_epi16 from saturating.
[[nodiscard]] inline std::int32_t dot_u8i8(const std::uint8_t* a, const std::int8_t* b, int size) noexcept {
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sum = _mm256_setzero_si256();
    for (int index = 0; index < size; index += 32) {
        const __m256i products = _mm256_maddubs_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(a + index)),
                                                       _mm256_load_si256(reinterpret_cast<const __m256i*>(b + index)));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(products, ones));
    }
    __m128i half = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    half = _mm_add_epi32(half, _mm_shuffle_epi32(half, 0x4E));
    half = _mm_add_epi32(half, _mm_shuffle_epi32(half, 0xB1));
    return _mm_cvtsi128_si32(half);
#else
    std::int32_t sum = 0;
    for (int index = 0; index < size; ++index) sum += static_cast<std::int32_t>(a[index]) * b[index];
    return sum;
#endif
}

// out = in + sum of added rows - sum of removed rows (length a multiple of 8).
inline void apply_rows(float* __restrict out, const float* __restrict in, const float* const* added, int added_count,
                       const float* const* removed, int removed_count, int size) noexcept {
    for (int index = 0; index < size; ++index) out[index] = in[index];
    for (int row = 0; row < added_count; ++row) {
        const float* __restrict values = added[row];
        for (int index = 0; index < size; ++index) out[index] += values[index];
    }
    for (int row = 0; row < removed_count; ++row) {
        const float* __restrict values = removed[row];
        for (int index = 0; index < size; ++index) out[index] -= values[index];
    }
}

}  // namespace nmo::neural
