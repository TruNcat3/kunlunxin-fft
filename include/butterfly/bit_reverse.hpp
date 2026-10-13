#pragma once
// Bit-reversal permutation for power-of-two FFTs.
//
// This header is the host-side (CPU) reference for the index math that the
// Triton device kernels implement. Keeping it header-only and dependency-free
// lets scripts/build.sh verify the butterfly layout with plain g++ and no
// device present.

#include <cstdint>
#include <cstddef>

namespace butterfly {

// Reverse the low `bits` bits of `x`. `bits` must be <= 31.
inline std::uint32_t bit_reverse(std::uint32_t x, int bits) {
    std::uint32_t rev = 0;
    for (int i = 0; i < bits; ++i) {
        rev = (rev << 1) | (x & 1u);
        x >>= 1;
    }
    return rev;
}

// True iff `n` is a power of two and non-zero.
inline bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

// log2 of a power-of-two `n`; requires is_power_of_two(n).
inline int log2_exact(std::size_t n) {
    int b = 0;
    while ((std::size_t{1} << b) != n) {
        ++b;
    }
    return b;
}

}  // namespace butterfly
