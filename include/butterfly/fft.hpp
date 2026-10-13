#pragma once
// Host-side (CPU) reference for the radix-2 DIT/DIF butterfly dataflow that the
// Triton device kernels implement.
//
// The index mapping matches src/xpu/kernels.py exactly, so this header is the
// specification the device code is checked against:
//
//   stage with half-span n: for k in [0, N/2)
//       j = k % n
//       i = ((k // n) * 2n) + j
//       pair (i, i+n), twiddle W[j * M], M = N / (2n)
//
// DIT applies the twiddle to the second input then writes (a+b, a-b).
// DIF writes (a+b, (a-b)*w). Both run log2(N) stages; the input or output
// bit-reversal permutation can be folded into the first or last stage.

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "butterfly/bit_reverse.hpp"
#include "butterfly/twiddle.hpp"

namespace butterfly {

// One radix-2 DIT butterfly stage, in place.
inline void dit_stage(std::vector<std::complex<double>>& a, std::size_t n_stage) {
    const std::size_t N = a.size();
    const std::size_t M = twiddle_stride(N, n_stage);
    const auto w = twiddle_table(N);
    for (std::size_t k = 0; k < N / 2; ++k) {
        const std::size_t j = k % n_stage;
        const std::size_t i = ((k / n_stage) * 2 * n_stage) + j;
        const std::complex<double> x = a[i];
        const std::complex<double> y = a[i + n_stage] * w[j * M];
        a[i] = x + y;
        a[i + n_stage] = x - y;
    }
}

// One radix-2 DIF butterfly stage, in place.
inline void dif_stage(std::vector<std::complex<double>>& a, std::size_t n_stage) {
    const std::size_t N = a.size();
    const std::size_t M = twiddle_stride(N, n_stage);
    const auto w = twiddle_table(N);
    for (std::size_t k = 0; k < N / 2; ++k) {
        const std::size_t j = k % n_stage;
        const std::size_t i = ((k / n_stage) * 2 * n_stage) + j;
        const std::complex<double> x = a[i];
        const std::complex<double> y = a[i + n_stage];
        a[i] = x + y;
        a[i + n_stage] = (x - y) * w[j * M];
    }
}

// Permute `a` into bit-reversed order, out of place.
inline std::vector<std::complex<double>> bit_reverse_permute(
    const std::vector<std::complex<double>>& a) {
    const std::size_t N = a.size();
    const int bits = log2_exact(N);
    std::vector<std::complex<double>> out(N);
    for (std::size_t i = 0; i < N; ++i) {
        out[bit_reverse(static_cast<std::uint32_t>(i), bits)] = a[i];
    }
    return out;
}

// Forward complex FFT via DIT stages (explicit bit-reversal first).
inline std::vector<std::complex<double>> fft_dit(std::vector<std::complex<double>> a) {
    const std::size_t N = a.size();
    if (!is_power_of_two(N)) {
        throw std::invalid_argument("fft_dit requires a power-of-two length");
    }
    a = bit_reverse_permute(a);
    for (std::size_t n_stage = 1; n_stage < N; n_stage *= 2) {
        dit_stage(a, n_stage);
    }
    return a;
}

// Forward complex FFT via DIF stages (bit-reversal folded into the last stage
// by permuting the finished result, matching the kernel's BITREV_OUTPUT).
inline std::vector<std::complex<double>> fft_dif(std::vector<std::complex<double>> a) {
    const std::size_t N = a.size();
    if (!is_power_of_two(N)) {
        throw std::invalid_argument("fft_dif requires a power-of-two length");
    }
    for (std::size_t n_stage = N / 2; n_stage >= 1; n_stage /= 2) {
        dif_stage(a, n_stage);
        if (n_stage == 1) {
            break;  // guard against unsigned wrap
        }
    }
    return bit_reverse_permute(a);
}

}  // namespace butterfly
