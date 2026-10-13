#pragma once
// Twiddle-factor helpers for the radix-2 butterfly.
//
// The device kernels index a planar cos/sin table of exp(-2*pi*i*idx/N) for
// idx in [0, N/2). A stage with butterfly half-span `n` reads W[j * M] where
// M = N / (2n) and j = k % n. This header mirrors that addressing on the CPU.

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace butterfly {

constexpr double kPi = 3.14159265358979323846;

// exp(-2*pi*i*idx/N)
inline std::complex<double> twiddle(std::size_t idx, std::size_t n) {
    const double ang = -2.0 * kPi * static_cast<double>(idx) / static_cast<double>(n);
    return {std::cos(ang), std::sin(ang)};
}

// Planar table of exp(-2*pi*i*idx/N) for idx in [0, n/2). The stage with
// half-span `n_stage` reads entry j * M with M = N / (2 * n_stage).
inline std::vector<std::complex<double>> twiddle_table(std::size_t n) {
    std::vector<std::complex<double>> w(n / 2);
    for (std::size_t idx = 0; idx < n / 2; ++idx) {
        w[idx] = twiddle(idx, n);
    }
    return w;
}

// Twiddle stride M = N / (2 * n_stage) for a stage with half-span n_stage.
inline std::size_t twiddle_stride(std::size_t n, std::size_t n_stage) {
    return n / (2 * n_stage);
}

}  // namespace butterfly
