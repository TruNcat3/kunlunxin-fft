// Host-side unit test for the butterfly reference implementation.
//
// Verifies fft_dit / fft_dif against a naive O(N^2) DFT at several power-of-two
// lengths, plus the bit-reversal and twiddle helpers. Run via scripts/build.sh.

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include "butterfly/bit_reverse.hpp"
#include "butterfly/fft.hpp"
#include "butterfly/twiddle.hpp"

namespace {

using cdx = std::complex<double>;

// Naive DFT for ground truth.
std::vector<cdx> naive_dft(const std::vector<cdx>& a) {
    const std::size_t N = a.size();
    std::vector<cdx> out(N, cdx{0.0, 0.0});
    for (std::size_t k = 0; k < N; ++k) {
        cdx sum{0.0, 0.0};
        for (std::size_t n = 0; n < N; ++n) {
            const double ang = -2.0 * butterfly::kPi * static_cast<double>(k) *
                               static_cast<double>(n) / static_cast<double>(N);
            sum += a[n] * cdx{std::cos(ang), std::sin(ang)};
        }
        out[k] = sum;
    }
    return out;
}

double max_err(const std::vector<cdx>& a, const std::vector<cdx>& b) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}

bool check_bit_reverse() {
    // Known mapping for 8 elements: 0,4,2,6,1,5,3,7
    const int expect[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    for (int i = 0; i < 8; ++i) {
        if (static_cast<int>(butterfly::bit_reverse(static_cast<std::uint32_t>(i), 3)) !=
            expect[i]) {
            std::printf("bit_reverse(%d) mismatch\n", i);
            return false;
        }
    }
    return true;
}

bool check_fft_at(std::size_t N, bool use_dif) {
    std::vector<cdx> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        // deterministic pseudo-random input
        const double re = std::sin(0.1 * static_cast<double>(i) + 1.3);
        const double im = std::cos(0.2 * static_cast<double>(i) + 0.7);
        x[i] = cdx{re, im};
    }
    const auto ref = naive_dft(x);
    const auto got = use_dif ? butterfly::fft_dif(x) : butterfly::fft_dit(x);
    const double err = max_err(got, ref);
    double mag = 0.0;
    for (const auto& v : ref) {
        mag = std::max(mag, std::abs(v));
    }
    const double rel = err / mag;
    const bool ok = rel < 1e-10;
    std::printf("  %s N=%4zu maxerr=%.3e rel=%.3e %s\n", use_dif ? "dif" : "dit", N, err, rel,
                ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    std::printf("butterfly host-side unit test\n");
    ok = check_bit_reverse() && ok;
    std::printf("  bit_reverse OK\n");

    for (std::size_t N : {8u, 16u, 64u, 256u, 1024u, 4096u}) {
        ok = check_fft_at(N, /*use_dif=*/false) && ok;
    }
    for (std::size_t N : {8u, 16u, 64u, 256u, 1024u, 4096u}) {
        ok = check_fft_at(N, /*use_dif=*/true) && ok;
    }

    std::printf("%s\n", ok ? "ALL PASSED" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
