#!/usr/bin/env bash
# Build the host-side C++ reference and unit tests, and pre-warm the Triton
# device-code cache.
#
# The P800 device code itself is produced by Triton at runtime (Triton ->
# XMLIR -> XPU instructions). There is no separate device build step, and
# nvcc/PTX binaries cannot execute on this device (see docs/device-stack.md).
# What this script compiles is the CPU-side butterfly reference used by the
# g++ unit test, so the index/twiddle math is verifiable without a device.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -std=c++17 -Wall -Wextra}"

mkdir -p build

echo "==> building host-side C++ reference + unit test"
"$CXX" $CXXFLAGS -Iinclude \
    src/host/fft_check.cpp \
    -o build/fft_check

echo "==> running host-side unit test"
./build/fft_check

if [[ "${SKIP_TRITON_WARM:-0}" != "1" ]]; then
  echo "==> pre-warming Triton device-code cache (first run compiles, can take minutes)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" "$PYTHON_BIN" scripts/warm_triton.py
else
  echo "==> skipping Triton warm (SKIP_TRITON_WARM=1)"
fi

echo "==> build complete"
