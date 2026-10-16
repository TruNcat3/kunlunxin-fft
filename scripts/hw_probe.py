#!/usr/bin/env python3
"""P1.2 hardware probe + calibration profile for the P800.

Runs on the device (``CUDA_VISIBLE_DEVICES=6`` on this box) and writes
``config/calibration.json`` containing:

* **versions** — python/torch/triton, git revision, device profile hash;
* **verified paths** — execution paths re-checked this round (M0/M2/M5
  batch lowerings, Td=2 packet path, vendor ``torch.fft.fft``);
* **resource constraints** — static config plus observed device props;
* **baseline costs** — median launch / sync / H2D / D2H / matmul /
  vendor-FFT timings (the cost model's anchors);
* **vendor re-probe** — vendor support re-probed this round (the issue
  forbids reusing stale conclusions).

The file is self-hashed; ``src/xpu/calibration.load`` refuses mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_release_matrix as relm  # noqa: E402  (import-safe module)
import triton  # noqa: E402
import triton.language as tl  # noqa: E402  (kernel namespace)

CALIBRATION_PATH = ROOT / "config" / "calibration.json"

VENDOR_SWEEP = ((16, 1), (1024, 16), (2048, 64), (4096, 16),
                (16384, 1), (65536, 1), (1048576, 1))
COPY_MIB = 64
MM_N = 1024


# --------------------------------------------------------------------------
# small building blocks
# --------------------------------------------------------------------------
def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def _timeit(fn, warmup, trials):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _versions():
    import torch
    import triton

    try:
        rev = relm._git("rev-parse", "HEAD")
    except Exception:  # noqa: BLE001
        rev = "unknown"
    try:
        dirty = bool(relm._git("status", "--porcelain"))
    except Exception:  # noqa: BLE001
        dirty = None
    from src.xpu.config import DeviceProfile
    prof = DeviceProfile.load()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "git_rev": rev,
        "git_dirty": dirty,
        "device_profile_hash": prof.hash(),
        "device": prof.device,
        "driver": prof.driver,
        "device_code_path": prof.device_code_path,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _resource_constraints(observed):
    from src.xpu.config import DeviceProfile
    prof = DeviceProfile.load()
    return {
        "static": {
            "xpu_arch": prof.xpu_arch,
            "clusters": prof.clusters,
            "cores_per_cluster": prof.cores_per_cluster,
            "shared_mem_bytes": prof.shared_mem_bytes,
            "hbm_gib": prof.hbm_gib,
        },
        "observed": observed,
    }


def _observe_device():
    import torch

    out = {}
    try:
        props = torch.cuda.get_device_properties(0)
        out["name"] = getattr(props, "name", None)
        out["total_memory_bytes"] = getattr(props, "total_memory", None)
        out["multi_processor_count"] = getattr(props, "multi_processor_count",
                                               None)
    except Exception as e:  # noqa: BLE001
        out["properties_error"] = f"{type(e).__name__}: {e}"
    try:
        out["device_count"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        out["device_count_error"] = f"{type(e).__name__}: {e}"
    return out


def _launch_probe():
    """One tiny elementwise kernel: baseline launch + sync cost."""
    import torch

    @triton.jit
    def _bump(x_ptr, o_ptr):
        i = tl.arange(0, 64)
        v = tl.load(x_ptr + i)
        tl.store(o_ptr + i, v + 1.0)

    x = torch.zeros(64, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    grid = (1,)
    _bump[grid](x, o)
    torch.cuda.synchronize()
    return x, o, _bump, grid, 64


def _verified_paths():
    """Re-check the paths this round; record status, never reuse claims."""
    from src.xpu import runner

    paths = []

    def add(name, fn, ref=None):
        try:
            y = fn()
            detail = None
            if ref is not None:
                err = relm._err_vs_np(y, ref)
                if not relm._err_gate(err):
                    paths.append({"path": name, "status": "wrong-value",
                                  "max_rel_error": err})
                    return
                detail = f"max_rel_error={err:.2e}"
            paths.append({"path": name, "status": "ok", "detail": detail})
        except Exception as e:  # noqa: BLE001
            paths.append({"path": name, "status": "error",
                          "detail": f"{type(e).__name__}: {e}"})

    n, b = 64, 4
    seed = 20261015
    x = relm._make_input(n, b, seed)
    ref = relm._ref_np(n, b, seed)

    add("triton-elementwise-launch", lambda: _launch_probe()[1])
    add("m0-global-stage-dit-n64-b1",
        lambda: runner.fft_dit(relm._make_input(n, 1, seed), block=1024),
        relm._ref_np(n, 1, seed))
    add("m0-global-stage-batch-folded-n64-b4",
        lambda: runner.fft_dit_batch(x, block=1024, variant="folded"), ref)
    add("m2-matrix-batch-spread-n64-b4",
        lambda: runner.fft_dit_matrix_batch(x, block=1024, tile=64, rows=8,
                                            variant="spread"), ref)
    add("m5-radix4-batch-spread-n64-b4",
        lambda: runner.fft_dit_radix4_batch(x, block=1024, variant="spread"),
        ref)
    add("m2-packet-td2-n64-b4",
        lambda: runner.fft_dit_matrix_packet_batch(x, block=1024, tile=64,
                                                   rows=8, packets=2), ref)
    return paths


def _vendor_reprobe():
    """Vendor FFT support re-probed this round (never reuse old claims)."""
    import torch

    out = []
    for n, b in VENDOR_SWEEP:
        try:
            x = torch.randn(b, n, dtype=torch.float32, device="cuda")
            torch.fft.fft(x, dim=-1)
            torch.cuda.synchronize()
            out.append({"N": n, "batch": b, "status": "ok"})
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:200]
            out.append({"N": n, "batch": b,
                        "status": f"unsupported: {type(e).__name__}: {msg}"})
    return out


def _baseline_costs():
    import torch

    # launch + sync
    x, o, bump, grid, _n = _launch_probe()
    launch_ms = _median(_timeit(lambda: bump[grid](x, o), 5, 50))
    # FFT stage kernels launch multi-block grids; anchor the per-launch
    # term at the worse of {1-block, 1024-block} dispatch
    xb = torch.zeros(1024 * 64, device="cuda", dtype=torch.float32)
    ob = torch.empty_like(xb)
    bump[(1024,)](xb, ob)
    torch.cuda.synchronize()
    launch_ms_g1024 = _median(_timeit(lambda: bump[(1024,)](xb, ob), 5, 50))
    sync_ms = _median(_timeit(lambda: torch.cuda.synchronize(), 5, 50))

    # copies (64 MiB)
    elems = (COPY_MIB * 1024 * 1024) // 4
    host = torch.randn(elems, dtype=torch.float32)
    dev = host.to("cuda")
    torch.cuda.synchronize()
    h2d = _median(_timeit(lambda: host.to("cuda"), 2, 10))
    d2h = _median(_timeit(lambda: dev.cpu(), 2, 10))
    gib = COPY_MIB / 1024.0

    # device-internal copy: the cost model's global-memory bandwidth anchor
    # (H2D/D2H cross the host link and must not stand in for GM traffic)
    src = torch.randn(elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    torch.cuda.synchronize()
    d2d = _median(_timeit(lambda: dst.copy_(src), 2, 10))

    # compute anchor: fp32 matmul
    a = torch.randn(MM_N, MM_N, device="cuda", dtype=torch.float32)
    bmat = torch.randn(MM_N, MM_N, device="cuda", dtype=torch.float32)
    mm_ms = _median(_timeit(lambda: torch.mm(a, bmat), 3, 10))
    gflops = 2.0 * MM_N ** 3 / (mm_ms * 1e-3) / 1e9

    # vendor compute anchor at a supported shape
    vx = torch.randn(64, 2048, dtype=torch.float32, device="cuda")
    fft_ms = _median(_timeit(lambda: torch.fft.fft(vx, dim=-1), 3, 20))

    return {
        "triton_launch_ms": launch_ms,
        "triton_launch_ms_g1024": launch_ms_g1024,
        "cuda_synchronize_ms": sync_ms,
        "h2d_64mib_ms": h2d,
        "h2d_gib_per_s": gib / (h2d * 1e-3),
        "d2h_64mib_ms": d2h,
        "d2h_gib_per_s": gib / (d2h * 1e-3),
        "d2d_64mib_ms": d2d,
        "d2d_gib_per_s": gib / (d2d * 1e-3),
        "torch_mm_1024_fp32_ms": mm_ms,
        "torch_mm_gflops": gflops,
        "vendor_fft_b64n2048_ms": fft_ms,
        "timing_protocol": "warmup then N trials, wall time with explicit "
                           "torch.cuda.synchronize(), median",
    }


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(CALIBRATION_PATH))
    ap.add_argument("--vendor-only", action="store_true",
                    help="only re-probe vendor support (fast)")
    args = ap.parse_args()

    import torch
    try:
        import torch_xmlir  # noqa: F401
    except ImportError:
        print("torch_xmlir not importable; probing without the XPU runtime "
              "will misreport device paths", file=sys.stderr)

    if not torch.cuda.is_available() and not args.vendor_only:
        raise SystemExit("no device visible; run under CUDA_VISIBLE_DEVICES")

    profile = {
        "schema_version": 1,
        "kind": "p800-calibration",
        "versions": _versions(),
        "vendor_reprobe": _vendor_reprobe(),
    }
    if not args.vendor_only:
        profile["verified_paths"] = _verified_paths()
        profile["resource_constraints"] = _resource_constraints(
            _observe_device())
        profile["baseline_costs"] = _baseline_costs()

    from src.xpu.calibration import compute_hash
    # Bind the profile to the live device/runtime/compiler fingerprint
    # (issue #3 review fix): self-hash proves integrity, env_hash proves
    # provenance. An unprobeable environment refuses to write.
    from src.xpu.envcheck import collect as collect_env
    from src.xpu.envcheck import env_hash as fingerprint_env
    env = collect_env()
    if env is None:
        raise SystemExit(
            "live environment fingerprint unavailable (xpu-smi / "
            "torch / triton / device props); refusing to write an "
            "unbound calibration profile")
    profile["env"] = env
    profile["env_hash"] = fingerprint_env(env)
    profile["profile_hash"] = compute_hash(profile)

    out = Path(args.out)
    out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} profile_hash={profile['profile_hash']}")
    ok = [p["path"] for p in profile.get("verified_paths", [])
          if p["status"] == "ok"]
    bad = [p for p in profile.get("verified_paths", [])
           if p["status"] != "ok"]
    print(f"verified paths: {len(ok)} ok"
          + (f", {len(bad)} NOT ok: {bad}" if bad else ""))
    for v in profile["vendor_reprobe"]:
        print(f"  vendor N={v['N']} B={v['batch']}: {v['status'][:70]}")


if __name__ == "__main__":
    main()
