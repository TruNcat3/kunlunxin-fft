"""WP3 trace/profile: per-path device-time attribution on P800.

Requires (cupti device tracing):
  XPU_ENABLE_PROFILER_TRACING=1 XPU_CUPTI_ENABLE_DEVICE=0

Outputs:
  results/p800/issue10_capability/wp3_profile.json
  results/p800/issue10_capability/wp3_trace/*.json.gz  (chrome traces)
"""
import sys, json, time, importlib.util, subprocess
from pathlib import Path
from collections import Counter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch_xmlir  # noqa: F401  (must precede triton/xpu kernel launches)
from torch.profiler import profile, ProfilerActivity

spec = importlib.util.spec_from_file_location(
    "wp3", str(ROOT / "scripts/issue10_wp3_perf.py"))
wp3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wp3)

OUT = ROOT / "results/p800/issue10_capability"
TR = OUT / "wp3_trace"
TR.mkdir(parents=True, exist_ok=True)
ITERS = 10
CELLS = [(1024, 64), (2048, 16)]

summary = {
    "schema": "issue10-wp3-profile-v1",
    "env": {"XPU_ENABLE_PROFILER_TRACING": "1",
            "XPU_CUPTI_ENABLE_DEVICE": "0",
            "activities": ["CPU", "CUDA (cupti via xpu runtime)"]},
    "note": "device time from cupti kernel events; launch overhead from "
            "cudaLaunchKernel self CPU time; wall_ms_per_iter includes "
            "profiling overhead -- use wp3_summary.json p50 for latency",
    "cells": {},
}


def kernel_cat(name):
    return "vendor_fft_kernel" if "fft" in name.lower() else "triton_kernel"


def cpu_cat(name):
    nl = name.lower()
    if any(k in nl for k in ("clone", "permute", "contiguous", "as_strided",
                             "view", "detach")):
        return "host_layout_copy"
    if any(k in nl for k in ("mul", "div", "add", "sub", "real", "imag",
                             "conj", "neg", "empty", "zeros")):
        return "host_elementwise"
    if "cuda" in nl or "launch" in nl:
        return "launch_runtime"
    return "host_other"


for n, b in CELLS:
    x = wp3._data(b, n)
    cell = {}
    for pname, runner in wp3.PATHS.items():
        try:
            call, meta = runner(x, False)
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            t0 = time.time()
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA],
                         record_shapes=False) as prof:
                for _ in range(ITERS):
                    call()
                torch.cuda.synchronize()
            wall_ms = (time.time() - t0) * 1000.0 / ITERS
            cats, topl = Counter(), []
            for ev in prof.key_averages():
                if not ev.count:
                    continue
                dev = (getattr(ev, "self_device_time_total", 0)
                       or getattr(ev, "self_cuda_time_total", 0) or 0)
                cpu = ev.self_cpu_time_total or 0
                if "Launch" in ev.key or "cudaLaunch" in ev.key:
                    cats["launch_cpu_us"] += cpu
                    dev = 0
                elif dev > 0:
                    cats[kernel_cat(ev.key)] += dev
                else:
                    cats[cpu_cat(ev.key) + "_cpu_us"] += cpu
                topl.append({"op": ev.key, "count": ev.count,
                             "dev_us_total": round(dev, 1),
                             "dev_us_per": round(dev / ev.count, 1),
                             "cpu_us_per": round(cpu / ev.count, 1)})
            topl.sort(key=lambda d: -d["dev_us_total"])
            cats = Counter({k: v for k, v in cats.items() if v > 0})
            cats_ms = {k: round(v / 1000.0 / ITERS, 4) for k, v in
                       sorted(cats.items(), key=lambda kv: -kv[1])}
            cell[pname] = {
                "wall_ms_per_iter": round(wall_ms, 3),
                "launches": meta.get("launches"),
                "breakdown_ms_per_iter": cats_ms,
                "top_ops": topl[:8], "status": "OK"}
            if pname in ("triton_resident_loop", "hybrid_vendor_p",
                         "triton_multipass"):
                tmp = str(TR / f"{pname}_{n}x{b}_fwd.json")
                prof.export_chrome_trace(tmp)
                subprocess.run(["gzip", "-f", tmp], check=True)
                cell[pname]["trace"] = f"wp3_trace/{Path(tmp).name}.gz"
            print(f"n={n} b={b} {pname}: wall={wall_ms:.3f}ms {cats_ms}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            cell[pname] = {"status": "FAIL",
                           "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"n={n} b={b} {pname}: FAIL {e}", flush=True)
    summary["cells"][f"{n}x{b}"] = cell

(OUT / "wp3_profile.json").write_text(json.dumps(summary, indent=2))
print("written wp3_profile.json")
