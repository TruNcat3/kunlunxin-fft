"""Phase/launch ledger on device (issue #8 PR-F, WP2).

The ledger must be observability-only: values identical with the
ledger on or off, headline path (no active ledger) adds no sync.
Phase names cover the WP2 split (pack/crop, Hermitian, conjugate,
normalization epilogue, core, validate).
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

try:
    import torch
    import torch_xmlir  # noqa: F401
except Exception as e:  # pragma: no cover - absent in ordinary CI
    pytest.skip(f"vendor xpytorch stack unavailable: {e}",
                allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.hardware

from src.xpu.config import DeviceProfile  # noqa: E402
from src.xpu.entry import execute_fft_plan, plan_fft  # noqa: E402
from src.xpu.ledger import (  # noqa: E402
    ACTIVE, PhaseLedger, launch_ledger, measure_phases)

PROF = DeviceProfile.load().hash()


def _x(n, b, seed=7):
    g = torch.Generator().manual_seed(seed)
    if b == 1:
        re = torch.randn(n, dtype=torch.float32, generator=g)
        im = torch.randn(n, dtype=torch.float32, generator=g)
        return torch.complex(re, im).to("cuda")
    re = torch.randn(b, n, dtype=torch.float32, generator=g)
    im = torch.randn(b, n, dtype=torch.float32, generator=g)
    return torch.complex(re, im).to("cuda")


def _err(y, ref):
    a = y.detach().cpu().numpy()
    return float(np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-30))


def _run_with_ledger(plan, x, sync=True):
    led = PhaseLedger(sync=sync)
    tok = ACTIVE.set(led)
    try:
        y = execute_fft_plan(plan, x)
        torch.cuda.synchronize()
    finally:
        ACTIVE.reset(tok)
    return y, led


def test_ledger_values_identical_c2c():
    plan = plan_fft("dit", 512, 4, profile_hash=PROF)
    x = _x(512, 4)
    y0 = execute_fft_plan(plan, x)   # headline path (no ledger)
    y1, led = _run_with_ledger(plan, x, sync=False)
    # vendor stack has no torch.equal (xdnn equal.cpp unimplemented);
    # compare on host instead
    assert np.array_equal(y0.detach().cpu().numpy(),
                          y1.detach().cpu().numpy()), \
        "ledger changed execution results"
    names = [p["name"] for p in led.phases]
    assert names == ["plan-validate", "core-invoke", "epilogue-scale"]


def test_phase_names_r2c_c2r_inverse():
    x512 = _x(512, 4)
    real = x512.real.contiguous()

    r2c = plan_fft("dit", 512, 4, profile_hash=PROF, transform_type="r2c")
    _, led = _run_with_ledger(r2c, real, sync=False)
    assert [p["name"] for p in led.phases] == [
        "plan-validate", "real-input-pack", "core-invoke",
        "epilogue-scale", "half-spectrum-crop"]

    # issue #9 WP2: (512,4) c2r is a closed cell whose promoted
    # winner is the dedicated half-length real_flow chain (guard >=2%
    # over the full-length real_compat path, same err threshold) --
    # the ledger records the deployed flow, not the pre-WP2 one.
    c2r = plan_fft("dit", 512, 4, profile_hash=PROF,
                   transform_type="c2r", transform_direction="inverse")
    half = _x(512 // 2 + 1, 4)
    _, led = _run_with_ledger(c2r, half, sync=False)
    assert [p["name"] for p in led.phases] == [
        "plan-validate", "half-spectrum-pre",
        "core-invoke", "real-output-epilogue"]

    inv = plan_fft("dit", 512, 4, profile_hash=PROF,
                   transform_direction="inverse")
    _, led = _run_with_ledger(inv, x512, sync=False)
    assert [p["name"] for p in led.phases] == [
        "plan-validate", "inverse-conj", "core-invoke", "epilogue-scale"]


def test_ledger_correctness_vs_numpy_with_sync():
    x = _x(512, 4)
    plan = plan_fft("dit", 512, 4, profile_hash=PROF)
    y, led = _run_with_ledger(plan, x, sync=True)
    ref = np.fft.fft(x.detach().cpu().numpy().astype(np.complex128),
                     axis=-1)
    assert _err(y, ref) <= 1e-4
    s = led.summary()
    assert s["sync_added"] is True
    assert s["kind"].startswith("fresh measured")
    assert not s["errors"]


def test_measure_phases_aggregates():
    x = _x(512, 4)
    plan = plan_fft("dit", 512, 4, profile_hash=PROF)
    out = measure_phases(plan, x, trials=7, warmup=3)
    assert out["kind"].startswith("fresh measured")
    assert out["sync_added"] is True
    assert out["trials"] == 7
    assert out["trial_wall_p50_ms"] > 0
    assert out["phase_sum_mean_ms"] > 0
    for name in ("plan-validate", "core-invoke", "epilogue-scale"):
        assert out["phases"][name]["calls"] == 7, name
    # json-clean
    import json
    json.dumps(out)


def test_headline_path_has_no_active_ledger():
    assert ACTIVE.get() is None
    x = _x(256, 4)
    plan = plan_fft("dit", 256, 4, profile_hash=PROF)
    execute_fft_plan(plan, x)
    assert ACTIVE.get() is None


def test_launch_ledger_on_device_plan():
    x = _x(512, 4)
    plan = plan_fft("dit", 512, 4, profile_hash=PROF)
    lg = launch_ledger(plan)
    assert lg["kind"].startswith("analytic estimate")
    if lg["available"]:
        assert lg["launches"] > 0 and lg["gm_bytes"] > 0
        assert lg["family"] in (
            "baseline_m0_rank1_loop", "incumbent_m4_folded", "m5_batch",
            "m2_batch", "m2_packet")
    else:
        assert lg["why"]          # honest absence, never invented
    y = execute_fft_plan(plan, x)
    assert y is not None
