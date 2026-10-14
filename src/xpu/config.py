"""Configuration loading and run manifests (host-only, no torch import).

This module is importable without the vendor xpytorch stack so that host-only
tests and manifest generation run in ordinary CI. It loads the P800 hardware
profile, the design space, and the size/test matrix from ``config/``, validates
them against a small schema, and emits a per-run manifest that pins every field
that could affect generated code or launch geometry.

The manifest is the provenance record required by the migration convergence
plan: every published timing row must be regenerable from its manifest without
hand-editing.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_DIR = os.path.join(_ROOT, "config")


class ConfigError(ValueError):
    """Raised when a config file is missing, unreadable, or fails its schema."""


def _read_json(name: str) -> dict:
    path = os.path.join(_CONFIG_DIR, name)
    if not os.path.exists(path):
        raise ConfigError(f"missing config file: {path}")
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path} is not valid JSON: {e}") from e


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class DeviceProfile:
    """Validated P800 hardware profile (from config/device.json)."""

    device: str
    board: str
    driver: str
    xpu_arch: int
    clusters: int
    cores_per_cluster: int
    shared_mem_bytes: int
    hbm_gib: int
    device_code_path: str
    torch_device_key: str
    baseline: str
    vendor_fft_max_n: int

    @staticmethod
    def load() -> "DeviceProfile":
        d = _read_json("device.json")
        for key in ("device", "board", "driver", "device_code_path", "torch_device_key", "baseline"):
            _require(isinstance(d.get(key), str) and d[key], f"device.json: {key} must be a non-empty string")
        for key in ("xpu_arch", "clusters", "cores_per_cluster", "shared_mem_bytes", "hbm_gib", "vendor_fft_max_n"):
            _require(isinstance(d.get(key), int) and d[key] > 0, f"device.json: {key} must be a positive int")
        return DeviceProfile(
            device=d["device"],
            board=d["board"],
            driver=d["driver"],
            xpu_arch=d["xpu_arch"],
            clusters=d["clusters"],
            cores_per_cluster=d["cores_per_cluster"],
            shared_mem_bytes=d["shared_mem_bytes"],
            hbm_gib=d["hbm_gib"],
            device_code_path=d["device_code_path"],
            torch_device_key=d["torch_device_key"],
            baseline=d["baseline"],
            vendor_fft_max_n=d["vendor_fft_max_n"],
        )

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SizeMatrix:
    """Validated length lists (from config/sizes.json)."""

    correctness_sizes: tuple
    bench_sizes: tuple
    blocksizes: tuple
    vendor_baseline_max_n: int

    @staticmethod
    def load() -> "SizeMatrix":
        d = _read_json("sizes.json")
        for key in ("correctness_sizes", "bench_sizes", "blocksizes"):
            v = d.get(key)
            _require(isinstance(v, list) and v, f"sizes.json: {key} must be a non-empty list")
            _require(all(isinstance(x, int) and x > 0 for x in v), f"sizes.json: {key} must be positive ints")
            _require(all((x & (x - 1)) == 0 for x in v), f"sizes.json: {key} must be powers of two")
        _require(
            isinstance(d.get("vendor_baseline_max_n"), int),
            "sizes.json: vendor_baseline_max_n must be an int",
        )
        return SizeMatrix(
            correctness_sizes=tuple(d["correctness_sizes"]),
            bench_sizes=tuple(d["bench_sizes"]),
            blocksizes=tuple(d["blocksizes"]),
            vendor_baseline_max_n=d["vendor_baseline_max_n"],
        )


@dataclass(frozen=True)
class DesignSpace:
    """Validated design space (from config/butterfly_space.json)."""

    space_id: str
    target: str
    radix: tuple
    dit_bitrev_input_fold: tuple
    dif_bitrev_output_fold: tuple
    fused_tile_candidates: tuple
    batch_candidates: tuple
    batch_variants: tuple
    layout_in: str
    layout_out: str

    @staticmethod
    def load() -> "DesignSpace":
        d = _read_json("butterfly_space.json")
        _require(isinstance(d.get("space_id"), str) and d["space_id"], "butterfly_space.json: space_id required")
        _require(isinstance(d.get("target"), str) and d["target"], "butterfly_space.json: target required")

        p = d.get("P", {})
        _require(isinstance(p.get("radix"), list) and p["radix"], "butterfly_space.json: P.radix required")
        dit = p.get("dit_stage_kernel", {})
        dif = p.get("dif_stage_kernel", {})
        _require(
            isinstance(dit.get("bitrev_input_fold"), list),
            "butterfly_space.json: P.dit_stage_kernel.bitrev_input_fold required",
        )
        _require(
            isinstance(dif.get("bitrev_output_fold"), list),
            "butterfly_space.json: P.dif_stage_kernel.bitrev_output_fold required",
        )
        # Fused local micro-FFT tile sizes (M1 candidate axis). Read from
        # P.stage_fusion.fused_tiles if present, else default to a single-tile
        # list so the config never contains a field the kernel ignores.
        fusion = p.get("stage_fusion", {})
        tiles = fusion.get("fused_tiles", [1])
        _require(
            isinstance(tiles, list) and all(isinstance(t, int) and t > 0 and (t & (t - 1)) == 0 for t in tiles),
            "butterfly_space.json: P.stage_fusion.fused_tiles must be powers of two",
        )

        # M4 batch expansion axis (B values and layout variants the stage
        # batch kernels actually consume). Defaults keep the loader honest
        # when an older config omits the section.
        batch = p.get("batch_expansion", {})
        batch_candidates = batch.get("candidates", [1])
        _require(
            isinstance(batch_candidates, list)
            and all(isinstance(t, int) and t > 0 and (t & (t - 1)) == 0 for t in batch_candidates),
            "butterfly_space.json: P.batch_expansion.candidates must be powers of two",
        )
        batch_variants = batch.get("variants", [])
        _require(
            isinstance(batch_variants, list)
            and all(v in ("spread", "folded") for v in batch_variants),
            "butterfly_space.json: P.batch_expansion.variants must be 'spread'/'folded'",
        )

        lay = d.get("L", {})
        _require(isinstance(lay.get("in"), list) and lay["in"], "butterfly_space.json: L.in required")
        _require(isinstance(lay.get("out"), list) and lay["out"], "butterfly_space.json: L.out required")

        return DesignSpace(
            space_id=d["space_id"],
            target=d["target"],
            radix=tuple(p["radix"]),
            dit_bitrev_input_fold=tuple(dit["bitrev_input_fold"]),
            dif_bitrev_output_fold=tuple(dif["bitrev_output_fold"]),
            fused_tile_candidates=tuple(tiles),
            batch_candidates=tuple(batch_candidates),
            batch_variants=tuple(batch_variants),
            layout_in=lay["in"][0],
            layout_out=lay["out"][0],
        )


def git_commit() -> str:
    """Best-effort current commit hash; returns 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True when the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


@dataclass
class Manifest:
    """Per-run provenance record (migration convergence plan section 9)."""

    commit: str
    dirty: bool
    profile_hash: str
    device: str
    driver: str
    xpu_arch: int
    torch_device_key: str
    dtype: str = "complex64"
    direction: str = "forward"
    precision: str = "fp32"
    flow_id: str = "baseline-global-stage"
    processing_unit: str = "radix-2"
    us: int = 1
    ts: int = 1
    ud: int = 1
    td: int = 1
    residence: str = "gm"
    layout: str = "planar-soa"
    block: int = 1024
    fused_tile: int = 1
    warmup: int = 10
    trials: int = 50
    correctness_threshold: float = 1e-4
    command: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_manifest(profile: DeviceProfile, **overrides) -> Manifest:
    """Build a manifest pinned to the current commit and profile."""
    m = Manifest(
        commit=git_commit(),
        dirty=git_dirty(),
        profile_hash=profile.hash(),
        device=profile.device,
        driver=profile.driver,
        xpu_arch=profile.xpu_arch,
        torch_device_key=profile.torch_device_key,
    )
    for k, v in overrides.items():
        if not hasattr(m, k):
            raise ConfigError(f"unknown manifest field: {k}")
        setattr(m, k, v)
    return m
