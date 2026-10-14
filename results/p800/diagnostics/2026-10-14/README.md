# Diagnostics: 2026-10-14 intermittent XPU hang during the phase-3 bench

Raw trace artifacts referenced by
[docs/development/incidents/2026-10-14-p800-bench-hang.md](../../../../docs/development/incidents/2026-10-14-p800-bench-hang.md):

| File | What it is |
|---|---|
| `hung-process-pyspy.txt` | live `py-spy dump` of the hung bench (stuck in `torch.cuda.synchronize()`), re-sampled 3x; includes the `xpu-smi` snapshot |
| `watchdog-timeline.txt` | driver log of the stall-detection kills (one genuine, one false positive) |
| `bisect32768-65rounds.log` | 65-round interleaved re-run at `N=32768` with events — full candidate set, no hang |
| `bench65536-isolated.log` | isolated full bench at `N=65536` (correctness + timing) — completes; contains the `b=64` api_ms used for the slow-phase math |
| `vendor-assert-stderr.txt` | the expected `torch.fft.fft` >2048 stderr pattern (not a hang cause) |
