#!/usr/bin/env bash
# Initialise the development environment for kunlunxin-fft.
#
# This project needs a Python environment that exposes the Kunlunxin P800 through
# torch.device('cuda'). On the reference machine that is the vendor "xpytorch"
# stack: torch 2.9.0 + torch_xmlir + flag_gems. There is no separate CUDA device;
# importing torch_xmlir retargets the 'cuda' device key to the XPU.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> checking python"
"$PYTHON_BIN" - <<'PY'
import sys
print("python", sys.version)
PY

echo "==> checking torch + torch_xmlir (P800 device path)"
"$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("torch not installed; install the vendor xpytorch stack first")
try:
    import torch_xmlir  # noqa: F401
except ImportError:
    sys.exit("torch_xmlir not installed; this stack is required to address the P800")
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.zeros(4, device="cuda")
    print("device tensor ok:", x.device)
PY

echo "==> checking triton device backend"
"$PYTHON_BIN" - <<'PY'
import triton
print("triton", triton.__version__)
from triton.runtime.driver import driver
print("target", driver.active.get_current_target())
PY

echo "==> environment initialised"
