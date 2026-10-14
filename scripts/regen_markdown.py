#!/usr/bin/env python3
"""Regenerate a bench markdown report from a saved bench JSON.

The normal path is ``scripts/bench.py --markdown out.md`` during a run; this
exists so a report can be rebuilt from raw rows alone (the JSON is the source
of truth). Candidate axes are recovered from the row keys.

Usage::

    python scripts/regen_markdown.py results/p800/bench_phase3.json \\
        results/p800/bench_phase3.md
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bench import _write_markdown  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        data = json.load(f)
    rows = data["rows"]

    class _Man:
        def __init__(self, d):
            self.__dict__.update(d)

    man = _Man(data["manifest"])
    keys = set()
    for r in rows:
        keys.update(r.keys())
    tiles = sorted({int(m.group(1)) for k in keys
                    if (m := re.fullmatch(r"m1_t(\d+)_ms", k))})
    matrix_tiles = sorted({int(m.group(1)) for k in keys
                           if (m := re.fullmatch(r"m2_dit_t(\d+)_ms", k))})
    grouped_tiles = sorted({int(m.group(1)) for k in keys
                            if (m := re.fullmatch(r"m3_g_dit_t(\d+)_ms", k))})
    bm = {re.fullmatch(r"m4_(\w+)_b(\d+)_dit_ms", k) for k in keys}
    bm = {m for m in bm if m}
    batch_variants = sorted({m.group(1) for m in bm})
    batchs = sorted({int(m.group(2)) for m in bm})
    _write_markdown(dst, rows, tiles, matrix_tiles, grouped_tiles, man,
                    batchs=batchs, batch_variants=batch_variants)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
