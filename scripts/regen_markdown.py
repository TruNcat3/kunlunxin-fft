#!/usr/bin/env python3
"""Regenerate a bench markdown report from a saved bench JSON.

The normal path is ``scripts/bench.py --markdown out.md`` during a run; this
exists so a report can be rebuilt from raw rows alone (the JSON is the source
of truth). Candidate axes are recovered from the row keys.

This entry point is deliberately thin: all rendering lives in
``scripts/report.py``, which is torch-free so ordinary CI can regenerate the
committed markdown byte-identically (``tests/test_report.py``).

Usage::

    python scripts/regen_markdown.py results/p800/bench_phase3.json \
        results/p800/bench_phase3.md
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import render_from_json  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    render_from_json(src, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
