#!/usr/bin/env python3
"""Render documentation figures from the committed P800 release summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_shapes(path: Path):
    with path.open() as handle:
        document = json.load(handle)
    shapes = [row for row in document["shapes"]
              if not row["label"].startswith("fallback-")]
    return document["manifest"], shapes


def render(summary: Path, output: Path) -> None:
    manifest, shapes = load_shapes(summary)
    singles = sorted((row for row in shapes if row["batch"] == 1),
                     key=lambda row: row["N"])
    matched = sorted((row for row in shapes
                      if row.get("vendor_status") == "ok"
                      and row["N"] in (1024, 2048)),
                     key=lambda row: (row["N"], row["batch"]))

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
        "svg.hashsalt": "kunlunxin-fft-release-overview",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2),
                             gridspec_kw={"width_ratios": [1.08, 0.92]})

    labels = [f"{row['N']:,}" for row in singles]
    values = [row["speedup_median"] for row in singles]
    colors = ["#9ca3af" if row["fallback"] else "#0072B2"
              for row in singles]
    axes[0].bar(range(len(values)), values, color=colors, width=0.72)
    axes[0].axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(len(labels)), labels, rotation=32, ha="right")
    axes[0].set_ylabel("M0 latency / selected latency (x)")
    axes[0].set_xlabel("FFT length N, batch = 1")
    axes[0].set_title("Frozen selector vs same-block M0")
    axes[0].set_ylim(0, max(values) * 1.18)
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.035, f"{value:.2f}x",
                     ha="center", va="bottom", fontsize=8)
    axes[0].text(0.02, 0.97, "blue: promoted candidate\ngray: M0 fallback",
                 transform=axes[0].transAxes, va="top", fontsize=8,
                 color="#374151")

    vendor_labels = [f"N={row['N']:,}\nB={row['batch']}" for row in matched]
    vendor_values = [row["vendor_median_ms"] / row["selected_median_ms"]
                     for row in matched]
    batch_colors = {1: "#009E73", 16: "#E69F00", 64: "#CC79A7"}
    axes[1].bar(range(len(matched)), vendor_values,
                color=[batch_colors[row["batch"]] for row in matched],
                width=0.72)
    axes[1].axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(vendor_labels)), vendor_labels)
    axes[1].set_ylabel("vendor latency / selected latency (x)")
    axes[1].set_xlabel("Matched FP32 forward C2C shape")
    axes[1].set_title("Selected path vs recorded vendor baseline")
    axes[1].set_ylim(0, max(vendor_values) * 1.18)
    for index, value in enumerate(vendor_values):
        axes[1].text(index, value + 0.035, f"{value:.2f}x",
                     ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Kunlunxin-FFT on P800: release-matrix-v2",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01,
             f"API wall time, plan reuse, 3 sessions; profile {manifest['profile_hash']}",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    svg_path = output.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, facecolor="white")
    fig.savefig(svg_path, facecolor="white", metadata={"Date": None})
    plt.close(fig)

    # Matplotlib emits trailing spaces in multi-line SVG path data. Normalize
    # them so generated assets pass repository checks and remain reproducible.
    svg_text = svg_path.read_text()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path,
                        default=Path("results/p800/release_matrix/summary.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/assets/figures/release-overview"))
    args = parser.parse_args()
    render(args.summary, args.output)
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
