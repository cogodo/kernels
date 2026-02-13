#!/usr/bin/env python3
"""
Generate benchmark graphs from results.json.

Usage:
    python plot_bench.py                    # reads results.json in same dir
    python plot_bench.py path/to/results.json
"""

import json
import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# B100 specs (for roofline model)
# ---------------------------------------------------------------------------
PEAK_FP32_TFLOPS = 60.0      # FP32 TFLOPS
PEAK_BW_TBS = 8.0             # TB/s memory bandwidth
PEAK_FP32_GFLOPS = PEAK_FP32_TFLOPS * 1e3   # 60,000 GFLOPS
PEAK_BW_GBS = PEAK_BW_TBS * 1e3             # 8,000 GB/s


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_kernel_names(results: list[dict]) -> list[str]:
    """Discover kernel names from the results keys (anything ending in _gflops
    that isn't cublas)."""
    names = []
    for key in results[0]:
        if key.endswith("_gflops") and not key.startswith("cublas"):
            name = key.removesuffix("_gflops")
            names.append(name)
    return sorted(names)


def size_label(r: dict) -> str:
    if r["M"] == r["N"] == r["K"]:
        return str(r["M"])
    return f'{r["M"]}x{r["N"]}x{r["K"]}'


# ---------------------------------------------------------------------------
# Graph 1: Throughput vs Matrix Size
# ---------------------------------------------------------------------------
def plot_throughput_vs_size(results, kernels, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [size_label(r) for r in results]
    x = np.arange(len(labels))

    for name in kernels:
        gflops = [r.get(f"{name}_gflops") or 0 for r in results]
        ax.plot(x, gflops, "o-", label=name, linewidth=2, markersize=6)

    cublas_gflops = [r["cublas_gflops"] for r in results]
    ax.plot(x, cublas_gflops, "s--", label="cuBLAS", linewidth=2, markersize=6,
            color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Matrix Size (M = N = K)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("GEMM Throughput vs Matrix Size (FP32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    path = os.path.join(out_dir, "throughput_vs_size.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Graph 2: Kernel Comparison (grouped bar chart)
# ---------------------------------------------------------------------------
def plot_kernel_comparison(results, kernels, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [size_label(r) for r in results]
    x = np.arange(len(labels))
    all_names = kernels + ["cublas"]
    width = 0.8 / len(all_names)

    for i, name in enumerate(all_names):
        gflops = [r.get(f"{name}_gflops") or 0 for r in results]
        offset = (i - len(all_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, gflops, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Matrix Size (M = N = K)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Kernel Comparison (FP32)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    path = os.path.join(out_dir, "kernel_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Graph 3: Percent of cuBLAS
# ---------------------------------------------------------------------------
def plot_vs_cublas(results, kernels, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [size_label(r) for r in results]
    x = np.arange(len(labels))

    for name in kernels:
        pct = [r.get(f"{name}_pct_cublas") or 0 for r in results]
        ax.plot(x, pct, "o-", label=name, linewidth=2, markersize=6)

    ax.axhline(y=100, color="black", linestyle="--", linewidth=1, label="cuBLAS (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Matrix Size (M = N = K)")
    ax.set_ylabel("% of cuBLAS Performance")
    ax.set_title("Custom Kernels vs cuBLAS (FP32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=max(110, ax.get_ylim()[1]))

    path = os.path.join(out_dir, "vs_cublas.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Graph 4: Roofline Model
# ---------------------------------------------------------------------------
def plot_roofline(results, kernels, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline ceilings
    ai_range = np.logspace(-1, 4, 500)  # arithmetic intensity range
    mem_ceiling = PEAK_BW_GBS * ai_range      # memory-bound ceiling
    compute_ceiling = np.full_like(ai_range, PEAK_FP32_GFLOPS)
    roofline = np.minimum(mem_ceiling, compute_ceiling)

    ax.loglog(ai_range, roofline, "k-", linewidth=2, label="Roofline (B100)")
    ax.fill_between(ai_range, roofline, alpha=0.05, color="black")

    # Ridge point
    ridge_ai = PEAK_FP32_GFLOPS / PEAK_BW_GBS
    ax.axvline(x=ridge_ai, color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"Ridge: {ridge_ai:.1f} FLOP/B",
                xy=(ridge_ai, PEAK_FP32_GFLOPS * 0.5),
                fontsize=8, color="gray", ha="right", rotation=90)

    # Plot each kernel + cuBLAS at each size
    for name in kernels:
        ais, gflops_vals = [], []
        for r in results:
            gf = r.get(f"{name}_gflops")
            if gf is not None and gf > 0:
                ai = r["flops"] / r["mem_bytes"]
                ais.append(ai)
                gflops_vals.append(gf)
        if ais:
            ax.loglog(ais, gflops_vals, "o", label=name, markersize=8)
            # Annotate with size
            for ai_val, gf_val, r in zip(ais, gflops_vals, results):
                ax.annotate(size_label(r), (ai_val, gf_val),
                            fontsize=6, alpha=0.7, textcoords="offset points",
                            xytext=(4, 4))

    # cuBLAS points
    cublas_ais = [r["flops"] / r["mem_bytes"] for r in results]
    cublas_gf = [r["cublas_gflops"] for r in results]
    ax.loglog(cublas_ais, cublas_gf, "s", label="cuBLAS", markersize=8,
              color="black")

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("Roofline Model — B100 FP32")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1, PEAK_FP32_GFLOPS * 2)

    path = os.path.join(out_dir, "roofline.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        print("Run `modal run modal_bench.py` first to generate benchmark data.")
        sys.exit(1)

    results = load_results(results_path)
    kernels = get_kernel_names(results)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded {len(results)} benchmark entries from {results_path}")
    print(f"Kernels found: {', '.join(kernels)}")
    print(f"Generating graphs in {out_dir}/\n")

    plot_throughput_vs_size(results, kernels, out_dir)
    plot_kernel_comparison(results, kernels, out_dir)
    plot_vs_cublas(results, kernels, out_dir)
    plot_roofline(results, kernels, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
