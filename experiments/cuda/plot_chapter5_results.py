from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "public" / "images" / "cuda-optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_benchmark() -> None:
    shapes = ["2048x1024", "4096x1024", "1024x4096", "1x4096"]
    separate = np.array([0.1083, 0.2137, 0.2506, 0.0091])
    fused = np.array([0.0585, 0.1159, 0.0640, 0.0054])
    speedup = separate / fused
    x = np.arange(len(shapes))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    bars_a = ax.bar(x - width / 2, separate, width, label="Separate kernels", color="#4b5563")
    bars_b = ax.bar(x + width / 2, fused, width, label="Fused kernel", color="#1473e6")
    ax.set_ylabel("Latency per iteration (ms)")
    ax.set_xlabel("Matrix shape: rows x columns")
    ax.set_xticks(x, shapes)
    ax.set_title("Scale + causal mask + softmax on RTX 3060")
    ax.set_ylim(0, 0.29)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for i, value in enumerate(speedup):
        ax.text(i, max(separate[i], fused[i]) + 0.016, f"{value:.2f}x", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fused-softmax-benchmark.png", bbox_inches="tight")
    plt.close(fig)


def plot_nsys_breakdown() -> None:
    kernels = ["scale_mask_kernel", "softmax_kernel", "fused kernel"]
    latency_us = [58.5242, 191.1795, 63.7665]
    colors = ["#6b7280", "#6b7280", "#1473e6"]

    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=180)
    bars = ax.barh(kernels, latency_us, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Average GPU kernel time (us)")
    ax.set_title("Nsight Systems kernel summary, shape 1024x4096")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, latency_us):
        ax.text(value + 3, bar.get_y() + bar.get_height() / 2, f"{value:.1f} us", va="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fused-softmax-nsys-kernel-time.png", bbox_inches="tight")
    plt.close(fig)


def plot_ncu_metrics() -> None:
    metrics = ["Compute (SM)\nthroughput", "DRAM\nthroughput", "Achieved\noccupancy"]
    values = [83.31, 64.25, 92.66]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=180)
    bars = ax.bar(x, values, width=0.56, color=["#1473e6", "#4b5563", "#0f9d74"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(x, metrics)
    ax.set_title("Nsight Compute metrics for the fused kernel")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.5, f"{value:.2f}%", ha="center")
    ax.text(
        0.02,
        0.97,
        "18 registers/thread | 6.10 waves/SM",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fused-softmax-ncu-metrics.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_benchmark()
    plot_nsys_breakdown()
    plot_ncu_metrics()
