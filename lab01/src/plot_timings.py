"""Tables and plots of every individual timing run.

Reads results/raw_timings.json (from measure_system_cost.py) and writes:
  results/training_runs.csv     - each of the 5 timed training runs, seconds
  results/inference_stats.csv   - statistics over the 100 single-sample predictions
  results/training_runs.png     - the 5 training runs per model
  results/inference_runs.png    - the 100 prediction runs per model

Usage (from repo root):
    python lab01/src/plot_timings.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS_DIR  # noqa: E402

COLORS = {"LogisticRegression": "#2a78d6", "RandomForest": "#eb6834"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TINYML_MS = 10

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def write_tables(runs):
    train = pd.DataFrame({"run": range(1, 6)})
    for name, r in runs.items():
        train[f"{name}_s"] = r["train_s"]
    train.to_csv(RESULTS_DIR / "training_runs.csv", index=False, float_format="%.4f")

    stats = []
    for name, r in runs.items():
        t = np.array(r["train_s"])
        i = np.array(r["infer_s"]) * 1e3
        stats.append({
            "model": name,
            "train_runs": len(t),
            "train_min_s": t.min(), "train_median_s": np.median(t), "train_max_s": t.max(),
            "infer_runs": len(i),
            "infer_min_ms": i.min(), "infer_median_ms": np.median(i), "infer_mean_ms": i.mean(),
            "infer_p95_ms": np.percentile(i, 95), "infer_max_ms": i.max(),
            "infer_total_100_s": i.sum() / 1e3,
        })
    pd.DataFrame(stats).to_csv(RESULTS_DIR / "inference_stats.csv", index=False, float_format="%.4f")
    print(train.to_string(index=False))
    print(pd.DataFrame(stats).T.to_string(header=False))


def plot_training(runs):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, 6)
    for name, r in runs.items():
        t = np.array(r["train_s"])
        c = COLORS[name]
        ax.plot(x, t, color=c, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        med = np.median(t)
        ax.hlines(med, 0.8, 5.2, color=c, linewidth=1, linestyles="--")
        ax.text(5.3, med, f"median {med:.3f} s", color=TEXT, va="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("timed run (after 1 warm-up)")
    ax.set_ylabel("training time, s")
    ax.set_title("Training time: 5 runs per model")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_runs.png", dpi=150)
    plt.close(fig)


def plot_inference(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, r) in zip(axes, runs.items()):
        ms = np.array(r["infer_s"]) * 1e3
        c = COLORS[name]
        x = np.arange(1, len(ms) + 1)
        med, p95 = np.median(ms), np.percentile(ms, 95)

        ylim = max(TINYML_MS * 1.2, p95 * 1.5) if name == "RandomForest" else p95 * 1.5
        shown = ms <= ylim
        ax.scatter(x[shown], ms[shown], s=18, color=c, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for xi, v in zip(x[~shown], ms[~shown]):  # outliers above the visible range
            ax.annotate(f"run {xi}: {v:.1f} ms", xy=(xi, ylim * 0.98), xytext=(xi + 6, ylim * 0.85),
                        fontsize=9, color=TEXT, arrowprops={"arrowstyle": "->", "color": MUTED})
        ax.hlines(med, 0, 100, color=TEXT, linewidth=1.2)
        ax.hlines(p95, 0, 100, color=MUTED, linewidth=1, linestyles=":")
        ax.text(102, med, f"median {med:.3f}", va="center", fontsize=9)
        ax.text(102, p95, f"p95 {p95:.3f}", va="center", fontsize=9, color=MUTED)
        if ylim > TINYML_MS:
            ax.axhline(TINYML_MS, color="#e34948", linewidth=1, linestyle="--")
            ax.text(2, TINYML_MS, "TinyML budget 10 ms", va="bottom", fontsize=9, color=TEXT)
        ax.set_ylim(0, ylim)
        ax.set_xlim(0, 130)
        ax.set_xlabel("prediction run")
        ax.set_ylabel("latency, ms")
        ax.set_title(f"{name}: 100 single-sample predictions")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "inference_runs.png", dpi=150)
    plt.close(fig)


def main():
    runs = json.loads((RESULTS_DIR / "raw_timings.json").read_text())["runs"]
    write_tables(runs)
    plot_training(runs)
    plot_inference(runs)
    print("saved training_runs.csv/png, inference_stats.csv, inference_runs.png")


if __name__ == "__main__":
    main()
