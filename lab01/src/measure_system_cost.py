"""Step 3: measure training time, single-sample latency, model size and peak memory.

Usage (from repo root):
    python lab01/src/measure_system_cost.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage
from sklearn.base import clone

from common import RESULTS_DIR, load_split, make_models, set_seeds

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100
MB = 1024 ** 2
SHORT = {"LogisticRegression": "logreg", "RandomForest": "rf"}
STANDALONE = Path(__file__).resolve().parent / "infer_standalone.py"


def time_training(model, X, y):
    clone(model).fit(X, y)  # warm-up
    times = []
    for _ in range(N_TRAIN_RUNS):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X, y)
        times.append(time.perf_counter() - t0)
    return times


def time_single_inference(model, x):
    model.predict(x)  # warm-up
    times = []
    for _ in range(N_INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(x)
        times.append(time.perf_counter() - t0)
    return times


def peak_memory_mb(fn, *args):
    """Peak process RSS (MB) while fn runs, sampled from a separate process."""
    baseline = max(memory_usage(-1, interval=0.01, timeout=0.2))
    peak = memory_usage((fn, args), interval=0.001, max_usage=True)
    return baseline, peak


def predict_many(model, x):
    for _ in range(N_INFER_RUNS):
        model.predict(x)


def parameter_bytes(model):
    """Bytes of the numeric arrays that make up the fitted model."""
    if hasattr(model, "coef_"):
        return model.coef_.nbytes + model.intercept_.nbytes
    total = 0
    for est in model.estimators_:
        state = est.tree_.__getstate__()
        total += state["nodes"].nbytes + state["values"].nbytes
    return total


def standalone_inference_memory(model_path, sample_path):
    out = subprocess.run(
        [sys.executable, str(STANDALONE), str(model_path), str(sample_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    x_one = X_test[:1]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = RESULTS_DIR / "single_sample.npy"
    np.save(sample_path, x_one)

    rows, raw, fitted = [], {}, {}
    for name, model in make_models().items():
        print(f"== {name}")
        train_times = time_training(model, X_train, y_train)

        mem_base, mem_train_peak = peak_memory_mb(model.fit, X_train, y_train)
        fitted[name] = model

        infer_times = time_single_inference(model, x_one)
        _, mem_infer_peak = peak_memory_mb(predict_many, model, x_one)

        model_path = RESULTS_DIR / f"model_{SHORT[name]}.joblib"
        joblib.dump(model, model_path)
        size_bytes = model_path.stat().st_size

        sa = standalone_inference_memory(model_path, sample_path)

        row = {
            "model": name,
            "train_time_median_ms": np.median(train_times) * 1e3,
            "infer_latency_median_ms": np.median(infer_times) * 1e3,
            "infer_latency_p95_ms": np.percentile(infer_times, 95) * 1e3,
            "model_size_bytes": size_bytes,
            "model_size_kb": size_bytes / 1024,
            "param_bytes": parameter_bytes(model),
            "rss_baseline_mb": mem_base,
            "peak_rss_train_mb": mem_train_peak,
            "train_mem_delta_mb": mem_train_peak - mem_base,
            "peak_rss_infer_mb": mem_infer_peak,
            "deploy_rss_interpreter_mb": sa["rss_interpreter"] / MB,
            "deploy_rss_after_imports_mb": sa["rss_after_imports"] / MB,
            "deploy_rss_after_load_mb": sa["rss_after_model_load"] / MB,
            "deploy_peak_rss_mb": sa["rss_peak"] / MB,
        }
        rows.append(row)
        raw[name] = {"train_s": train_times, "infer_s": infer_times}
        for k, v in row.items():
            print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    # Single artifact required by the lab manual: both fitted models in one file.
    joblib.dump(fitted, RESULTS_DIR / "model.joblib")
    combined = (RESULTS_DIR / "model.joblib").stat().st_size
    print(f"model.joblib (both models): {combined} bytes = {combined / 1024:.2f} KB")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "system_cost.csv", index=False, float_format="%.4f")
    (RESULTS_DIR / "raw_timings.json").write_text(json.dumps({
        "cpu": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_total_mb": psutil.virtual_memory().total / MB,
        "runs": raw,
    }, indent=2))
    print("saved system_cost.csv, raw_timings.json")


if __name__ == "__main__":
    main()
