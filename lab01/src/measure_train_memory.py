"""Peak training memory, each run in a fresh process (5 runs, median).

Writes results/train_memory.csv.

Usage (from repo root):
    python lab01/src/measure_train_memory.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS_DIR

N_RUNS = 5
MB = 1024 ** 2
SCRIPT = Path(__file__).resolve().parent / "train_standalone.py"


def run_once(name):
    out = subprocess.run([sys.executable, str(SCRIPT), name],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    rows = []
    for name in ("LogisticRegression", "RandomForest"):
        runs = [run_once(name) for _ in range(N_RUNS)]
        med = {k: np.median([r[k] for r in runs]) / MB for k in runs[0]}
        rows.append({
            "model": name,
            "runs": N_RUNS,
            "rss_interpreter_mb": med["rss_interpreter"],
            "rss_after_imports_mb": med["rss_after_imports"],
            "rss_before_fit_mb": med["rss_before_fit"],
            "peak_rss_train_mb": med["rss_peak"],
            "train_mem_delta_mb": float(np.median([(r["rss_peak"] - r["rss_before_fit"]) / MB for r in runs])),
            "peak_rss_train_all_runs_mb": " ".join(f"{r['rss_peak'] / MB:.2f}" for r in runs),
        })
        print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "train_memory.csv", index=False, float_format="%.2f")
    print("saved train_memory.csv")


if __name__ == "__main__":
    main()
