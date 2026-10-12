"""Minimal training process: load the data split and fit one model once.

Runs in a fresh interpreter (no torch, nothing trained before), so the peak
memory reflects training alone rather than allocator reuse from earlier fits.
Prints JSON.

Usage:
    python train_standalone.py <LogisticRegression|RandomForest>
"""
import json
import random
import sys

import psutil

proc = psutil.Process()
rss_interpreter = proc.memory_info().rss

import numpy as np  # noqa: E402

from common import SEED, load_split, make_models  # noqa: E402

rss_after_imports = proc.memory_info().rss

random.seed(SEED)
np.random.seed(SEED)
X_train, X_test, y_train, y_test = load_split()
model = make_models()[sys.argv[1]]
rss_before_fit = proc.memory_info().rss

model.fit(X_train, y_train)

info = proc.memory_info()
# peak_wset is the OS-tracked peak working set (Windows); elsewhere fall back to RSS.
peak = max(info.rss, getattr(info, "peak_wset", 0))

print(json.dumps({
    "rss_interpreter": rss_interpreter,
    "rss_after_imports": rss_after_imports,
    "rss_before_fit": rss_before_fit,
    "rss_after_fit": info.rss,
    "rss_peak": peak,
}))
