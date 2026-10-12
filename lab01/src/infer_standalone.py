"""Minimal inference process: load one saved model, predict one sample 100 times.

Simulates a deployment runtime (no torch, no training code) so that its
memory is a fair estimate of what a device would need. Prints JSON.

Usage:
    python infer_standalone.py <model.joblib> <sample.npy>
"""
import json
import sys

import psutil

proc = psutil.Process()
rss_interpreter = proc.memory_info().rss

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn.ensemble  # noqa: E402,F401
import sklearn.linear_model  # noqa: E402,F401

rss_libs = proc.memory_info().rss

model = joblib.load(sys.argv[1])
x = np.load(sys.argv[2])
rss_model = proc.memory_info().rss

peak = rss_model
model.predict(x)  # warm-up
for _ in range(100):
    model.predict(x)
    peak = max(peak, proc.memory_info().rss)

info = proc.memory_info()
# On Windows psutil exposes the OS-tracked peak working set, which also
# catches short spikes between our samples.
peak = max(peak, getattr(info, "peak_wset", 0))

print(json.dumps({
    "rss_interpreter": rss_interpreter,
    "rss_after_imports": rss_libs,
    "rss_after_model_load": rss_model,
    "rss_peak": peak,
}))
