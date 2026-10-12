"""Step 4: check measured costs against Cloud / Edge / Mobile / TinyML budgets.

Reads results/system_cost.csv (from measure_system_cost.py) and writes
results/deployment_fit.csv.

Two scenarios are evaluated:
  * python  - the model served as measured: Python + numpy + sklearn process
              (memory = peak RSS of the standalone inference process).
  * native  - the model exported to plain code without the Python runtime
              (memory and size = bytes of the fitted parameter arrays).
Latency is the measured median on this machine in both cases.

Usage (from repo root):
    python lab01/src/deployment_fit.py
"""
import pandas as pd

from common import RESULTS_DIR

MB = 1024 ** 2
KB = 1024

# Budgets from the lab manual. Memory is a range of device sizes in the tier:
# mem_min = smallest device, mem_max = largest device (None = unbounded).
BUDGETS = {
    "Cloud":  {"mem_min": 1024 * MB, "mem_max": None,     "latency_ms": 100, "size": 500 * MB},
    "Edge":   {"mem_min": 256 * MB,  "mem_max": 1024 * MB, "latency_ms": 50,  "size": 50 * MB},
    "Mobile": {"mem_min": 64 * MB,   "mem_max": 256 * MB,  "latency_ms": 20,  "size": 10 * MB},
    "TinyML": {"mem_min": 256 * KB,  "mem_max": 256 * KB,  "latency_ms": 10,  "size": 100 * KB},
}


def verdict(mem_all, mem_any, lat_ok, size_ok):
    if not (lat_ok and size_ok and mem_any):
        return "NO"
    return "YES" if mem_all else "PARTIAL (larger devices only)"


def main():
    cost = pd.read_csv(RESULTS_DIR / "system_cost.csv")
    rows = []
    for _, c in cost.iterrows():
        scenarios = {
            "python": (c.deploy_peak_rss_mb * MB, c.model_size_bytes),
            "native": (c.param_bytes, c.param_bytes),
        }
        for scenario, (mem, size) in scenarios.items():
            for tier, b in BUDGETS.items():
                mem_all = mem <= b["mem_min"]
                mem_any = b["mem_max"] is None or mem <= b["mem_max"]
                lat_ok = c.infer_latency_median_ms <= b["latency_ms"]
                size_ok = size <= b["size"]
                rows.append({
                    "model": c.model,
                    "scenario": scenario,
                    "tier": tier,
                    "memory_kb": round(mem / KB, 1),
                    "latency_ms": round(c.infer_latency_median_ms, 4),
                    "size_kb": round(size / KB, 2),
                    "mem_fits_smallest_device": mem_all,
                    "mem_fits_largest_device": mem_any,
                    "latency_ok": lat_ok,
                    "size_ok": size_ok,
                    "verdict": verdict(mem_all, mem_any, lat_ok, size_ok),
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "deployment_fit.csv", index=False)
    pd.set_option("display.width", 200)
    for scenario in ("python", "native"):
        print(f"\n=== scenario: {scenario}")
        print(df[df.scenario == scenario].drop(columns="scenario").to_string(index=False))


if __name__ == "__main__":
    main()
