"""Step 2: train the two baselines and record test accuracy.

Usage (from repo root):
    python lab01/src/train_baselines.py
"""
import pandas as pd
from sklearn.metrics import accuracy_score

from common import RESULTS_DIR, load_split, make_models, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"train: {X_train.shape}, test: {X_test.shape}, features: {X_train.shape[1]}")
    print(f"class balance train={y_train.mean():.4f} test={y_test.mean():.4f} (share of benign)")

    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": round(acc, 4)})
        print(f"{name:<20} accuracy = {acc:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline_accuracy.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
