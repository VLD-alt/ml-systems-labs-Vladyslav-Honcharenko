# Lab 1 — Environment and first system measurements

## 1. Goal

Build a reproducible Python environment, train two baseline classifiers on the
Breast Cancer Wisconsin (Diagnostic) dataset, measure their system cost
(training time, single-sample latency, model size, peak memory), and decide on
which deployment tier — Cloud, Edge, Mobile, TinyML — each model could run and
where it is best to run it.

| Learning outcome | Evidence in this report |
|---|---|
| Create and verify an isolated Python environment with pinned dependencies | §2 Environment; `results/versions.txt`: isolated venv = YES, Python 3.11.8, 25/25 packages match their pins |
| Train baseline linear and ensemble models using a reproducible data split | §2 Data and models; §3.1 Accuracy (identical on re-run) |
| Measure training time, inference latency, model size and peak memory correctly | §2 Measurements; §3.2 System cost; §3.3 Every timing run; §3.4 Measurement pitfalls |
| Evaluate deployment suitability against Cloud, Edge, Mobile and TinyML budgets | §3.5 Budget check; §3.6 Where each model can run; §3.7 Where it is best to run |

## 2. Method

**Environment.** Python 3.11.8 in an isolated `.venv`, all 25 packages pinned
to the versions in the lab manual. `src/print_versions.py` prints Python,
platform and every package; it also checks that it runs inside the virtual
environment (`sys.prefix` = `.venv`, different from the base install) and
compares each installed version with its pin. Output: `results/versions.txt`
— isolated venv YES, 0 mismatches. The pinned set has one internal conflict:
`mlflow==2.14.1` requires `pyarrow<16` while the manual pins
`pyarrow==16.1.0`. To keep every pin exact, mlflow was installed without
dependency resolution and its other dependencies were installed separately
(steps in the repo `README.md`). mlflow is not used in this lab.

**Hardware.** AMD Ryzen 7 2700X (8 cores / 16 threads), 16 GB RAM,
Windows 10 Pro 19045, CPU only.

**Data and models.** `load_breast_cancer(return_X_y=True)`: 569 samples, 30
features. Split 70/30, stratified, `random_state=42` → 398 train / 171 test,
benign share 0.628 / 0.626. Seeds 42 for `random`, `numpy`, `torch`,
`torch.cuda`. Models exactly as specified, no preprocessing:
`LogisticRegression(max_iter=1000, random_state=42)` and
`RandomForestClassifier(n_estimators=100, random_state=42)`.

**Measurements.**

| Metric | Procedure |
|---|---|
| Training time | 1 warm-up fit, then 5 timed fits on a fresh clone each; median |
| Single-sample latency | `predict` on one test row; 1 warm-up, then 100 timed calls; median (p95 and max also reported) |
| Model size | `joblib.dump` of each model; file size in bytes and KB (1 KB = 1024 B). Both models are also saved together in `results/model.joblib` (291 658 B) |
| Peak memory, training | Each run in a **fresh process** that imports only numpy and sklearn, loads the split and fits the model once; peak taken from the OS peak working set (psutil `peak_wset`); 5 runs, median. "Added by training" = peak − RSS just before `fit` |
| Peak memory, inference | A **fresh process** that imports only numpy, sklearn and joblib, loads one saved model and predicts the sample 100 times; OS peak working set. This is the process that would be deployed |
| Parameter bytes | Size of the fitted numeric arrays (coefficients, or tree node and value arrays) — the footprint if the model is exported to native code |

All timers use `time.perf_counter`. Every individual run is saved in
`results/raw_timings.json`; per-run tables and plots are produced from it by
`src/plot_timings.py`.

**Deployment check.** Each model is compared with the manual's budgets on all
three constraints — memory, latency, size — in two scenarios:

* *python* — the model served as measured: memory = peak RSS of the
  inference process, size = joblib file;
* *native* — the model exported to plain code with no Python runtime (e.g.
  ONNX Runtime Mobile or generated C): memory and size = parameter bytes.

Latency is the measured median (p95 also checked). Memory budgets for Edge and
Mobile are ranges of device sizes: a model *fits* the tier if it is below the
smallest device, and *partially fits* if it is below only the largest one.

## 3. Results

### 3.1 Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForest | 0.9357 |

### 3.2 System cost (`results/system_cost.csv`, `results/train_memory.csv`)

| Metric | LogisticRegression | RandomForest |
|---|---:|---:|
| Training time, median of 5 | 0.288 s | 0.170 s |
| Single-sample latency, median of 100 | 0.088 ms | 4.189 ms |
| Single-sample latency, p95 | 0.096 ms | 6.108 ms |
| Model size (joblib) | 1 055 B = 1.03 KB | 290 889 B = 284.07 KB |
| Parameter bytes | 248 B | 251 040 B = 245.2 KB |
| **Peak memory, training** (fresh process, median of 5) | **101.4 MB** | **101.6 MB** |
| — Python interpreter alone | 16.5 MB | 16.5 MB |
| — after importing numpy + sklearn | 99.9 MB | 100.0 MB |
| — after loading the data split | 101.0 MB | 101.2 MB |
| — memory added by `fit` itself | 0.40 MB | 0.51 MB |
| **Peak memory, inference** (deployment process) | **95.8 MB** | **97.0 MB** |
| — Python interpreter alone | 16.5 MB | 16.4 MB |
| — after importing numpy + sklearn + joblib | 95.6 MB | 95.7 MB |
| — after loading the model | 95.7 MB | 96.8 MB |

The five training-memory runs differ by less than 0.3 MB
(LogReg 101.19–101.49 MB, RF 101.46–101.75 MB).

### 3.3 Every timing run (`results/training_runs.csv`, `results/inference_stats.csv`)

Following the manual, training was timed **5 times** per model (after 1
warm-up fit) and single-sample prediction **100 times** per model (after 1
warm-up call). All individual runs are shown below, not only the medians.

**Training — 5 runs, seconds:**

| Run | LogisticRegression | RandomForest (100 trees) |
|---:|---:|---:|
| 1 | 0.2833 | 0.1736 |
| 2 | 0.2928 | 0.1687 |
| 3 | 0.2794 | 0.1696 |
| 4 | 0.2881 | 0.1693 |
| 5 | 0.3013 | 0.1745 |
| min | 0.2794 | 0.1687 |
| **median** | **0.2881** | **0.1696** |
| max | 0.3013 | 0.1745 |
| spread (max − min) / median | 7.6 % | 3.4 % |

![Training time, 5 runs per model](results/training_runs.png)

**Single-sample prediction — 100 runs:**

| Statistic over 100 runs | LogisticRegression | RandomForest |
|---|---:|---:|
| min | 0.0863 ms | 3.996 ms |
| **median** | **0.0876 ms** | **4.189 ms** |
| mean | 0.0890 ms | 4.892 ms |
| p95 | 0.0960 ms | 6.108 ms |
| max | 0.1113 ms | 56.818 ms |
| total time of all 100 predictions | 0.0089 s | 0.4892 s |
| predictions per second (1 / median) | ≈ 11 400 | ≈ 240 |

![Latency of 100 single-sample predictions per model](results/inference_runs.png)

What the individual runs show:

* **Training is stable.** All five runs of each model lie within a few percent
  of the median, so the median of 5 is a reliable number. Training 100 trees on
  398 samples takes ≈ 0.17 s; the logistic regression takes ≈ 0.29 s because
  lbfgs runs the full 1000 iterations.
* **One warm-up call is not always enough.** The first ~10 logistic-regression
  predictions are up to 27 % slower (0.11 ms vs 0.088 ms) before the latency
  settles, which is CPU caches and branch predictors warming up.
* **A single outlier can wreck the mean.** Run 84 of the forest took 56.8 ms,
  ≈ 14× its median, most likely because the OS scheduled another process at that
  moment. That one run pulls the mean up to 4.89 ms, while the median stays at
  4.19 ms. This is why the manual asks for the median, not the mean or a single
  run.

### 3.4 Measurement pitfalls found and fixed

* **Memory of the profiler counted as memory of the model.** With
  `include_children=True`, `memory_profiler` on Windows also counted its own
  monitor process, which re-imports the script, and reported a false doubling
  (≈ 529 MB).
* **Allocator reuse hid the training memory.** Measured inside the timing
  script, training appeared to add < 0.1 MB, because the same process had
  already trained the model six times and simply reused that memory. That
  process also had torch loaded for seeding (≈ 265 MB total). Moving the
  measurement to a fresh process per run gives the numbers in §3.2:
  0.40 / 0.51 MB added by `fit` on top of ≈ 101 MB.

### 3.5 Budget check (`results/deployment_fit.csv`)

Budgets from the lab manual:

| Tier | Device memory | Latency | Model size |
|---|---|---|---|
| Cloud | ≥ 1 GB | ≤ 100 ms | ≤ 500 MB |
| Edge | 256–1024 MB | ≤ 50 ms | ≤ 50 MB |
| Mobile | 64–256 MB | ≤ 20 ms | ≤ 10 MB |
| TinyML | ≤ 256 KB | ≤ 10 ms | ≤ 100 KB |

Each cell below shows the result and the headroom (budget ÷ measured; "over"
means measured ÷ budget).

**LogisticRegression, served with Python** — memory 95.8 MB, latency 0.088 ms
(p95 0.096 ms), size 1.03 KB:

| Tier | Memory | Latency | Size | Verdict |
|---|---|---|---|---|
| Cloud | ✅ 10.7× below 1 GB | ✅ 1142× | ✅ ≈ 500 000× | **Fits** |
| Edge | ✅ 2.7× below the smallest 256 MB device | ✅ 571× | ✅ ≈ 50 000× | **Fits** |
| Mobile | ⚠️ fits a 256 MB device (2.7×), not a 64 MB one | ✅ 228× | ✅ ≈ 9 900× | **Partial** |
| TinyML | ❌ 383× over 256 KB | ✅ 114× | ✅ 97× | **No** (memory) |

**RandomForest, served with Python** — memory 97.0 MB, latency 4.19 ms
(p95 6.11 ms), size 284.07 KB:

| Tier | Memory | Latency | Size | Verdict |
|---|---|---|---|---|
| Cloud | ✅ 10.6× below 1 GB | ✅ 23.9× | ✅ 1802× | **Fits** |
| Edge | ✅ 2.6× below the smallest 256 MB device | ✅ 11.9× | ✅ 180× | **Fits** |
| Mobile | ⚠️ fits a 256 MB device (2.6×), not a 64 MB one | ✅ 4.8× (p95 3.3×) | ✅ 36× | **Partial** |
| TinyML | ❌ 388× over 256 KB | ✅ 2.4× (p95 1.6×) | ❌ 2.8× over 100 KB | **No** (memory and size) |

**Exported to native code** (no Python runtime) — memory and size = parameter
bytes, latency as measured:

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogisticRegression (248 B) | ✅ | ✅ | ✅ | ✅ 248 B ≪ 100 KB size, ≪ 256 KB memory |
| RandomForest (245.2 KB) | ✅ | ✅ | ✅ | ❌ 245.2 KB > 100 KB size; memory 245.2 KB leaves almost no headroom in 256 KB |

All latencies were measured on a desktop CPU. A phone CPU is several times
slower and a microcontroller much slower than that, so a small latency
headroom (RandomForest on Mobile 4.8×, on TinyML 2.4×) is not a safe margin,
while the large headroom of LogisticRegression (> 100× everywhere) is.

### 3.6 Where each model can run

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogisticRegression | ✅ Yes | ✅ Yes | ✅ Yes after native export; with Python only on devices with > ~100 MB free | ✅ Yes, only after native export |
| RandomForest | ✅ Yes | ✅ Yes | ⚠️ Yes after native export, but latency margin is thin on a phone CPU; with Python only on devices with > ~100 MB free | ❌ No — too big even without Python |

What limits each tier: on Cloud and Edge nothing is close to a limit. On
Mobile the limit is **memory of the Python runtime** (≈ 96 MB, of which the
models are < 1.5 MB). On TinyML it is the **runtime** for both models and the
**model size** for the forest: at ≈ 2.8 KB per tree, it would need fewer than
~35 trees to fit 100 KB (accuracy of such a forest was not measured).

### 3.7 Where it is best to run

Fitting the budget only says where a model *can* run. Where it *should* run
also depends on the application (Reddi, Ch. 2 deployment decision framework:
latency need, privacy, connectivity, cost and energy). Here the task is to
classify a breast tumour from 30 features computed from a biopsy image: the
data is sensitive medical data, the decision is not real-time (seconds are
fine), and one prediction is needed per patient, not thousands per second.

* **LogisticRegression → best on Edge or Mobile** (a hospital workstation, the
  imaging device itself, or a clinician's tablet). It needs 1 KB of model,
  0.09 ms and < 100 MB even with Python, so any local device runs it with
  huge headroom. Patient data never has to leave the hospital, and the model
  works without network. Sending the data to the Cloud would add a network
  round-trip — typically tens of milliseconds, hundreds of times longer than
  the inference itself — plus privacy and compliance concerns, for no gain.
  Exported to native code it could even be embedded in the measuring
  instrument (TinyML tier).
* **RandomForest → best on Edge** (a hospital server or workstation). It has
  comfortable headroom there (latency 12×, memory 2.6× even on the smallest
  256 MB device), and the same privacy argument applies. Mobile is possible
  but risky: on a phone CPU the 4.8× latency headroom would shrink, and the
  measured tail (max 56.8 ms) already exceeds the 20 ms Mobile budget on a
  desktop. Cloud works but is unnecessary. TinyML is ruled out by size.
* **Overall choice: LogisticRegression.** It is slightly more accurate
  (0.9415 vs 0.9357), 275× smaller, ~48× faster, and the only one of the two
  that can run on every tier. RandomForest buys no accuracy for its extra cost.
* **Training** needs ≈ 101 MB and < 0.3 s for both models, so it can be done
  on any Cloud, Edge or Mobile device, but it is best done once offline on a
  workstation or in the Cloud, and only the fitted model deployed.

## 4. Conclusions

1. **The runtime, not the model, sets the memory cost.** Both models need
   ≈ 96–101 MB to serve or train, and ≈ 80–84 MB of that is just numpy and
   sklearn on top of the 16.5 MB interpreter. The models themselves add
   0.1–1.3 MB, and training adds only 0.4–0.5 MB. So "where can it run"
   depends on the serving stack at least as much as on the model: exporting
   LogisticRegression to native code shrinks it from ~96 MB to 248 bytes and
   moves it from "Partial on Mobile" to "fits TinyML".

2. **Model complexity shows up in latency and size, not in accuracy.** The
   100-tree forest is 275× larger and ~48× slower per prediction than the
   linear model, and it is not more accurate on this split
   (0.9357 vs 0.9415). For a small tabular problem like this, the extra
   capacity buys only system cost, and the size budget is what rules the
   forest out of TinyML. The best place to run a model also depends on the
   application: for private, non-real-time medical data, local Edge or Mobile
   deployment is preferable to the Cloud even though all three fit.

3. **Measuring correctly takes care, and naïve numbers can be wrong by a lot.**
   A profiler setting doubled the memory (529 MB instead of ~265 MB), and
   measuring training memory in a reused process made it look like < 0.1 MB.
   A single latency outlier (56.8 ms) moved the mean by 17 %, while the median
   did not move. Fresh processes, warm-up, medians over repeated runs and a
   sanity check of every number were needed for results I can trust. The
   logistic regression is also slower to train (0.29 s vs 0.17 s, because
   lbfgs hits its 1000-iteration limit on unscaled features) but faster to
   serve: training cost and serving cost must be measured separately.

## Reproduce

From the repo root, with the `.venv` active:
`python lab01/src/print_versions.py > lab01/results/versions.txt`, then
`python lab01/src/train_baselines.py`, `python lab01/src/measure_system_cost.py`,
`python lab01/src/measure_train_memory.py`, `python lab01/src/plot_timings.py`,
`python lab01/src/deployment_fit.py`.
