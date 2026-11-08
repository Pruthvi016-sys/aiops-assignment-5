# DA3408 A5 — SwiftParcel ETA: Serving a Model Four Ways (AIOps Module 4)

**Mudda Manikanta Pruthvi Raj — DA24B016**

Two pre-trained delivery-time regressors (**v1** Ridge, **v2** Gradient Boosting) are served four ways:

* **Part A:** a validated, instrumented FastAPI REST service, with tests and load tests.
* **Part B:** an MLflow registry with a multi-criteria promotion gate and a shadow comparison.
* **Part C:** a Ray Serve application with a sticky, hash-based canary.
* **Part D:** a release-gate smoke test, plus abort criteria for the canary.

| Video | Link |
|---|---|
| 🎥 Demo + explanation (2–5 min) | _add Google Drive link here_ |

**Report:** [`REPORT.pdf`](REPORT.pdf) (4 pages) has every table, every written answer, the machine description and the list of what was not tested.

---

## Where to find each requirement

"Report" means [`REPORT.pdf`](REPORT.pdf), with the page number in brackets. Screenshot numbers refer to [`evidence_screenshots/`](evidence_screenshots).

### Part A — REST service

| Requirement | Code | Answer / output | Screenshot |
|---|---|---|---|
| **A1.1** `POST /v1/models/eta:predict`, 1-decimal predictions, same order | [`partA/eta_app.py`](partA/eta_app.py) → `predict()` | Report A1 (p.1) | 02 |
| **A1.2** Feature ranges, batch size 1–500 | `eta_app.py` → class `Delivery` (`Field` limits), class `PredictRequest` (`MAX_BATCH`) | Report A1 (p.1) | 02 |
| **A1.3** Cross-field rule → 422 "infeasible" | `eta_app.py` → `Delivery._feasible()` | Report A1 (p.1) | 02 |
| **A1.4** `X-Request-ID` echo or UUID on every response | `eta_app.py` → middleware `request_id_and_metrics()` | Report A1 (p.1) | 02 |
| **A1.5** 422 body `{"error","details":[{field,msg}],"request_id"}` | `eta_app.py` → `validation_handler()` | Report A1 (p.1) | 02 |
| **A1.6** `/metrics`: counter and histogram, `/metrics` not counted | `eta_app.py` → class `Metrics`, `metrics()` | Report A1 (p.1) | 03 |
| **A1.7** `/healthz`; generic 500 without internals | `eta_app.py` → `healthz()`; `predict()` `try/except`; `generic_handler()` | Report A1 (p.1) | 03 |
| **A2** ≥ 10 pytest cases (TestClient) | [`partA/test_eta_app.py`](partA/test_eta_app.py) (23 cases) | Report A2 (p.1) | 04 |
| **A3.1** Load-test table, concurrency 1/2/4/8/16 | [`partA/run_loadtests.py`](partA/run_loadtests.py) runs the provided [`load_test.py`](load_test.py) | Report A3 table (p.1) + Fig. 1 (p.2); [`partA/loadtest_results.json`](partA/loadtest_results.json) | 05, 08 |
| **A3.2** Saturation point and capacity under p95 ≤ 80 ms (with reasoning) | `run_loadtests.py` → `analyse()` (Little's law) | Report A3 (2) (p.1); [`partA/loadtest_analysis.json`](partA/loadtest_analysis.json) | 06, 08 |
| **A3.3** ms/row for batch 1 vs batch 100 | `run_loadtests.py` → `main()` | Report A3 (3) (p.1); [`partA/loadtest_batch.json`](partA/loadtest_batch.json) | 07, 08 |
| **A3.4** One change to double capacity, measured honestly | `--workers 2` (in `run_loadtests.py`) | Report A3 (4) (p.1); [`partA/loadtest_2workers.json`](partA/loadtest_2workers.json) | 07, 08 |
| **A3.5** Why p95/p99, not the mean | — | Report A3 (5) (p.1) | — |

### Part B — MLflow

| Requirement | Code | Answer / output | Screenshot |
|---|---|---|---|
| **B1** Custom pyfunc → `eta_minutes`, `model_version` | [`partB/eta_pyfunc.py`](partB/eta_pyfunc.py) → `EtaPyfunc` | Report B1 (p.2) | 10 |
| **B1** Signature, input example, pinned `pip_requirements` | [`partB/eta_mlflow.py`](partB/eta_mlflow.py) → `register()`, `_pinned_requirements()` | Report B1 (p.2) | 10 |
| **B1** Registered as `eta-model`, aliases `prod`→v1, `candidate`→v2, SQLite store | `eta_mlflow.py` → `register()`, `registry_table()` | Report B1 registry + Fig. 2 (p.2); [`partB/gate_output.txt`](partB/gate_output.txt) | 09, 12, 13 |
| **B2** Pure `decide(prod, cand, …)` | `eta_mlflow.py` → `decide()` | Report B2 (p.2) | — |
| **B2** `promotion_gate(**thresholds)` on fresh data, p95 over 200 rows | `eta_mlflow.py` → `promotion_gate()`, `_evaluate()` | Report B2 (p.2) | 09, 14 |
| **B2** ≥ 6 tests with boundary cases | [`partB/test_gate.py`](partB/test_gate.py) (21 cases) | Report B2 (p.2) | 16 |
| **B2** Gate output: defaults and `min_rel_gain=0.60` | `eta_mlflow.py` → `main()` | Report B2 table (p.2); [`partB/gate_output.txt`](partB/gate_output.txt) | 14, 15 |
| **B3** `mlflow models serve` on 5101 / 5102; same 200 requests to both | [`partB/shadow.py`](partB/shadow.py) | [`partB/shadow_output.txt`](partB/shadow_output.txt) | 17, 18, 19 |
| **B3** (i) mean abs diff, (ii) % > 10 min, (iii) p95 per server | `shadow.py` | Report B3 (p.2) | 19 |
| **B3** Promote? Why moving an alias doesn't change a running server; two zero-downtime ways | — (plus a live alias-move demo) | Report B3 (p.2) | 19 |

### Part C — Ray Serve sticky canary

| Requirement | Code | Answer / output | Screenshot |
|---|---|---|---|
| **C1.1** `EtaModel`: load once per replica, batch predict; prod 2 replicas; candidate autoscaling 1–3 | [`partC/eta_ray.py`](partC/eta_ray.py) → `EtaModel`, `build()`, `CANDIDATE_AUTOSCALING` | Report C1 (p.3) | 21, 22 |
| **C1.2** Router: `{customer_id, instances}` → output + `"arm"` | `eta_ray.py` → `Router.predict()` | Report C1 (p.3) | 23 |
| **C1.3** Sticky canary by stable hash, customers not requests | `eta_ray.py` → `bucket()`, `choose_arm()` | Report C1 (p.3) | 23 |
| **C1.4** `X-Beta: 1` forces the candidate | `eta_ray.py` → `Router.predict()` | Report C1 (p.3) | 23 |
| **C1.5** Fractional CPUs fit 2 CPUs; `build(weight=0.1, extra_work_ms=0.0)` | `eta_ray.py` → `PROD_CPU`, `CAND_CPU`, `ROUTER_CPU`, `build()` | Report C1 (p.3) | 22 |
| **C2** (a)–(e) with assertions, 1000 ids, weight 0.2, 2-CPU Ray | [`partC/verify_ray.py`](partC/verify_ray.py) | Report C2 + Fig. 4 (p.3); [`partC/verify_output.txt`](partC/verify_output.txt) | 24 |
| **C3.1** Why not Python's `hash()` | — (plus a live demo) | Report C3 (p.3) | 25 |
| **C3.2** Two reasons stickiness matters | — | Report C3 (p.3) | — |
| **C3.3** Justify the autoscaling parameters | — (evidence: [`partC/autoscale_demo.py`](partC/autoscale_demo.py)) | Report C3 (p.3); [`partC/autoscale_output.txt`](partC/autoscale_output.txt) | 25 |

### Part D — Release gate

| Requirement | Code | Answer / output | Screenshot |
|---|---|---|---|
| **D1.1** `--url --expect-version [--max-p95-ms]`, exit 0/1, ≥ 5 checks, ≥ 2 model-sanity checks, a 422 check | [`partD/smoke_eta.py`](partD/smoke_eta.py) → class `Smoke` (`c_health`, `c_predict`, `c_plausible`, `c_distance`, `c_traffic`, `c_stops`, `c_invalid`, `c_latency`) | Report D1.1 + Fig. 5 (p.3–4); [`partD/smoke_output.txt`](partD/smoke_output.txt) | 26, 27 |
| **D1.2** Four abort criteria (metric, threshold, window) | — | Report D1.2 table (p.4) | — |

### Submission items

| Item | Where |
|---|---|
| Report ≤ 4 pages, machine description, list of what was not tested | [`REPORT.pdf`](REPORT.pdf): header (machine) and §5 "Not tested / limitations" (p.4) |
| Interface contract (§2.1 of the assignment) | Names exactly as required: `app`, `model`, `ETA_MODEL`, `register`, `decide`, `promotion_gate`, `MLFLOW_TRACKING_URI`, `build`, `--url --expect-version --max-p95-ms` |
| No models or databases in the repo | [`.gitignore`](.gitignore) excludes `data/`, `mlflow.db`, `mlruns/`; regenerate with `python eta_data.py` |
| Setup and run guide | [How we ran it, end to end](#how-we-ran-it-end-to-end) below |

---

## Repository structure

```
.
├── README.md                     # this guide
├── REPORT.pdf                    # the report (4 pages): all tables, answers, machine, not-tested list
├── requirements.txt              # starter file (unchanged)
├── eta_data.py                   # starter file (unchanged): creates data/ and the v1/v2 model bundles
├── load_test.py                  # starter file (unchanged): load generator from the lecture
├── run_all.sh                    # re-runs every step in one go
├── partA/
│   ├── eta_app.py                # A1  FastAPI service (app, model, ETA_MODEL)
│   ├── test_eta_app.py           # A2  23 pytest cases
│   ├── run_loadtests.py          # A3  runs load_test.py for every measurement + Little's-law analysis
│   ├── loadtest_results.json     # A3  concurrency sweep, 1 worker
│   ├── loadtest_batch.json       # A3  batch 1 vs 100
│   ├── loadtest_2workers.json    # A3  sweep with 2 workers
│   └── loadtest_analysis.json    # A3  saturation, capacity, ratios
├── partB/
│   ├── eta_pyfunc.py             # B1  custom pyfunc (EtaPyfunc)
│   ├── eta_mlflow.py             # B1/B2  register(), decide(), promotion_gate()
│   ├── test_gate.py              # B2  21 tests of decide()
│   ├── gate_output.txt / .json   # B1/B2  registry table + gate outputs
│   ├── shadow.py                 # B3  shadow comparison
│   └── shadow_output.txt / .json # B3  results
├── partC/
│   ├── eta_ray.py                # C1  EtaModel, Router, bucket(), build()
│   ├── verify_ray.py             # C2  checks (a)–(e) with assertions
│   ├── verify_output.txt         # C2  printed evidence
│   ├── autoscale_demo.py         # C3  autoscaling evidence
│   └── autoscale_output.txt
├── partD/
│   ├── smoke_eta.py              # D1  release-gate smoke test
│   └── smoke_output.txt
└── evidence_screenshots/         # 27 screenshots of the end-to-end run
```

`data/`, `mlflow.db` and `mlruns/` are created when you run the code and are not committed.

---

## How we ran it, end to end

Tested on an HP Laptop 15s (Intel i5-1235U, 12 logical CPUs, 14 GiB RAM), Ubuntu, Python 3.14.7. Run every command
from the repository folder. "Terminal 1" is for servers that keep running; "Terminal 2" is for everything else.
Run `source .venv/bin/activate` in every new terminal.

> Shortcut: after step 0, `bash run_all.sh` re-runs every result in one go (~10 min). The steps below are the
> exact commands we ran one by one to take the screenshots.

### 0. Setup and data
```bash
git clone https://github.com/Pruthvi016-sys/aiops-assignment5.git
cd aiops-assignment5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
```
Expected: `v1 … MAE = 11.95 min`, `v2 … MAE = 6.10 min` (screenshot 01).

### Part A1 — REST service
**Terminal 1**
```bash
ETA_MODEL=data/eta_v1.joblib uvicorn partA.eta_app:app --port 8000
```
**Terminal 2**
```bash
U=localhost:8000/v1/models/eta:predict; H='content-type: application/json'
R='{"distance_km":12.5,"n_stops":4,"hour":8,"weekday":2,"traffic_index":0.6,"rain":1,"weight_kg":5}'

# A1.1 valid prediction + batch order
curl -s -i -X POST $U -H "$H" -d "{\"instances\":[$R]}"; echo
curl -s -X POST $U -H "$H" -d '{"instances":[{"distance_km":5,"n_stops":2,"hour":3,"weekday":1,"traffic_index":0.1,"rain":0,"weight_kg":2},{"distance_km":120,"n_stops":15,"hour":17,"weekday":4,"traffic_index":0.9,"rain":1,"weight_kg":40},{"distance_km":30,"n_stops":6,"hour":12,"weekday":2,"traffic_index":0.5,"rain":0,"weight_kg":8}]}'; echo

# A1.2 ranges + empty batch
curl -s -X POST $U -H "$H" -d '{"instances":[{"distance_km":600,"n_stops":4,"hour":25,"weekday":2,"traffic_index":0.6,"rain":3,"weight_kg":5}]}'; echo
curl -s -X POST $U -H "$H" -d '{"instances":[]}'; echo

# A1.3 infeasible rule
curl -s -X POST $U -H "$H" -d '{"instances":[{"distance_km":200,"n_stops":30,"hour":8,"weekday":2,"traffic_index":0.6,"rain":1,"weight_kg":5}]}'; echo

# A1.4 / A1.5 request id echoed (also on an error) + error format
curl -s -i -X POST $U -H "$H" -H "X-Request-ID: my-test-123" -d "{\"instances\":[$R]}"; echo
curl -s -i -X POST $U -H "$H" -H "X-Request-ID: err-456" -d '{"instances":[]}'; echo

# A1.6 metrics
curl -s localhost:8000/metrics

# A1.7 health + generic 500 (via the monkeypatch test)
curl -s localhost:8000/healthz; echo
python -m pytest -v partA -k generic_500
```
Screenshots 02, 03. Then stop the server in Terminal 1 (Ctrl+C).

### Part A2 — Tests
```bash
python -m pytest -v partA
```
Expected: `23 passed` (screenshot 04).

### Part A3 — Load test (server stopped first; the script starts its own on port 8000)
```bash
nproc; free -h; lscpu | grep "Model name"
python partA/run_loadtests.py
```
Summary of all A3 answers:
```bash
python - <<'PY'
import json, pandas as pd
a = json.load(open('partA/loadtest_analysis.json'))
print("=== A3.1  v1, 1 worker, batch 1 (partA/loadtest_results.json) ===")
print(pd.DataFrame(json.load(open('partA/loadtest_results.json'))).drop(columns=['replicas']).to_string(index=False))
o = a['one_worker']
print(f"\n=== A3.2  service demand D = {o['service_demand_ms']} ms | X_max = {o['max_rps']} req/s | Little N* = {o['saturation_concurrency_littles_law']} | knee (95% of max) at N = {o['observed_knee_concurrency_95pct_of_max']} | capacity under p95<=80ms = {o['capacity_under_slo_rps']} req/s at N = {o['capacity_under_slo_at_concurrency']}")
print("\n=== A3.3  batch 1 vs 100 at concurrency 1 ===")
print(pd.DataFrame(a['batch']).to_string(index=False))
print("\n=== A3.4  2 uvicorn workers ===")
print(pd.DataFrame(json.load(open('partA/loadtest_2workers.json'))).drop(columns=['replicas']).to_string(index=False))
print(f"\ncapacity 1 worker = {o['capacity_under_slo_rps']} -> 2 workers = {a['two_workers']['capacity_under_slo_rps']} req/s  => {a['capacity_ratio_2w_over_1w']}x")
PY
```
Screenshots 05–08.

### Part B1 + B2 — Registry and promotion gate
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
rm -rf mlflow.db mlruns mlartifacts
python partB/eta_mlflow.py
```
This prints the registry table and the gate output for defaults and for `min_rel_gain=0.60`, and writes
`partB/gate_output.txt` (screenshots 09, 14, 15).

Signature, input example and pinned requirements:
```bash
f=$(ls mlruns/*/models/*/artifacts/MLmodel | head -1); d=$(dirname $f)
ls $d
grep -A10 "^signature" $f
cat $d/requirements.txt; echo
```
Pyfunc output through both aliases:
```bash
python - <<'PY' 2>&1 | grep -v -E "INFO|Warning|color_warning"
import mlflow, eta_data
mlflow.set_tracking_uri("sqlite:///mlflow.db")
X = eta_data.make_data(3, seed=7)[eta_data.FEATURES]
for a in ("prod", "candidate"):
    print(f"--- models:/eta-model@{a}")
    print(mlflow.pyfunc.load_model(f"models:/eta-model@{a}").predict(X))
PY
```
Screenshot 10. MLflow UI (open http://127.0.0.1:5000 → **Model registry** → **eta-model**, then Ctrl+C):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Screenshots 11–13. Gate tests:
```bash
python -m pytest -v partB
```
Expected: `21 passed` (screenshot 16).

### Part B3 — Shadow comparison
**Terminal 1**
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m "models:/eta-model@prod" -p 5101 --env-manager local
```
**Terminal 3**
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
```
**Terminal 2**
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
for p in 5101 5102; do curl -s -X POST localhost:$p/invocations -H 'content-type: application/json' -d '{"dataframe_records":[{"distance_km":12.5,"n_stops":4,"hour":8,"weekday":2,"traffic_index":0.6,"rain":1,"weight_kg":5}]}'; echo; done
python partB/shadow.py --no-start
```
Live proof that moving an alias does not change the running server (then the alias is restored):
```bash
python -c "import mlflow; from mlflow import MlflowClient; mlflow.set_tracking_uri('sqlite:///mlflow.db'); c=MlflowClient(); c.set_registered_model_alias('eta-model','prod','2'); print('alias prod now ->', c.get_model_version_by_alias('eta-model','prod').version)"
curl -s -X POST localhost:5101/invocations -H 'content-type: application/json' -d '{"dataframe_records":[{"distance_km":12.5,"n_stops":4,"hour":8,"weekday":2,"traffic_index":0.6,"rain":1,"weight_kg":5}]}'; echo
python -c "import mlflow; from mlflow import MlflowClient; mlflow.set_tracking_uri('sqlite:///mlflow.db'); c=MlflowClient(); c.set_registered_model_alias('eta-model','prod','1'); print('alias prod restored ->', c.get_model_version_by_alias('eta-model','prod').version)"
```
Screenshots 17–19. Stop both servers with Ctrl+C. (`python partB/shadow.py` without `--no-start` starts and stops both servers by itself.)

### Part C1 — Ray Serve app on 2 CPUs
**Terminal 1**
```bash
ray stop --force
ray start --head --num-cpus 2
serve run partC.eta_ray:app
```
**Terminal 2** — deployment status, replicas and CPU reservation:
```bash
python - <<'PY' 2>&1 | grep -vE "INFO|WARNING|^\s*$|Connecting to existing"
import ray
from ray import serve
ray.init(address="auto", log_to_driver=False)
st = serve.status().applications["default"]
print("app status:", st.status)
for n, d in st.deployments.items():
    print(f"  {n:13s} {d.status} replicas={dict(d.replica_states)}")
print("CPUs total =", ray.cluster_resources().get("CPU"), "| reserved =", round(ray.cluster_resources().get("CPU") - ray.available_resources().get("CPU", 0), 2))
PY
curl -s localhost:8000/canary; echo
```
Routing by customer, stickiness and the `X-Beta` override:
```bash
U=localhost:8000/v1/models/eta:predict; H='content-type: application/json'
R='{"distance_km":12.5,"n_stops":4,"hour":8,"weekday":2,"traffic_index":0.6,"rain":1,"weight_kg":5}'
for c in alice bob carol dave erin frank grace heidi ivan judy kim leo mia nina oscar; do printf "%-6s " $c; curl -s -X POST $U -H "$H" -d "{\"customer_id\":\"$c\",\"instances\":[$R]}" | python -c "import sys,json; j=json.load(sys.stdin); print('arm',j['arm'],j['predictions'],j['served_by'])"; done
echo "--- bob 3 times (sticky):"
for i in 1 2 3; do curl -s -X POST $U -H "$H" -d "{\"customer_id\":\"bob\",\"instances\":[$R]}"; echo; done
echo "--- alice with X-Beta: 1:"
curl -s -X POST $U -H "$H" -H "X-Beta: 1" -d "{\"customer_id\":\"alice\",\"instances\":[$R]}"; echo
```
Screenshots 20–23. Then Ctrl+C in Terminal 1.

### Part C2 — Verification (starts its own 2-CPU Ray)
```bash
ray stop --force
python partC/verify_ray.py 2>&1 | grep -vE "^\(|WARNING|INFO"
cat partC/verify_output.txt
```
Expected: `ALL CHECKS PASSED: ['a', 'b', 'c', 'd', 'e']` (screenshot 24).

### Part C3 — Evidence for the short answers
```bash
for i in 1 2 3; do python -c "import hashlib; c='cust-0042'; print('hash():', hash(c), '| sha256 bucket:', round(int.from_bytes(hashlib.sha256(('swiftparcel-eta-canary-v2:'+c).encode()).digest()[:8],'big')/2**64, 6))"; done
ray stop --force
python partC/autoscale_demo.py 2>&1 | grep -E "^t=|peak|autoscaling_config"
```
`hash()` changes on every run while the SHA-256 bucket stays the same. The candidate scales 1 → 3 → 1 (screenshot 25).

### Part D1 — Release gate
**Terminal 1**
```bash
ETA_MODEL=data/eta_v1.joblib uvicorn partA.eta_app:app --port 8000
```
**Terminal 2**
```bash
{ python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80; echo "exit code: $?"; python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v2; echo "exit code: $?"; } | tee partD/smoke_output.txt
```
Expected: `8/8 checks passed` with exit code 0, then the wrong-version control with exit code 1 (screenshots 26, 27).
Then Ctrl+C in Terminal 1.


### Troubleshooting
| Symptom | Fix |
|---|---|
| `FileNotFoundError: data/eta_v1.joblib` | Run step 0 (`python eta_data.py`) |
| `address already in use` on 8000 | `pkill -f uvicorn; ray stop --force` |
| Ray will not start or connect | `ray stop --force`, then retry; Part C2 must start with no Ray running |
| `mlflow models serve` tries to build a virtualenv | Keep `--env-manager local` |
| Token-authentication warning from Ray | Only a notice; it can be ignored |

---

## Evidence screenshots

All in [`evidence_screenshots/`](evidence_screenshots).

| # | Shows |
|---|---|
| [01](evidence_screenshots/01_data_generated_and_partA_server_started.png) | Data and models generated (v1 MAE 11.95, v2 MAE 6.10); Part A server started |
| [02](evidence_screenshots/02_A1_predict_batch_ranges_infeasible_request_id.png) | A1: prediction, batch order, range errors, infeasible rule, request-id echo (also on a 422) |
| [03](evidence_screenshots/03_A1_metrics_healthz_generic500.png) | A1: Prometheus metrics, `/healthz`, generic-500 test passed |
| [04](evidence_screenshots/04_A2_pytest_partA_23_passed.png) | A2: 23 tests passed |
| [05](evidence_screenshots/05_A3_machine_and_loadtest_run.png) | A3: machine (12 CPUs, 14 GiB, i5-1235U) and load-test run |
| [06](evidence_screenshots/06_A3_analysis_1worker.png) | A3: Little's-law analysis, 1 worker |
| [07](evidence_screenshots/07_A3_analysis_2workers_batch.png) | A3: 2 workers and batch analysis |
| [08](evidence_screenshots/08_A3_summary_all_answers.png) | A3: summary of all answers (capacity 456.5 → 831.5 req/s, 1.82×) |
| [09](evidence_screenshots/09_B1_register_registry_table.png) | B1: models registered, registry table (prod → v1, candidate → v2) |
| [10](evidence_screenshots/10_B1_signature_requirements_pyfunc_by_alias.png) | B1: signature, input example, pinned requirements, pyfunc output via both aliases |
| [11](evidence_screenshots/11_B1_mlflow_ui_started.png) | B1: MLflow UI started |
| [12](evidence_screenshots/12_B1_mlflow_ui_logged_models.png) | B1: MLflow UI — both logged models |
| [13](evidence_screenshots/13_B1_mlflow_ui_registry_aliases.png) | B1: MLflow registry — Version 1 @prod, Version 2 @candidate |
| [14](evidence_screenshots/14_B2_gate_default_output.png) | B2: gate with default thresholds → promote true |
| [15](evidence_screenshots/15_B2_gate_min_rel_gain_0.60.png) | B2: gate with `min_rel_gain=0.60` → promote false |
| [16](evidence_screenshots/16_B2_pytest_partB_21_passed.png) | B2: 21 tests of `decide()` passed (boundary cases) |
| [17](evidence_screenshots/17_B3_mlflow_serve_prod_5101.png) | B3: prod served on port 5101 |
| [18](evidence_screenshots/18_B3_mlflow_serve_candidate_5102.png) | B3: candidate served on port 5102 |
| [19](evidence_screenshots/19_B3_shadow_metrics_and_alias_move_demo.png) | B3: shadow metrics (9.48 min, 35 %, p95 5.2 / 6.66 ms); alias moved, server still v1 |
| [20](evidence_screenshots/20_C1_ray_start_2cpus.png) | C1: Ray started with 2 CPUs |
| [21](evidence_screenshots/21_C1_serve_run_deployments.png) | C1: EtaProd ×2, EtaCandidate (autoscaling), Router deployed |
| [22](evidence_screenshots/22_C1_status_replicas_cpu_canary_config.png) | C1: all healthy, 0.95 of 2 CPUs reserved, canary config |
| [23](evidence_screenshots/23_C1_sticky_routing_and_xbeta.png) | C1: sticky routing per customer, X-Beta override |
| [24](evidence_screenshots/24_C2_verify_ray_all_checks_passed.png) | C2: checks (a)–(e) all passed |
| [25](evidence_screenshots/25_C3_hash_vs_sha256_and_autoscaling.png) | C3: `hash()` differs per process, SHA-256 stable; autoscaling 1 → 3 → 1 |
| [26](evidence_screenshots/26_D1_partA_server_started.png) | D1: Part A server started for the smoke test |
| [27](evidence_screenshots/27_D1_smoke_pass_exit0_and_fail_exit1.png) | D1: smoke test 8/8 (exit 0); wrong version (exit 1) |

---

# AI Usage

