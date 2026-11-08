#!/usr/bin/env bash
# run_all.sh -- reproduce EVERY result of the assignment from a clean checkout (report: REPORT.pdf).
#   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#   bash run_all.sh            (~8-10 minutes on a 2-core laptop)
# Ports used: 8000 (Part A / Ray), 5101 + 5102 (MLflow). Make sure nothing else is listening there.
set -euo pipefail
cd "$(dirname "$0")"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
export MLFLOW_DISABLE_AGENT_HINT=1
PY=python

step() { echo; echo "==================== $* ===================="; }

step "0. data + models"
$PY eta_data.py | tee data_output.txt
$PY -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

step "A2 + B2. unit tests"
$PY -m pytest -q partA 2>&1 | tee partA/pytest_output.txt
$PY -m pytest -q partB 2>&1 | tee partB/pytest_output.txt

step "A3. load tests (about 2 minutes)"
$PY partA/run_loadtests.py

step "B1 + B2. MLflow registry + promotion gate"
rm -rf mlflow.db mlruns mlartifacts
$PY partB/eta_mlflow.py

step "B3. shadow comparison (mlflow models serve on 5101/5102)"
$PY partB/shadow.py

step "C2. Ray sticky-canary verification"
ray stop --force >/dev/null 2>&1 || true
$PY partC/verify_ray.py

step "C (extra). autoscaling demo (about 2 minutes)"
$PY partC/autoscale_demo.py

step "D. smoke test against the Part A service"
ETA_MODEL=data/eta_v1.joblib $PY -m uvicorn partA.eta_app:app --port 8000 --log-level warning &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 50); do curl -sf localhost:8000/healthz >/dev/null && break; sleep 0.3; done
{
  echo '$ python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80'
  set +e
  $PY partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80; echo "exit code: $?"
  echo
  echo '# negative control: the gate must FAIL when the wrong version is deployed'
  echo '$ python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v2'
  $PY partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v2; echo "exit code: $?"
  set -e
} 2>&1 | tee partD/smoke_output.txt
kill $SRV; trap - EXIT

echo "Done. All result files regenerated (partA/*.json, partB/*_output.txt, partC/*_output.txt, partD/smoke_output.txt)."
echo "The written report is REPORT.pdf (LaTeX source: report/REPORT.tex)."
