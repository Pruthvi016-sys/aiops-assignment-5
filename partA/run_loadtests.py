"""partA/run_loadtests.py -- runs every A3 measurement with the provided load_test.py and analyses it.

    python partA/run_loadtests.py            (from the repository root, venv active)

Steps (each server is started and stopped automatically):
  1. v1 served by 1 uvicorn worker  -> concurrency 1 2 4 8 16, 4 s each, batch 1  -> partA/loadtest_results.json
  2. same server, concurrency 1     -> batch 1 and batch 100                       -> partA/loadtest_batch.json
  3. the capacity change: 2 uvicorn workers (one per core) -> same sweep           -> partA/loadtest_2workers.json
  4. analysis (Little's law, saturation, SLO capacity)                             -> partA/loadtest_analysis.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "partA"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/models/eta:predict"
SLO_P95_MS = 80.0
PY = sys.executable


def start_server(workers: int):
    env = dict(os.environ, ETA_MODEL="data/eta_v1.joblib")
    cmd = [PY, "-m", "uvicorn", "partA.eta_app:app", "--host", "127.0.0.1", "--port", str(PORT),
           "--workers", str(workers), "--log-level", "warning"]
    p = subprocess.Popen(cmd, cwd=ROOT, env=env)
    for _ in range(100):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/healthz", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        p.kill()
        raise RuntimeError("server did not start")
    for _ in range(50):                                   # warm-up (imports, first predict)
        requests.post(URL, json={"instances": [{"distance_km": 10, "n_stops": 3, "hour": 8, "weekday": 1,
                                                "traffic_index": 0.5, "rain": 0, "weight_kg": 4}]})
    return p


def stop_server(p):
    p.terminate()
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(1)


def load_test(out: Path, conc, batch=1, duration=4):
    cmd = [PY, "load_test.py", "--url", URL, "--data", "data/eta_sample.csv", "--concurrency",
           *map(str, conc), "--duration", str(duration), "--batch", str(batch), "--out", str(out)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(out.read_text())


def analyse(rows):
    """Little's law for a closed loop: N = X * R  ->  mean response time R = N / X.
    Service demand D = R at N=1 (no queueing).  Max throughput X_max ~ highest measured rps.
    Saturation N* ~ X_max * D: beyond it, extra clients only queue (R grows ~linearly, X flat)."""
    out = []
    for r in rows:
        n, x = r["concurrency"], r["rps"]
        out.append(dict(concurrency=n, rps=x, p95_ms=r["p95_ms"],
                        littles_law_mean_ms=round(1000 * n / x, 2) if x else None,
                        meets_slo=r["p95_ms"] <= SLO_P95_MS and r["errors"] == 0))
    d_ms = out[0]["littles_law_mean_ms"]
    x_max = max(r["rps"] for r in rows)
    n_star = x_max * d_ms / 1000
    ok = [r for r in out if r["meets_slo"]]
    best = max(ok, key=lambda r: r["rps"]) if ok else None
    knee = next(r["concurrency"] for r in out if r["rps"] >= 0.95 * x_max)   # observed saturation
    return dict(per_level=out, service_demand_ms=d_ms, max_rps=x_max,
                saturation_concurrency_littles_law=round(n_star, 2),
                observed_knee_concurrency_95pct_of_max=knee,
                slo_p95_ms=SLO_P95_MS,
                capacity_under_slo_rps=best["rps"] if best else 0,
                capacity_under_slo_at_concurrency=best["concurrency"] if best else None)


def main():
    conc = [1, 2, 4, 8, 16]
    p = start_server(1)
    try:
        sweep = load_test(A / "loadtest_results.json", conc)
        b1 = load_test(A / "_b1.json", [1], batch=1)[0]
        b100 = load_test(A / "_b100.json", [1], batch=100)[0]
    finally:
        stop_server(p)
    batch = []
    for b, r in ((1, b1), (100, b100)):
        mean_ms = 1000 / r["rps"]                          # conc 1: mean latency = 1 / throughput
        batch.append(dict(batch=b, rps=r["rps"], p50_ms=r["p50_ms"], mean_request_ms=round(mean_ms, 2),
                          ms_per_row=round(mean_ms / b, 4), rows_per_s=round(r["rps"] * b, 1)))
    (A / "loadtest_batch.json").write_text(json.dumps(batch, indent=2))
    for f in ("_b1.json", "_b100.json"):
        (A / f).unlink()

    p = start_server(2)
    try:
        sweep2 = load_test(A / "loadtest_2workers.json", conc)
    finally:
        stop_server(p)

    a1, a2 = analyse(sweep), analyse(sweep2)
    res = dict(one_worker=a1, two_workers=a2, batch=batch,
               capacity_ratio_2w_over_1w=round(a2["capacity_under_slo_rps"] / max(a1["capacity_under_slo_rps"], 1e-9), 2),
               max_rps_ratio_2w_over_1w=round(a2["max_rps"] / a1["max_rps"], 2))
    (A / "loadtest_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
