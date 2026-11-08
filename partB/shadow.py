"""partB/shadow.py -- B3 shadow comparison of @prod vs @candidate served by `mlflow models serve`.

    python partB/shadow.py                 # starts both servers itself, writes partB/shadow_output.txt
    python partB/shadow.py --no-start      # if you already started them by hand:
        mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local
        mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local

Sends the SAME 200 single-row requests (first 200 rows of data/eta_fresh.csv) to both servers,
alternating which server is hit first so neither gets a systematic warm-cache advantage.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import eta_data  # noqa: E402

SERVERS = {"prod": 5101, "candidate": 5102}


def start(alias, port):
    env = dict(os.environ, MLFLOW_TRACKING_URI=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
               MLFLOW_DISABLE_AGENT_HINT="1")
    cmd = ["mlflow", "models", "serve", "-m", f"models:/eta-model@{alias}", "-p", str(port),
           "--host", "127.0.0.1", "--env-manager", "local", "--workers", "1"]
    log = open(HERE / f"_serve_{alias}.log", "w")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait(port, timeout=120):
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            if requests.get(f"http://127.0.0.1:{port}/ping", timeout=1).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server on {port} did not come up (see partB/_serve_*.log)")


def call(sess, port, rec):
    t0 = time.perf_counter()
    r = sess.post(f"http://127.0.0.1:{port}/invocations", json={"dataframe_records": [rec]}, timeout=10)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    p = r.json()["predictions"][0]
    return p["eta_minutes"], p["model_version"], ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    procs = [] if a.no_start else [start(al, port) for al, port in SERVERS.items()]
    try:
        for port in SERVERS.values():
            wait(port)
        recs = pd.read_csv(ROOT / "data" / "eta_fresh.csv")[eta_data.FEATURES].head(a.n).to_dict("records")
        sess = {k: requests.Session() for k in SERVERS}
        for k, port in SERVERS.items():                 # warm-up, not measured
            for rec in recs[:10]:
                call(sess[k], port, rec)
        res = {k: {"pred": [], "ver": set(), "ms": []} for k in SERVERS}
        for i, rec in enumerate(recs):
            order = list(SERVERS) if i % 2 == 0 else list(SERVERS)[::-1]
            for k in order:
                pred, ver, ms = call(sess[k], SERVERS[k], rec)
                res[k]["pred"].append(pred); res[k]["ver"].add(ver); res[k]["ms"].append(ms)
        p, c = np.array(res["prod"]["pred"]), np.array(res["candidate"]["pred"])
        diff = np.abs(p - c)
        y = pd.read_csv(ROOT / "data" / "eta_fresh.csv")["eta_minutes"].head(a.n).to_numpy()
        summary = {
            "requests_per_server": len(recs),
            "prod_model_version": sorted(res["prod"]["ver"]),
            "candidate_model_version": sorted(res["candidate"]["ver"]),
            "mean_abs_diff_minutes": round(float(diff.mean()), 2),
            "pct_diff_gt_10_min": round(float((diff > 10).mean() * 100), 1),
            "max_abs_diff_minutes": round(float(diff.max()), 1),
            "prod_p50_ms": round(float(np.percentile(res["prod"]["ms"], 50)), 2),
            "prod_p95_ms": round(float(np.percentile(res["prod"]["ms"], 95)), 2),
            "candidate_p50_ms": round(float(np.percentile(res["candidate"]["ms"], 50)), 2),
            "candidate_p95_ms": round(float(np.percentile(res["candidate"]["ms"], 95)), 2),
            "info_prod_mae_on_these_200": round(float(np.abs(p - y).mean()), 2),
            "info_candidate_mae_on_these_200": round(float(np.abs(c - y).mean()), 2),
        }
        text = "== B3 shadow comparison: same 200 single-row requests to both servers ==\n" + json.dumps(summary, indent=2)
        print(text)
        (HERE / "shadow_output.txt").write_text(text + "\n")
        (HERE / "shadow_output.json").write_text(json.dumps(summary, indent=2))
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(15)
            except subprocess.TimeoutExpired:
                pr.kill()


if __name__ == "__main__":
    main()
