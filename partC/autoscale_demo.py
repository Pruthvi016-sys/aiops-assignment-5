"""partC/autoscale_demo.py -- optional evidence that the candidate's autoscaling_config works.

    python partC/autoscale_demo.py        (from the repository root, venv active; ~2 min)

Deploys build(weight=1.0, extra_work_ms=30) so every customer goes to the candidate and each request
costs ~30 ms of CPU, drives it with 8 concurrent clients for 30 s, then idles, and prints the number of
running EtaCandidate replicas every 3 s. Output -> partC/autoscale_output.txt
"""
import sys
import threading
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ray  # noqa: E402
from ray import serve  # noqa: E402
import eta_ray  # noqa: E402

URL = "http://127.0.0.1:8000/v1/models/eta:predict"
ROW = {"distance_km": 18.5, "n_stops": 6, "hour": 17, "weekday": 3, "traffic_index": 0.7, "rain": 1, "weight_kg": 9.0}
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def n_candidate():
    d = serve.status().applications["eta"].deployments["EtaCandidate"]
    return dict(d.replica_states).get("RUNNING", 0)


def main(load_s=30, idle_s=90, clients=8):
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": f"{HERE}:{ROOT}"}})
    serve.start(http_options={"host": "127.0.0.1", "port": 8000})
    try:
        serve.run(eta_ray.build(weight=1.0, extra_work_ms=30), name="eta", route_prefix="/")
        say(f"autoscaling_config = {eta_ray.CANDIDATE_AUTOSCALING}")
        stop = threading.Event()
        done = [0]

        def client(i):
            s = requests.Session()
            while not stop.is_set():
                s.post(URL, json={"customer_id": f"load-{i}", "instances": [ROW]}, timeout=30)
                done[0] += 1

        threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(clients)]
        t0 = time.time()
        for t in threads:
            t.start()
        peak = 1
        while time.time() - t0 < load_s + idle_s:
            el = time.time() - t0
            if el >= load_s and not stop.is_set():
                stop.set()
                say(f"t={el:5.1f}s  -- load stopped --")
            n = n_candidate()
            peak = max(peak, n)
            say(f"t={el:5.1f}s  phase={'load' if el < load_s else 'idle'}  EtaCandidate RUNNING replicas={n}  "
                f"requests served={done[0]}")
            time.sleep(3)
        say(f"peak candidate replicas = {peak} (min 1, max {eta_ray.CANDIDATE_AUTOSCALING['max_replicas']}); "
            f"downscale_delay_s={eta_ray.CANDIDATE_AUTOSCALING['downscale_delay_s']} so scale-down happens "
            f"only after a minute of low load")
    finally:
        (HERE / "autoscale_output.txt").write_text("\n".join(LOG) + "\n")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
