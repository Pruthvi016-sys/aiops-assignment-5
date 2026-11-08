"""partC/verify_ray.py -- C2 verification of the sticky canary on a local 2-CPU Ray.

    python partC/verify_ray.py            (from the repository root, venv active)

Checks (each one is an assert; evidence is printed and also written to partC/verify_output.txt):
  (a) with weight 0.2, the share of 1000 distinct customers routed to v2 is within 0.2 +- 0.04
  (b) 200 customers get the same arm on a second call
  (c) the same holds after deleting and redeploying the identical app (new replica processes)
  (d) after raising the weight 0.2 -> 0.5 no customer that was on v2 moves back to v1
  (e) X-Beta: 1 forces v2 for 50 requests (all from customers normally on v1)
plus: deployment / replica / CPU-reservation evidence from Ray itself.
"""
import concurrent.futures as cf
import json
import sys
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
APP = "eta"
ROW = {"distance_km": 18.5, "n_stops": 6, "hour": 17, "weekday": 3, "traffic_index": 0.7,
       "rain": 1, "weight_kg": 9.0}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name=APP, route_prefix="/")
    for _ in range(120):                               # wait until the HTTP route answers
        try:
            if requests.get("http://127.0.0.1:8000/healthz", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("app did not become healthy")


def ask(cid, beta=False, sess=None):
    h = {"X-Beta": "1"} if beta else {}
    r = (sess or requests).post(URL, json={"customer_id": cid, "instances": [ROW]}, headers=h, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["arm"], j["model_version"], j["served_by"]


def ask_many(ids, beta=False, workers=8):
    def job(chunk):
        s = requests.Session()
        return [(cid, *ask(cid, beta, s)) for cid in chunk]
    chunks = [ids[i::workers] for i in range(workers)]
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(job, chunks):
            for cid, arm, ver, who in res:
                out[cid] = (arm, ver, who)
    return {cid: out[cid] for cid in ids}


def replica_evidence():
    st = serve.status().applications[APP]
    for name, d in st.deployments.items():
        say(f"   deployment {name:13s} status={d.status.value if hasattr(d.status, 'value') else d.status} "
            f"replicas={dict(d.replica_states)}")
    say(f"   cluster CPUs={ray.cluster_resources().get('CPU')}  "
        f"available CPUs={round(ray.available_resources().get('CPU', 0), 2)}")


def main():
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": f"{HERE}:{ROOT}"}})
    serve.start(http_options={"host": "127.0.0.1", "port": 8000})
    ids = [f"cust-{i:04d}" for i in range(1000)]
    passed = []
    try:
        say("== deploy weight=0.2 ==")
        deploy(0.2)
        replica_evidence()
        cfg = requests.get("http://127.0.0.1:8000/canary").json()
        say("   canary config:", json.dumps(cfg))
        say(f"   CPU reservation: router {eta_ray.ROUTER_CPU} + prod 2x{eta_ray.PROD_CPU} + "
            f"candidate max {eta_ray.CANDIDATE_AUTOSCALING['max_replicas']}x{eta_ray.CAND_CPU} = "
            f"{eta_ray.ROUTER_CPU + 2 * eta_ray.PROD_CPU + 3 * eta_ray.CAND_CPU} CPU <= 2")

        # (a) share --------------------------------------------------------------------------
        t0 = time.time()
        first = ask_many(ids)
        v2 = [c for c in ids if first[c][0] == "v2"]
        share = len(v2) / len(ids)
        mism = [c for c in ids if first[c][0] != eta_ray.choose_arm(c, 0.2)]
        arm_ver_ok = all((a == "v1" and v == "v1") or (a == "v2" and v == "v2") for a, v, _ in first.values())
        say(f"(a) 1000 customers in {time.time() - t0:.1f}s: v2 share = {share:.3f} ({len(v2)}/1000); "
            f"expected 0.200 +- 0.040; server vs local hash mismatches = {len(mism)}; "
            f"arm matches model_version: {arm_ver_ok}")
        say(f"    replicas that answered: {sorted({w for _, _, w in first.values()})}")
        assert abs(share - 0.2) <= 0.04, share
        assert not mism and arm_ver_ok
        passed.append("a")

        # (b) stickiness on a second call -------------------------------------------------------
        sample = ids[:200]
        second = ask_many(sample)
        changed = [c for c in sample if second[c][0] != first[c][0]]
        n_v2 = sum(first[c][0] == "v2" for c in sample)
        say(f"(b) second call for 200 customers ({n_v2} on v2, {200 - n_v2} on v1): arm changed for {len(changed)}")
        assert not changed
        passed.append("b")

        # (c) after redeploying the identical app -----------------------------------------------
        old_pids = {w for _, _, w in first.values()}
        serve.delete(APP)
        say("(c) app deleted; redeploying identical app (weight=0.2) ...")
        deploy(0.2)
        third = ask_many(sample)
        new_pids = {w for _, _, w in third.values()}
        changed = [c for c in sample if third[c][0] != first[c][0]]
        say(f"    new replica processes: {sorted(new_pids)} (overlap with old: {sorted(old_pids & new_pids)})")
        say(f"    arm changed for {len(changed)} of 200 customers after redeploy")
        assert not changed
        passed.append("c")

        # (d) raise weight 0.2 -> 0.5 -----------------------------------------------------------
        deploy(0.5)
        after = ask_many(ids)
        back = [c for c in v2 if after[c][0] != "v2"]
        share5 = sum(after[c][0] == "v2" for c in ids) / len(ids)
        moved_up = sum(first[c][0] == "v1" and after[c][0] == "v2" for c in ids)
        say(f"(d) weight 0.5: v2 share = {share5:.3f}; of {len(v2)} customers previously on v2, "
            f"{len(back)} moved back to v1; {moved_up} customers moved v1 -> v2")
        assert not back
        assert abs(share5 - 0.5) <= 0.04
        passed.append("d")

        # (e) header override ---------------------------------------------------------------------
        on_v1 = [c for c in ids if after[c][0] == "v1"][:50]
        forced = ask_many(on_v1, beta=True)
        n_forced = sum(forced[c][0] == "v2" and forced[c][1] == "v2" for c in on_v1)
        normal = ask_many(on_v1[:10])
        say(f"(e) X-Beta: 1 for 50 v1 customers -> {n_forced}/50 served by v2; "
            f"same customers without header -> {sorted({normal[c][0] for c in normal})}")
        assert n_forced == 50 and all(normal[c][0] == "v1" for c in normal)
        passed.append("e")

        say("== final deployment state ==")
        replica_evidence()
        say(f"ALL CHECKS PASSED: {passed}")
    finally:
        (HERE / "verify_output.txt").write_text("\n".join(LOG) + "\n")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
