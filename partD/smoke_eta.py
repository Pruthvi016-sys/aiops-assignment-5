"""partD/smoke_eta.py -- release gate / smoke test for an ETA REST endpoint (Module 4, Part D).

    python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 [--max-p95-ms 80]
    python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v2 --beta   # Ray: force candidate

--url may be the server base URL or the full .../v1/models/eta:predict URL.
Works against the Part A service and the Part C Ray router (a customer_id is always sent; Part A ignores it).
Exit code 0 = every check passed, 1 = at least one failed (or the server was unreachable).

Checks
  1. health       GET /healthz -> 200, status ok (and model_version == expected if the server reports one)
  2. predict      valid single row -> 200, one finite prediction, model_version == --expect-version
  3. plausible    predictions for a realistic batch lie in (0, 1000) minutes, batch order preserved
  4. sanity-dist  longer distance never gives a shorter ETA (5 -> 20 -> 60 -> 140 km, all else fixed)
  5. sanity-traf  heavier traffic never gives a shorter ETA (traffic 0.0 -> 0.5 -> 1.0)
  6. sanity-stops more stops never gives a shorter ETA (2 -> 10 -> 20 stops)
  7. invalid-422  out-of-range input -> 422 ; infeasible (200 km, 30 stops) -> 422 mentioning 'infeasible'
  8. latency      p95 of 50 sequential single-row requests <= --max-p95-ms
"""
import argparse
import math
import sys
import time

import requests

BASE_ROW = {"distance_km": 25.0, "n_stops": 6, "hour": 17, "weekday": 2,
            "traffic_index": 0.5, "rain": 0, "weight_kg": 8.0}


class Smoke:
    def __init__(self, url, expect_version, max_p95_ms, beta, timeout):
        url = url.rstrip("/")
        self.predict_url = url if url.endswith(":predict") else url + "/v1/models/eta:predict"
        self.base = self.predict_url.rsplit("/v1/models/", 1)[0]
        self.expect = expect_version
        self.max_p95 = max_p95_ms
        self.headers = {"X-Beta": "1"} if beta else {}
        self.timeout = timeout
        self.s = requests.Session()
        self.results = []

    def post(self, rows, cid="smoke-test-customer"):
        return self.s.post(self.predict_url, json={"customer_id": cid, "instances": rows},
                           headers=self.headers, timeout=self.timeout)

    def preds(self, rows):
        r = self.post(rows)
        r.raise_for_status()
        return r.json()["predictions"]

    def check(self, name, fn):
        try:
            ok, info = fn()
        except Exception as e:                       # any exception = failed check
            ok, info = False, f"{type(e).__name__}: {e}"
        self.results.append((name, ok, info))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:12s} {info}", flush=True)

    # ---------------------------------------------------------------- checks
    def c_health(self):
        r = self.s.get(self.base + "/healthz", timeout=self.timeout)
        j = r.json()
        ver = j.get("model_version")
        ok = r.status_code == 200 and j.get("status") == "ok"
        if ver is not None and not self.headers:        # with --beta the health shows prod's version
            ok = ok and ver == self.expect
        return ok, f"HTTP {r.status_code} {j}"

    def c_predict(self):
        r = self.post([BASE_ROW])
        j = r.json()
        p = j.get("predictions", [None])
        ok = (r.status_code == 200 and len(p) == 1 and isinstance(p[0], (int, float))
              and math.isfinite(p[0]) and j.get("model_version") == self.expect)
        return ok, f"HTTP {r.status_code} prediction={p} model_version={j.get('model_version')} expected={self.expect}"

    def c_plausible(self):
        rows = [dict(BASE_ROW, distance_km=d, n_stops=n, hour=h, rain=rn)
                for d, n, h, rn in [(1, 1, 3, 0), (12, 4, 8, 1), (45, 12, 18, 1), (120, 20, 12, 0), (300, 15, 22, 1)]]
        batch = self.preds(rows)
        singles = [self.preds([r])[0] for r in rows]
        ok = len(batch) == len(rows) and all(0 < p < 1000 for p in batch) and batch == singles
        return ok, f"batch={batch} order_preserved={batch == singles}"

    def _monotone(self, field, values):
        p = self.preds([dict(BASE_ROW, **{field: v}) for v in values])
        return all(b >= a for a, b in zip(p, p[1:])), f"{field} {values} -> ETA {p}"

    def c_distance(self):
        return self._monotone("distance_km", [5.0, 20.0, 60.0, 140.0])

    def c_traffic(self):
        return self._monotone("traffic_index", [0.0, 0.5, 1.0])

    def c_stops(self):
        return self._monotone("n_stops", [2, 10, 20])

    def c_invalid(self):
        r1 = self.post([dict(BASE_ROW, hour=25)])
        r2 = self.post([dict(BASE_ROW, distance_km=200, n_stops=30)])
        r3 = self.s.post(self.predict_url, json={"customer_id": "x", "instances": []},
                         headers=self.headers, timeout=self.timeout)
        ok = r1.status_code == 422 and r2.status_code == 422 and "infeasible" in r2.text and r3.status_code == 422
        return ok, f"hour=25 -> {r1.status_code}; infeasible -> {r2.status_code} ('infeasible' in body: " \
                   f"{'infeasible' in r2.text}); empty batch -> {r3.status_code}"

    def c_latency(self):
        for _ in range(5):
            self.post([BASE_ROW])
        lat = []
        for i in range(50):
            t0 = time.perf_counter()
            r = self.post([dict(BASE_ROW, distance_km=5 + i)])
            lat.append((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                return False, f"request {i} returned HTTP {r.status_code}"
        lat.sort()
        p95 = lat[int(0.95 * len(lat)) - 1]
        return p95 <= self.max_p95, f"p50={lat[len(lat) // 2]:.1f} ms p95={p95:.1f} ms (limit {self.max_p95} ms)"

    def run(self):
        print(f"smoke test -> {self.predict_url}  expect-version={self.expect}  "
              f"max-p95-ms={self.max_p95}  headers={self.headers}")
        for name, fn in [("health", self.c_health), ("predict", self.c_predict), ("plausible", self.c_plausible),
                         ("sanity-dist", self.c_distance), ("sanity-traf", self.c_traffic),
                         ("sanity-stops", self.c_stops), ("invalid-422", self.c_invalid),
                         ("latency", self.c_latency)]:
            self.check(name, fn)
        n_ok = sum(ok for _, ok, _ in self.results)
        verdict = n_ok == len(self.results)
        print(f"RESULT: {'PASS' if verdict else 'FAIL'} ({n_ok}/{len(self.results)} checks passed)")
        return 0 if verdict else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--beta", action="store_true", help="send X-Beta: 1 (Ray router: force the candidate)")
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args(argv)
    try:
        return Smoke(a.url, a.expect_version, a.max_p95_ms, a.beta, a.timeout).run()
    except requests.RequestException as e:
        print(f"RESULT: FAIL (server unreachable: {e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
