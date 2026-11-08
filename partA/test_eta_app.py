"""partA/test_eta_app.py -- pytest suite for eta_app (A2).   Run:  pytest -q partA"""
import os
import sys
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("ETA_MODEL", str(ROOT / "data" / "eta_v1.joblib"))

from fastapi.testclient import TestClient  # noqa: E402
import eta_app  # noqa: E402

URL = "/v1/models/eta:predict"
GOOD = {"distance_km": 12.5, "n_stops": 4, "hour": 8, "weekday": 2,
        "traffic_index": 0.6, "rain": 1, "weight_kg": 5.0}


@pytest.fixture
def client():
    # raise_server_exceptions=False so a 500 is returned as a response, like a real server.
    return TestClient(eta_app.app, raise_server_exceptions=False)


def row(**kw):
    r = dict(GOOD)
    r.update(kw)
    return r


# ---------------------------------------------------------------- happy path
def test_valid_request(client):
    r = client.post(URL, json={"instances": [GOOD]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predictions", "model_version", "request_id"}
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert isinstance(p, float) and p == round(p, 1)            # one decimal
    expected = round(float(eta_app.model.predict(
        __import__("pandas").DataFrame([GOOD], columns=eta_app.FEATURES))[0]), 1)
    assert p == expected


def test_batch_order_preserved(client):
    rows = [row(distance_km=d) for d in (1.0, 80.0, 5.0, 140.0, 30.0)]
    batch = client.post(URL, json={"instances": rows}).json()["predictions"]
    singles = [client.post(URL, json={"instances": [r]}).json()["predictions"][0] for r in rows]
    assert batch == singles
    assert batch != sorted(batch)          # the inputs were deliberately not monotone


# ---------------------------------------------------------------- invalid inputs (3 kinds + format)
def _assert_validation(r, field_part):
    assert r.status_code == 422
    b = r.json()
    assert b["error"] == "validation_failed"
    assert isinstance(b["details"], list) and b["details"]
    assert all({"field", "msg"} <= set(d) for d in b["details"])
    assert any(field_part in d["field"] for d in b["details"]), b["details"]
    assert b["request_id"] == r.headers["X-Request-ID"]


def test_invalid_out_of_range(client):
    _assert_validation(client.post(URL, json={"instances": [row(hour=24)]}), "hour")


def test_invalid_missing_field(client):
    bad = dict(GOOD)
    del bad["traffic_index"]
    _assert_validation(client.post(URL, json={"instances": [bad]}), "traffic_index")


def test_invalid_wrong_type(client):
    _assert_validation(client.post(URL, json={"instances": [row(n_stops="many")]}), "n_stops")


@pytest.mark.parametrize("field,value", [("weight_kg", 0), ("weight_kg", 1000.5), ("rain", 2),
                                         ("distance_km", 0.05), ("traffic_index", 1.01),
                                         ("weekday", -1), ("n_stops", 61)])
def test_invalid_range_edges(client, field, value):
    _assert_validation(client.post(URL, json={"instances": [row(**{field: value})]}), field)


def test_valid_range_edges(client):
    edge = {"distance_km": 0.1, "n_stops": 60, "hour": 23, "weekday": 0,
            "traffic_index": 1.0, "rain": 0, "weight_kg": 1000}
    assert client.post(URL, json={"instances": [edge]}).status_code == 200


# ---------------------------------------------------------------- cross-field rule
def test_cross_field_infeasible(client):
    r = client.post(URL, json={"instances": [row(distance_km=151, n_stops=21)]})
    assert r.status_code == 422
    assert "infeasible" in r.text
    assert r.json()["error"] == "validation_failed"


def test_cross_field_boundary_is_feasible(client):
    # exactly 150 km / 20 stops is allowed (rule uses strict >)
    assert client.post(URL, json={"instances": [row(distance_km=150, n_stops=60)]}).status_code == 200
    assert client.post(URL, json={"instances": [row(distance_km=500, n_stops=20)]}).status_code == 200


# ---------------------------------------------------------------- batch-size limits
def test_batch_empty_rejected(client):
    _assert_validation(client.post(URL, json={"instances": []}), "instances")


def test_batch_501_rejected_500_accepted(client):
    _assert_validation(client.post(URL, json={"instances": [GOOD] * 501}), "instances")
    r = client.post(URL, json={"instances": [GOOD] * 500})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 500


# ---------------------------------------------------------------- request id
def test_request_id_echoed(client):
    r = client.post(URL, json={"instances": [GOOD]}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated(client):
    r = client.post(URL, json={"instances": [GOOD]})
    rid = r.headers["X-Request-ID"]
    uuid.UUID(rid)                                   # valid UUID
    assert r.json()["request_id"] == rid
    assert client.post(URL, json={"instances": [GOOD]}).headers["X-Request-ID"] != rid


def test_request_id_on_error(client):
    r = client.post(URL, json={"instances": [row(hour=99)]}, headers={"X-Request-ID": "err-7"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-7" and r.json()["request_id"] == "err-7"
    r2 = client.post(URL, json={"instances": []})
    uuid.UUID(r2.headers["X-Request-ID"])
    assert r2.json()["request_id"] == r2.headers["X-Request-ID"]


# ---------------------------------------------------------------- health + metrics
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1"}
    assert "X-Request-ID" in r.headers


def _counter(text, path, status):
    key = f'eta_requests_total{{path="{path}",status="{status}"}} '
    for line in text.splitlines():
        if line.startswith(key):
            return int(float(line.split()[-1]))
    return 0


def test_metrics_output(client):
    before = client.get("/metrics").text
    client.post(URL, json={"instances": [GOOD]})
    client.post(URL, json={"instances": [row(hour=-1)]})
    text = client.get("/metrics").text
    after = client.get("/metrics").text
    assert _counter(text, URL, 200) == _counter(before, URL, 200) + 1
    assert _counter(text, URL, 422) == _counter(before, URL, 422) + 1
    # scraping /metrics must not change any counter or the histogram count
    assert "/metrics" not in "".join(l for l in after.splitlines() if l.startswith("eta_requests_total"))
    def counts(t):
        return [l for l in t.splitlines() if l.startswith(("eta_requests_total", "eta_request_latency_seconds_count"))]
    assert counts(after) == counts(text)
    assert "# TYPE eta_request_latency_seconds histogram" in text
    buckets = [l for l in text.splitlines() if l.startswith("eta_request_latency_seconds_bucket{le=")]
    assert len(buckets) >= 5 and 'le="+Inf"' in buckets[-1]
    counts = [int(l.split()[-1]) for l in buckets]
    assert counts == sorted(counts)                  # cumulative
    count = int([l for l in text.splitlines() if l.startswith("eta_request_latency_seconds_count")][0].split()[-1])
    assert counts[-1] == count
    assert any(l.startswith("eta_request_latency_seconds_sum") for l in text.splitlines())


# ---------------------------------------------------------------- generic 500
def test_generic_500_on_model_failure(client, monkeypatch):
    class Boom:
        def predict(self, X):
            raise RuntimeError("SECRET-internal-detail /opt/model.bin")

    monkeypatch.setattr(eta_app, "model", Boom())
    r = client.post(URL, json={"instances": [GOOD]}, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert "SECRET" not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text
    assert r.json() == {"error": "internal_error", "request_id": "boom-1"}
    assert r.headers["X-Request-ID"] == "boom-1"
