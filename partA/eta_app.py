"""partA/eta_app.py -- SwiftParcel ETA REST service (Module 4, Part A).

Run from the repository root:
    ETA_MODEL=data/eta_v1.joblib uvicorn partA.eta_app:app --port 8000
    # or:  cd partA && uvicorn eta_app:app --port 8000

Endpoints
    POST /v1/models/eta:predict   {"instances": [{7 features}, ...]} -> predictions (minutes, 1 decimal)
    GET  /healthz                 {"status": "ok", "model_version": ...}
    GET  /metrics                 Prometheus text format (counter + latency histogram)

Design notes
  * The model bundle is loaded ONCE at import time into the module-level variable `model`
    (the fitted scikit-learn estimator). The handler looks `model` up at call time, so tests can
    monkeypatch it.
  * Every response -- success, validation error, 404, 500 -- carries X-Request-ID; the same id is
    in the JSON body as `request_id` for the predict / error responses.
  * Metrics are implemented by hand (no extra dependency) and protected by a lock because sync
    endpoints run in a thread pool. Requests to /metrics are not counted.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- model loading
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "data/eta_v1.joblib"


def _resolve(path: str) -> Path:
    """Accept a path relative to the CWD; fall back to the repository root so the app also
    works when started from inside partA/."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    alt = REPO_ROOT / path
    return alt if alt.exists() else p


MODEL_PATH = _resolve(os.environ.get("ETA_MODEL", DEFAULT_MODEL))
_bundle = joblib.load(MODEL_PATH)                    # loaded once per process
model = _bundle["model"]                             # module-level variable required by the contract
FEATURES: List[str] = list(_bundle["feature_columns"])
MODEL_VERSION: str = str(_bundle.get("version", "unknown"))

# --------------------------------------------------------------------------- schemas
MAX_BATCH = 500


class Delivery(BaseModel):
    """One delivery. Ranges from the assignment spec (A1.2)."""
    model_config = ConfigDict(extra="ignore")

    distance_km: float = Field(..., ge=0.1, le=500)
    n_stops: int = Field(..., ge=1, le=60)
    hour: int = Field(..., ge=0, le=23)
    weekday: int = Field(..., ge=0, le=6)
    traffic_index: float = Field(..., ge=0, le=1)
    rain: int = Field(..., ge=0, le=1)
    weight_kg: float = Field(..., gt=0, le=1000)

    @model_validator(mode="after")
    def _feasible(self):
        # Cross-field rule (A1.3): long-haul multi-drop runs are not offered.
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(..., min_length=1, max_length=MAX_BATCH)


class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str
    request_id: str


# --------------------------------------------------------------------------- metrics
BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5)
KNOWN_PATHS = {"/v1/models/eta:predict", "/healthz"}


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: dict[tuple[str, str], int] = {}
        self.bucket_counts = [0] * len(BUCKETS)   # non-cumulative; cumulated at render time
        self.lat_sum = 0.0
        self.lat_count = 0

    def observe(self, path: str, status: int, seconds: float) -> None:
        with self.lock:
            key = (path, str(status))
            self.requests[key] = self.requests.get(key, 0) + 1
            for i, le in enumerate(BUCKETS):
                if seconds <= le:
                    self.bucket_counts[i] += 1
                    break
            self.lat_sum += seconds
            self.lat_count += 1

    def render(self) -> str:
        with self.lock:
            out = ["# HELP eta_requests_total Total HTTP requests by path and status.",
                   "# TYPE eta_requests_total counter"]
            for (path, status), n in sorted(self.requests.items()):
                out.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
            out += ["# HELP eta_request_latency_seconds Request latency in seconds.",
                    "# TYPE eta_request_latency_seconds histogram"]
            cum = 0
            for le, c in zip(BUCKETS, self.bucket_counts):
                cum += c
                out.append(f'eta_request_latency_seconds_bucket{{le="{le}"}} {cum}')
            out.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {self.lat_count}')
            out.append(f"eta_request_latency_seconds_sum {self.lat_sum:.6f}")
            out.append(f"eta_request_latency_seconds_count {self.lat_count}")
            return "\n".join(out) + "\n"


METRICS = Metrics()

# --------------------------------------------------------------------------- app
app = FastAPI(title="SwiftParcel ETA service", version="1.0")


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # last-resort safety net: never leak internals
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    path = request.url.path
    if path != "/metrics":
        METRICS.observe(path if path in KNOWN_PATHS else "other", response.status_code,
                        time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = []
    for e in exc.errors():
        loc = [str(x) for x in e.get("loc", ()) if x != "body"]
        details.append({"field": ".".join(loc) or "body", "msg": e.get("msg", "invalid")})
    rid = _rid(request)
    return JSONResponse({"error": "validation_failed", "details": details, "request_id": rid},
                        status_code=422, headers={"X-Request-ID": rid})


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    rid = _rid(request)
    return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500,
                        headers={"X-Request-ID": rid})


@app.post("/v1/models/eta:predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    rid = request.state.request_id
    X = pd.DataFrame([d.model_dump() for d in req.instances], columns=FEATURES)
    try:
        y = model.predict(X)                     # global lookup -> monkeypatch-able
        preds = [round(float(v), 1) for v in y]
    except Exception:
        # Generic 500: no stack trace, no exception text in the body.
        return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")
