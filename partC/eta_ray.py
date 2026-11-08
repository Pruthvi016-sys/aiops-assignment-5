"""partC/eta_ray.py -- Ray Serve application with a *sticky* canary (Module 4, Part C).

    ray start --head --num-cpus 2            # or let verify_ray.py start a local 2-CPU Ray
    serve run partC.eta_ray:app              # default weight 0.1 (run from repo root)
    curl -X POST localhost:8000/v1/models/eta:predict -H 'content-type: application/json' \
         -d '{"customer_id":"c-42","instances":[{"distance_km":12,"n_stops":3,"hour":8,"weekday":1,"traffic_index":0.5,"rain":0,"weight_kg":4}]}'

Architecture
    Router (FastAPI ingress, 1 replica, 0.2 CPU)
       |-- EtaProd      = EtaModel(v1 bundle), 2 fixed replicas, 0.25 CPU each
       '-- EtaCandidate = EtaModel(v2 bundle), autoscaling 1..3 replicas, 0.25 CPU each
    Worst-case CPU reservation: 0.2 + 2*0.25 + 3*0.25 = 1.45 CPU  (fits a 2-CPU Ray with headroom
    for the Serve controller / proxy, which reserve 0 CPU).

Sticky canary
    bucket(id) = first 8 bytes of SHA-256(SALT + ":" + id) / 2**64  in [0, 1)
    arm = "v2" if bucket(id) < weight else "v1"
    -> deterministic across processes, machines and restarts; customers (not requests) are split;
       raising the weight only ADDS customers to v2 (the set {b < 0.2} is a subset of {b < 0.5}).
    X-Beta: 1 forces v2 for that single request.
"""
import hashlib
import os
import time
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray import serve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SALT = "swiftparcel-eta-canary-v2"          # change the salt => new, independent experiment

PROD_CPU = 0.25
CAND_CPU = 0.25
ROUTER_CPU = 0.2
CANDIDATE_AUTOSCALING = {
    "min_replicas": 1,
    "initial_replicas": 1,
    "max_replicas": 3,
    "target_ongoing_requests": 2,
    "upscale_delay_s": 5,
    "downscale_delay_s": 60,
}


# ------------------------------------------------------------------ bucketing (pure, testable)
def bucket(customer_id: str, salt: str = SALT) -> float:
    """Stable value in [0, 1) for a customer id. Uses SHA-256, NOT Python's salted hash()."""
    h = hashlib.sha256(f"{salt}:{customer_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def choose_arm(customer_id: str, weight: float, beta: bool = False) -> str:
    if beta:
        return "v2"
    return "v2" if bucket(customer_id) < weight else "v1"


# ------------------------------------------------------------------ request schema (same ranges as Part A)
class Delivery(BaseModel):
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
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class CanaryRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=200)
    instances: List[Delivery] = Field(..., min_length=1, max_length=500)


# ------------------------------------------------------------------ deployments
@serve.deployment(ray_actor_options={"num_cpus": PROD_CPU}, max_ongoing_requests=16)
class EtaModel:
    """Loads one bundle ONCE per replica (in __init__) and predicts a batch of rows."""

    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        b = joblib.load(bundle_path)
        self.model, self.features, self.version = b["model"], list(b["feature_columns"]), str(b["version"])
        self.extra_work_ms = float(extra_work_ms)
        self.replica = f"{self.version}-pid{os.getpid()}"

    def predict(self, rows: List[dict]) -> dict:
        if self.extra_work_ms > 0:            # simulated extra CPU work (for autoscaling experiments)
            t_end = time.perf_counter() + self.extra_work_ms / 1000
            while time.perf_counter() < t_end:
                pass
        y = self.model.predict(pd.DataFrame(rows, columns=self.features))
        return {"predictions": [round(float(v), 1) for v in y], "model_version": self.version,
                "served_by": self.replica}


api = FastAPI(title="SwiftParcel ETA - sticky canary")


@serve.deployment(ray_actor_options={"num_cpus": ROUTER_CPU}, num_replicas=1, max_ongoing_requests=64)
@serve.ingress(api)
class Router:
    def __init__(self, prod, candidate, weight: float = 0.1):
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        self.arms = {"v1": prod, "v2": candidate}
        self.weight = float(weight)

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: CanaryRequest, x_beta: Optional[str] = Header(default=None)):
        arm = choose_arm(req.customer_id, self.weight, beta=(x_beta or "").strip() == "1")
        out = await self.arms[arm].predict.remote([d.model_dump() for d in req.instances])
        return {**out, "arm": arm, "customer_id": req.customer_id}

    @api.get("/healthz")
    async def healthz(self):
        return {"status": "ok", "model_version": "v1", "candidate_version": "v2", "weight": self.weight}

    @api.get("/canary")
    async def canary(self):
        return {"weight": self.weight, "salt": SALT, "candidate_autoscaling": CANDIDATE_AUTOSCALING}


def _bundle(v: str) -> str:
    p = Path("data") / f"eta_{v}.joblib"
    return str((p if p.exists() else ROOT / p).resolve())


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Return the bound Serve application (pass it to serve.run)."""
    prod = EtaModel.options(name="EtaProd", num_replicas=2,
                            ray_actor_options={"num_cpus": PROD_CPU}).bind(_bundle("v1"), extra_work_ms)
    cand = EtaModel.options(name="EtaCandidate",
                            autoscaling_config=CANDIDATE_AUTOSCALING,
                            ray_actor_options={"num_cpus": CAND_CPU}).bind(_bundle("v2"), extra_work_ms)
    return Router.bind(prod, cand, weight)


# `serve run partC.eta_ray:app`
app = build()
