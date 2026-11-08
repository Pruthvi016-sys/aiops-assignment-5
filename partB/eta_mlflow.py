"""partB/eta_mlflow.py -- MLflow registry, multi-criteria promotion gate (Module 4, Part B).

    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db      # default if unset
    python partB/eta_mlflow.py                          # register + gate (default and min_rel_gain=0.60)

Public functions (interface contract):
    register()                                  -> logs v1/v2 as pyfuncs, registers "eta-model", sets aliases
    decide(prod, cand, **thresholds)            -> pure function, {"criteria": {...}, "promote": bool}
    promotion_gate(**thresholds)                -> evaluates @prod and @candidate on fresh data, calls decide()
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_NAME = "eta-model"
EXPERIMENT = "swiftparcel-eta"
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
DEFAULT_URI = "sqlite:///mlflow.db"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_registry_uri(tracking_uri())
    return mlflow


def _bundle_path(version: str) -> Path:
    p = Path("data") / f"eta_{version}.joblib"
    return p if p.exists() else ROOT / p


def _pinned_requirements():
    import cloudpickle, joblib, mlflow, numpy, pandas, sklearn
    return [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pandas.__version__}", f"numpy=={numpy.__version__}",
            f"joblib=={joblib.__version__}", f"cloudpickle=={cloudpickle.__version__}"]


# ============================================================================ B1 register
def register():
    """Log v1 (Ridge) and v2 (GBR) as custom pyfuncs with signature + input example + pinned
    requirements, register both under eta-model, alias prod -> v1, candidate -> v2.
    Returns the registry table as a list of dicts."""
    import joblib
    import pandas as pd
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.metrics import mean_absolute_error
    import eta_data
    from eta_pyfunc import EtaPyfunc

    mlflow = _mlflow()
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    fresh = eta_data.make_data(2000, seed=99)
    example = fresh[eta_data.FEATURES].head(3).reset_index(drop=True)

    registered = {}
    for v in ("v1", "v2"):
        path = _bundle_path(v)
        bundle = joblib.load(path)
        wrapper = EtaPyfunc()
        wrapper.model, wrapper.features, wrapper.version = bundle["model"], bundle["feature_columns"], v
        signature = infer_signature(example, wrapper.predict(None, example))
        mae = mean_absolute_error(fresh.eta_minutes, bundle["model"].predict(fresh[eta_data.FEATURES]))
        with mlflow.start_run(run_name=f"eta-{v}"):
            mlflow.log_params({"bundle_version": v, "estimator": type(bundle["model"]).__name__})
            mlflow.log_metric("fresh_mae", round(float(mae), 4))
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=EtaPyfunc(),
                artifacts={"bundle": str(path)},
                code_paths=[str(HERE / "eta_pyfunc.py")],
                signature=signature,
                input_example=example,
                pip_requirements=_pinned_requirements(),
                registered_model_name=MODEL_NAME,
            )
        mv = info.registered_model_version
        client.set_model_version_tag(MODEL_NAME, str(mv), "bundle_version", v)
        registered[v] = str(mv)

    client.set_registered_model_alias(MODEL_NAME, "prod", registered["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", registered["v2"])
    return registry_table()


def registry_table():
    from mlflow import MlflowClient
    _mlflow()
    client = MlflowClient()
    rows = []
    for found in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        mv = client.get_model_version(MODEL_NAME, found.version)    # search results omit aliases
        rows.append({"version": int(mv.version), "aliases": ",".join(mv.aliases) or "-",
                     "bundle": mv.tags.get("bundle_version", "?"), "run_id": mv.run_id})
    return rows


# ============================================================================ B2 gate
EPS = 1e-9   # tolerance so that a value exactly on a threshold counts as "inside"


def decide(prod: dict, cand: dict, min_rel_gain: float = 0.03, max_p95_ms: float = 10.0,
           max_slice_regression: float = 0.05) -> dict:
    """Pure multi-criteria promotion decision.

    prod / cand: dicts with keys mae, mae_rush, mae_rain, p95_ms.
    * mae_gain:      (prod.mae - cand.mae) / prod.mae >= min_rel_gain
    * latency:       cand.p95_ms <= max_p95_ms
    * rush / rain:   cand.mae_slice <= prod.mae_slice * (1 + max_slice_regression)
                     (max_slice_regression is a *relative* tolerance, 0.05 = 5 %)
    Returns {"criteria": {name: bool}, "promote": all(criteria), "details": {...numbers...}}.
    """
    for d, who in ((prod, "prod"), (cand, "cand")):
        missing = {"mae", "mae_rush", "mae_rain", "p95_ms"} - set(d)
        if missing:
            raise ValueError(f"{who} metrics missing {sorted(missing)}")

    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"] if prod["mae"] > 0 else float("-inf")
    rush_lim = prod["mae_rush"] * (1 + max_slice_regression)
    rain_lim = prod["mae_rain"] * (1 + max_slice_regression)
    criteria = {
        "mae_gain": rel_gain >= min_rel_gain - EPS,
        "latency_p95": cand["p95_ms"] <= max_p95_ms + EPS,
        "slice_rush_no_regression": cand["mae_rush"] <= rush_lim + EPS,
        "slice_rain_no_regression": cand["mae_rain"] <= rain_lim + EPS,
    }
    details = {
        "rel_mae_gain": round(rel_gain, 4), "min_rel_gain": min_rel_gain,
        "cand_p95_ms": cand["p95_ms"], "max_p95_ms": max_p95_ms,
        "cand_mae_rush": cand["mae_rush"], "max_allowed_mae_rush": round(rush_lim, 4),
        "cand_mae_rain": cand["mae_rain"], "max_allowed_mae_rain": round(rain_lim, 4),
    }
    return {"criteria": criteria, "promote": all(criteria.values()), "details": details}


def _evaluate(pyfunc_model, fresh, features, n_latency=200):
    import numpy as np
    out = pyfunc_model.predict(fresh[features])
    pred = out["eta_minutes"].to_numpy()
    err = np.abs(pred - fresh["eta_minutes"].to_numpy())
    rush = fresh["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (fresh["rain"] == 1).to_numpy()
    rows = [fresh[features].iloc[[i]] for i in range(n_latency)]
    for r in rows[:20]:                                   # warm-up, not timed
        pyfunc_model.predict(r)
    lat = []
    for r in rows:
        t0 = time.perf_counter()
        pyfunc_model.predict(r)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"mae": round(float(err.mean()), 4), "mae_rush": round(float(err[rush].mean()), 4),
            "mae_rain": round(float(err[rain].mean()), 4),
            "p95_ms": round(float(np.percentile(lat, 95)), 4),
            "model_version": str(out["model_version"].iloc[0])}


def promotion_gate(**thresholds) -> dict:
    """Load @prod and @candidate from the registry, evaluate both on fresh data
    (eta_data.make_data(2000, seed=99)), measure single-row p95 over 200 predictions, call decide()."""
    import eta_data
    mlflow = _mlflow()
    fresh = eta_data.make_data(2000, seed=99)
    prod = _evaluate(mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@prod"), fresh, eta_data.FEATURES)
    cand = _evaluate(mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate"), fresh, eta_data.FEATURES)
    metric_keys = ("mae", "mae_rush", "mae_rain", "p95_ms")
    result = decide({k: prod[k] for k in metric_keys}, {k: cand[k] for k in metric_keys}, **thresholds)
    result["prod_metrics"], result["cand_metrics"] = prod, cand
    result["thresholds"] = thresholds or "defaults (min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05)"
    return result


def main():
    import pandas as pd
    lines = [f"tracking URI: {tracking_uri()}", "", "== B1 registry table (eta-model) =="]
    table = register()
    lines.append(pd.DataFrame(table).to_string(index=False))
    machine = {"registry": table}
    for label, th in (("default thresholds", {}), ("min_rel_gain=0.60", {"min_rel_gain": 0.60})):
        res = promotion_gate(**th)
        machine["default" if not th else "min_rel_gain_0.60"] = res
        lines += ["", f"== B2 promotion_gate({', '.join(f'{k}={v}' for k, v in th.items())})  [{label}] ==",
                  json.dumps(res, indent=2)]
    text = "\n".join(lines)
    print(text)
    (HERE / "gate_output.txt").write_text(text + "\n")
    (HERE / "gate_output.json").write_text(json.dumps(machine, indent=2))


if __name__ == "__main__":
    main()
