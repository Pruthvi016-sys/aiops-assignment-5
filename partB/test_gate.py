"""partB/test_gate.py -- unit tests for the pure decide() function (B2).   Run:  pytest -q partB
Synthetic numbers only; no MLflow server or registry is needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eta_mlflow import decide  # noqa: E402

PROD = {"mae": 10.0, "mae_rush": 20.0, "mae_rain": 12.0, "p95_ms": 3.0}


def cand(**kw):
    c = {"mae": 9.0, "mae_rush": 19.0, "mae_rain": 11.0, "p95_ms": 4.0}   # clearly better everywhere
    c.update(kw)
    return c


def test_clear_improvement_promotes():
    r = decide(PROD, cand())
    assert r["promote"] is True
    assert set(r["criteria"]) == {"mae_gain", "latency_p95", "slice_rush_no_regression",
                                  "slice_rain_no_regression"}
    assert all(r["criteria"].values())


def test_result_shape_and_purity():
    p, c = dict(PROD), cand()
    r = decide(p, c)
    assert set(r) >= {"criteria", "promote"} and isinstance(r["promote"], bool)
    assert p == PROD and c == cand()                 # inputs not mutated
    assert decide(p, c) == r                         # deterministic


# --------------------------------------------------------------- overall MAE gain (default 3 %)
def test_gain_exactly_on_threshold_promotes():
    assert decide(PROD, cand(mae=9.7))["promote"] is True             # gain = 3.0 %


def test_gain_just_inside_threshold_promotes():
    assert decide(PROD, cand(mae=9.69))["criteria"]["mae_gain"] is True   # 3.1 %


def test_gain_just_outside_threshold_blocks():
    r = decide(PROD, cand(mae=9.71))                                  # 2.9 %
    assert r["criteria"]["mae_gain"] is False and r["promote"] is False


def test_worse_candidate_blocks():
    assert decide(PROD, cand(mae=10.5))["promote"] is False


def test_custom_min_rel_gain():
    assert decide(PROD, cand(mae=5.0), min_rel_gain=0.60)["promote"] is False   # 50 % < 60 %
    assert decide(PROD, cand(mae=3.9), min_rel_gain=0.60)["promote"] is True    # 61 %


# --------------------------------------------------------------- latency (default 10 ms)
@pytest.mark.parametrize("p95,ok", [(9.99, True), (10.0, True), (10.01, False)])
def test_latency_boundary(p95, ok):
    r = decide(PROD, cand(p95_ms=p95))
    assert r["criteria"]["latency_p95"] is ok and r["promote"] is ok


def test_latency_uses_candidate_not_prod():
    slow_prod = dict(PROD, p95_ms=50.0)
    assert decide(slow_prod, cand(p95_ms=5.0))["promote"] is True


# --------------------------------------------------------------- slices (default 5 % relative)
@pytest.mark.parametrize("rush,ok", [(20.99, True), (21.0, True), (21.01, False)])
def test_rush_slice_boundary(rush, ok):
    r = decide(PROD, cand(mae_rush=rush))           # limit = 20 * 1.05 = 21.0
    assert r["criteria"]["slice_rush_no_regression"] is ok and r["promote"] is ok


@pytest.mark.parametrize("rain,ok", [(12.59, True), (12.6, True), (12.61, False)])
def test_rain_slice_boundary(rain, ok):
    r = decide(PROD, cand(mae_rain=rain))           # limit = 12 * 1.05 = 12.6
    assert r["criteria"]["slice_rain_no_regression"] is ok and r["promote"] is ok


def test_better_overall_but_slice_regresses_blocks():
    # the classic trap: big overall gain hides a regression on rainy deliveries
    r = decide(PROD, cand(mae=6.0, mae_rain=14.0))
    assert r["criteria"]["mae_gain"] is True
    assert r["criteria"]["slice_rain_no_regression"] is False
    assert r["promote"] is False


def test_zero_slice_tolerance():
    assert decide(PROD, cand(mae_rush=20.0), max_slice_regression=0.0)["promote"] is True
    assert decide(PROD, cand(mae_rush=20.01), max_slice_regression=0.0)["promote"] is False


def test_all_criteria_fail():
    r = decide(PROD, {"mae": 11, "mae_rush": 30, "mae_rain": 20, "p95_ms": 99})
    assert r["promote"] is False and not any(r["criteria"].values())


def test_missing_key_raises():
    with pytest.raises(ValueError):
        decide(PROD, {"mae": 9.0, "mae_rush": 19.0, "p95_ms": 4.0})
