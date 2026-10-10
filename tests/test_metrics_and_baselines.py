"""Metric definitions, baselines, origin schedule, and committed-result consistency."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandforecast.baselines import moving_average_forecast, seasonal_naive_forecast
from demandforecast.download import combine_sheets, normalize_stock_code
from demandforecast.metrics import regression_metrics, score_predictions
from demandforecast.paths import project_root
from demandforecast.schedule import origin_dates


def test_seasonal_naive_weekly_cycle():
    history = np.array([10, 11, 12, 13, 14, 15, 16], dtype=float)
    forecast = seasonal_naive_forecast(history, horizon=8, season=7)
    assert forecast.tolist() == [10, 11, 12, 13, 14, 15, 16, 10]


def test_seasonal_naive_short_season():
    history = np.arange(6, dtype=float)
    forecast = seasonal_naive_forecast(history, horizon=4, season=3)
    assert forecast.tolist() == [3, 4, 5, 3]


def test_moving_average_is_flat():
    forecast = moving_average_forecast(np.array([1, 2, 3, 4], dtype=float), horizon=2, window=3)
    assert forecast.tolist() == [3, 3]


def test_metrics_known_values_and_zero_zero_smape():
    metrics = regression_metrics([0, 2], [0, 4])
    assert metrics["n_points"] == 2
    assert metrics["mae"] == pytest.approx(1)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2))
    assert metrics["wape"] == pytest.approx(1)
    assert metrics["smape"] == pytest.approx((0 + 2 * 2 / 6) / 2)
    zeros = regression_metrics([0], [0])
    assert zeros["smape"] == 0
    assert np.isnan(zeros["wape"])


def test_origin_grid_keeps_a_full_horizon_and_warmup():
    min_date = pd.Timestamp("2020-01-01")
    max_date = pd.Timestamp("2020-06-01")
    origins = origin_dates(
        min_date,
        max_date,
        horizon=3,
        step=3,
        yearly_lag_days=10,
        warmup_days=4,
    )
    first_lag = min_date + pd.Timedelta(days=10)
    assert origins[0] == first_lag + pd.Timedelta(days=4 + 3)
    assert all(origin + pd.Timedelta(days=3) <= max_date for origin in origins)
    assert origins[-1] + pd.Timedelta(days=3) > max_date - pd.Timedelta(days=3)
    gaps = [origins[i + 1] - origins[i] for i in range(len(origins) - 1)]
    assert all(gap == pd.Timedelta(days=3) for gap in gaps)


def test_normalize_stock_code_and_sheet_overlap():
    assert normalize_stock_code(pd.Series(["85099b", " 84077.0 ", "post"])).tolist() == [
        "85099B",
        "84077",
        "POST",
    ]
    early = pd.DataFrame(
        {
            "invoice_date": pd.to_datetime(["2010-11-30", "2010-12-01"]),
            "invoice": ["a", "overlap"],
        }
    )
    late = pd.DataFrame(
        {
            "invoice_date": pd.to_datetime(["2010-12-01", "2010-12-02"]),
            "invoice": ["from-late", "c"],
        }
    )
    combined = combine_sheets(early, late)
    assert combined["invoice"].tolist() == ["a", "from-late", "c"]


def test_lightgbm_fits_a_tiny_problem():
    lightgbm = pytest.importorskip("lightgbm")
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = x.sum(axis=1)
    dataset = lightgbm.Dataset(x, label=y)
    booster = lightgbm.train(
        {"objective": "regression_l1", "verbosity": -1, "num_threads": 1},
        dataset,
        num_boost_round=5,
    )
    pred = booster.predict(x)
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


def test_committed_metrics_match_predictions():
    root = project_root()
    predictions_path = root / "reports" / "predictions.csv"
    metrics_path = root / "reports" / "metrics.csv"
    if not predictions_path.is_file() or not metrics_path.is_file():
        pytest.skip("Backtest reports are not in the tree yet")
    preds = pd.read_csv(predictions_path, parse_dates=["target_date"])
    expected = pd.read_csv(metrics_path, dtype={"fold": str, "origin_date": str, "model": str})
    expected["origin_date"] = expected["origin_date"].fillna("")
    got = score_predictions(preds)
    merged = expected.merge(got, on=["fold", "origin_date", "model"], suffixes=("_file", "_recomputed"))
    assert len(merged) == len(expected)
    assert len(merged) == len(got)
    assert (merged["n_points_file"].astype(int) == merged["n_points_recomputed"].astype(int)).all()
    for column in ("mae", "rmse", "wape", "smape"):
        np.testing.assert_allclose(
            merged[f"{column}_file"].to_numpy(dtype=float),
            merged[f"{column}_recomputed"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-8,
        )
    assert Path(root / "reports" / "headline.txt").is_file()
