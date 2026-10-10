"""Metric definitions, baselines, origin schedule, and committed-result consistency."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandforecast.analysis import write_analysis
from demandforecast.baselines import (
    croston_forecast,
    demand_pattern,
    moving_average_forecast,
    sba_forecast,
    seasonal_naive_forecast,
    tsb_forecast,
)
from demandforecast.boosting import fit_lightgbm, predict_lightgbm
from demandforecast.config import DAILY_MODELS, FEATURE_COLUMNS, INVENTORY_MODELS, Config
from demandforecast.decision import decision_metrics
from demandforecast.download import combine_sheets, normalize_stock_code
from demandforecast.inventory import inventory_evaluation
from demandforecast.metrics import regression_metrics, score_predictions
from demandforecast.paths import project_root
from demandforecast.reporting import headline_block
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
    assert metrics["bias"] == pytest.approx(1)
    assert metrics["smape"] == pytest.approx((0 + 2 * 2 / 6) / 2)
    zeros = regression_metrics([0], [0])
    assert zeros["smape"] == 0
    assert np.isnan(zeros["wape"])
    assert np.isnan(zeros["bias"])


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


def test_croston_sba_tsb_hand_values():
    history = np.array([0, 2, 0, 4], dtype=float)
    assert croston_forecast(history, 3, 0.1).tolist() == pytest.approx([1.1, 1.1, 1.1])
    assert sba_forecast(history, 1, 0.1)[0] == pytest.approx(1.045)
    assert tsb_forecast(history, 1, 0.1)[0] == pytest.approx(2.002)
    trailed = np.array([0, 2, 0, 4, 0], dtype=float)
    assert croston_forecast(trailed, 1, 0.1)[0] == pytest.approx(1.1)
    assert tsb_forecast(trailed, 1, 0.1)[0] == pytest.approx(0.819 * 2.2)


def test_demand_pattern_uses_syntetos_boylan_cutoffs():
    assert demand_pattern(np.ones(10), 1.32, 0.49)[0] == "smooth"
    name, adi, cv2 = demand_pattern(np.array([0, 5, 0, 5, 0, 5], dtype=float), 1.32, 0.49)
    assert name == "intermittent"
    assert adi == pytest.approx(5 / 3)
    assert cv2 == pytest.approx(0)
    name, adi, cv2 = demand_pattern(np.array([0, 1, 0, 10, 0, 1], dtype=float), 1.32, 0.49)
    assert name == "lumpy"
    assert adi == pytest.approx(5 / 3)
    assert cv2 == pytest.approx(27 / 16)
    assert demand_pattern(np.array([1, 10, 1, 10], dtype=float), 1.32, 0.49)[0] == "erratic"


def test_order_total_wape_is_not_the_daily_wape():
    preds = pd.DataFrame(
        {
            "fold": [1, 1, 1],
            "origin_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "sku": ["A", "A", "B"],
            "target_date": ["2020-01-02", "2020-01-03", "2020-01-02"],
            "actual": [10.0, 0.0, 4.0],
            "m": [1.0, 8.0, 4.0],
        }
    )
    scored = decision_metrics(preds, ["m"]).set_index(["grain", "model"])
    assert scored.loc[("daily", "m"), "wape"] == pytest.approx(17 / 14)
    assert scored.loc[("daily", "m"), "bias"] == pytest.approx(-1 / 14)
    assert scored.loc[("order_total", "m"), "wape"] == pytest.approx(1 / 14)
    assert scored.loc[("order_total", "m"), "n_points"] == 2
    assert scored.loc[("daily_aggregate", "m"), "wape"] == pytest.approx(17 / 14)


def test_inventory_cost_uses_origin_price_and_stated_ratio():
    preds = pd.DataFrame(
        {
            "fold": [1, 1, 1],
            "origin_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "sku": ["A", "A", "B"],
            "actual": [10.0, 0.0, 5.0],
            "lightgbm_tweedie": [4.0, 0.0, 8.0],
            "price_asof": [2.0, 2.0, np.nan],
            "demand_class": ["lumpy", "lumpy", "erratic"],
        }
    )
    detail, summary = inventory_evaluation(preds, ["lightgbm_tweedie"], stockout_to_holding=3.0)
    row_a = detail.loc[detail["sku"] == "A"].iloc[0]
    assert row_a["units_short"] == pytest.approx(6)
    assert row_a["units_over"] == pytest.approx(0)
    assert row_a["cost"] == pytest.approx(36)
    row_b = detail.loc[detail["sku"] == "B"].iloc[0]
    assert row_b["units_over"] == pytest.approx(3)
    assert bool(row_b["price_missing"])
    assert np.isnan(row_b["cost"])
    totals = summary.iloc[0]
    assert totals["n_missing_price"] == 1
    assert totals["units_short"] == pytest.approx(6)
    assert totals["units_over"] == pytest.approx(3)
    assert totals["cost"] == pytest.approx(36)
    assert totals["stockout_to_holding"] == pytest.approx(3)


def test_headline_names_the_order_quantity_winner():
    rows = []
    wapes = {
        "moving_average_28": 0.40,
        "lightgbm_tweedie": 0.50,
        "lightgbm_l1": 0.60,
        "seasonal_naive_w7": 0.70,
        "seasonal_naive_y364": 0.80,
    }
    for grain in ("daily", "order_total", "daily_aggregate"):
        for name, wape in wapes.items():
            rows.append(
                {
                    "grain": grain,
                    "model": name,
                    "n_points": 10,
                    "mae": wape,
                    "rmse": wape,
                    "wape": wape,
                    "smape": 0.2,
                    "bias": -0.05 if name == "moving_average_28" else -0.20,
                }
            )
    decision = pd.DataFrame(rows)
    metric_rows = []
    for name, wape in wapes.items():
        metric_rows.append(
            {
                "fold": "pooled",
                "origin_date": "",
                "model": name,
                "n_points": 10,
                "mae": wape,
                "rmse": wape,
                "wape": wape,
                "smape": 0.2,
                "bias": -0.05,
            }
        )
    paragraphs, summary = headline_block(pd.DataFrame(metric_rows), decision, n_folds=10, horizon=28)
    assert paragraphs[0].startswith(
        "At the 28-day order quantity (10 SKU-origins across 10 origins), "
        "the 28-day moving average has the lowest WAPE, 0.400000, with bias -0.050000."
    )
    assert "LightGBM Tweedie order-quantity WAPE was 0.500000" in paragraphs[1]
    assert summary["order_winner"] == "moving_average_28"


def test_analysis_quotes_the_order_wape_and_the_cost_ratio():
    rows = []
    for grain in ("daily", "order_total", "daily_aggregate"):
        for i, name in enumerate(DAILY_MODELS):
            rows.append(
                {
                    "grain": grain,
                    "model": name,
                    "n_points": 4,
                    "mae": 1.0,
                    "rmse": 1.0,
                    "wape": 0.2 + i * 0.05,
                    "smape": 0.3,
                    "bias": -0.01 * i,
                }
            )
    decision = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        [
            {
                "fold": "pooled",
                "origin_date": "",
                "model": name,
                "n_points": 4,
                "mae": 1.0,
                "rmse": 1.0,
                "wape": 0.2,
                "smape": 0.3,
                "bias": 0.0,
            }
            for name in DAILY_MODELS
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "segment_type": "demand_class",
                "segment": "lumpy",
                "grain": "order_total",
                "model": "lightgbm_tweedie",
                "actual_units": 10.0,
                "n_points": 1,
                "mae": 1.0,
                "rmse": 1.0,
                "wape": 0.2,
                "smape": 0.3,
                "bias": 0.0,
            }
        ]
    )
    inventory_summary = pd.DataFrame(
        [
            {
                "model": name,
                "stockout_to_holding": 3.0,
                "holding_cost_basis": "origin unit price",
                "n_sku_origins": 1,
                "n_missing_price": 0,
                "actual_units": 10.0,
                "forecast_units": 8.0,
                "units_short": 2.0,
                "units_over": 0.0,
                "value_short": 4.0,
                "value_over": 0.0,
                "cost": 12.0,
            }
            for name in INVENTORY_MODELS
        ]
    )
    inventory_detail = pd.DataFrame(
        [
            {
                "origin_date": "2020-01-01",
                "sku": "A",
                "demand_class": "lumpy",
                "model": "lightgbm_tweedie",
                "units_short": 2.0,
                "units_over": 0.0,
                "cost": 12.0,
            }
        ]
    )
    weekly = decision.loc[decision["model"].isin(["moving_average_28", "lightgbm_tweedie"])].copy()
    weekly["model"] = weekly["model"].map(
        {"moving_average_28": "moving_average_4", "lightgbm_tweedie": "lightgbm_tweedie"}
    )
    # The weekly table printer accepts whatever rows it is given.
    stats = {
        "selection_cutoff": "2011-02-22",
        "n_skus_selected": 1,
        "n_skus_eligible_before_cutoff": 2,
        "min_sale_days": 180,
        "selected_unit_share_before_cutoff": 0.5,
        "selected_unit_share_full_period": 0.25,
        "date_min": "2009-12-01",
        "date_max": "2011-12-09",
        "n_weekly_unit_mismatches": 0,
        "quality": {
            "n_matched_sales": 1,
            "n_sales_lines": 2,
            "n_clean_lines": 1,
            "clean_quantity": 5,
            "daily_units": 5,
            "n_panel": 10,
            "n_panel_keys": 10,
            "n_bad_asof": 0,
        },
        "skus": [
            {
                "sku": "A",
                "description": "MUG",
                "units_before_cutoff": 10,
                "sale_days_before_cutoff": 180,
            }
        ],
    }
    text = write_analysis(
        Path("/tmp/analysis-unit.md"),
        decision=decision,
        metrics=metrics,
        segments=segments,
        lumpy=None,
        inventory_summary=inventory_summary,
        inventory_detail=inventory_detail,
        weekly_decision=weekly,
        build_stats=stats,
    )
    assert "0.200000" in text
    assert "Stockout-to-holding ratio 3:1" in text
    assert "2011-02-22" in text


def test_fit_lightgbm_l1_and_tweedie():
    pytest.importorskip("lightgbm")
    train = _boost_frame(30)
    val = _boost_frame(12)
    cfg = Config(
        lgb_num_boost_round=8,
        lgb_early_stopping_rounds=2,
        lgb_min_data_in_leaf=1,
        lgb_num_leaves=4,
    )
    for objective in ("regression_l1", "tweedie"):
        booster, n_iter = fit_lightgbm(train, val, cfg, objective)
        pred = predict_lightgbm(booster, val, n_iter)
        assert pred.shape == (12,)
        assert np.isfinite(pred).all()
        assert (pred >= 0).all()
        assert n_iter >= 1


def _boost_frame(n: int) -> pd.DataFrame:
    data = {column: np.linspace(1.0, 2.0, n) for column in FEATURE_COLUMNS if column != "sku"}
    data["sku"] = pd.Categorical(["A"] * n)
    data["target_units"] = np.linspace(1.0, 3.0, n)
    return pd.DataFrame(data)


def test_committed_metrics_match_predictions():
    root = project_root()
    predictions_path = root / "reports" / "predictions.csv"
    metrics_path = root / "reports" / "metrics.csv"
    if not predictions_path.is_file() or not metrics_path.is_file():
        pytest.skip("Backtest reports are not in the tree yet")
    preds = pd.read_csv(predictions_path, parse_dates=["target_date"])
    if not set(DAILY_MODELS).issubset(preds.columns):
        pytest.skip("predictions.csv does not have the current model columns")
    expected = pd.read_csv(metrics_path, dtype={"fold": str, "origin_date": str, "model": str})
    expected["origin_date"] = expected["origin_date"].fillna("")
    got = score_predictions(preds)
    merged = expected.merge(got, on=["fold", "origin_date", "model"], suffixes=("_file", "_recomputed"))
    assert len(merged) == len(expected)
    assert len(merged) == len(got)
    assert (merged["n_points_file"].astype(int) == merged["n_points_recomputed"].astype(int)).all()
    for column in ("mae", "rmse", "wape", "smape", "bias"):
        np.testing.assert_allclose(
            merged[f"{column}_file"].to_numpy(dtype=float),
            merged[f"{column}_recomputed"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-8,
        )
    headline_path = root / "reports" / "headline.txt"
    assert headline_path.is_file()
    lead = headline_path.read_text(encoding="utf-8").splitlines()[0]
    if "order quantity" not in lead:
        pytest.skip("headline has not been regenerated")
    assert lead in (root / "README.md").read_text(encoding="utf-8")

    decision_path = root / "reports" / "decision_metrics.csv"
    if decision_path.is_file():
        decision_file = pd.read_csv(decision_path)
        decision_got = decision_metrics(preds, DAILY_MODELS)
        check = decision_file.merge(decision_got, on=["grain", "model"], suffixes=("_file", "_recomputed"))
        assert len(check) == len(decision_file)
        for column in ("wape", "bias"):
            np.testing.assert_allclose(
                check[f"{column}_file"].to_numpy(dtype=float),
                check[f"{column}_recomputed"].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-6,
            )

    summary_path = root / "reports" / "inventory_summary.csv"
    if summary_path.is_file() and {"price_asof", "demand_class"}.issubset(preds.columns):
        summary_file = pd.read_csv(summary_path)
        ratio = float(summary_file["stockout_to_holding"].iloc[0])
        _, summary_got = inventory_evaluation(preds, list(summary_file["model"]), ratio)
        check = summary_file.merge(summary_got, on="model", suffixes=("_file", "_recomputed"))
        assert len(check) == len(summary_file)
        for column in ("units_short", "units_over", "cost"):
            np.testing.assert_allclose(
                check[f"{column}_file"].to_numpy(dtype=float),
                check[f"{column}_recomputed"].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-4,
            )
