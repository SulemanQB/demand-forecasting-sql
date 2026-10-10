"""Order-quantity interval, activity window, horizon cuts, and committed tables."""

import numpy as np
import pandas as pd
import pytest

from demandforecast.config import Config
from demandforecast.decision import decision_metrics
from demandforecast.paths import project_root
from demandforecast.reporting import hedged_order_sentence
from demandforecast.robustness import (
    activity_flags,
    fold_record,
    horizon_cut_metrics,
    largest_contributor,
    order_wape_by_origin,
    origin_cluster_interval,
    sku_origin_contributions,
)


def _history(origin: pd.Timestamp, sale_offsets: list[int]) -> pd.DataFrame:
    dates = pd.date_range(origin - pd.Timedelta(days=100), origin + pd.Timedelta(days=2), freq="D")
    units = np.zeros(len(dates), dtype=float)
    for offset in sale_offsets:
        sale_day = origin + pd.Timedelta(days=offset)
        units[dates == sale_day] = 4.0
    return pd.DataFrame({"date": dates, "units": units})


def test_activity_window_is_open_on_the_left_and_closed_on_the_origin():
    origin = pd.Timestamp("2020-04-10")
    pairs = pd.DataFrame(
        {
            "sku": ["far", "edge", "inside", "origin_day", "after"],
            "origin_date": [origin] * 5,
        }
    )
    histories = {
        "far": _history(origin, [-91]),
        "edge": _history(origin, [-90]),
        "inside": _history(origin, [-89]),
        "origin_day": _history(origin, [0]),
        "after": _history(origin, [1]),
    }
    flags = activity_flags(pairs, histories, window_days=90).set_index("sku")
    assert flags.loc["far", "active"] == False
    assert flags.loc["edge", "active"] == False
    assert flags.loc["inside", "active"] == True
    assert flags.loc["origin_day", "active"] == True
    assert flags.loc["after", "active"] == False
    assert flags.loc["edge", "units_in_window"] == 0
    assert flags.loc["inside", "units_in_window"] == pytest.approx(4)


def test_horizon_cut_sums_only_the_requested_days():
    preds = pd.DataFrame(
        {
            "fold": [1, 1],
            "origin_date": ["2020-01-01", "2020-01-01"],
            "sku": ["A", "A"],
            "target_date": ["2020-01-02", "2020-01-03"],
            "horizon": [1, 2],
            "actual": [10.0, 0.0],
            "moving_average_28": [10.0, 5.0],
            "lightgbm_l1": [10.0, 0.0],
            "lightgbm_tweedie": [12.0, 0.0],
        }
    )
    scored = horizon_cut_metrics(preds, cuts=((1, 1), (1, 2)))
    tw = scored.loc[scored["model"] == "lightgbm_tweedie"].set_index("h_max")
    ma = scored.loc[scored["model"] == "moving_average_28"].set_index("h_max")
    assert tw.loc[1, "wape"] == pytest.approx(0.2)
    assert ma.loc[1, "wape"] == pytest.approx(0.0)
    assert tw.loc[2, "wape"] == pytest.approx(0.2)
    assert ma.loc[2, "wape"] == pytest.approx(0.5)
    assert set(scored["evaluation"]) == {"partial_sum_not_a_refit"}
    assert "Not a refit" in scored["note"].iloc[0]


def test_origin_cluster_interval_is_reproducible_and_sorted():
    rows = []
    for fold, ae_model, ae_base in ((1, 1.0, 5.0), (2, 2.0, 4.0), (3, 3.0, 3.0), (4, 0.0, 6.0)):
        rows.append(
            {
                "fold": fold,
                "origin_date": f"2020-01-0{fold}",
                "model": "lightgbm_tweedie",
                "actual_units": 10.0,
                "abs_error": ae_model,
            }
        )
        rows.append(
            {
                "fold": fold,
                "origin_date": f"2020-01-0{fold}",
                "model": "moving_average_28",
                "actual_units": 10.0,
                "abs_error": ae_base,
            }
        )
    by_origin = pd.DataFrame(rows)
    first = origin_cluster_interval(by_origin, n_draws=2000, seed=42)
    second = origin_cluster_interval(by_origin.sample(frac=1, random_state=1), n_draws=2000, seed=42)
    assert first["ci_low"] <= first["ci_high"]
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
    assert first["point"] == second["point"]
    assert first["point"] < 0
    # Every origin has lower absolute error for Tweedie, so the resampled gap stays negative.
    assert first["includes_zero"] == False
    assert first["ci_high"] < 0


def test_largest_contributor_share_uses_the_net_reduction():
    preds = pd.DataFrame(
        {
            "fold": [1, 1],
            "sku": ["A", "B"],
            "origin_date": ["2020-01-01", "2020-01-01"],
            "actual": [10.0, 10.0],
            "lightgbm_tweedie": [10.0, 12.0],
            "moving_average_28": [20.0, 10.0],
        }
    )
    contributions = sku_origin_contributions(preds)
    top = largest_contributor(contributions)
    assert top["sku"] == "A"
    assert top["ae_reduction"] == pytest.approx(10)
    assert top["net_ae_reduction"] == pytest.approx(8)
    assert top["share_of_net"] == pytest.approx(10 / 8)
    record = fold_record(
        order_wape_by_origin(preds, ["lightgbm_tweedie", "moving_average_28"])
    )
    assert record["n_lower"] == 1
    assert record["n_origins"] == 1


def test_hedged_sentence_states_whether_zero_is_inside_the_interval():
    rows = []
    for grain in ("daily", "order_total", "daily_aggregate"):
        for name, wape in (("moving_average_28", 0.40), ("lightgbm_tweedie", 0.36)):
            rows.append(
                {
                    "grain": grain,
                    "model": name,
                    "n_points": 400,
                    "wape": wape,
                    "bias": -0.063205 if name == "lightgbm_tweedie" else -0.05,
                }
            )
    decision = pd.DataFrame(rows)
    robustness = {
        "interval": {"ci_low": -0.09, "ci_high": 0.01, "includes_zero": True},
        "folds": {"n_lower": 6, "n_origins": 10, "n_ties": 0},
    }
    text = hedged_order_sentence(decision, robustness, n_folds=10, horizon=28)
    assert text.startswith("At the 28-day order quantity (400 SKU-origins across 10 origins)")
    assert "0.360000" in text
    assert "includes zero" in text
    assert "lower on 6 of 10 origins" in text
    robustness["interval"] = {"ci_low": -0.09, "ci_high": -0.01, "includes_zero": False}
    excluded = hedged_order_sentence(decision, robustness, n_folds=10, horizon=28)
    assert "does not include zero" in excluded


def test_committed_robustness_matches_predictions():
    root = project_root()
    reports = root / "reports"
    predictions_path = reports / "predictions.csv"
    fold_path = reports / "order_wape_by_fold.csv"
    if not predictions_path.is_file() or not fold_path.is_file():
        pytest.skip("Robustness reports are not in the tree yet")
    preds = pd.read_csv(predictions_path)
    got = order_wape_by_origin(preds)
    saved = pd.read_csv(fold_path)
    merged = saved.merge(got, on=["fold", "origin_date", "model"], suffixes=("_file", "_recomputed"))
    assert len(merged) == len(saved)
    np.testing.assert_allclose(merged["wape_file"], merged["wape_recomputed"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        merged["abs_error_file"], merged["abs_error_recomputed"], rtol=1e-6, atol=1e-6
    )

    interval_path = reports / "origin_cluster_interval.csv"
    saved_interval = pd.read_csv(interval_path).iloc[0]
    cfg = Config()
    interval = origin_cluster_interval(got, n_draws=int(saved_interval["n_draws"]), seed=int(saved_interval["seed"]))
    assert interval["includes_zero"] == bool(saved_interval["includes_zero"])
    assert interval["ci_low"] == pytest.approx(float(saved_interval["ci_low"]), rel=0, abs=1e-12)
    assert interval["ci_high"] == pytest.approx(float(saved_interval["ci_high"]), rel=0, abs=1e-12)
    assert interval["n_draws"] == cfg.bootstrap_draws
    assert interval["seed"] == cfg.seed

    contributions = sku_origin_contributions(preds)
    top = largest_contributor(contributions)
    saved_top = pd.read_csv(reports / "sku_origin_contribution.csv").iloc[0]
    assert top["sku"] == str(saved_top["sku"])
    assert top["origin_date"] == str(saved_top["origin_date"])
    assert top["share_of_net"] == pytest.approx(float(saved_top["share_of_net"]))

    flags = pd.read_csv(reports / "sku_origin_activity.csv")
    flags["origin_date"] = flags["origin_date"].astype(str)
    active_ids = flags.loc[flags["active"], ["sku", "origin_date"]]
    active_ids["sku"] = active_ids["sku"].astype(str)
    sliced = preds.copy()
    sliced["sku"] = sliced["sku"].astype(str)
    sliced["origin_date"] = sliced["origin_date"].astype(str)
    sliced = sliced.merge(active_ids, on=["sku", "origin_date"], how="inner")
    active_got = decision_metrics(sliced, ["moving_average_28", "lightgbm_l1", "lightgbm_tweedie"])
    active_saved = pd.read_csv(reports / "active_window_metrics.csv")
    active_saved = active_saved.loc[active_saved["grain"] == "order_total"]
    check = active_saved.merge(active_got, on=["grain", "model"], suffixes=("_file", "_recomputed"))
    np.testing.assert_allclose(check["wape_file"], check["wape_recomputed"], rtol=1e-6, atol=1e-8)
    assert "not a refit" in active_saved["note"].iloc[0].lower()

    horizon_saved = pd.read_csv(reports / "horizon_cut_metrics.csv")
    horizon_got = horizon_cut_metrics(preds)
    horizon_check = horizon_saved.merge(
        horizon_got,
        on=["h_min", "h_max", "model"],
        suffixes=("_file", "_recomputed"),
    )
    assert len(horizon_check) == len(horizon_saved)
    np.testing.assert_allclose(
        horizon_check["wape_file"], horizon_check["wape_recomputed"], rtol=1e-6, atol=1e-8
    )
    assert set(horizon_saved["evaluation"]) == {"partial_sum_not_a_refit"}

    top_skus = pd.read_csv(reports / "top20_pre_cutoff_skus.csv")
    assert len(top_skus) == cfg.top_refit_n
    sku_list = set(top_skus["sku"].astype(str))
    refit_preds = pd.read_csv(reports / "top20_refit_predictions.csv")
    assert set(refit_preds["sku"].astype(str)) == sku_list
    assert set(refit_preds["evaluation"]) == {"refit_pre_cutoff_top_n"}
    refit_saved = pd.read_csv(reports / "top20_refit_decision_metrics.csv")
    refit_got = decision_metrics(refit_preds, ["moving_average_28", "lightgbm_l1", "lightgbm_tweedie"])
    refit_check = refit_saved.merge(refit_got, on=["grain", "model"], suffixes=("_file", "_recomputed"))
    np.testing.assert_allclose(refit_check["wape_file"], refit_check["wape_recomputed"], rtol=1e-6, atol=1e-8)
    assert "not a slice" in refit_saved["note"].iloc[0].lower()

    slice_saved = pd.read_csv(reports / "top20_slice_decision_metrics.csv")
    slice_preds = preds.loc[preds["sku"].astype(str).isin(sku_list)]
    slice_got = decision_metrics(slice_preds, ["moving_average_28", "lightgbm_l1", "lightgbm_tweedie"])
    slice_check = slice_saved.merge(slice_got, on=["grain", "model"], suffixes=("_file", "_recomputed"))
    np.testing.assert_allclose(slice_check["wape_file"], slice_check["wape_recomputed"], rtol=1e-6, atol=1e-8)
    assert "not refit" in slice_saved["note"].str.lower().iloc[0]
    assert "slice" in slice_saved["evaluation"].iloc[0]

    lead = (reports / "headline.txt").read_text(encoding="utf-8").splitlines()[0]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert lead in readme
    assert "make all" in readme
