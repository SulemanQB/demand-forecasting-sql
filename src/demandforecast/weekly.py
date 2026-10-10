"""ISO-week backtest. Origins are Mondays produced by the daily schedule in 7-day steps."""

import duckdb
import numpy as np
import pandas as pd

from demandforecast.baselines import moving_average_forecast, seasonal_naive_forecast
from demandforecast.boosting import fit_lightgbm, predict_lightgbm, prepare_frame
from demandforecast.config import TARGET_COLUMN, WEEKLY_FEATURE_COLUMNS, WEEKLY_MODELS, Config
from demandforecast.decision import decision_metrics
from demandforecast.metrics import regression_metrics, score_predictions
from demandforecast.paths import duckdb_path, reports_dir
from demandforecast.reporting import weekly_order_sentence, write_metrics_csv, write_metrics_md
from demandforecast.schedule import origin_dates


def _log(message: str) -> None:
    print(message, flush=True)


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    db_path = duckdb_path()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        supervised = con.execute("SELECT * FROM weekly_supervised").fetchdf()
        panel = con.execute("SELECT sku, week_start, units FROM weekly_panel").fetchdf()
    finally:
        con.close()
    return supervised, panel


def _histories(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = panel.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"]).dt.normalize()
    frame["sku"] = frame["sku"].astype(str)
    histories: dict[str, pd.DataFrame] = {}
    for sku, grp in frame.groupby("sku", sort=True):
        ordered = grp.sort_values("week_start").reset_index(drop=True)
        gaps = ordered["week_start"].diff().dt.days.iloc[1:]
        if len(ordered) == 0 or not bool((gaps == 7).all()):
            raise RuntimeError(f"SKU {sku} is not a complete weekly panel")
        if not bool((ordered["week_start"].dt.dayofweek == 0).all()):
            raise RuntimeError(f"SKU {sku} has a week that does not start on Monday")
        histories[str(sku)] = ordered
    return histories


def _baseline_frame(histories, origin, skus, cfg: Config) -> pd.DataFrame:
    rows = []
    for sku in skus:
        values = histories[sku].loc[histories[sku]["week_start"] <= origin, "units"].to_numpy(dtype=float)
        repeat = seasonal_naive_forecast(values, cfg.weekly_horizon, season=1)
        yearly = seasonal_naive_forecast(values, cfg.weekly_horizon, season=cfg.weekly_season_year)
        average = moving_average_forecast(values, cfg.weekly_horizon, cfg.weekly_ma_window)
        for step in range(cfg.weekly_horizon):
            rows.append(
                {
                    "sku": sku,
                    "horizon": step + 1,
                    "py_w1": repeat[step],
                    "py_w52": yearly[step],
                    "py_ma": average[step],
                }
            )
    return pd.DataFrame(rows)


def _assert_close(left, right, label: str) -> None:
    if np.asarray(left).shape != np.asarray(right).shape or np.isnan(np.asarray(right, dtype=float)).any():
        raise RuntimeError(f"{label} is missing values")
    if not np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{label} does not match the SQL feature")


def run_weekly_backtest(cfg: Config) -> dict:
    supervised, panel = _load()
    histories = _histories(panel)
    skus = sorted(histories)
    supervised = prepare_frame(
        supervised,
        WEEKLY_FEATURE_COLUMNS,
        skus,
        ["asof_date", "target_date", "naive_w1_date", "same_week_last_year_date"],
    )
    min_date = min(frame["week_start"].min() for frame in histories.values())
    max_date = max(frame["week_start"].max() for frame in histories.values())
    origins = origin_dates(
        min_date,
        max_date,
        horizon=cfg.weekly_horizon * 7,
        step=cfg.weekly_step * 7,
        yearly_lag_days=cfg.weekly_season_year * 7,
        warmup_days=cfg.weekly_warmup_weeks * 7,
    )
    if len(origins) < 4:
        raise RuntimeError(f"Only {len(origins)} weekly origins available")
    if any(pd.Timestamp(origin).dayofweek != 0 for origin in origins):
        raise RuntimeError("A weekly origin is not a Monday")
    _log(
        f"Weekly backtest: {len(skus)} SKUs, {len(supervised):,} rows, "
        f"{len(origins)} origins from {origins[0].date()} to {origins[-1].date()}"
    )

    frames = []
    for fold, origin in enumerate(origins, start=1):
        origin = pd.Timestamp(origin)
        val_start = origin - pd.Timedelta(days=cfg.weekly_horizon * 7)
        train = supervised.loc[supervised["target_date"] <= val_start].sort_values(
            ["sku", "target_date", "horizon"]
        )
        val = supervised.loc[
            (supervised["target_date"] > val_start) & (supervised["target_date"] <= origin)
        ].sort_values(["sku", "target_date", "horizon"])
        test = (
            supervised.loc[supervised["asof_date"] == origin]
            .sort_values(["sku", "horizon"])
            .reset_index(drop=True)
        )
        if len(test) != len(skus) * cfg.weekly_horizon:
            raise RuntimeError(
                f"Weekly fold {fold} has {len(test)} rows, expected {len(skus) * cfg.weekly_horizon}"
            )
        if test["target_date"].min() <= origin:
            raise RuntimeError(f"Weekly fold {fold} overlaps the origin")
        if len(train) < 200:
            raise RuntimeError(f"Weekly fold {fold} training set is too small ({len(train)} rows)")

        booster, n_iter = fit_lightgbm(train, val, cfg, "tweedie", WEEKLY_FEATURE_COLUMNS)
        pred = predict_lightgbm(booster, test, n_iter, WEEKLY_FEATURE_COLUMNS)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Weekly fold {fold} produced non-finite predictions")
        names = list(booster.feature_name())
        if names != WEEKLY_FEATURE_COLUMNS:
            raise RuntimeError(f"Unexpected weekly feature names: {names}")

        baselines = _baseline_frame(histories, origin, skus, cfg)
        scored = test.copy()
        scored["sku"] = scored["sku"].astype(str)
        scored = scored.merge(baselines, on=["sku", "horizon"], how="left", validate="one_to_one")
        _assert_close(scored["seasonal_naive_w1"], scored["py_w1"], "1-week seasonal naive")
        _assert_close(scored["units_same_week_last_year"], scored["py_w52"], "52-week seasonal naive")
        _assert_close(scored["roll_mean_4"], scored["py_ma"], "4-week moving average")

        frame = pd.DataFrame(
            {
                "fold": fold,
                "origin_date": origin.date().isoformat(),
                "sku": scored["sku"].to_numpy(),
                "target_date": pd.to_datetime(scored["target_date"]).dt.strftime("%Y-%m-%d"),
                "horizon": scored["horizon"].astype(int).to_numpy(),
                "actual": scored[TARGET_COLUMN].to_numpy(dtype=float),
                "seasonal_naive_w1": scored["py_w1"].to_numpy(dtype=float),
                "seasonal_naive_w52": scored["py_w52"].to_numpy(dtype=float),
                "moving_average_4": scored["py_ma"].to_numpy(dtype=float),
                "lightgbm_tweedie": pred,
            }
        )
        frames.append(frame)
        wape_model = regression_metrics(frame["actual"], frame["lightgbm_tweedie"])["wape"]
        wape_naive = regression_metrics(frame["actual"], frame["seasonal_naive_w1"])["wape"]
        _log(
            f"week fold {fold:02d} origin {origin.date()} train {len(train):,} "
            f"trees {n_iter} wape_naive1 {wape_naive:.4f} wape_tweedie {wape_model:.4f}"
        )

    preds = pd.concat(frames, ignore_index=True)
    metrics = score_predictions(preds, WEEKLY_MODELS)
    decision = decision_metrics(preds, WEEKLY_MODELS)
    sentence = weekly_order_sentence(decision, len(origins), cfg.weekly_horizon)
    _log(sentence)

    reports = reports_dir()
    preds.to_csv(reports / "weekly_predictions.csv", index=False)
    write_metrics_csv(metrics, reports / "weekly_metrics.csv")
    write_metrics_md(metrics, [sentence], reports / "weekly_metrics.md", title="Weekly backtest metrics")
    decision.to_csv(reports / "weekly_decision_metrics.csv", index=False)
    return {
        "decision": decision,
        "metrics": metrics,
        "n_folds": len(origins),
        "origins": [item.date().isoformat() for item in origins],
        "n_points": int(len(preds)),
        "sentence": sentence,
    }
