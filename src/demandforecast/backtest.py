"""Rolling-origin backtest of the SQL feature table."""

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from demandforecast.baselines import moving_average_forecast, seasonal_naive_forecast
from demandforecast.boosting import fit_lightgbm, predict_lightgbm
from demandforecast.config import FEATURE_COLUMNS, TARGET_COLUMN, Config
from demandforecast.metrics import regression_metrics, score_predictions
from demandforecast.paths import duckdb_path, reports_dir
from demandforecast.plots import plot_forecast_vs_actual, plot_wape_by_fold
from demandforecast.reporting import headline_block, write_json, write_metrics_csv, write_metrics_md
from demandforecast.schedule import origin_dates

import duckdb


def _log(message: str) -> None:
    print(message, flush=True)


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    db_path = duckdb_path()
    if not db_path.is_file():
        raise FileNotFoundError(
            f"DuckDB file not found at {db_path}. Run: python -m demandforecast build-features"
        )
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        supervised = con.execute("SELECT * FROM supervised").fetchdf()
        panel = con.execute("SELECT sku, date, units FROM panel").fetchdf()
    finally:
        con.close()
    stats_path = reports_dir() / "build_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    return supervised, panel, stats


def _histories(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["sku"] = frame["sku"].astype(str)
    histories: dict[str, pd.DataFrame] = {}
    for sku, grp in frame.groupby("sku", sort=True):
        ordered = grp.sort_values("date").reset_index(drop=True)
        gaps = ordered["date"].diff().dt.days.iloc[1:]
        if len(ordered) == 0 or not bool((gaps == 1).all()):
            raise RuntimeError(f"SKU {sku} is not a complete daily panel")
        histories[str(sku)] = ordered
    return histories


def _prepare(supervised: pd.DataFrame, skus: list[str]) -> pd.DataFrame:
    frame = supervised.copy()
    for column in ("asof_date", "target_date", "naive_w7_date", "same_day_last_year_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    frame["sku"] = pd.Categorical(frame["sku"].astype(str), categories=skus)
    numeric = [column for column in FEATURE_COLUMNS if column != "sku"] + [TARGET_COLUMN]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    return frame


def _baseline_frame(histories: dict[str, pd.DataFrame], origin: pd.Timestamp, skus: list[str], cfg: Config) -> pd.DataFrame:
    rows = []
    for sku in skus:
        values = histories[sku].loc[histories[sku]["date"] <= origin, "units"].to_numpy(dtype=float)
        weekly = seasonal_naive_forecast(values, cfg.horizon, cfg.season_week)
        yearly = seasonal_naive_forecast(values, cfg.horizon, cfg.season_year)
        average = moving_average_forecast(values, cfg.horizon, cfg.moving_average_window)
        for step in range(cfg.horizon):
            rows.append(
                {
                    "sku": sku,
                    "horizon": step + 1,
                    "py_w7": weekly[step],
                    "py_y": yearly[step],
                    "py_ma": average[step],
                }
            )
    return pd.DataFrame(rows)


def _assert_close(left, right, label: str) -> None:
    if np.asarray(left).shape != np.asarray(right).shape or np.isnan(np.asarray(right, dtype=float)).any():
        raise RuntimeError(f"{label} is missing values")
    if not np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{label} does not match the SQL feature")


def run_backtest(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    started = time.perf_counter()
    supervised, panel, build_stats = _load_tables()
    histories = _histories(panel)
    skus = sorted(histories)
    supervised = _prepare(supervised, skus)
    min_date = min(frame["date"].min() for frame in histories.values())
    max_date = max(frame["date"].max() for frame in histories.values())
    origins = origin_dates(
        min_date,
        max_date,
        horizon=cfg.horizon,
        step=cfg.fold_step_days,
        yearly_lag_days=cfg.season_year,
        warmup_days=cfg.lag_warmup_days,
    )
    if len(origins) < 4:
        raise RuntimeError(f"Only {len(origins)} origins available; expected several folds")
    _log(
        f"Backtest: {len(skus)} SKUs, {len(supervised):,} supervised rows, "
        f"{len(origins)} origins from {origins[0].date()} to {origins[-1].date()}"
    )

    prediction_frames = []
    fold_info = []
    gain = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    feature_names: list[str] | None = None

    for fold, origin in enumerate(origins, start=1):
        origin = pd.Timestamp(origin)
        val_start = origin - pd.Timedelta(days=cfg.horizon)
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
        if len(test) != len(skus) * cfg.horizon:
            raise RuntimeError(
                f"Fold {fold} has {len(test)} test rows, expected {len(skus) * cfg.horizon}"
            )
        if test["target_date"].min() <= origin:
            raise RuntimeError(f"Fold {fold} test window overlaps the origin")
        if (test["naive_w7_date"] > origin).any() or (test["same_day_last_year_date"] > origin).any():
            raise RuntimeError(f"Fold {fold} uses a feature date after the origin")
        if len(train) < 1000:
            raise RuntimeError(f"Fold {fold} training set is too small ({len(train)} rows)")

        booster, n_iter = fit_lightgbm(train, val, cfg)
        pred = predict_lightgbm(booster, test, n_iter)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Fold {fold} produced non-finite predictions")
        names = list(booster.feature_name())
        if names != FEATURE_COLUMNS:
            raise RuntimeError(f"Unexpected LightGBM feature names: {names}")
        feature_names = names
        gain += np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)

        baselines = _baseline_frame(histories, origin, skus, cfg)
        scored = test.copy()
        scored["sku"] = scored["sku"].astype(str)
        scored = scored.merge(baselines, on=["sku", "horizon"], how="left", validate="one_to_one")
        _assert_close(scored["seasonal_naive_w7"], scored["py_w7"], "Weekly seasonal naive")
        _assert_close(scored["units_same_day_last_year"], scored["py_y"], "Yearly seasonal naive")
        _assert_close(scored["roll_mean_28"], scored["py_ma"], "28-day moving average")

        frame = pd.DataFrame(
            {
                "fold": fold,
                "origin_date": origin.date().isoformat(),
                "sku": scored["sku"].to_numpy(),
                "target_date": scored["target_date"].to_numpy(),
                "horizon": scored["horizon"].astype(int).to_numpy(),
                "actual": scored[TARGET_COLUMN].to_numpy(dtype=float),
                "seasonal_naive_w7": scored["py_w7"].to_numpy(dtype=float),
                "seasonal_naive_y364": scored["py_y"].to_numpy(dtype=float),
                "moving_average_28": scored["py_ma"].to_numpy(dtype=float),
                "lightgbm": pred,
            }
        )
        prediction_frames.append(frame)
        wape_model = regression_metrics(frame["actual"], frame["lightgbm"])["wape"]
        wape_week = regression_metrics(frame["actual"], frame["seasonal_naive_w7"])["wape"]
        _log(
            f"fold {fold:02d} origin {origin.date()} train {len(train):,} "
            f"trees {n_iter} wape_naive7 {wape_week:.4f} wape_lgb {wape_model:.4f}"
        )
        fold_info.append(
            {
                "fold": fold,
                "origin": origin.date().isoformat(),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "best_iteration": int(n_iter),
                "wape_lightgbm": wape_model,
                "wape_seasonal_naive_w7": wape_week,
            }
        )

    preds = pd.concat(prediction_frames, ignore_index=True)
    metrics = score_predictions(preds)
    headline, mae_line, summary = headline_block(metrics, len(origins), cfg.horizon)
    _log(headline)
    _log(mae_line)

    reports = reports_dir()
    preds.to_csv(reports / "predictions.csv", index=False)
    write_metrics_csv(metrics, reports / "metrics.csv")
    write_metrics_md(metrics, headline, mae_line, reports / "metrics.md")
    (reports / "headline.txt").write_text(headline + "\n" + mae_line + "\n", encoding="utf-8")

    if feature_names is None:
        raise RuntimeError("No model was fit")
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_gain": gain / len(origins)}
    ).sort_values("mean_gain", ascending=False)
    importance.to_csv(reports / "feature_importance.csv", index=False)

    last = preds.loc[preds["fold"] == int(preds["fold"].max())].copy()
    daily = last.groupby("target_date", as_index=False)[
        ["actual", "seasonal_naive_w7", "seasonal_naive_y364", "lightgbm"]
    ].sum()
    origin_label = last["origin_date"].iloc[0]
    plot_forecast_vs_actual(
        daily,
        reports / "forecast_vs_actual.png",
        f"Last window (origin {origin_label}): sum of {last['sku'].nunique()} SKUs",
    )
    plot_wape_by_fold(metrics, reports / "wape_by_fold.png")

    elapsed = time.perf_counter() - started
    metadata = {
        "horizon_days": cfg.horizon,
        "fold_step_days": cfg.fold_step_days,
        "n_folds": len(origins),
        "n_skus": len(skus),
        "origins": [item["origin"] for item in fold_info],
        "elapsed_seconds": elapsed,
        "config": asdict(cfg),
        "build": build_stats,
        "folds": fold_info,
        "feature_importance_mean_gain": importance.to_dict(orient="records"),
        **summary,
    }
    write_json(metadata, reports / "run_metadata.json")
    _log(f"Wrote reports to {reports} in {elapsed:.1f}s")
    return metadata
