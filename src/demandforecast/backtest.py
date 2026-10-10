"""Rolling-origin backtest of the SQL feature table."""

import json
import time
from dataclasses import asdict

import duckdb
import numpy as np
import pandas as pd

from demandforecast.analysis import write_analysis
from demandforecast.baselines import (
    croston_forecast,
    demand_pattern,
    moving_average_forecast,
    sba_forecast,
    seasonal_naive_forecast,
    tsb_forecast,
)
from demandforecast.boosting import fit_lightgbm, predict_lightgbm, prepare_frame
from demandforecast.config import (
    DAILY_MODELS,
    FEATURE_COLUMNS,
    INVENTORY_MODELS,
    LUMPY_COMPARE_MODELS,
    TARGET_COLUMN,
    Config,
)
from demandforecast.decision import decision_metrics
from demandforecast.inventory import inventory_evaluation
from demandforecast.metrics import regression_metrics, score_predictions
from demandforecast.paths import duckdb_path, reports_dir
from demandforecast.plots import plot_decision_summary, plot_forecast_vs_actual, plot_wape_by_fold
from demandforecast.reporting import headline_block, write_json, write_metrics_csv, write_metrics_md
from demandforecast.schedule import origin_dates
from demandforecast.segments import segment_metrics
from demandforecast.weekly import run_weekly_backtest


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


def _baseline_frame(histories, origin, skus, cfg: Config) -> pd.DataFrame:
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


def _intermittent_frame(histories, origin, skus, cfg: Config) -> pd.DataFrame:
    """Class and Croston-family forecasts from history through the origin, first positive day onward."""
    rows = []
    for sku in skus:
        values = histories[sku].loc[histories[sku]["date"] <= origin, "units"].to_numpy(dtype=float)
        idx = np.flatnonzero(values > 0.0)
        observed = values if idx.size == 0 else values[int(idx[0]) :]
        name, adi, cv2 = demand_pattern(observed, cfg.adi_threshold, cfg.cv2_threshold)
        croston = croston_forecast(observed, cfg.horizon, cfg.croston_alpha)
        sba = sba_forecast(observed, cfg.horizon, cfg.croston_alpha)
        tsb = tsb_forecast(observed, cfg.horizon, cfg.croston_alpha)
        for step in range(cfg.horizon):
            rows.append(
                {
                    "sku": sku,
                    "horizon": step + 1,
                    "croston": croston[step],
                    "sba": sba[step],
                    "tsb": tsb[step],
                    "demand_class": name,
                    "adi": adi,
                    "cv2": cv2,
                }
            )
    return pd.DataFrame(rows)


def _assert_close(left, right, label: str) -> None:
    if np.asarray(left).shape != np.asarray(right).shape or np.isnan(np.asarray(right, dtype=float)).any():
        raise RuntimeError(f"{label} is missing values")
    if not np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{label} does not match the SQL feature")


def _gain_frame(names: list[str], gain_l1: np.ndarray, gain_tweedie: np.ndarray, n_folds: int) -> pd.DataFrame:
    parts = []
    for model, gain in (("lightgbm_tweedie", gain_tweedie), ("lightgbm_l1", gain_l1)):
        part = pd.DataFrame({"model": model, "feature": names, "mean_gain": gain / n_folds})
        parts.append(part.sort_values("mean_gain", ascending=False))
    return pd.concat(parts, ignore_index=True)


def run_backtest(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    started = time.perf_counter()
    supervised, panel, build_stats = _load_tables()
    histories = _histories(panel)
    skus = sorted(histories)
    supervised = prepare_frame(
        supervised,
        FEATURE_COLUMNS,
        skus,
        ["asof_date", "target_date", "naive_w7_date", "same_day_last_year_date"],
    )
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
    recorded = build_stats.get("selection_cutoff")
    if recorded is None:
        raise RuntimeError("build_stats.json has no selection_cutoff. Rebuild features.")
    if pd.Timestamp(recorded) != pd.Timestamp(origins[0]):
        raise RuntimeError(
            f"Selection cutoff {recorded} is not the first origin {origins[0].date().isoformat()}"
        )
    _log(
        f"Backtest: {len(skus)} SKUs, {len(supervised):,} supervised rows, "
        f"{len(origins)} origins from {origins[0].date()} to {origins[-1].date()}"
    )

    prediction_frames = []
    fold_info = []
    gain_l1 = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    gain_tweedie = np.zeros(len(FEATURE_COLUMNS), dtype=float)
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
        price_n = test.groupby(test["sku"].astype(str))["price_asof"].nunique(dropna=False)
        if bool((price_n > 1).any()):
            raise RuntimeError(f"Fold {fold} has more than one origin price per SKU")

        l1_booster, l1_iter = fit_lightgbm(train, val, cfg, "regression_l1")
        l1_pred = predict_lightgbm(l1_booster, test, l1_iter)
        tw_booster, tw_iter = fit_lightgbm(train, val, cfg, "tweedie")
        tw_pred = predict_lightgbm(tw_booster, test, tw_iter)
        if not np.isfinite(l1_pred).all() or not np.isfinite(tw_pred).all():
            raise RuntimeError(f"Fold {fold} produced non-finite predictions")
        names = list(tw_booster.feature_name())
        if names != FEATURE_COLUMNS or list(l1_booster.feature_name()) != FEATURE_COLUMNS:
            raise RuntimeError(f"Unexpected LightGBM feature names: {names}")
        feature_names = names
        gain_l1 += np.asarray(l1_booster.feature_importance(importance_type="gain"), dtype=float)
        gain_tweedie += np.asarray(tw_booster.feature_importance(importance_type="gain"), dtype=float)

        baselines = _baseline_frame(histories, origin, skus, cfg)
        intermittent = _intermittent_frame(histories, origin, skus, cfg)
        scored = test.copy()
        scored["sku"] = scored["sku"].astype(str)
        scored = scored.merge(baselines, on=["sku", "horizon"], how="left", validate="one_to_one")
        scored = scored.merge(intermittent, on=["sku", "horizon"], how="left", validate="one_to_one")
        _assert_close(scored["seasonal_naive_w7"], scored["py_w7"], "Weekly seasonal naive")
        _assert_close(scored["units_same_day_last_year"], scored["py_y"], "Yearly seasonal naive")
        _assert_close(scored["roll_mean_28"], scored["py_ma"], "28-day moving average")

        frame = pd.DataFrame(
            {
                "fold": fold,
                "origin_date": origin.date().isoformat(),
                "sku": scored["sku"].to_numpy(),
                "target_date": pd.to_datetime(scored["target_date"]).dt.strftime("%Y-%m-%d"),
                "horizon": scored["horizon"].astype(int).to_numpy(),
                "actual": scored[TARGET_COLUMN].to_numpy(dtype=float),
                "seasonal_naive_w7": scored["py_w7"].to_numpy(dtype=float),
                "seasonal_naive_y364": scored["py_y"].to_numpy(dtype=float),
                "moving_average_28": scored["py_ma"].to_numpy(dtype=float),
                "lightgbm_l1": l1_pred,
                "lightgbm_tweedie": tw_pred,
                "croston": scored["croston"].to_numpy(dtype=float),
                "sba": scored["sba"].to_numpy(dtype=float),
                "tsb": scored["tsb"].to_numpy(dtype=float),
                "demand_class": scored["demand_class"].to_numpy(),
                "adi": scored["adi"].to_numpy(dtype=float),
                "cv2": scored["cv2"].to_numpy(dtype=float),
                "price_asof": scored["price_asof"].to_numpy(dtype=float),
            }
        )
        prediction_frames.append(frame)
        wape_tw = regression_metrics(frame["actual"], frame["lightgbm_tweedie"])["wape"]
        wape_l1 = regression_metrics(frame["actual"], frame["lightgbm_l1"])["wape"]
        wape_week = regression_metrics(frame["actual"], frame["seasonal_naive_w7"])["wape"]
        _log(
            f"fold {fold:02d} origin {origin.date()} train {len(train):,} "
            f"trees_l1 {l1_iter} trees_tweedie {tw_iter} "
            f"wape_naive7 {wape_week:.4f} wape_l1 {wape_l1:.4f} wape_tweedie {wape_tw:.4f}"
        )
        fold_info.append(
            {
                "fold": fold,
                "origin": origin.date().isoformat(),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "best_iteration_l1": int(l1_iter),
                "best_iteration_tweedie": int(tw_iter),
                "wape_lightgbm_l1": wape_l1,
                "wape_lightgbm_tweedie": wape_tw,
                "wape_seasonal_naive_w7": wape_week,
            }
        )

    preds = pd.concat(prediction_frames, ignore_index=True)
    metrics = score_predictions(preds, DAILY_MODELS)
    decision = decision_metrics(preds, DAILY_MODELS)
    segments = segment_metrics(preds, DAILY_MODELS)
    lumpy_rows = preds.loc[preds["demand_class"] == "lumpy"].copy()
    lumpy = score_predictions(lumpy_rows, LUMPY_COMPARE_MODELS) if not lumpy_rows.empty else None
    inventory_detail, inventory_summary = inventory_evaluation(
        preds, INVENTORY_MODELS, cfg.stockout_to_holding
    )
    weekly = run_weekly_backtest(cfg)

    paragraphs, summary = headline_block(
        metrics,
        decision,
        len(origins),
        cfg.horizon,
        weekly_decision=weekly["decision"],
        weekly_folds=weekly["n_folds"],
        weekly_horizon=cfg.weekly_horizon,
    )
    for paragraph in paragraphs:
        _log(paragraph)

    if feature_names is None:
        raise RuntimeError("No model was fit")
    importance = _gain_frame(feature_names, gain_l1, gain_tweedie, len(origins))

    reports = reports_dir()
    preds.to_csv(reports / "predictions.csv", index=False)
    write_metrics_csv(metrics, reports / "metrics.csv")
    write_metrics_md(metrics, paragraphs, reports / "metrics.md")
    (reports / "headline.txt").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    decision.to_csv(reports / "decision_metrics.csv", index=False)
    segments.to_csv(reports / "segment_metrics.csv", index=False)
    if lumpy is None:
        pd.DataFrame(columns=["fold", "origin_date", "model"]).to_csv(
            reports / "lumpy_metrics.csv", index=False
        )
    else:
        write_metrics_csv(lumpy, reports / "lumpy_metrics.csv")
    inventory_detail.to_csv(reports / "inventory_by_sku_origin.csv", index=False)
    inventory_summary.to_csv(reports / "inventory_summary.csv", index=False)
    importance.to_csv(reports / "feature_importance.csv", index=False)

    last = preds.loc[preds["fold"] == int(preds["fold"].max())].copy()
    daily = last.groupby("target_date", as_index=False)[
        ["actual", "moving_average_28", "lightgbm_l1", "lightgbm_tweedie"]
    ].sum()
    origin_label = last["origin_date"].iloc[0]
    plot_forecast_vs_actual(
        daily,
        reports / "forecast_vs_actual.png",
        f"Last window (origin {origin_label}): sum of {last['sku'].nunique()} SKUs",
    )
    plot_wape_by_fold(metrics, reports / "wape_by_fold.png")
    plot_decision_summary(decision, reports / "bias.png")
    write_analysis(
        reports / "analysis.md",
        decision=decision,
        metrics=metrics,
        segments=segments,
        lumpy=lumpy,
        inventory_summary=inventory_summary,
        inventory_detail=inventory_detail,
        weekly_decision=weekly["decision"],
        build_stats=build_stats,
    )

    elapsed = time.perf_counter() - started
    metadata = {
        "horizon_days": cfg.horizon,
        "fold_step_days": cfg.fold_step_days,
        "n_folds": len(origins),
        "n_skus": len(skus),
        "n_lumpy_rows": int(len(lumpy_rows)),
        "origins": [item["origin"] for item in fold_info],
        "weekly_origins": weekly["origins"],
        "weekly_n_folds": weekly["n_folds"],
        "elapsed_seconds": elapsed,
        "config": asdict(cfg),
        "build": build_stats,
        "folds": fold_info,
        "feature_importance_mean_gain": importance.to_dict(orient="records"),
        "inventory_summary": inventory_summary.to_dict(orient="records"),
        **summary,
    }
    write_json(metadata, reports / "run_metadata.json")
    _log(f"Wrote reports to {reports} in {elapsed:.1f}s")
    return metadata
