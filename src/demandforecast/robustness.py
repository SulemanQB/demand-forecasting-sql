"""Order-quantity checks that do not refit the top-40 model.

The origin-cluster interval resamples origins, because the ten test windows share
calendar shocks. Horizon cuts and the 90-day activity filter score the forecasts
already produced. A separate module refits the pre-cutoff top 20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demandforecast.config import DAILY_MODELS, Config
from demandforecast.decision import decision_metrics
from demandforecast.metrics import regression_metrics
from demandforecast.segments import volume_quartile


MODEL = "lightgbm_tweedie"
BASELINE = "moving_average_28"
HORIZON_CUTS: tuple[tuple[int, int], ...] = (
    (1, 7),
    (8, 14),
    (15, 21),
    (22, 28),
    (1, 14),
    (1, 28),
)
PARTIAL_SUM_NOTE = "Partial sum of the fitted 28-day forecast. Not a refit."
ACTIVE_SLICE_NOTE = "SKU-origins with a sale in the activity window. Slice of the fitted model, not a refit."


def order_wape_by_origin(preds: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Pooled order-quantity error inside each origin."""
    names = list(DAILY_MODELS if models is None else models)
    required = {"fold", "origin_date", "sku", "actual", *names}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    rows: list[dict] = []
    grouped = preds.groupby(["fold", "origin_date"], sort=True)
    for (fold, origin), part in grouped:
        ordered = part.groupby("sku", as_index=False)[["actual", *names]].sum()
        origin_text = pd.Timestamp(origin).date().isoformat()
        for model in names:
            actual = ordered["actual"].to_numpy(dtype=float)
            pred = ordered[model].to_numpy(dtype=float)
            rows.append(
                {
                    "fold": int(fold),
                    "origin_date": origin_text,
                    "model": model,
                    "actual_units": float(np.sum(actual)),
                    "abs_error": float(np.sum(np.abs(pred - actual))),
                    **regression_metrics(actual, pred),
                }
            )
    return pd.DataFrame(rows)


def fold_record(by_origin: pd.DataFrame, model: str = MODEL, baseline: str = BASELINE) -> dict:
    """Count origins whose order-quantity WAPE is lower for `model`."""
    model_rows = by_origin.loc[by_origin["model"] == model].set_index("origin_date")
    base_rows = by_origin.loc[by_origin["model"] == baseline].set_index("origin_date")
    origins = sorted(model_rows.index.intersection(base_rows.index))
    if not origins:
        raise ValueError("No origins to compare")
    wins = 0
    ties = 0
    for origin in origins:
        gap = float(model_rows.loc[origin, "wape"]) - float(base_rows.loc[origin, "wape"])
        if gap < 0.0:
            wins += 1
        elif gap == 0.0:
            ties += 1
    return {
        "model": model,
        "baseline": baseline,
        "n_origins": len(origins),
        "n_lower": wins,
        "n_ties": ties,
        "n_higher": len(origins) - wins - ties,
    }


def sku_origin_contributions(
    preds: pd.DataFrame,
    model: str = MODEL,
    baseline: str = BASELINE,
) -> pd.DataFrame:
    """Order-level absolute-error reduction, model versus baseline. Positive favors the model."""
    ordered = preds.groupby(["sku", "origin_date"], as_index=False)[["actual", model, baseline]].sum()
    ordered["sku"] = ordered["sku"].astype(str)
    ordered["origin_date"] = ordered["origin_date"].map(lambda value: pd.Timestamp(value).date().isoformat())
    actual = ordered["actual"].to_numpy(dtype=float)
    ae_model = np.abs(ordered[model].to_numpy(dtype=float) - actual)
    ae_base = np.abs(ordered[baseline].to_numpy(dtype=float) - actual)
    reduction = ae_base - ae_model
    net = float(np.sum(reduction))
    share = np.full(reduction.shape, np.nan) if net == 0.0 else reduction / net
    out = pd.DataFrame(
        {
            "sku": ordered["sku"].to_numpy(),
            "origin_date": ordered["origin_date"].to_numpy(),
            "actual": actual,
            "forecast_model": ordered[model].to_numpy(dtype=float),
            "forecast_baseline": ordered[baseline].to_numpy(dtype=float),
            "ae_model": ae_model,
            "ae_baseline": ae_base,
            "ae_reduction": reduction,
            "share_of_net": share,
            "model": model,
            "baseline": baseline,
            "net_ae_reduction": net,
        }
    )
    return out.sort_values(
        ["ae_reduction", "sku", "origin_date"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def largest_contributor(contributions: pd.DataFrame) -> dict:
    if contributions.empty:
        raise ValueError("No SKU-origins to rank")
    row = contributions.iloc[0]
    return {
        "sku": str(row.sku),
        "origin_date": str(row.origin_date),
        "actual": float(row.actual),
        "forecast_model": float(row.forecast_model),
        "forecast_baseline": float(row.forecast_baseline),
        "ae_reduction": float(row.ae_reduction),
        "net_ae_reduction": float(row.net_ae_reduction),
        "share_of_net": float(row.share_of_net),
        "model": str(row.model),
        "baseline": str(row.baseline),
    }


def origin_cluster_interval(
    by_origin: pd.DataFrame,
    *,
    model: str = MODEL,
    baseline: str = BASELINE,
    n_draws: int,
    seed: int,
) -> dict:
    """Percentile interval for WAPE(model) - WAPE(baseline), resampling whole origins.

    Origins are sorted before the draw so the seed does not depend on row order.
    Within an origin, absolute error is the sum of SKU order-quantity absolute errors.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    model_rows = by_origin.loc[by_origin["model"] == model, ["origin_date", "actual_units", "abs_error"]]
    base_rows = by_origin.loc[by_origin["model"] == baseline, ["origin_date", "actual_units", "abs_error"]]
    merged = model_rows.merge(base_rows, on="origin_date", suffixes=("_model", "_baseline"))
    if merged.empty:
        raise ValueError("No origins for the cluster interval")
    merged = merged.sort_values("origin_date", kind="mergesort").reset_index(drop=True)
    actual = merged["actual_units_model"].to_numpy(dtype=float)
    if not np.allclose(actual, merged["actual_units_baseline"].to_numpy(dtype=float)):
        raise RuntimeError("Origin actuals differ between model and baseline")
    ae_model = merged["abs_error_model"].to_numpy(dtype=float)
    ae_base = merged["abs_error_baseline"].to_numpy(dtype=float)
    point = float(ae_model.sum() / actual.sum() - ae_base.sum() / actual.sum())
    rng = np.random.default_rng(seed)
    n_origins = int(actual.size)
    draw = rng.integers(0, n_origins, size=(n_draws, n_origins))
    drawn_actual = actual[draw].sum(axis=1)
    gap = ae_model[draw].sum(axis=1) / drawn_actual - ae_base[draw].sum(axis=1) / drawn_actual
    low, high = (float(v) for v in np.quantile(gap, [0.025, 0.975], method="linear"))
    return {
        "statistic": "order_wape_model_minus_baseline",
        "model": model,
        "baseline": baseline,
        "n_origins": n_origins,
        "n_draws": int(n_draws),
        "seed": int(seed),
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "share_draws_negative": float(np.mean(gap < 0.0)),
        "includes_zero": bool(low <= 0.0 <= high),
        "method": "origin_cluster_percentile_linear",
    }


def activity_flags(
    pairs: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    window_days: int,
) -> pd.DataFrame:
    """Mark SKU-origins with a positive unit on (origin - window_days, origin].

    The left endpoint is excluded. A sale on the origin day counts. History after
    the origin is ignored.
    """
    if window_days < 1:
        raise ValueError("window_days must be positive")
    required = {"sku", "origin_date"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Activity pairs are missing columns: {sorted(missing)}")
    rows: list[dict] = []
    work = pairs[["sku", "origin_date"]].drop_duplicates().copy()
    work["sku"] = work["sku"].astype(str)
    work["origin_date"] = work["origin_date"].map(lambda value: pd.Timestamp(value).normalize())
    for sku, part in work.groupby("sku", sort=True):
        history = histories.get(str(sku))
        if history is None:
            raise KeyError(f"No history for SKU {sku}")
        dates = pd.to_datetime(history["date"]).dt.normalize()
        units = history["units"].to_numpy(dtype=float)
        positive = dates.to_numpy()[units > 0.0]
        for origin in part["origin_date"]:
            start = origin - pd.Timedelta(days=window_days)
            in_window = (positive > np.datetime64(start)) & (positive <= np.datetime64(origin))
            known = positive[positive <= np.datetime64(origin)]
            last_positive = ""
            if known.size:
                last_positive = pd.Timestamp(known.max()).date().isoformat()
            date_values = dates.to_numpy()
            window = (date_values > np.datetime64(start)) & (date_values <= np.datetime64(origin)) & (units > 0.0)
            rows.append(
                {
                    "sku": str(sku),
                    "origin_date": origin.date().isoformat(),
                    "active": bool(np.any(in_window)),
                    "units_in_window": float(units[window].sum()),
                    "last_positive_on_or_before_origin": last_positive,
                    "window_days": int(window_days),
                }
            )
    return pd.DataFrame(rows)


def activity_decision(
    preds: pd.DataFrame,
    flags: pd.DataFrame,
    models: list[str] | None = None,
) -> pd.DataFrame:
    names = list(models or [BASELINE, "lightgbm_l1", MODEL])
    flagged = preds.copy()
    flagged["sku"] = flagged["sku"].astype(str)
    flagged["origin_date"] = flagged["origin_date"].map(lambda value: pd.Timestamp(value).date().isoformat())
    keep = flags.loc[flags["active"], ["sku", "origin_date"]]
    sliced = flagged.merge(keep, on=["sku", "origin_date"], how="inner")
    if sliced.empty:
        raise ValueError("Activity slice is empty")
    scored = decision_metrics(sliced, names)
    scored.insert(0, "evaluation", "active_last_window_slice")
    scored.insert(1, "note", ACTIVE_SLICE_NOTE)
    scored.insert(2, "n_active_sku_origins", int(keep.drop_duplicates().shape[0]))
    scored.insert(3, "n_sku_origins", int(flagged[["sku", "origin_date"]].drop_duplicates().shape[0]))
    return scored


def horizon_cut_metrics(
    preds: pd.DataFrame,
    cuts: tuple[tuple[int, int], ...] = HORIZON_CUTS,
    models: list[str] | None = None,
) -> pd.DataFrame:
    names = list(models or [BASELINE, "lightgbm_l1", MODEL])
    if "horizon" not in preds.columns:
        raise ValueError("Predictions have no horizon column")
    frames: list[pd.DataFrame] = []
    for h_min, h_max in cuts:
        part = preds.loc[(preds["horizon"] >= h_min) & (preds["horizon"] <= h_max)]
        if part.empty:
            raise ValueError(f"Horizon cut {h_min}-{h_max} has no rows")
        scored = decision_metrics(part, names)
        scored = scored.loc[scored["grain"] == "order_total"].copy()
        scored.insert(0, "h_min", h_min)
        scored.insert(1, "h_max", h_max)
        scored.insert(2, "evaluation", "partial_sum_not_a_refit")
        scored.insert(3, "note", PARTIAL_SUM_NOTE)
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def quartile_order_summary(preds: pd.DataFrame, quartile: str = "Q1") -> dict:
    """Order-quantity summary for one volume quartile. Quartiles are descriptive."""
    labels = volume_quartile(preds)
    part = preds.loc[labels.to_numpy() == quartile].copy()
    if part.empty:
        raise ValueError(f"Quartile {quartile} is empty")
    models = [BASELINE, "lightgbm_l1", MODEL]
    ordered = part.groupby(["sku", "origin_date"], as_index=False)[["actual", *models]].sum()
    summary = {
        "quartile": quartile,
        "n_sku_origins": int(len(ordered)),
        "actual_units": float(ordered["actual"].sum()),
        "mean_actual": float(ordered["actual"].mean()),
        "wape": {},
        "bias": {},
        "mean_forecast": {},
        "n_above_actual": {},
    }
    for model in models:
        metrics = regression_metrics(ordered["actual"], ordered[model])
        summary["wape"][model] = float(metrics["wape"])
        summary["bias"][model] = float(metrics["bias"])
        summary["mean_forecast"][model] = float(ordered[model].mean())
        summary["n_above_actual"][model] = int((ordered[model] > ordered["actual"]).sum())
    return summary


def quiet_skus(
    preds: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    contributions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    descriptions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """SKUs whose scored actuals are all zero, with history known by the cutoff."""
    totals = preds.groupby(preds["sku"].astype(str))["actual"].sum()
    names = [sku for sku, total in totals.items() if float(total) <= 0.0]
    descriptions = descriptions or {}
    rows: list[dict] = []
    cutoff = pd.Timestamp(cutoff).normalize()
    net = float(contributions["net_ae_reduction"].iloc[0]) if len(contributions) else 0.0
    for sku in sorted(names):
        history = histories[sku]
        dates = pd.to_datetime(history["date"]).dt.normalize()
        units = history["units"].to_numpy(dtype=float)
        date_values = dates.to_numpy()
        positive = date_values[units > 0.0]
        last_positive = ""
        if positive.size:
            last_positive = pd.Timestamp(positive.max()).date().isoformat()
        before = float(units[date_values < np.datetime64(cutoff)].sum())
        part = preds.loc[preds["sku"].astype(str) == sku]
        share_rows = contributions.loc[contributions["sku"] == sku, "share_of_net"]
        row = {
            "sku": sku,
            "description": descriptions.get(sku, ""),
            "n_origins": int(part["origin_date"].nunique()),
            "actual_units": float(part["actual"].sum()),
            "last_positive_date": last_positive,
            "units_before_cutoff": before,
            "share_of_net_ae_reduction": float(share_rows.sum()) if len(share_rows) else float("nan"),
            "net_ae_reduction": net,
        }
        for model in (BASELINE, "lightgbm_l1", MODEL):
            row[f"forecast_{model}"] = float(part[model].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_robustness(
    preds: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    cfg: Config,
    *,
    descriptions: dict[str, str] | None = None,
    cutoff: pd.Timestamp,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    by_origin = order_wape_by_origin(preds)
    record = fold_record(by_origin)
    contributions = sku_origin_contributions(preds)
    contributor = largest_contributor(contributions)
    interval = origin_cluster_interval(
        by_origin,
        n_draws=cfg.bootstrap_draws,
        seed=cfg.seed,
    )
    flags = activity_flags(preds[["sku", "origin_date"]], histories, cfg.active_window_days)
    active = activity_decision(preds, flags)
    cuts = horizon_cut_metrics(preds)
    quartile = quartile_order_summary(preds, "Q1")
    quiet = quiet_skus(
        preds,
        histories,
        contributions,
        cutoff=cutoff,
        descriptions=descriptions,
    )
    tables = {
        "order_wape_by_fold": by_origin,
        "sku_origin_contribution": contributions,
        "sku_origin_activity": flags,
        "active_window_metrics": active,
        "horizon_cut_metrics": cuts,
        "quiet_skus": quiet,
    }
    bundle = {
        "interval": interval,
        "folds": record,
        "contributor": contributor,
        "active": _active_numbers(active, cfg.active_window_days),
        "horizon": _horizon_numbers(cuts),
        "quartile": quartile,
        "quiet": quiet.to_dict(orient="records"),
    }
    return bundle, tables


def _active_numbers(active: pd.DataFrame, window_days: int) -> dict:
    order = active.loc[active["grain"] == "order_total"].set_index("model")
    return {
        "window_days": int(window_days),
        "n_active_sku_origins": int(order["n_active_sku_origins"].iloc[0]),
        "n_sku_origins": int(order["n_sku_origins"].iloc[0]),
        "wape": {model: float(order.loc[model, "wape"]) for model in order.index},
        "bias": {model: float(order.loc[model, "bias"]) for model in order.index},
    }


def _horizon_numbers(cuts: pd.DataFrame) -> list[dict]:
    rows = []
    for (h_min, h_max), part in cuts.groupby(["h_min", "h_max"], sort=False):
        indexed = part.set_index("model")
        rows.append(
            {
                "h_min": int(h_min),
                "h_max": int(h_max),
                "wape": {model: float(indexed.loc[model, "wape"]) for model in indexed.index},
                "bias": {model: float(indexed.loc[model, "bias"]) for model in indexed.index},
            }
        )
    return rows


def refit_summary(refit_decision: pd.DataFrame, slice_decision: pd.DataFrame, skus: list[str]) -> dict:
    def _pull(frame: pd.DataFrame) -> tuple[dict, dict, int]:
        order = frame.loc[frame["grain"] == "order_total"].set_index("model")
        wape = {model: float(order.loc[model, "wape"]) for model in order.index}
        bias = {model: float(order.loc[model, "bias"]) for model in order.index}
        return wape, bias, int(order["n_points"].iloc[0])

    refit_wape, refit_bias, n_orders = _pull(refit_decision)
    slice_wape, slice_bias, slice_orders = _pull(slice_decision)
    if n_orders != slice_orders:
        raise RuntimeError("Refit and slice do not cover the same number of SKU-origins")
    return {
        "n_skus": len(skus),
        "skus": list(skus),
        "n_sku_origins": n_orders,
        "refit_wape": refit_wape,
        "refit_bias": refit_bias,
        "slice_wape": slice_wape,
        "slice_bias": slice_bias,
    }
