"""Horizon-total inventory cost at a stated stockout-to-holding ratio.

Holding cost per unit is the unit price known at the origin (`price_asof`), a
retail-value proxy, not a measured warehouse cost. A missing or non-positive
price stays in the unit counts and is dropped from the money totals.
"""

import numpy as np
import pandas as pd


def inventory_evaluation(
    preds: pd.DataFrame,
    models: list[str],
    stockout_to_holding: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stockout_to_holding <= 0:
        raise ValueError("stockout_to_holding must be positive")
    keys = ["fold", "origin_date", "sku"]
    required = set(keys) | {"actual", "price_asof", "demand_class", *models}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    price_n = preds.groupby(keys)["price_asof"].nunique(dropna=False)
    if bool((price_n > 1).any()):
        raise RuntimeError("price_asof is not constant inside a SKU-origin")
    class_n = preds.groupby(keys)["demand_class"].nunique(dropna=False)
    if bool((class_n > 1).any()):
        raise RuntimeError("demand_class is not constant inside a SKU-origin")

    base = preds.groupby(keys, as_index=False).agg(
        actual=("actual", "sum"),
        price_asof=("price_asof", "first"),
        demand_class=("demand_class", "first"),
    )
    frames = []
    for model in models:
        forecast = preds.groupby(keys, as_index=False)[model].sum()
        part = base.merge(forecast, on=keys, validate="one_to_one")
        pred = part[model].to_numpy(dtype=float)
        actual = part["actual"].to_numpy(dtype=float)
        price = part["price_asof"].to_numpy(dtype=float)
        short = np.maximum(actual - pred, 0.0)
        over = np.maximum(pred - actual, 0.0)
        priced = np.isfinite(price) & (price > 0.0)
        value_short = np.where(priced, short * price, np.nan)
        value_over = np.where(priced, over * price, np.nan)
        cost = np.where(priced, value_over + stockout_to_holding * value_short, np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "fold": part["fold"].to_numpy(),
                    "origin_date": part["origin_date"].to_numpy(),
                    "sku": part["sku"].to_numpy(),
                    "demand_class": part["demand_class"].to_numpy(),
                    "model": model,
                    "actual_units": actual,
                    "forecast_units": pred,
                    "price_asof": price,
                    "units_short": short,
                    "units_over": over,
                    "value_short": value_short,
                    "value_over": value_over,
                    "cost": cost,
                    "price_missing": ~priced,
                    "stockout_to_holding": stockout_to_holding,
                    "holding_cost_basis": "origin unit price",
                }
            )
        )
    detail = pd.concat(frames, ignore_index=True)
    rows = []
    for model, part in detail.groupby("model", sort=False):
        priced = part.loc[~part["price_missing"]]
        rows.append(
            {
                "model": model,
                "stockout_to_holding": stockout_to_holding,
                "holding_cost_basis": "origin unit price",
                "n_sku_origins": int(len(part)),
                "n_missing_price": int(part["price_missing"].sum()),
                "actual_units": float(part["actual_units"].sum()),
                "forecast_units": float(part["forecast_units"].sum()),
                "units_short": float(part["units_short"].sum()),
                "units_over": float(part["units_over"].sum()),
                "value_short": float(priced["value_short"].sum()) if len(priced) else float("nan"),
                "value_over": float(priced["value_over"].sum()) if len(priced) else float("nan"),
                "cost": float(priced["cost"].sum()) if len(priced) else float("nan"),
            }
        )
    return detail, pd.DataFrame(rows)
