"""Decision metrics beside the daily SKU-day score.

Daily WAPE treats every SKU-day as a point. The replenishment quantity in this
project is the sum of the horizon for one SKU at one origin. Aggregate WAPE
sums the assortment on each date first, then scores those dates.
"""

import pandas as pd

from demandforecast.metrics import regression_metrics


def decision_metrics(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    required = {"sku", "origin_date", "target_date", "actual", *models}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if preds.empty:
        raise ValueError("Cannot score an empty forecast")

    rows: list[dict] = []
    for model in models:
        rows.append({"grain": "daily", "model": model, **regression_metrics(preds["actual"], preds[model])})
        ordered = preds.groupby(["sku", "origin_date"], as_index=False)[["actual", model]].sum()
        rows.append(
            {
                "grain": "order_total",
                "model": model,
                **regression_metrics(ordered["actual"], ordered[model]),
            }
        )
        by_day = preds.groupby("target_date", as_index=False)[["actual", model]].sum()
        rows.append(
            {
                "grain": "daily_aggregate",
                "model": model,
                **regression_metrics(by_day["actual"], by_day[model]),
            }
        )
    return pd.DataFrame(rows)
