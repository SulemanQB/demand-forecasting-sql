"""Where the error sits. Volume quartiles describe the backtest; they are not features.

Demand class is attached upstream from history through the origin, not from the
scored window.
"""

import numpy as np
import pandas as pd

from demandforecast.metrics import regression_metrics


def volume_quartile(preds: pd.DataFrame) -> pd.Series:
    """Equal-count quartiles of total backtest actual units, Q1 lowest."""
    totals = preds.groupby("sku", sort=False)["actual"].sum().sort_values(kind="mergesort")
    n = int(totals.shape[0])
    if n == 0:
        raise ValueError("No SKUs to segment")
    labels = {}
    for i, sku in enumerate(totals.index):
        labels[sku] = f"Q{min(int(i * 4 / n), 3) + 1}"
    return preds["sku"].map(labels)


def segment_metrics(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if "demand_class" not in preds.columns:
        raise ValueError("Predictions have no demand_class column")
    frame = preds.copy()
    frame["volume_quartile"] = volume_quartile(frame)
    rows: list[dict] = []
    slices = [
        ("volume_quartile", "volume_quartile"),
        ("demand_class", "demand_class"),
    ]
    for segment_type, column in slices:
        for segment, part in frame.groupby(column, sort=True):
            if part.empty:
                continue
            for model in models:
                rows.extend(_rows(segment_type, str(segment), part, model))
    return pd.DataFrame(rows)


def _rows(segment_type: str, segment: str, part: pd.DataFrame, model: str) -> list[dict]:
    out = []
    daily = regression_metrics(part["actual"], part[model])
    out.append(
        {
            "segment_type": segment_type,
            "segment": segment,
            "grain": "daily",
            "model": model,
            "actual_units": float(np.sum(part["actual"].to_numpy(dtype=float))),
            **daily,
        }
    )
    ordered = part.groupby(["sku", "origin_date"], as_index=False)[["actual", model]].sum()
    total = regression_metrics(ordered["actual"], ordered[model])
    out.append(
        {
            "segment_type": segment_type,
            "segment": segment,
            "grain": "order_total",
            "model": model,
            "actual_units": float(np.sum(ordered["actual"].to_numpy(dtype=float))),
            **total,
        }
    )
    return out
