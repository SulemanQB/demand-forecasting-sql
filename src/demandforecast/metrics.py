"""Point-forecast error metrics."""

import numpy as np
import pandas as pd

from demandforecast.config import DAILY_MODELS


def regression_metrics(y, yhat) -> dict[str, float | int]:
    actual = np.asarray(y, dtype=float)
    pred = np.asarray(yhat, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, predicted {pred.shape}")
    if actual.size == 0:
        raise ValueError("Cannot score an empty forecast")
    error = pred - actual
    abs_error = np.abs(error)
    denom = float(np.sum(np.abs(actual)))
    wape = float(np.sum(abs_error) / denom) if denom > 0 else float("nan")
    bias = float(np.sum(error) / denom) if denom > 0 else float("nan")
    smape_denom = np.abs(actual) + np.abs(pred)
    smape_terms = np.zeros_like(actual, dtype=float)
    np.divide(2.0 * abs_error, smape_denom, out=smape_terms, where=smape_denom != 0.0)
    return {
        "n_points": int(actual.size),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": wape,
        "smape": float(np.mean(smape_terms)),
        "bias": bias,
    }


def score_predictions(preds: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Per-fold and pooled metrics. `fold` is stored as text, with pooled last."""
    names = list(DAILY_MODELS if models is None else models)
    required = {"fold", "origin_date", "actual", *names}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if preds.empty:
        raise ValueError("Cannot score an empty forecast")

    rows: list[dict] = []
    grouped = preds.groupby(["fold", "origin_date"], sort=True)
    for (fold, origin), part in grouped:
        origin_text = pd.Timestamp(origin).date().isoformat()
        for model in names:
            rows.append(
                {
                    "fold": str(fold),
                    "origin_date": origin_text,
                    "model": model,
                    **regression_metrics(part["actual"], part[model]),
                }
            )
    for model in names:
        rows.append(
            {
                "fold": "pooled",
                "origin_date": "",
                "model": model,
                **regression_metrics(preds["actual"], preds[model]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["_fold_order"] = frame["fold"].map(lambda value: 10**9 if value == "pooled" else int(value))
    frame["_model_order"] = frame["model"].map({name: i for i, name in enumerate(names)})
    frame = frame.sort_values(["_fold_order", "_model_order"]).drop(
        columns=["_fold_order", "_model_order"]
    )
    return frame.reset_index(drop=True)


def relative_reduction_pct(baseline: float, model: float) -> float:
    if baseline == 0:
        raise ZeroDivisionError("Baseline metric is zero")
    return (baseline - model) / baseline * 100.0
