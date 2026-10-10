"""Write metric tables and the headline sentences used in the README."""

import json
from pathlib import Path

import pandas as pd

from demandforecast.config import MODEL_LABELS, MODEL_NAMES
from demandforecast.metrics import relative_reduction_pct


def _fmt_rate(value: float) -> str:
    return f"{value:.6f}"


def _fmt_error(value: float) -> str:
    return f"{value:.4f}"


def _fmt_int(value) -> str:
    return f"{int(value)}"


def comparison_phrase(baseline_name: str, baseline_wape: float, model_wape: float) -> str:
    reduction = relative_reduction_pct(baseline_wape, model_wape)
    if reduction >= 0:
        direction = "lower"
        shown = reduction
    else:
        direction = "higher"
        shown = -reduction
    return (
        f"{shown:.2f}% {direction} than the {baseline_name} (WAPE {baseline_wape:.6f})"
    )


def write_metrics_csv(metrics: pd.DataFrame, path: Path) -> None:
    metrics.to_csv(path, index=False)


def write_metrics_md(metrics: pd.DataFrame, headline: str, mae_line: str, path: Path) -> None:
    lines = [
        "# Backtest metrics",
        "",
        headline,
        "",
        mae_line,
        "",
        "WAPE and sMAPE in this table are ratios (sMAPE uses the 0/0 = 0 convention and is not multiplied by 100). "
        "The CSV stores the same figures at full precision. Displayed rates are rounded to 6 decimal places and MAE/RMSE to 4.",
        "",
        "| fold | origin | model | points | MAE | RMSE | WAPE | sMAPE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.fold),
                    row.origin_date or "",
                    MODEL_LABELS[row.model],
                    _fmt_int(row.n_points),
                    _fmt_error(row.mae),
                    _fmt_error(row.rmse),
                    _fmt_rate(row.wape),
                    _fmt_rate(row.smape),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def headline_block(metrics: pd.DataFrame, n_folds: int, horizon: int) -> tuple[str, str, dict]:
    pooled = metrics.loc[metrics["fold"] == "pooled"].set_index("model")
    wape = {name: float(pooled.loc[name, "wape"]) for name in MODEL_NAMES}
    mae = {name: float(pooled.loc[name, "mae"]) for name in MODEL_NAMES}
    rmse = {name: float(pooled.loc[name, "rmse"]) for name in MODEL_NAMES}
    smape = {name: float(pooled.loc[name, "smape"]) for name in MODEL_NAMES}
    phrases = [
        comparison_phrase(MODEL_LABELS[name], wape[name], wape["lightgbm"])
        for name in MODEL_NAMES
        if name != "lightgbm"
    ]
    headline = (
        f"On a {horizon}-day horizon across {n_folds} rolling origins, "
        f"LightGBM pooled WAPE was {wape['lightgbm']:.6f}: " + "; ".join(phrases) + "."
    )
    mae_phrase = comparison_phrase(
        MODEL_LABELS["seasonal_naive_w7"], mae["seasonal_naive_w7"], mae["lightgbm"]
    ).replace("WAPE", "MAE")
    mae_line = f"Pooled MAE was {mae['lightgbm']:.4f}, which is {mae_phrase}."
    reductions = {
        f"wape_reduction_pct_vs_{name}": relative_reduction_pct(wape[name], wape["lightgbm"])
        for name in MODEL_NAMES
        if name != "lightgbm"
    }
    summary = {
        "wape": wape,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        **reductions,
        "headline": headline,
        "mae_line": mae_line,
    }
    return headline, mae_line, summary


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
