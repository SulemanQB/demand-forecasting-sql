"""Write metric tables and the headline sentences used in the README."""

import json
from pathlib import Path

import pandas as pd

from demandforecast.config import DAILY_MODELS, MODEL_LABELS, WEEKLY_MODELS
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


def write_metrics_md(
    metrics: pd.DataFrame,
    paragraphs: list[str],
    path: Path,
    title: str = "Backtest metrics",
) -> None:
    lines = [f"# {title}", ""]
    for paragraph in paragraphs:
        lines.append(paragraph)
        lines.append("")
    lines.extend(
        [
            "WAPE, bias, and sMAPE in this table are ratios (sMAPE uses the 0/0 = 0 convention and is not multiplied by 100). "
            "Bias is (sum of forecast minus actual) / sum of actual. Negative bias is an under-forecast. "
            "The CSV stores the same figures at full precision. Displayed rates are rounded to 6 decimal places and MAE/RMSE to 4.",
            "",
            "| fold | origin | model | points | MAE | RMSE | WAPE | sMAPE | bias |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
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
                    f"{row.bias:+.6f}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def weekly_order_sentence(decision: pd.DataFrame, n_folds: int, horizon_weeks: int) -> str:
    order = decision.loc[decision["grain"] == "order_total"].set_index("model")
    winner = order["wape"].idxmin()
    n_orders = int(order.loc[winner, "n_points"])
    bits = []
    for name in WEEKLY_MODELS:
        row = order.loc[name]
        bits.append(
            f"{MODEL_LABELS[name]} WAPE {float(row.wape):.6f} (bias {float(row.bias):+.6f})"
        )
    return (
        f"On the weekly track (ISO weeks, horizon {horizon_weeks}, {n_folds} origins, "
        f"{n_orders} SKU-origins), the {MODEL_LABELS[winner]} has the lowest summed-horizon WAPE. "
        + "; ".join(bits)
        + "."
    )


def headline_block(
    metrics: pd.DataFrame,
    decision: pd.DataFrame,
    n_folds: int,
    horizon: int,
    weekly_decision: pd.DataFrame | None = None,
    weekly_folds: int | None = None,
    weekly_horizon: int | None = None,
) -> tuple[list[str], dict]:
    order = decision.loc[decision["grain"] == "order_total"].set_index("model")
    daily = decision.loc[decision["grain"] == "daily"].set_index("model")
    aggregate = decision.loc[decision["grain"] == "daily_aggregate"].set_index("model")
    winner = str(order["wape"].idxmin())
    n_orders = int(order.loc[winner, "n_points"])
    lead = (
        f"At the {horizon}-day order quantity ({n_orders} SKU-origins across {n_folds} origins), "
        f"the {MODEL_LABELS[winner]} has the lowest WAPE, {float(order.loc[winner, 'wape']):.6f}, "
        f"with bias {float(order.loc[winner, 'bias']):+.6f}."
    )
    tweedie = order.loc["lightgbm_tweedie"]
    median = order.loc["lightgbm_l1"]
    average = order.loc["moving_average_28"]
    second = (
        f"LightGBM Tweedie order-quantity WAPE was {float(tweedie.wape):.6f} "
        f"(bias {float(tweedie.bias):+.6f}): "
        f"{comparison_phrase(MODEL_LABELS['moving_average_28'], float(average.wape), float(tweedie.wape))}. "
        f"LightGBM L1, fit on absolute error, had order-quantity WAPE {float(median.wape):.6f} "
        f"(bias {float(median.bias):+.6f})."
    )
    bias_bits = [
        f"{MODEL_LABELS[name]} {float(order.loc[name, 'bias']):+.6f}" for name in DAILY_MODELS
    ]
    bias_line = (
        "Order-quantity bias, (forecast minus actual) / actual, was " + "; ".join(bias_bits) + "."
    )
    agg_bits = [
        f"{float(aggregate.loc[name, 'wape']):.6f} for the {MODEL_LABELS[name]}"
        for name in DAILY_MODELS
    ]
    agg_line = (
        "Summing the assortment before scoring, daily aggregate WAPE was " + "; ".join(agg_bits) + "."
    )
    daily_line = (
        f"On individual SKU-days, Tweedie WAPE was {float(daily.loc['lightgbm_tweedie', 'wape']):.6f} "
        f"(bias {float(daily.loc['lightgbm_tweedie', 'bias']):+.6f}) and L1 WAPE was "
        f"{float(daily.loc['lightgbm_l1', 'wape']):.6f} "
        f"(bias {float(daily.loc['lightgbm_l1', 'bias']):+.6f}), against "
        f"{float(daily.loc['moving_average_28', 'wape']):.6f} for the 28-day moving average and "
        f"{float(daily.loc['seasonal_naive_w7', 'wape']):.6f} for the weekly seasonal naive. "
        "The daily score is a shape score; the order quantity above is the replenishment comparison."
    )
    paragraphs = [lead, second, bias_line, agg_line, daily_line]
    if weekly_decision is not None:
        if weekly_folds is None or weekly_horizon is None:
            raise ValueError("Weekly headline needs the fold count and horizon")
        paragraphs.append(weekly_order_sentence(weekly_decision, weekly_folds, weekly_horizon))

    pooled = metrics.loc[metrics["fold"] == "pooled"].set_index("model")
    mae = {name: float(pooled.loc[name, "mae"]) for name in DAILY_MODELS}
    rmse = {name: float(pooled.loc[name, "rmse"]) for name in DAILY_MODELS}
    smape = {name: float(pooled.loc[name, "smape"]) for name in DAILY_MODELS}
    mae_line = (
        f"Pooled daily MAE was {mae['lightgbm_tweedie']:.4f} for Tweedie, "
        f"{mae['lightgbm_l1']:.4f} for L1, "
        f"{mae['moving_average_28']:.4f} for the 28-day moving average, "
        f"{mae['seasonal_naive_w7']:.4f} for the weekly seasonal naive, and "
        f"{mae['seasonal_naive_y364']:.4f} for the yearly seasonal naive. "
        "On the same rows, the relative MAE gap equals the relative WAPE gap because both scale with total absolute error."
    )
    rmse_line = (
        f"Pooled daily RMSE was {rmse['lightgbm_tweedie']:.4f} for Tweedie, "
        f"{rmse['lightgbm_l1']:.4f} for L1, "
        f"{rmse['moving_average_28']:.4f} for the 28-day moving average, "
        f"{rmse['seasonal_naive_w7']:.4f} for the weekly seasonal naive, and "
        f"{rmse['seasonal_naive_y364']:.4f} for the yearly seasonal naive."
    )
    smape_line = (
        f"Pooled daily sMAPE was {smape['lightgbm_tweedie']:.6f} for Tweedie, "
        f"{smape['lightgbm_l1']:.6f} for L1, "
        f"{smape['moving_average_28']:.6f} for the 28-day moving average, "
        f"{smape['seasonal_naive_w7']:.6f} for the weekly seasonal naive, and "
        f"{smape['seasonal_naive_y364']:.6f} for the yearly seasonal naive."
    )
    paragraphs.extend([mae_line, rmse_line, smape_line])

    def _column(frame: pd.DataFrame, column: str) -> dict[str, float]:
        return {name: float(frame.loc[name, column]) for name in DAILY_MODELS}

    summary = {
        "order_winner": winner,
        "order_wape": _column(order, "wape"),
        "order_bias": _column(order, "bias"),
        "daily_wape": _column(daily, "wape"),
        "daily_bias": _column(daily, "bias"),
        "aggregate_wape": _column(aggregate, "wape"),
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "headline": lead,
        "paragraphs": paragraphs,
    }
    return paragraphs, summary


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
