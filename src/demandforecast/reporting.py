"""Write metric tables and the headline sentences used in the README."""

import json
from pathlib import Path

import pandas as pd

from demandforecast.config import DAILY_MODELS, MODEL_LABELS, WEEKLY_MODELS
from demandforecast.metrics import relative_reduction_pct
from demandforecast.refit import REFIT_NOTE, SLICE_NOTE


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


def _gap_phrase(baseline_wape: float, model_wape: float) -> str:
    reduction = relative_reduction_pct(baseline_wape, model_wape)
    if reduction >= 0:
        return f"{reduction:.2f}% lower"
    return f"{-reduction:.2f}% higher"


def hedged_order_sentence(
    decision: pd.DataFrame,
    robustness: dict,
    n_folds: int,
    horizon: int,
) -> str:
    order = decision.loc[decision["grain"] == "order_total"].set_index("model")
    tweedie = order.loc["lightgbm_tweedie"]
    average = order.loc["moving_average_28"]
    n_orders = int(tweedie.n_points)
    interval = robustness["interval"]
    folds = robustness["folds"]
    if interval["includes_zero"]:
        zero_text = "includes zero"
    else:
        zero_text = "does not include zero"
    tie_text = f", with {int(folds['n_ties'])} ties" if int(folds["n_ties"]) else ""
    return (
        f"At the {horizon}-day order quantity ({n_orders} SKU-origins across {n_folds} origins), "
        f"LightGBM Tweedie WAPE was {float(tweedie.wape):.6f} (bias {float(tweedie.bias):+.6f}), "
        f"{comparison_phrase(MODEL_LABELS['moving_average_28'], float(average.wape), float(tweedie.wape))}; "
        f"an origin-cluster 95% interval for the WAPE gap (Tweedie minus the moving average) is "
        f"[{float(interval['ci_low']):+.6f}, {float(interval['ci_high']):+.6f}] and {zero_text}; "
        f"Tweedie was lower on {int(folds['n_lower'])} of {int(folds['n_origins'])} origins{tie_text}."
    )


def contributor_sentence(contributor: dict) -> str:
    net = float(contributor["net_ae_reduction"])
    share = float(contributor["share_of_net"])
    if net > 0.0:
        role = "of the net absolute-error reduction versus the moving average"
    elif net < 0.0:
        role = "of the net absolute-error increase versus the moving average"
    else:
        role = "of a zero net absolute-error gap versus the moving average"
    return (
        f"The largest SKU-origin share {role} was SKU {contributor['sku']} "
        f"at origin {contributor['origin_date']} "
        f"({share:.2%} of the net; actual {float(contributor['actual']):.0f}, "
        f"Tweedie {float(contributor['forecast_model']):.2f}, "
        f"moving average {float(contributor['forecast_baseline']):.2f})."
    )


def active_slice_sentence(active: dict) -> str:
    tw = float(active["wape"]["lightgbm_tweedie"])
    ma = float(active["wape"]["moving_average_28"])
    bias = float(active["bias"]["lightgbm_tweedie"])
    return (
        f"Dropping SKU-origins with no sale in the {int(active['window_days'])} days before the origin "
        f"(dates after origin minus {int(active['window_days'])} days, through the origin), without refitting, "
        f"leaves {int(active['n_active_sku_origins'])} of {int(active['n_sku_origins'])} SKU-origins. "
        f"On that slice, order-quantity WAPE was {tw:.6f} for Tweedie (bias {bias:+.6f}) and {ma:.6f} "
        f"for the moving average, {_gap_phrase(ma, tw)} for Tweedie."
    )


def horizon_sentence(cuts: list[dict]) -> str:
    wanted = {(1, 7), (1, 14)}
    bits = []
    for row in cuts:
        key = (int(row["h_min"]), int(row["h_max"]))
        if key not in wanted:
            continue
        tw = float(row["wape"]["lightgbm_tweedie"])
        ma = float(row["wape"]["moving_average_28"])
        bits.append(
            f"days {key[0]}-{key[1]}: Tweedie WAPE {tw:.6f} is {_gap_phrase(ma, tw)} "
            f"than the moving average (WAPE {ma:.6f})"
        )
    if len(bits) != 2:
        raise ValueError("Horizon sentence needs the 1-7 and 1-14 cuts")
    return (
        "Horizon cuts are partial sums of the fitted 28-day forecasts, not refits. "
        + "; ".join(bits)
        + "."
    )


def quartile_sentence(quartile: dict) -> str:
    tw = float(quartile["wape"]["lightgbm_tweedie"])
    ma = float(quartile["wape"]["moving_average_28"])
    l1 = float(quartile["wape"]["lightgbm_l1"])
    n = int(quartile["n_sku_origins"])
    above_tw = int(quartile["n_above_actual"]["lightgbm_tweedie"])
    above_ma = int(quartile["n_above_actual"]["moving_average_28"])
    return (
        f"In {quartile['quartile']}, the lowest backtest-volume quartile (descriptive, not a feature; "
        f"{float(quartile['actual_units']):.0f} units, mean order {float(quartile['mean_actual']):.2f}), "
        f"order-quantity WAPE was {tw:.6f} for Tweedie (bias {float(quartile['bias']['lightgbm_tweedie']):+.6f}), "
        f"{ma:.6f} for the moving average (bias {float(quartile['bias']['moving_average_28']):+.6f}), "
        f"and {l1:.6f} for L1 (bias {float(quartile['bias']['lightgbm_l1']):+.6f}). "
        f"Tweedie was above the 28-day actual on {above_tw} of {n} SKU-origins, "
        f"against {above_ma} of {n} for the moving average."
    )


def quiet_sentence(rows: list[dict]) -> str:
    if not rows:
        return "No scored SKU had zero actual units across the test origins."
    parts = []
    for row in rows:
        description = str(row.get("description") or "").strip()
        name = f"SKU {row['sku']}"
        if description:
            name = f"{name} ({description})"
        share = float(row["share_of_net_ae_reduction"])
        drag = " A negative share increases Tweedie's absolute error." if share < 0.0 else ""
        parts.append(
            f"{name} has actual {float(row['actual_units']):.0f} on all {int(row['n_origins'])} scored origins. "
            f"The last positive day in the panel is {row['last_positive_date'] or 'none'}, "
            f"and it sold {float(row['units_before_cutoff']):.0f} units before the first origin. "
            f"The moving average forecasts {float(row['forecast_moving_average_28']):.2f} units in total; "
            f"Tweedie forecasts {float(row['forecast_lightgbm_tweedie']):.2f}. "
            f"Its share of the net absolute-error reduction versus the moving average is {share:.2%}.{drag}"
        )
    return " ".join(parts)


def refit_sentence(refit: dict, quiet: list[dict]) -> str:
    tw = float(refit["refit_wape"]["lightgbm_tweedie"])
    ma = float(refit["refit_wape"]["moving_average_28"])
    stw = float(refit["slice_wape"]["lightgbm_tweedie"])
    sma = float(refit["slice_wape"]["moving_average_28"])
    quiet_in_refit = [str(row["sku"]) for row in quiet if str(row["sku"]) in set(refit["skus"])]
    extra = ""
    if len(quiet_in_refit) == 1:
        extra = (
            f" The refit assortment includes {quiet_in_refit[0]}, "
            "which has no sales in the scored window."
        )
    elif quiet_in_refit:
        listed = ", ".join(quiet_in_refit)
        extra = f" The refit assortment includes {listed}, which have no sales in the scored window."
    return (
        f"{REFIT_NOTE} "
        f"On this {int(refit['n_skus'])}-SKU refit, order-quantity WAPE was {tw:.6f} for Tweedie "
        f"(bias {float(refit['refit_bias']['lightgbm_tweedie']):+.6f}) and {ma:.6f} for the moving average "
        f"({_gap_phrase(ma, tw)} for Tweedie), across {int(refit['n_sku_origins'])} SKU-origins. "
        f"{SLICE_NOTE} "
        f"That slice has Tweedie WAPE {stw:.6f} and moving-average WAPE {sma:.6f} "
        f"({_gap_phrase(sma, stw)} for Tweedie).{extra}"
    )


def headline_block(
    metrics: pd.DataFrame,
    decision: pd.DataFrame,
    n_folds: int,
    horizon: int,
    weekly_decision: pd.DataFrame | None = None,
    weekly_folds: int | None = None,
    weekly_horizon: int | None = None,
    robustness: dict | None = None,
) -> tuple[list[str], dict]:
    order = decision.loc[decision["grain"] == "order_total"].set_index("model")
    daily = decision.loc[decision["grain"] == "daily"].set_index("model")
    aggregate = decision.loc[decision["grain"] == "daily_aggregate"].set_index("model")
    winner = str(order["wape"].idxmin())
    n_orders = int(order.loc[winner, "n_points"])
    tweedie = order.loc["lightgbm_tweedie"]
    median = order.loc["lightgbm_l1"]
    average = order.loc["moving_average_28"]
    if robustness is None:
        lead = (
            f"At the {horizon}-day order quantity ({n_orders} SKU-origins across {n_folds} origins), "
            f"the {MODEL_LABELS[winner]} has the lowest WAPE, {float(order.loc[winner, 'wape']):.6f}, "
            f"with bias {float(order.loc[winner, 'bias']):+.6f}."
        )
        second = (
            f"LightGBM Tweedie order-quantity WAPE was {float(tweedie.wape):.6f} "
            f"(bias {float(tweedie.bias):+.6f}): "
            f"{comparison_phrase(MODEL_LABELS['moving_average_28'], float(average.wape), float(tweedie.wape))}. "
            f"LightGBM L1, fit on absolute error, had order-quantity WAPE {float(median.wape):.6f} "
            f"(bias {float(median.bias):+.6f})."
        )
        context: list[str] = []
    else:
        lead = hedged_order_sentence(decision, robustness, n_folds, horizon)
        second = contributor_sentence(robustness["contributor"])
        context = [
            (
                f"LightGBM L1, fit on absolute error, had order-quantity WAPE {float(median.wape):.6f} "
                f"(bias {float(median.bias):+.6f})."
            ),
            active_slice_sentence(robustness["active"]),
            horizon_sentence(robustness["horizon"]),
            quartile_sentence(robustness["quartile"]),
            quiet_sentence(robustness["quiet"]),
            refit_sentence(robustness["refit"], robustness["quiet"]),
        ]
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
    paragraphs = [lead, second, *context, bias_line, agg_line, daily_line]
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
        "robustness": robustness,
    }
    return paragraphs, summary


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
