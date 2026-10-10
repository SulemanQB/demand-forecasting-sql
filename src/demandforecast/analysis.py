"""Short note assembled from the report tables so the prose cannot drift from them."""

from pathlib import Path

import numpy as np
import pandas as pd

from demandforecast.config import DAILY_MODELS, INVENTORY_MODELS, LUMPY_COMPARE_MODELS, MODEL_LABELS


def write_analysis(
    path: Path,
    *,
    decision: pd.DataFrame,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    lumpy: pd.DataFrame | None,
    inventory_summary: pd.DataFrame,
    inventory_detail: pd.DataFrame,
    weekly_decision: pd.DataFrame,
    build_stats: dict,
    robustness_notes: list[str] | None = None,
    horizon: pd.DataFrame | None = None,
    active: pd.DataFrame | None = None,
    refit_decision: pd.DataFrame | None = None,
    slice_decision: pd.DataFrame | None = None,
) -> str:
    text = _render(
        decision=decision,
        metrics=metrics,
        segments=segments,
        lumpy=lumpy,
        inventory_summary=inventory_summary,
        inventory_detail=inventory_detail,
        weekly_decision=weekly_decision,
        build_stats=build_stats,
        robustness_notes=robustness_notes,
        horizon=horizon,
        active=active,
        refit_decision=refit_decision,
        slice_decision=slice_decision,
    )
    path.write_text(text, encoding="utf-8")
    return text


def _render(
    *,
    decision: pd.DataFrame,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    lumpy: pd.DataFrame | None,
    inventory_summary: pd.DataFrame,
    inventory_detail: pd.DataFrame,
    weekly_decision: pd.DataFrame,
    build_stats: dict,
    robustness_notes: list[str] | None = None,
    horizon: pd.DataFrame | None = None,
    active: pd.DataFrame | None = None,
    refit_decision: pd.DataFrame | None = None,
    slice_decision: pd.DataFrame | None = None,
) -> str:
    order = decision.loc[decision["grain"] == "order_total"].set_index("model")
    winner = str(order["wape"].idxmin())
    quality = build_stats.get("quality", {})
    lines = [
        "# Evaluation note",
        "",
        "This note is written from the CSV files in `reports/`. "
        "The replenishment comparison is the 28-day total for one SKU at one origin. "
        "Daily WAPE is reported beside it as a shape score. "
        "Volume quartiles are computed from backtest actuals and are not model features. "
        "Demand class uses only history through the origin, starting at the first positive day.",
        "",
    ]
    if robustness_notes:
        lines.extend([robustness_notes[0], ""])
        lines.extend(["## Order-quantity uncertainty and labeled cuts", ""])
        for note in robustness_notes[1:]:
            lines.extend([note, ""])
    else:
        lines.extend(
            [
                f"The lowest order-quantity WAPE is the {MODEL_LABELS[winner]} at "
                f"{float(order.loc[winner, 'wape']):.6f} "
                f"(bias {float(order.loc[winner, 'bias']):+.6f}, "
                f"{int(order.loc[winner, 'n_points'])} SKU-origins).",
                "",
            ]
        )
    if horizon is not None and not horizon.empty:
        lines.extend(["Horizon cuts (order quantity):", "", _horizon_table(horizon), ""])
    if active is not None and not active.empty:
        lines.extend(
            [
                "Active-in-window slice (order quantity):",
                "",
                _decision_table(active.loc[active["grain"] == "order_total"]),
                "",
            ]
        )
    if refit_decision is not None and slice_decision is not None:
        lines.extend(
            [
                "Pre-cutoff top-SKU refit, then the same SKUs sliced from the top-40 model:",
                "",
                _labeled_decision_table(refit_decision),
                "",
                _labeled_decision_table(slice_decision),
                "",
            ]
        )
    lines.extend(
        [
        "## Order quantity, bias, and the daily aggregate",
        "",
        _decision_table(decision),
        "",
        "Bias is (sum of forecast minus actual) / sum of actual. A negative value under-forecasts the total.",
        "",
        "## Daily point metrics",
        "",
        _pooled_table(metrics),
        "",
        "Per-fold rows are in `metrics.md`. On the same rows, a relative MAE gap equals the relative WAPE gap.",
        "",
        "## Weekly track",
        "",
        "ISO weeks (Monday start), horizon 4. The 1-week naive repeats the origin week. "
        "The 52-week naive is the target week minus 364 days. "
        "The first and last weeks of the panel can be shorter than seven days.",
        "",
        _decision_table(weekly_decision),
        "",
        "## Lumpy SKUs",
        "",
        "A SKU-origin is lumpy when ADI > 1.32 and CV² > 0.49 on history from the first positive day through the origin. "
        "Croston, SBA, and TSB use alpha 0.1, are flat across the horizon, and are not included in the all-SKU WAPE. "
        "Croston and SBA do not update on trailing zeros. TSB updates the demand probability every period after the first demand.",
        "",
        ]
    )
    if lumpy is None or lumpy.empty:
        lines.append("No SKU-origin was classed as lumpy.")
    else:
        pooled = lumpy.loc[lumpy["fold"] == "pooled"]
        lines.append(_model_table(pooled, LUMPY_COMPARE_MODELS))
    lines.extend(["", "## Error by segment", ""])
    lines.append(_segment_table(segments))
    lines.extend(
        [
            "",
            "The full segment table, including the naive forecasts, is `segment_metrics.csv`.",
            "",
            "## Inventory cost",
            "",
            "For each SKU-origin the order quantity is the 28-day sum. "
            "Units short are actual minus forecast when that is positive; units over are the reverse. "
            "Holding cost per unit equals the origin unit price (the last realized price, not the price on the target day). "
            "Stockout cost per unit is 3 times holding cost. "
            "Cost = units over × price + 3 × units short × price. "
            "Rows with a missing or non-positive price stay in the unit totals and drop out of the money totals.",
            "",
            _inventory_table(inventory_summary),
            "",
            _top_costs(inventory_detail),
            "",
            "The SKU-origin table is `inventory_by_sku_origin.csv`. L1 is the absolute-error contrast. "
            "The moving average is included because it is the order-quantity baseline.",
            "",
            "## Sample and reconciliation",
            "",
            _sample_paragraph(build_stats, quality),
            "",
            "Charts: `order_wape_by_fold.png` (28-day order-quantity WAPE by origin), "
            "`bias.png` (bias by grain, and order-quantity WAPE), `forecast_vs_actual.png`, `wape_by_fold.png`.",
            "",
        ]
    )
    return "\n".join(lines)


def _horizon_table(horizon: pd.DataFrame) -> str:
    header = "| days | evaluation | model | points | WAPE | bias |"
    rule = "| --- | --- | --- | --- | --- | --- |"
    body = []
    for row in horizon.itertuples(index=False):
        body.append(
            f"| {int(row.h_min)}-{int(row.h_max)} | {row.evaluation} | "
            f"{MODEL_LABELS.get(row.model, str(row.model))} | {int(row.n_points)} | "
            f"{row.wape:.6f} | {row.bias:+.6f} |"
        )
    return "\n".join([header, rule, *body])


def _labeled_decision_table(decision: pd.DataFrame) -> str:
    order = decision.loc[decision["grain"] == "order_total"]
    label = str(order["evaluation"].iloc[0]) if "evaluation" in order.columns else ""
    note = str(order["note"].iloc[0]) if "note" in order.columns else ""
    return f"{label}. {note}\n\n" + _decision_table(order)


def _decision_table(decision: pd.DataFrame) -> str:
    header = "| grain | model | points | WAPE | bias |"
    rule = "| --- | --- | --- | --- | --- |"
    body = []
    for row in decision.itertuples(index=False):
        body.append(
            "| "
            + " | ".join(
                [
                    str(row.grain),
                    MODEL_LABELS.get(row.model, str(row.model)),
                    str(int(row.n_points)),
                    f"{row.wape:.6f}",
                    f"{row.bias:+.6f}",
                ]
            )
            + " |"
        )
    return "\n".join([header, rule, *body])


def _pooled_table(metrics: pd.DataFrame) -> str:
    pooled = metrics.loc[metrics["fold"] == "pooled"].set_index("model")
    header = "| model | points | MAE | RMSE | WAPE | sMAPE | bias |"
    rule = "| --- | --- | --- | --- | --- | --- | --- |"
    body = []
    for name in DAILY_MODELS:
        row = pooled.loc[name]
        body.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[name],
                    str(int(row.n_points)),
                    f"{row.mae:.4f}",
                    f"{row.rmse:.4f}",
                    f"{row.wape:.6f}",
                    f"{row.smape:.6f}",
                    f"{row.bias:+.6f}",
                ]
            )
            + " |"
        )
    return "\n".join([header, rule, *body])


def _model_table(pooled: pd.DataFrame, models: list[str]) -> str:
    frame = pooled.set_index("model")
    header = "| model | points | WAPE | bias |"
    rule = "| --- | --- | --- | --- |"
    body = []
    for name in models:
        if name not in frame.index:
            continue
        row = frame.loc[name]
        body.append(
            f"| {MODEL_LABELS[name]} | {int(row.n_points)} | {row.wape:.6f} | {row.bias:+.6f} |"
        )
    return "\n".join([header, rule, *body])


def _segment_table(segments: pd.DataFrame) -> str:
    keep_models = ["moving_average_28", "lightgbm_l1", "lightgbm_tweedie"]
    use = segments.loc[segments["model"].isin(keep_models)].copy()
    header = "| segment | grain | model | points | actual units | WAPE | bias |"
    rule = "| --- | --- | --- | --- | --- | --- | --- |"
    body = []
    for row in use.itertuples(index=False):
        wape = "n/a" if not np.isfinite(row.wape) else f"{row.wape:.6f}"
        bias = "n/a" if not np.isfinite(row.bias) else f"{row.bias:+.6f}"
        body.append(
            "| "
            + " | ".join(
                [
                    f"{row.segment_type}: {row.segment}",
                    str(row.grain),
                    MODEL_LABELS[row.model],
                    str(int(row.n_points)),
                    f"{row.actual_units:.1f}",
                    wape,
                    bias,
                ]
            )
            + " |"
        )
    return "\n".join([header, rule, *body])


def _inventory_table(summary: pd.DataFrame) -> str:
    frame = summary.set_index("model")
    header = "| model | SKU-origins | missing price | units short | units over | value short | value over | cost |"
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    body = []
    for name in INVENTORY_MODELS:
        if name not in frame.index:
            continue
        row = frame.loc[name]
        body.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[name],
                    str(int(row.n_sku_origins)),
                    str(int(row.n_missing_price)),
                    f"{row.units_short:.2f}",
                    f"{row.units_over:.2f}",
                    _money(row.value_short),
                    _money(row.value_over),
                    _money(row.cost),
                ]
            )
            + " |"
        )
    ratio = float(summary["stockout_to_holding"].iloc[0])
    basis = str(summary["holding_cost_basis"].iloc[0])
    lead = (
        f"Stockout-to-holding ratio {ratio:g}:1. Holding basis: {basis}. "
        "Value and cost are in the dataset currency (sterling)."
    )
    return lead + "\n\n" + "\n".join([header, rule, *body])


def _top_costs(detail: pd.DataFrame) -> str:
    part = detail.loc[detail["model"] == "lightgbm_tweedie"].dropna(subset=["cost"])
    if part.empty:
        return "No priced Tweedie SKU-origin was available for a cost ranking."
    top = part.nlargest(5, "cost")
    header = "| origin | sku | class | units short | units over | cost |"
    rule = "| --- | --- | --- | --- | --- | --- |"
    body = []
    for row in top.itertuples(index=False):
        body.append(
            "| "
            + " | ".join(
                [
                    str(row.origin_date),
                    str(row.sku),
                    str(row.demand_class),
                    f"{row.units_short:.2f}",
                    f"{row.units_over:.2f}",
                    _money(row.cost),
                ]
            )
            + " |"
        )
    return "Highest Tweedie costs:\n\n" + "\n".join([header, rule, *body])


def _sample_paragraph(build_stats: dict, quality: dict) -> str:
    share_before = float(build_stats["selected_unit_share_before_cutoff"])
    share_full = float(build_stats["selected_unit_share_full_period"])
    examples = []
    for row in build_stats.get("skus", [])[:3]:
        description = row.get("description") or ""
        examples.append(
            f"{row['sku']} {description} ({float(row['units_before_cutoff']):,.0f} units before the cutoff, "
            f"{int(row['sale_days_before_cutoff'])} sale days)"
        )
    example_text = "; ".join(examples) if examples else "none listed"
    return (
        f"Selection cutoff {build_stats['selection_cutoff']}, which is the first daily origin. "
        f"Ranks use dates strictly before that day. "
        f"{int(build_stats['n_skus_selected'])} SKUs were taken from "
        f"{int(build_stats['n_skus_eligible_before_cutoff'])} with at least "
        f"{int(build_stats['min_sale_days'])} sale days in the window. "
        f"They are {share_before:.2%} of units before the cutoff and, as a descriptive full-period share only, "
        f"{share_full:.2%} of cleaned units from {build_stats['date_min']} to {build_stats['date_max']}. "
        f"Examples: {example_text}. "
        f"Cancellation matches: {int(quality.get('n_matched_sales', 0))} "
        f"(sales lines {int(quality.get('n_sales_lines', 0))}, "
        f"clean lines {int(quality.get('n_clean_lines', 0))}). "
        f"Clean quantity {float(quality.get('clean_quantity', 0)):.0f} equals daily units "
        f"{float(quality.get('daily_units', 0)):.0f}. "
        f"Panel rows {int(quality.get('n_panel', 0))} and distinct keys {int(quality.get('n_panel_keys', 0))}. "
        f"Supervised rows with as-of on or after the target: {int(quality.get('n_bad_asof', 0))}. "
        f"Weekly versus daily unit mismatches: {int(build_stats.get('n_weekly_unit_mismatches', 0))}."
    )


def _money(value) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):,.2f}"
