"""Backtest figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demandforecast.config import DAILY_MODELS, MODEL_LABELS


_SHORT = {
    "seasonal_naive_w7": "Naive 7",
    "seasonal_naive_y364": "Naive 364",
    "moving_average_28": "MA 28",
    "lightgbm_l1": "LGBM L1",
    "lightgbm_tweedie": "LGBM Tweedie",
}


def plot_forecast_vs_actual(daily: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.plot(daily["target_date"], daily["actual"], color="#222222", lw=1.7, label="Actual")
    ax.plot(
        daily["target_date"],
        daily["moving_average_28"],
        color="#8a8f98",
        lw=1.3,
        ls="--",
        label="MA 28",
    )
    ax.plot(
        daily["target_date"],
        daily["lightgbm_l1"],
        color="#9bb4c8",
        lw=1.3,
        label="LightGBM L1",
    )
    ax.plot(
        daily["target_date"],
        daily["lightgbm_tweedie"],
        color="#1f4e79",
        lw=1.7,
        label="LightGBM Tweedie",
    )
    ax.set_title(title)
    ax.set_ylabel("Units")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#e6e6e6")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_wape_by_fold(metrics: pd.DataFrame, path: Path) -> None:
    use = metrics.loc[metrics["fold"] != "pooled"].copy()
    models = ["moving_average_28", "lightgbm_l1", "lightgbm_tweedie"]
    colors = {
        "moving_average_28": "#8a8f98",
        "lightgbm_l1": "#9bb4c8",
        "lightgbm_tweedie": "#1f4e79",
    }
    origins = list(dict.fromkeys(use["origin_date"].tolist()))
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    x = range(len(origins))
    width = 0.25
    for i, model in enumerate(models):
        part = use.loc[use["model"] == model].set_index("origin_date").loc[origins]
        offset = (i - 1) * width
        ax.bar(
            [v + offset for v in x],
            part["wape"],
            width=width,
            color=colors[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(origins, rotation=45, ha="right")
    ax.set_ylabel("WAPE")
    ax.set_title("Daily WAPE by backtest fold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e6e6")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_order_wape_by_fold(by_origin: pd.DataFrame, path: Path) -> None:
    """28-day order-quantity WAPE at each origin."""
    models = ["moving_average_28", "lightgbm_l1", "lightgbm_tweedie"]
    colors = {
        "moving_average_28": "#8a8f98",
        "lightgbm_l1": "#9bb4c8",
        "lightgbm_tweedie": "#1f4e79",
    }
    origins = list(dict.fromkeys(by_origin["origin_date"].tolist()))
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    x = range(len(origins))
    width = 0.25
    for i, model in enumerate(models):
        part = by_origin.loc[by_origin["model"] == model].set_index("origin_date").loc[origins]
        offset = (i - 1) * width
        ax.bar(
            [v + offset for v in x],
            part["wape"],
            width=width,
            color=colors[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(origins, rotation=45, ha="right")
    ax.set_ylabel("WAPE")
    ax.set_title("28-day order-quantity WAPE by origin")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e6e6")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_decision_summary(decision: pd.DataFrame, path: Path) -> None:
    """Bias at the daily and order grains, and order-quantity WAPE."""
    models = [name for name in DAILY_MODELS if name in set(decision["model"])]
    labels = [_SHORT[name] for name in models]
    bias_daily = _aligned(decision, "daily", "bias", models)
    bias_order = _aligned(decision, "order_total", "bias", models)
    wape_order = _aligned(decision, "order_total", "wape", models)
    x = range(len(models))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    axes[0].bar([i - width / 2 for i in x], bias_daily, width=width, color="#9bb4c8", label="Daily")
    axes[0].bar([i + width / 2 for i in x], bias_order, width=width, color="#1f4e79", label="28-day total")
    axes[0].axhline(0.0, color="#222222", lw=0.8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Bias")
    axes[0].set_title("Bias by grain")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#e6e6e6")

    colors = ["#1f4e79" if name == "lightgbm_tweedie" else "#8a8f98" for name in models]
    axes[1].bar(list(x), wape_order, color=colors)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("WAPE")
    axes[1].set_title("28-day order quantity")
    axes[1].grid(axis="y", color="#e6e6e6")

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _aligned(decision: pd.DataFrame, grain: str, column: str, models: list[str]) -> list[float]:
    part = decision.loc[decision["grain"] == grain].set_index("model")
    return [float(part.loc[name, column]) for name in models]
