"""Backtest figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_vs_actual(daily: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.plot(daily["target_date"], daily["actual"], color="#222222", lw=1.7, label="Actual")
    ax.plot(
        daily["target_date"],
        daily["seasonal_naive_w7"],
        color="#8a8f98",
        lw=1.3,
        ls="--",
        label="Seasonal naive (7)",
    )
    ax.plot(
        daily["target_date"],
        daily["seasonal_naive_y364"],
        color="#c47b2b",
        lw=1.3,
        ls="--",
        label="Seasonal naive (364)",
    )
    ax.plot(
        daily["target_date"],
        daily["lightgbm"],
        color="#1f4e79",
        lw=1.7,
        label="LightGBM",
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
    models = ["seasonal_naive_w7", "seasonal_naive_y364", "lightgbm"]
    labels = {
        "seasonal_naive_w7": "Seasonal naive (7)",
        "seasonal_naive_y364": "Seasonal naive (364)",
        "lightgbm": "LightGBM",
    }
    colors = {
        "seasonal_naive_w7": "#8a8f98",
        "seasonal_naive_y364": "#c47b2b",
        "lightgbm": "#1f4e79",
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
            part["wape"] * 100.0,
            width=width,
            color=colors[model],
            label=labels[model],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(origins, rotation=45, ha="right")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("WAPE by backtest fold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e6e6")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
