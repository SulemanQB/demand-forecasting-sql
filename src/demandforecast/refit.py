"""A second LightGBM fit on the pre-cutoff top SKUs.

The moving average is still the per-SKU baseline. LightGBM is trained only on
the listed SKUs, so this is not a filter of the top-40 model.
"""

import duckdb
import numpy as np
import pandas as pd

from demandforecast.baselines import moving_average_forecast
from demandforecast.boosting import fit_lightgbm, predict_lightgbm, prepare_frame
from demandforecast.config import FEATURE_COLUMNS, TARGET_COLUMN, Config
from demandforecast.paths import duckdb_path

REFIT_EVALUATION = "refit_pre_cutoff_top_n"
REFIT_NOTE = (
    "LightGBM refit on the SKUs with the most units before the first origin. "
    "Not a slice of the top-40 model."
)
SLICE_EVALUATION = "slice_of_top40_not_a_refit"
SLICE_NOTE = "Same SKUs scored from the top-40 model. The model was not refit."


def load_pre_cutoff_top_skus(n: int) -> pd.DataFrame:
    if n < 1:
        raise ValueError("n must be positive")
    db_path = duckdb_path()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        frame = con.execute(
            """
            SELECT sku, total_units, sale_days
            FROM selected_skus
            ORDER BY total_units DESC, sku
            LIMIT ?
            """,
            [n],
        ).fetchdf()
    finally:
        con.close()
    if len(frame) != n:
        raise RuntimeError(f"Expected {n} pre-cutoff SKUs, found {len(frame)}")
    frame["sku"] = frame["sku"].astype(str)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame.insert(1, "evaluation", "pre_cutoff_rank")
    frame.insert(
        2,
        "note",
        "Ranked by units on dates strictly before the first origin.",
    )
    return frame


def refit_predictions(
    supervised: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    origins: list[pd.Timestamp],
    cfg: Config,
    skus: list[str],
) -> pd.DataFrame:
    skus = [str(sku) for sku in skus]
    subset = supervised.loc[supervised["sku"].astype(str).isin(skus)].copy()
    subset = prepare_frame(
        subset,
        FEATURE_COLUMNS,
        skus,
        ["asof_date", "target_date", "naive_w7_date", "same_day_last_year_date"],
    )
    frames: list[pd.DataFrame] = []
    for fold, origin in enumerate(origins, start=1):
        origin = pd.Timestamp(origin)
        val_start = origin - pd.Timedelta(days=cfg.horizon)
        train = subset.loc[subset["target_date"] <= val_start].sort_values(
            ["sku", "target_date", "horizon"]
        )
        val = subset.loc[
            (subset["target_date"] > val_start) & (subset["target_date"] <= origin)
        ].sort_values(["sku", "target_date", "horizon"])
        test = (
            subset.loc[subset["asof_date"] == origin]
            .sort_values(["sku", "horizon"])
            .reset_index(drop=True)
        )
        if len(test) != len(skus) * cfg.horizon:
            raise RuntimeError(
                f"Refit fold {fold} has {len(test)} test rows, expected {len(skus) * cfg.horizon}"
            )
        if len(train) < 1000:
            raise RuntimeError(f"Refit fold {fold} training set is too small ({len(train)} rows)")
        l1_booster, l1_iter = fit_lightgbm(train, val, cfg, "regression_l1")
        tw_booster, tw_iter = fit_lightgbm(train, val, cfg, "tweedie")
        l1_pred = predict_lightgbm(l1_booster, test, l1_iter)
        tw_pred = predict_lightgbm(tw_booster, test, tw_iter)
        if list(tw_booster.feature_name()) != FEATURE_COLUMNS:
            raise RuntimeError("Refit feature names differ from the feature list")
        scored = test.copy()
        scored["sku"] = scored["sku"].astype(str)
        scored = _attach_moving_average(scored, histories, origin, skus, cfg)
        frames.append(
            pd.DataFrame(
                {
                    "evaluation": REFIT_EVALUATION,
                    "fold": fold,
                    "origin_date": origin.date().isoformat(),
                    "sku": scored["sku"].to_numpy(),
                    "target_date": pd.to_datetime(scored["target_date"]).dt.strftime("%Y-%m-%d"),
                    "horizon": scored["horizon"].astype(int).to_numpy(),
                    "actual": scored[TARGET_COLUMN].to_numpy(dtype=float),
                    "moving_average_28": scored["py_ma"].to_numpy(dtype=float),
                    "lightgbm_l1": l1_pred,
                    "lightgbm_tweedie": tw_pred,
                    "best_iteration_l1": int(l1_iter),
                    "best_iteration_tweedie": int(tw_iter),
                }
            )
        )
        print(
            f"refit fold {fold:02d} origin {origin.date()} train {len(train):,} "
            f"trees_l1 {l1_iter} trees_tweedie {tw_iter}",
            flush=True,
        )
    return pd.concat(frames, ignore_index=True)


def _attach_moving_average(scored, histories, origin, skus, cfg: Config) -> pd.DataFrame:
    rows = []
    for sku in skus:
        values = histories[sku].loc[histories[sku]["date"] <= origin, "units"].to_numpy(dtype=float)
        average = moving_average_forecast(values, cfg.horizon, cfg.moving_average_window)
        for step in range(cfg.horizon):
            rows.append({"sku": sku, "horizon": step + 1, "py_ma": float(average[step])})
    attached = scored.merge(pd.DataFrame(rows), on=["sku", "horizon"], how="left", validate="one_to_one")
    if not np.allclose(
        attached["roll_mean_28"].to_numpy(dtype=float),
        attached["py_ma"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=False,
    ):
        raise RuntimeError("Refit moving average does not match the SQL feature")
    return attached
