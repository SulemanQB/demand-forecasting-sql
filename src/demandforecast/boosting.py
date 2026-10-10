"""Global LightGBM models on a SQL feature table."""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from demandforecast.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN, Config


def lightgbm_params(cfg: Config, objective: str) -> dict:
    if objective == "regression_l1":
        metric = "l1"
        extra: dict = {}
    elif objective == "tweedie":
        # RMSE keeps early stopping aimed at the mean. The L1 model keeps its own metric.
        metric = "rmse"
        extra = {"tweedie_variance_power": cfg.tweedie_variance_power}
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    return {
        "objective": objective,
        "metric": metric,
        "learning_rate": cfg.lgb_learning_rate,
        "num_leaves": cfg.lgb_num_leaves,
        "min_data_in_leaf": cfg.lgb_min_data_in_leaf,
        "feature_fraction": cfg.lgb_feature_fraction,
        "bagging_fraction": cfg.lgb_bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": cfg.lgb_lambda_l2,
        "seed": cfg.seed,
        "bagging_seed": cfg.seed,
        "feature_fraction_seed": cfg.seed,
        "data_random_seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "num_threads": min(4, os.cpu_count() or 1),
        "device_type": "cpu",
        **extra,
    }


def prepare_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    skus: list[str],
    date_columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    for column in date_columns:
        out[column] = pd.to_datetime(out[column]).dt.normalize()
    out["sku"] = pd.Categorical(out["sku"].astype(str), categories=skus)
    numeric = [column for column in feature_columns if column != "sku"] + [TARGET_COLUMN]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


def fit_lightgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: Config,
    objective: str,
    feature_columns: list[str] | None = None,
) -> tuple[lgb.Booster, int]:
    if train.empty or val.empty:
        raise RuntimeError("LightGBM training or validation slice is empty")
    columns = list(feature_columns or FEATURE_COLUMNS)
    if objective == "tweedie":
        labels = train[TARGET_COLUMN].to_numpy(dtype=float)
        if np.nanmin(labels) < 0.0 or np.nanmin(val[TARGET_COLUMN].to_numpy(dtype=float)) < 0.0:
            raise RuntimeError("Tweedie labels must be non-negative")
    params = lightgbm_params(cfg, objective)
    dtrain = lgb.Dataset(
        train[columns],
        label=train[TARGET_COLUMN].to_numpy(dtype=float),
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        val[columns],
        label=val[TARGET_COLUMN].to_numpy(dtype=float),
        categorical_feature=CATEGORICAL_FEATURES,
        reference=dtrain,
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.lgb_num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(cfg.lgb_early_stopping_rounds, verbose=False)],
    )
    n_iter = int(booster.best_iteration or 0)
    if n_iter <= 0:
        n_iter = int(booster.current_iteration())
    return booster, n_iter


def predict_lightgbm(
    booster: lgb.Booster,
    frame: pd.DataFrame,
    n_iter: int,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    columns = list(feature_columns or FEATURE_COLUMNS)
    pred = booster.predict(frame[columns], num_iteration=n_iter)
    return np.maximum(np.asarray(pred, dtype=float), 0.0)
