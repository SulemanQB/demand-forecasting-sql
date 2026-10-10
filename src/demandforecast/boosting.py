"""Global LightGBM model on the SQL feature table."""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from demandforecast.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN, Config


def lightgbm_params(cfg: Config) -> dict:
    return {
        "objective": "regression_l1",
        "metric": "l1",
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
    }


def fit_lightgbm(train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> tuple[lgb.Booster, int]:
    if train.empty or val.empty:
        raise RuntimeError("LightGBM training or validation slice is empty")
    params = lightgbm_params(cfg)
    dtrain = lgb.Dataset(
        train[FEATURE_COLUMNS],
        label=train[TARGET_COLUMN].to_numpy(dtype=float),
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        val[FEATURE_COLUMNS],
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


def predict_lightgbm(booster: lgb.Booster, frame: pd.DataFrame, n_iter: int) -> np.ndarray:
    pred = booster.predict(frame[FEATURE_COLUMNS], num_iteration=n_iter)
    return np.maximum(np.asarray(pred, dtype=float), 0.0)
