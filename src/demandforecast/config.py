"""Fixed project settings. The backtest uses this configuration as specified."""

from dataclasses import dataclass


DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
DATASET_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
# The 2009-2010 sheet and the 2010-2011 sheet both contain 1-9 December 2010.
SHEET_OVERLAP_CUT = "2010-12-01"
SHEET_EARLY = "Year 2009-2010"
SHEET_LATE = "Year 2010-2011"

FEATURE_COLUMNS = [
    "units_asof",
    "units_lag_7",
    "units_lag_14",
    "units_lag_21",
    "units_lag_28",
    "units_lag_364",
    "roll_mean_7",
    "roll_mean_28",
    "roll_std_7",
    "roll_std_28",
    "same_weekday_mean_4",
    "seasonal_naive_w7",
    "units_same_day_last_year",
    "level_ratio_7_28",
    "price_asof",
    "price_ratio",
    "history_days",
    "horizon",
    "target_dow",
    "target_month",
    "target_day",
    "target_week",
    "target_is_weekend",
    "days_to_christmas",
    "sku",
]
CATEGORICAL_FEATURES = ["sku"]
TARGET_COLUMN = "target_units"

MODEL_NAMES = [
    "seasonal_naive_w7",
    "seasonal_naive_y364",
    "moving_average_28",
    "lightgbm",
]
MODEL_LABELS = {
    "seasonal_naive_w7": "weekly seasonal naive",
    "seasonal_naive_y364": "yearly seasonal naive",
    "moving_average_28": "28-day moving average",
    "lightgbm": "LightGBM",
}


@dataclass(frozen=True)
class Config:
    country: str = "United Kingdom"
    top_n: int = 40
    min_sale_days: int = 180
    horizon: int = 28
    # Full 28-day rolling window and lag 28 require 29 daily rows (current + 28 back).
    min_history_days: int = 29
    cancel_match_days: int = 21
    fold_step_days: int = 28
    season_week: int = 7
    season_year: int = 364
    # Days of as-of history with a defined yearly lag required inside the training cut.
    lag_warmup_days: int = 56
    moving_average_window: int = 28
    lgb_num_boost_round: int = 400
    lgb_early_stopping_rounds: int = 40
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 63
    lgb_min_data_in_leaf: int = 40
    lgb_feature_fraction: float = 0.9
    lgb_bagging_fraction: float = 0.8
    lgb_lambda_l2: float = 1.0
    seed: int = 42
