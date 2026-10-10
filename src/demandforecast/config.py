"""Fixed project settings. Nothing here is tuned on the backtest."""

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

WEEKLY_FEATURE_COLUMNS = [
    "units_asof",
    "units_lag_1",
    "units_lag_4",
    "units_lag_52",
    "roll_mean_4",
    "roll_std_4",
    "seasonal_naive_w1",
    "units_same_week_last_year",
    "price_asof",
    "history_weeks",
    "horizon",
    "target_week",
    "target_month",
    "weeks_to_christmas",
    "sku",
]

# Scored on every SKU-day. Intermittent baselines are not in this list.
DAILY_MODELS = [
    "seasonal_naive_w7",
    "seasonal_naive_y364",
    "moving_average_28",
    "lightgbm_l1",
    "lightgbm_tweedie",
]
MODEL_NAMES = DAILY_MODELS

WEEKLY_MODELS = [
    "seasonal_naive_w1",
    "seasonal_naive_w52",
    "moving_average_4",
    "lightgbm_tweedie",
]

# Croston-family forecasts are scored only on SKU-origins classed as lumpy.
INTERMITTENT_MODELS = ["croston", "sba", "tsb"]
LUMPY_COMPARE_MODELS = [
    "croston",
    "sba",
    "tsb",
    "moving_average_28",
    "lightgbm_tweedie",
    "lightgbm_l1",
]

# Decision model, median-model contrast, and the flat baseline a buyer would use.
INVENTORY_MODELS = [
    "lightgbm_tweedie",
    "lightgbm_l1",
    "moving_average_28",
]

MODEL_LABELS = {
    "seasonal_naive_w7": "weekly seasonal naive",
    "seasonal_naive_y364": "yearly seasonal naive",
    "moving_average_28": "28-day moving average",
    "lightgbm_l1": "LightGBM L1",
    "lightgbm_tweedie": "LightGBM Tweedie",
    "croston": "Croston",
    "sba": "SBA",
    "tsb": "TSB",
    "seasonal_naive_w1": "1-week seasonal naive",
    "seasonal_naive_w52": "52-week seasonal naive",
    "moving_average_4": "4-week moving average",
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
    # Library default. The Tweedie objective was chosen after an earlier L1 run; this power was not searched.
    tweedie_variance_power: float = 1.5
    # Shared smoothing constant for Croston, SBA, and both TSB updates.
    croston_alpha: float = 0.1
    # Syntetos-Boylan cutoffs, fixed before looking at test error.
    adi_threshold: float = 1.32
    cv2_threshold: float = 0.49
    # One stockout unit costs this many holding units. Holding uses the origin price.
    stockout_to_holding: float = 3.0
    weekly_horizon: int = 4
    weekly_step: int = 4
    weekly_season_year: int = 52
    weekly_warmup_weeks: int = 8
    weekly_min_history: int = 5
    weekly_ma_window: int = 4
    # Origin-cluster interval for the order-quantity WAPE gap. Fixed evaluation settings.
    bootstrap_draws: int = 10000
    # A SKU-origin is active when it has a sale in (origin - window, origin]. Not a model feature.
    active_window_days: int = 90
    # Second LightGBM fit on this many pre-cutoff SKUs. Not a filter of the top-40 fit.
    top_refit_n: int = 20
