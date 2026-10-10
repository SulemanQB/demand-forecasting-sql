"""Forecast baselines that use only history through the origin."""

import numpy as np


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Forecast y[T+h] with y[T+h-season*k], k chosen so the source date is on or before T.

    `history` is chronological and ends at the origin T. Its last element is y[T].
    """
    if season < 1:
        raise ValueError("season must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = np.asarray(history, dtype=float)
    n = int(values.shape[0])
    out = np.empty(horizon, dtype=float)
    for h in range(1, horizon + 1):
        k = int(np.ceil(h / season))
        idx = n - 1 + h - season * k
        if idx < 0 or idx >= n:
            raise ValueError(
                f"History length {n} is too short for horizon {h} and season {season}"
            )
        out[h - 1] = values[idx]
    return out


def moving_average_forecast(history: np.ndarray, horizon: int, window: int) -> np.ndarray:
    """Flat forecast equal to the mean of the last `window` observations, inclusive of T."""
    if window < 1:
        raise ValueError("window must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = np.asarray(history, dtype=float)
    if values.shape[0] < window:
        raise ValueError(f"History length {values.shape[0]} is shorter than window {window}")
    level = float(np.mean(values[-window:]))
    return np.repeat(level, horizon)
