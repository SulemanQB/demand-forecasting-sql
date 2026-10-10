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


def croston_level(history: np.ndarray, alpha: float) -> float:
    """Croston size/interval level. Updates only on positive demand.

    The first interval is the index of the first demand plus one. Trailing zeros
    do not update either smoother. Returns 0 when the history has no demand.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    values = np.asarray(history, dtype=float)
    idx = np.flatnonzero(values > 0.0)
    if idx.size == 0:
        return 0.0
    size = float(values[idx[0]])
    interval = float(idx[0] + 1)
    for prev, cur in zip(idx[:-1], idx[1:]):
        gap = float(cur - prev)
        size = size + alpha * (float(values[cur]) - size)
        interval = interval + alpha * (gap - interval)
    if interval <= 0.0:
        raise RuntimeError("Croston interval collapsed to a non-positive value")
    return size / interval


def croston_forecast(history: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """Flat Croston forecast over the horizon."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    level = croston_level(history, alpha)
    return np.repeat(level, horizon)


def sba_forecast(history: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """Syntetos-Boylan approximation: Croston times (1 - alpha/2), flat over the horizon."""
    level = croston_level(history, alpha) * (1.0 - alpha / 2.0)
    return np.repeat(level, horizon)


def tsb_forecast(history: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """Teunter-Syntetos-Babai forecast. Probability updates every period after the first demand.

    Size updates only when demand is positive. Both smoothers use `alpha`.
    Periods before the first demand are ignored. The forecast is probability times size,
    repeated across the horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    values = np.asarray(history, dtype=float)
    idx = np.flatnonzero(values > 0.0)
    if idx.size == 0:
        return np.zeros(horizon, dtype=float)
    first = int(idx[0])
    size = float(values[first])
    probability = 1.0
    for t in range(first + 1, int(values.shape[0])):
        demand = float(values[t])
        if demand > 0.0:
            probability = probability + alpha * (1.0 - probability)
            size = size + alpha * (demand - size)
        else:
            probability = probability + alpha * (0.0 - probability)
    return np.repeat(probability * size, horizon)


def demand_pattern(
    history: np.ndarray,
    adi_threshold: float,
    cv2_threshold: float,
) -> tuple[str, float, float]:
    """Syntetos-Boylan class from the first positive day through the end of `history`.

    Leading zeros are dropped so a SKU that starts mid-panel is not called intermittent
    for the weeks before it was sold. ADI is periods per positive demand. CV² is the
    sample variance of positive sizes divided by the square of their mean. A single
    positive size has CV² 0.
    """
    values = np.asarray(history, dtype=float)
    idx = np.flatnonzero(values > 0.0)
    if idx.size == 0:
        return "no_demand", float("inf"), float("nan")
    observed = values[int(idx[0]) :]
    positive = observed[observed > 0.0]
    adi = float(observed.shape[0] / positive.shape[0])
    if positive.shape[0] == 1 or float(np.mean(positive)) == 0.0:
        cv2 = 0.0
    else:
        cv2 = float(np.var(positive, ddof=1) / (float(np.mean(positive)) ** 2))
    lumpy_gap = adi > adi_threshold
    lumpy_size = cv2 > cv2_threshold
    if lumpy_gap and lumpy_size:
        name = "lumpy"
    elif lumpy_gap:
        name = "intermittent"
    elif lumpy_size:
        name = "erratic"
    else:
        name = "smooth"
    return name, adi, cv2


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
