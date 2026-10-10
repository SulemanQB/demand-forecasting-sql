"""Rolling-origin schedule."""

import pandas as pd


def origin_dates(
    min_date,
    max_date,
    *,
    horizon: int,
    step: int,
    yearly_lag_days: int,
    warmup_days: int,
) -> list[pd.Timestamp]:
    """Origins with a full horizon and a training cut that has seen the yearly lag.

    The first date with a defined lag of `yearly_lag_days` on a complete daily spine
    is `min_date + yearly_lag_days` (history length yearly_lag_days + 1). Training for
    origin O uses targets on or before O - horizon, so the latest training as-of date
    is O - horizon - 1. The first origin is set so that at least `warmup_days` as-of
    dates in that training cut have the yearly lag.
    """
    if horizon < 1 or step < 1 or warmup_days < 1 or yearly_lag_days < 1:
        raise ValueError("horizon, step, warmup_days, and yearly_lag_days must be positive")
    first_lag = pd.Timestamp(min_date) + pd.Timedelta(days=yearly_lag_days)
    first = first_lag + pd.Timedelta(days=warmup_days + horizon)
    last = pd.Timestamp(max_date) - pd.Timedelta(days=horizon)
    origins: list[pd.Timestamp] = []
    day = first
    while day <= last:
        origins.append(pd.Timestamp(day))
        day = day + pd.Timedelta(days=step)
    return origins
