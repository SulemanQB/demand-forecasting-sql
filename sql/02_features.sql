-- As-of features and a direct multi-horizon supervised table.
-- For each target date the feature windows close on asof_date = target_date - horizon.
-- Nothing on the target date, and nothing after the origin, enters the feature vector.
-- Calendar fields of the target date are known when the forecast is issued.
-- Price is the last realized unit price as of the origin, not the price paid on the target day.
-- seasonal_naive_w7 and units_same_day_last_year are baseline forecasts known at the origin
-- (season length 7, and 364 days so the weekday is preserved).

CREATE OR REPLACE TABLE asof_features AS
WITH priced AS (
    SELECT
        sku,
        date,
        units,
        avg_price,
        last_value(avg_price IGNORE NULLS) OVER (
            PARTITION BY sku
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS price_filled
    FROM panel
)
SELECT
    sku,
    date AS asof_date,
    units AS units_asof,
    lag(units, 7) OVER w AS units_lag_7,
    lag(units, 14) OVER w AS units_lag_14,
    lag(units, 21) OVER w AS units_lag_21,
    lag(units, 28) OVER w AS units_lag_28,
    lag(units, 364) OVER w AS units_lag_364,
    avg(units) OVER (
        PARTITION BY sku ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS roll_mean_7,
    avg(units) OVER (
        PARTITION BY sku ORDER BY date
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ) AS roll_mean_28,
    stddev_samp(units) OVER (
        PARTITION BY sku ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS roll_std_7,
    stddev_samp(units) OVER (
        PARTITION BY sku ORDER BY date
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ) AS roll_std_28,
    price_filled AS price_asof,
    avg(price_filled) OVER (
        PARTITION BY sku ORDER BY date
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ) AS price_roll_mean_28,
    count(*) OVER (
        PARTITION BY sku ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )::INTEGER AS history_days
FROM priced
WINDOW w AS (PARTITION BY sku ORDER BY date);

CREATE OR REPLACE TABLE supervised AS
WITH targets AS (
    SELECT
        f.sku,
        f.asof_date,
        d.date AS target_date,
        date_diff('day', f.asof_date, d.date)::INTEGER AS horizon,
        f.units_asof,
        f.units_lag_7,
        f.units_lag_14,
        f.units_lag_21,
        f.units_lag_28,
        f.units_lag_364,
        f.roll_mean_7,
        f.roll_mean_28,
        f.roll_std_7,
        f.roll_std_28,
        f.price_asof,
        f.price_roll_mean_28,
        f.history_days,
        (
            coalesce(f.units_lag_7, 0) + coalesce(f.units_lag_14, 0)
            + coalesce(f.units_lag_21, 0) + coalesce(f.units_lag_28, 0)
        ) / nullif(
            (f.units_lag_7 IS NOT NULL)::INTEGER
            + (f.units_lag_14 IS NOT NULL)::INTEGER
            + (f.units_lag_21 IS NOT NULL)::INTEGER
            + (f.units_lag_28 IS NOT NULL)::INTEGER,
            0
        ) AS same_weekday_mean_4,
        f.roll_mean_7 / nullif(f.roll_mean_28, 0) AS level_ratio_7_28,
        f.price_asof / nullif(f.price_roll_mean_28, 0) AS price_ratio
    FROM asof_features f
    JOIN config cfg ON true
    JOIN calendar d
        ON d.date > f.asof_date
       AND d.date <= f.asof_date + cfg.horizon
    WHERE f.history_days >= cfg.min_history_days
)
SELECT
    t.sku,
    t.asof_date,
    t.target_date,
    t.horizon,
    t.units_asof,
    t.units_lag_7,
    t.units_lag_14,
    t.units_lag_21,
    t.units_lag_28,
    t.units_lag_364,
    t.roll_mean_7,
    t.roll_mean_28,
    t.roll_std_7,
    t.roll_std_28,
    t.same_weekday_mean_4,
    nw.units AS seasonal_naive_w7,
    ny.units AS units_same_day_last_year,
    t.level_ratio_7_28,
    t.price_asof,
    t.price_ratio,
    t.history_days,
    cast(isodow(t.target_date) AS INTEGER) AS target_dow,
    cast(date_part('month', t.target_date) AS INTEGER) AS target_month,
    cast(date_part('day', t.target_date) AS INTEGER) AS target_day,
    cast(date_part('week', t.target_date) AS INTEGER) AS target_week,
    CASE WHEN isodow(t.target_date) >= 6 THEN 1 ELSE 0 END AS target_is_weekend,
    date_diff(
        'day',
        t.target_date,
        make_date(cast(date_part('year', t.target_date) AS INTEGER), 12, 25)
    )::INTEGER AS days_to_christmas,
    tgt.units AS target_units,
    t.target_date - cast(7 * ceil(t.horizon / 7.0) AS INTEGER) AS naive_w7_date,
    t.target_date - 364 AS same_day_last_year_date
FROM targets t
JOIN panel tgt
    ON tgt.sku = t.sku
   AND tgt.date = t.target_date
LEFT JOIN panel nw
    ON nw.sku = t.sku
   AND nw.date = t.target_date - cast(7 * ceil(t.horizon / 7.0) AS INTEGER)
LEFT JOIN panel ny
    ON ny.sku = t.sku
   AND ny.date = t.target_date - 364;
