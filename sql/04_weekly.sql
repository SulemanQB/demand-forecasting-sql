-- ISO-week panel on the same selected SKUs.
-- date_trunc('week') is Monday. The first and last weeks can be partial because
-- the daily panel does not start or end on a Monday; week totals still equal the
-- sum of the daily rows that fall in that week.
-- Horizon is weekly_horizon steps of 7 days. The 1-week naive repeats the origin
-- week (season 1). The 52-week naive reads the target week minus 364 days.

CREATE OR REPLACE TABLE weekly_panel AS
WITH priced AS (
    SELECT
        sku,
        date,
        units,
        last_value(avg_price IGNORE NULLS) OVER (
            PARTITION BY sku
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS price_filled
    FROM panel
),
weeks AS (
    SELECT
        sku,
        cast(date_trunc('week', date) AS DATE) AS week_start,
        sum(units)::DOUBLE AS units,
        arg_max(price_filled, date)::DOUBLE AS avg_price
    FROM priced
    GROUP BY sku, cast(date_trunc('week', date) AS DATE)
),
spine AS (
    SELECT cast(unnest(generate_series(
        (SELECT min(week_start) FROM weeks),
        (SELECT max(week_start) FROM weeks),
        INTERVAL 7 DAY
    )) AS DATE) AS week_start
)
SELECT
    s.sku,
    c.week_start,
    coalesce(w.units, 0)::DOUBLE AS units,
    w.avg_price
FROM selected_skus s
CROSS JOIN spine c
LEFT JOIN weeks w
    ON w.sku = s.sku
   AND w.week_start = c.week_start;

CREATE OR REPLACE TABLE weekly_calendar AS
SELECT DISTINCT week_start
FROM weekly_panel
ORDER BY week_start;

CREATE OR REPLACE TABLE weekly_asof AS
WITH filled AS (
    SELECT
        sku,
        week_start,
        units,
        avg_price,
        last_value(avg_price IGNORE NULLS) OVER (
            PARTITION BY sku
            ORDER BY week_start
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS price_filled
    FROM weekly_panel
)
SELECT
    sku,
    week_start AS asof_date,
    units AS units_asof,
    lag(units, 1) OVER w AS units_lag_1,
    lag(units, 4) OVER w AS units_lag_4,
    lag(units, 52) OVER w AS units_lag_52,
    avg(units) OVER (
        PARTITION BY sku ORDER BY week_start
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS roll_mean_4,
    stddev_samp(units) OVER (
        PARTITION BY sku ORDER BY week_start
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS roll_std_4,
    price_filled AS price_asof,
    count(*) OVER (
        PARTITION BY sku ORDER BY week_start
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )::INTEGER AS history_weeks
FROM filled
WINDOW w AS (PARTITION BY sku ORDER BY week_start);

CREATE OR REPLACE TABLE weekly_supervised AS
WITH targets AS (
    SELECT
        f.sku,
        f.asof_date,
        d.week_start AS target_date,
        cast(date_diff('day', f.asof_date, d.week_start) / 7 AS INTEGER) AS horizon,
        f.units_asof,
        f.units_lag_1,
        f.units_lag_4,
        f.units_lag_52,
        f.roll_mean_4,
        f.roll_std_4,
        f.price_asof,
        f.history_weeks
    FROM weekly_asof f
    JOIN config cfg ON true
    JOIN weekly_calendar d
        ON d.week_start > f.asof_date
       AND d.week_start <= f.asof_date + cfg.weekly_horizon * 7
    WHERE f.history_weeks >= cfg.weekly_min_history
)
SELECT
    t.sku,
    t.asof_date,
    t.target_date,
    t.horizon,
    t.units_asof,
    t.units_lag_1,
    t.units_lag_4,
    t.units_lag_52,
    t.roll_mean_4,
    t.roll_std_4,
    nw.units AS seasonal_naive_w1,
    ny.units AS units_same_week_last_year,
    t.price_asof,
    t.history_weeks,
    cast(date_part('week', t.target_date) AS INTEGER) AS target_week,
    cast(date_part('month', t.target_date) AS INTEGER) AS target_month,
    cast(
        date_diff(
            'day',
            t.target_date,
            make_date(cast(date_part('year', t.target_date) AS INTEGER), 12, 25)
        ) / 7
        AS INTEGER
    ) AS weeks_to_christmas,
    tgt.units AS target_units,
    t.asof_date AS naive_w1_date,
    t.target_date - 364 AS same_week_last_year_date
FROM targets t
JOIN weekly_panel tgt
    ON tgt.sku = t.sku
   AND tgt.week_start = t.target_date
LEFT JOIN weekly_panel nw
    ON nw.sku = t.sku
   AND nw.week_start = t.asof_date
LEFT JOIN weekly_panel ny
    ON ny.sku = t.sku
   AND ny.week_start = t.target_date - 364;

CREATE OR REPLACE TABLE weekly_duplicate_keys AS
SELECT
    sku,
    week_start,
    count(*) OVER (PARTITION BY sku, week_start) AS n_rows
FROM weekly_panel
QUALIFY count(*) OVER (PARTITION BY sku, week_start) > 1;
