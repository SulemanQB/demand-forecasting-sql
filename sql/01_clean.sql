-- Clean UK product sales and build a daily SKU panel.
-- A return cancels at most one sale, and a sale is cancelled by at most one return.
-- The match is greedy: each return takes its closest sale (then the lowest sale id),
-- and a sale claimed by several returns keeps the closest return. It is not a
-- minimum-cost assignment, so a return can be left unmatched when its sale was
-- taken by a nearer return.
-- The panel is one row per selected SKU per calendar day. Later window functions use
-- ROWS frames and assume that grain (no missing dates, no duplicate keys).
-- SKU rank uses only dates strictly before the first backtest origin. That origin is
-- min(sale date) + yearly_lag_days + lag_warmup_days + horizon, the same expression
-- as origin_dates in the Python schedule.

CREATE OR REPLACE TABLE scoped_lines AS
WITH typed AS (
    SELECT
        upper(trim(cast(invoice AS VARCHAR))) AS invoice,
        upper(trim(cast(stock_code AS VARCHAR))) AS stock_code,
        nullif(trim(cast(description AS VARCHAR)), '') AS description,
        cast(quantity AS BIGINT) AS quantity,
        cast(invoice_date AS TIMESTAMP) AS invoice_ts,
        cast(price AS DOUBLE) AS price,
        cast(customer_id AS BIGINT) AS customer_id,
        trim(cast(country AS VARCHAR)) AS country
    FROM raw_invoices
)
SELECT *
FROM typed
WHERE country = (SELECT country FROM config)
  AND price > 0
  AND quantity <> 0
  AND stock_code NOT IN ('', 'NAN', 'NONE', 'NULL')
  AND regexp_matches(stock_code, '^[0-9]');

CREATE OR REPLACE TABLE sales_lines AS
SELECT
    invoice,
    stock_code,
    description,
    quantity,
    invoice_ts,
    price,
    customer_id,
    country,
    row_number() OVER (
        ORDER BY invoice_ts, invoice, stock_code, quantity
    ) AS sale_id
FROM scoped_lines
WHERE quantity > 0
  AND left(invoice, 1) <> 'C';

CREATE OR REPLACE TABLE return_lines AS
SELECT
    stock_code,
    abs(quantity) AS quantity,
    customer_id,
    cast(invoice_ts AS DATE) AS return_date,
    row_number() OVER (
        ORDER BY invoice_ts, invoice, stock_code, quantity
    ) AS return_id
FROM scoped_lines
WHERE quantity < 0
  AND customer_id IS NOT NULL;

CREATE OR REPLACE TABLE cancel_matches AS
WITH candidates AS (
    SELECT
        s.sale_id,
        r.return_id,
        s.stock_code,
        abs(date_diff('day', cast(s.invoice_ts AS DATE), r.return_date)) AS day_gap
    FROM sales_lines s
    JOIN return_lines r
        ON s.stock_code = r.stock_code
       AND s.quantity = r.quantity
       AND s.customer_id = r.customer_id
    CROSS JOIN config cfg
    WHERE abs(date_diff('day', cast(s.invoice_ts AS DATE), r.return_date))
        <= cfg.cancel_match_days
),
closest_sale AS (
    SELECT sale_id, return_id, stock_code, day_gap
    FROM candidates
    QUALIFY row_number() OVER (
        PARTITION BY return_id
        ORDER BY day_gap, sale_id
    ) = 1
)
SELECT sale_id, return_id, stock_code, day_gap
FROM closest_sale
QUALIFY row_number() OVER (
    PARTITION BY sale_id
    ORDER BY day_gap, return_id
) = 1;

CREATE OR REPLACE TABLE clean_lines AS
SELECT
    s.invoice,
    s.stock_code,
    s.description,
    s.quantity,
    s.invoice_ts,
    s.price,
    s.customer_id,
    s.country
FROM sales_lines s
ANTI JOIN cancel_matches m
    ON s.sale_id = m.sale_id;

CREATE OR REPLACE TABLE daily_sku AS
SELECT
    stock_code AS sku,
    cast(invoice_ts AS DATE) AS date,
    sum(quantity)::DOUBLE AS units,
    sum(quantity * price) / nullif(sum(quantity), 0) AS avg_price,
    count(*)::BIGINT AS n_lines
FROM clean_lines
GROUP BY stock_code, cast(invoice_ts AS DATE);

CREATE OR REPLACE TABLE selection_cutoff AS
SELECT
    (SELECT min(date) FROM daily_sku)
        + (
            SELECT yearly_lag_days + lag_warmup_days + horizon
            FROM config
        ) AS selection_cutoff;

CREATE OR REPLACE TABLE sku_catalog AS
SELECT
    stock_code AS sku,
    arg_max(description, invoice_ts) AS description
FROM clean_lines
WHERE cast(invoice_ts AS DATE) < (SELECT selection_cutoff FROM selection_cutoff)
GROUP BY stock_code;

CREATE OR REPLACE TABLE selected_skus AS
SELECT
    sku,
    total_units,
    sale_days
FROM (
    SELECT
        d.sku,
        sum(d.units) AS total_units,
        count(*)::INTEGER AS sale_days
    FROM daily_sku d
    CROSS JOIN selection_cutoff c
    WHERE d.date < c.selection_cutoff
    GROUP BY d.sku
    HAVING count(*) >= (SELECT min_sale_days FROM config)
)
QUALIFY row_number() OVER (ORDER BY total_units DESC, sku)
    <= (SELECT top_n FROM config);

CREATE OR REPLACE TABLE calendar AS
SELECT cast(unnest(generate_series(
    (SELECT min(date) FROM daily_sku),
    (SELECT max(date) FROM daily_sku),
    INTERVAL 1 DAY
)) AS DATE) AS date;

CREATE OR REPLACE TABLE panel AS
SELECT
    s.sku,
    c.date,
    coalesce(d.units, 0)::DOUBLE AS units,
    d.avg_price
FROM selected_skus s
CROSS JOIN calendar c
LEFT JOIN daily_sku d
    ON d.sku = s.sku
   AND d.date = c.date;
