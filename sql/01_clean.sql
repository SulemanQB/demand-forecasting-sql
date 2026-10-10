-- Clean UK product sales and build a daily SKU panel.
-- Cancellation matching uses the return lines, so those lines are kept until the anti join.
-- The panel is one row per selected SKU per calendar day. Later window functions use ROWS
-- frames and assume that grain (no missing dates, no duplicate keys).

CREATE OR REPLACE TABLE clean_lines AS
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
),
scoped AS (
    SELECT *
    FROM typed
    WHERE country = (SELECT country FROM config)
      AND price > 0
      AND quantity <> 0
      AND stock_code NOT IN ('', 'NAN', 'NONE', 'NULL')
      AND regexp_matches(stock_code, '^[0-9]')
),
sales AS (
    SELECT *
    FROM scoped
    WHERE quantity > 0
      AND left(invoice, 1) <> 'C'
),
returns AS (
    SELECT
        stock_code,
        abs(quantity) AS quantity,
        customer_id,
        cast(invoice_ts AS DATE) AS return_date
    FROM scoped
    WHERE quantity < 0
      AND customer_id IS NOT NULL
)
SELECT
    s.invoice,
    s.stock_code,
    s.description,
    s.quantity,
    s.invoice_ts,
    s.price,
    s.customer_id,
    s.country
FROM sales s
CROSS JOIN config cfg
ANTI JOIN returns r
    ON s.stock_code = r.stock_code
   AND s.quantity = r.quantity
   AND s.customer_id = r.customer_id
   AND abs(date_diff('day', cast(s.invoice_ts AS DATE), r.return_date)) <= cfg.cancel_match_days;

CREATE OR REPLACE TABLE daily_sku AS
SELECT
    stock_code AS sku,
    cast(invoice_ts AS DATE) AS date,
    sum(quantity)::DOUBLE AS units,
    sum(quantity * price) / nullif(sum(quantity), 0) AS avg_price,
    count(*)::BIGINT AS n_lines
FROM clean_lines
GROUP BY stock_code, cast(invoice_ts AS DATE);

CREATE OR REPLACE TABLE sku_catalog AS
SELECT
    stock_code AS sku,
    arg_max(description, invoice_ts) AS description
FROM clean_lines
GROUP BY stock_code;

CREATE OR REPLACE TABLE selected_skus AS
WITH stats AS (
    SELECT
        sku,
        sum(units) AS total_units,
        count(*)::INTEGER AS sale_days
    FROM daily_sku
    GROUP BY sku
),
ranked AS (
    SELECT
        sku,
        total_units,
        sale_days,
        row_number() OVER (ORDER BY total_units DESC, sku) AS rk
    FROM stats
    WHERE sale_days >= (SELECT min_sale_days FROM config)
)
SELECT sku, total_units, sale_days
FROM ranked
WHERE rk <= (SELECT top_n FROM config);

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
