-- Reconciliation after cleaning and the supervised build.
-- Duplicate keys and broken cancellation pairs are listed with QUALIFY.
-- Both violation tables are empty when the panel is one row per SKU-day and
-- each matched sale has one return and each matched return has one sale.
-- Python reads quality_checks and fails the build if a count disagrees.

CREATE OR REPLACE TABLE panel_duplicate_keys AS
SELECT
    sku,
    date,
    count(*) OVER (PARTITION BY sku, date) AS n_rows
FROM panel
QUALIFY count(*) OVER (PARTITION BY sku, date) > 1;

CREATE OR REPLACE TABLE cancel_match_violations AS
SELECT
    sale_id,
    return_id,
    count(*) OVER (PARTITION BY sale_id) AS n_for_sale,
    count(*) OVER (PARTITION BY return_id) AS n_for_return
FROM cancel_matches
QUALIFY count(*) OVER (PARTITION BY sale_id) > 1
    OR count(*) OVER (PARTITION BY return_id) > 1;

CREATE OR REPLACE TABLE quality_checks AS
SELECT
    (SELECT count(*) FROM panel) AS n_panel,
    (SELECT count(*) FROM (SELECT sku, date FROM panel GROUP BY 1, 2)) AS n_panel_keys,
    (SELECT count(*) FROM calendar) AS n_calendar,
    (SELECT date_diff('day', min(date), max(date)) + 1 FROM calendar) AS calendar_span,
    (SELECT coalesce(sum(quantity), 0) FROM clean_lines) AS clean_quantity,
    (SELECT coalesce(sum(units), 0) FROM daily_sku) AS daily_units,
    (SELECT count(*) FROM cancel_matches) AS n_matched_sales,
    (SELECT count(DISTINCT return_id) FROM cancel_matches) AS n_matched_returns,
    (SELECT count(DISTINCT sale_id) FROM cancel_matches) AS n_matched_sale_ids,
    (SELECT count(*) FROM sales_lines) AS n_sales_lines,
    (SELECT count(*) FROM return_lines) AS n_return_lines,
    (SELECT count(*) FROM clean_lines) AS n_clean_lines,
    (SELECT count(*) FROM supervised WHERE asof_date >= target_date) AS n_bad_asof,
    (SELECT count(*) FROM selected_skus) AS n_selected,
    (SELECT selection_cutoff FROM selection_cutoff) AS selection_cutoff,
    (SELECT count(*) FROM panel_duplicate_keys) AS n_duplicate_key_rows,
    (SELECT count(*) FROM cancel_match_violations) AS n_match_violations;
