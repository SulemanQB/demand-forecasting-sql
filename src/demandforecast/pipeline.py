"""Load the raw parquet into DuckDB and build the SQL feature tables."""

import json
from datetime import date, datetime
from decimal import Decimal

import duckdb

from demandforecast.config import DATASET_PAGE, DATASET_SHA256, DATASET_URL, Config
from demandforecast.paths import duckdb_path, raw_parquet_path, reports_dir
from demandforecast.sql_runner import assert_supervised_time, run_sql_file


def _log(message: str) -> None:
    print(message, flush=True)


def create_config_table(con, cfg: Config) -> None:
    con.execute("DROP TABLE IF EXISTS config")
    con.execute(
        """
        CREATE TABLE config (
            country VARCHAR,
            top_n INTEGER,
            min_sale_days INTEGER,
            horizon INTEGER,
            min_history_days INTEGER,
            cancel_match_days INTEGER,
            yearly_lag_days INTEGER,
            lag_warmup_days INTEGER,
            weekly_horizon INTEGER,
            weekly_min_history INTEGER
        )
        """
    )
    con.execute(
        "INSERT INTO config VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            cfg.country,
            cfg.top_n,
            cfg.min_sale_days,
            cfg.horizon,
            cfg.min_history_days,
            cfg.cancel_match_days,
            cfg.season_year,
            cfg.lag_warmup_days,
            cfg.weekly_horizon,
            cfg.weekly_min_history,
        ],
    )


def _jsonable(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_features(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    parquet = raw_parquet_path()
    if not parquet.is_file():
        raise FileNotFoundError(
            f"Raw parquet not found at {parquet}. Run: python -m demandforecast download"
        )
    db_path = duckdb_path()
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE raw_invoices AS SELECT * FROM read_parquet(?)",
            [str(parquet)],
        )
        create_config_table(con, cfg)
        _log("Running sql/01_clean.sql")
        run_sql_file(con, "01_clean.sql")
        _log("Running sql/02_features.sql")
        run_sql_file(con, "02_features.sql")
        assert_supervised_time(con)
        _log("Running sql/03_quality.sql")
        run_sql_file(con, "03_quality.sql")
        quality = _assert_quality(con, cfg)
        _log("Running sql/04_weekly.sql")
        run_sql_file(con, "04_weekly.sql")
        weekly_mismatches = _assert_weekly(con, cfg)
        stats = _validate_and_collect(con, cfg, quality, weekly_mismatches)
    finally:
        con.close()
    out = reports_dir() / "build_stats.json"
    out.write_text(json.dumps(stats, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    _log(
        f"Selected {stats['n_skus_selected']} SKUs before {stats['selection_cutoff']}, "
        f"{stats['n_supervised_rows']:,} supervised rows, "
        f"panel {stats['date_min']} to {stats['date_max']}"
    )
    return stats


def _assert_quality(con, cfg: Config) -> dict:
    checks = con.execute("SELECT * FROM quality_checks").fetchdf().iloc[0]
    if int(checks["n_duplicate_key_rows"]) != 0:
        raise RuntimeError("Panel has duplicate sku-date keys")
    if int(checks["n_match_violations"]) != 0:
        raise RuntimeError("Cancellation match is not one sale to one return")
    if int(checks["n_panel"]) != int(checks["n_panel_keys"]):
        raise RuntimeError("Panel key count does not match the row count")
    if int(checks["n_calendar"]) != int(checks["calendar_span"]):
        raise RuntimeError(
            f"Calendar is not a continuous daily spine ({int(checks['n_calendar'])} vs {int(checks['calendar_span'])})"
        )
    if abs(float(checks["clean_quantity"]) - float(checks["daily_units"])) > 1e-4:
        raise RuntimeError("Clean-line quantity does not match daily SKU units")
    matched = int(checks["n_matched_sales"])
    if matched != int(checks["n_matched_returns"]) or matched != int(checks["n_matched_sale_ids"]):
        raise RuntimeError("Matched sale count does not equal matched return count")
    if int(checks["n_clean_lines"]) != int(checks["n_sales_lines"]) - matched:
        raise RuntimeError("Clean lines are not sales minus one-to-one cancellations")
    if int(checks["n_bad_asof"]) != 0:
        raise RuntimeError("Supervised rows have an as-of date on or after the target")
    if int(checks["n_selected"]) < cfg.top_n:
        raise RuntimeError(
            f"Only {int(checks['n_selected'])} SKUs have at least {cfg.min_sale_days} "
            "sale days before the first origin; lower min_sale_days or top_n."
        )
    return {column: _jsonable(checks[column]) for column in checks.index}


def _assert_weekly(con, cfg: Config) -> int:
    duplicates = con.execute("SELECT count(*) FROM weekly_duplicate_keys").fetchone()[0]
    if duplicates:
        raise RuntimeError("Weekly panel has duplicate sku-week keys")
    bad_monday = con.execute(
        "SELECT count(*) FROM weekly_panel WHERE isodow(week_start) <> 1"
    ).fetchone()[0]
    if bad_monday:
        raise RuntimeError("Weekly panel has a week that does not start on Monday")
    mismatches = con.execute(
        """
        SELECT count(*) FROM (
            SELECT sku, sum(units) AS units FROM panel GROUP BY sku
        ) d
        JOIN (
            SELECT sku, sum(units) AS units FROM weekly_panel GROUP BY sku
        ) w USING (sku)
        WHERE abs(d.units - w.units) > 1e-4
        """
    ).fetchone()[0]
    if mismatches:
        raise RuntimeError(f"{mismatches} SKUs have weekly totals that differ from the daily panel")
    bad_time = con.execute(
        """
        SELECT count(*) FROM weekly_supervised
        WHERE asof_date >= target_date
           OR horizon < 1
           OR horizon > (SELECT weekly_horizon FROM config)
           OR date_diff('day', asof_date, target_date) <> horizon * 7
           OR seasonal_naive_w1 IS NULL
           OR abs(seasonal_naive_w1 - units_asof) > 1e-6
           OR naive_w1_date <> asof_date
        """
    ).fetchone()[0]
    if bad_time:
        raise RuntimeError(f"{bad_time} weekly supervised rows break the as-of or season-1 rule")
    n_weekly = con.execute("SELECT count(*) FROM weekly_supervised").fetchone()[0]
    if n_weekly == 0:
        raise RuntimeError("Weekly supervised table is empty")
    if cfg.weekly_horizon < 1:
        raise RuntimeError("weekly_horizon must be positive")
    return int(mismatches)


def _validate_and_collect(con, cfg: Config, quality: dict, weekly_mismatches: int) -> dict:
    n_panel = int(quality["n_panel"])
    n_selected = int(quality["n_selected"])
    n_supervised = con.execute("SELECT count(*) FROM supervised").fetchone()[0]
    if n_supervised == 0:
        raise RuntimeError("Supervised table is empty")
    n_weekly = con.execute("SELECT count(*) FROM weekly_supervised").fetchone()[0]
    n_raw = con.execute("SELECT count(*) FROM raw_invoices").fetchone()[0]

    n_eligible = con.execute(
        """
        SELECT count(*) FROM (
            SELECT d.sku
            FROM daily_sku d
            CROSS JOIN selection_cutoff c
            WHERE d.date < c.selection_cutoff
            GROUP BY d.sku
            HAVING count(*) >= (SELECT min_sale_days FROM config)
        )
        """
    ).fetchone()[0]
    selected_before, before_all, selected_full, all_units = con.execute(
        """
        SELECT
            (SELECT sum(total_units) FROM selected_skus),
            (
                SELECT sum(d.units)
                FROM daily_sku d
                CROSS JOIN selection_cutoff c
                WHERE d.date < c.selection_cutoff
            ),
            (
                SELECT sum(units)
                FROM daily_sku
                WHERE sku IN (SELECT sku FROM selected_skus)
            ),
            (SELECT sum(units) FROM daily_sku)
        """
    ).fetchone()
    date_min, date_max = con.execute("SELECT min(date), max(date) FROM panel").fetchone()
    week_min, week_max = con.execute(
        "SELECT min(week_start), max(week_start) FROM weekly_panel"
    ).fetchone()
    skus = con.execute(
        """
        SELECT
            s.sku,
            c.description,
            s.total_units AS units_before_cutoff,
            s.sale_days AS sale_days_before_cutoff,
            b.first_sale,
            b.last_sale,
            b.full_units,
            b.full_sale_days
        FROM selected_skus s
        LEFT JOIN sku_catalog c USING (sku)
        JOIN (
            SELECT
                sku,
                min(date) AS first_sale,
                max(date) AS last_sale,
                sum(units) AS full_units,
                count(*)::INTEGER AS full_sale_days
            FROM daily_sku
            GROUP BY sku
        ) b USING (sku)
        ORDER BY s.total_units DESC, s.sku
        """
    ).fetchdf()
    sku_rows = []
    for row in skus.to_dict(orient="records"):
        sku_rows.append({key: _jsonable(value) for key, value in row.items()})

    return {
        "dataset_url": DATASET_URL,
        "dataset_page": DATASET_PAGE,
        "dataset_sha256": DATASET_SHA256,
        "country": cfg.country,
        "top_n": cfg.top_n,
        "min_sale_days": cfg.min_sale_days,
        "cancel_match_days": cfg.cancel_match_days,
        "horizon": cfg.horizon,
        "min_history_days": cfg.min_history_days,
        "selection_rule": (
            "Top SKUs by units on dates strictly before the first origin, "
            "among SKUs with at least min_sale_days in that window. "
            "The cutoff equals min(sale date) + yearly_lag_days + lag_warmup_days + horizon."
        ),
        "selection_cutoff": quality["selection_cutoff"],
        "n_raw_rows": int(n_raw),
        "n_clean_lines": int(quality["n_clean_lines"]),
        "n_sales_lines": int(quality["n_sales_lines"]),
        "n_return_lines": int(quality["n_return_lines"]),
        "n_cancel_matches": int(quality["n_matched_sales"]),
        "n_skus_eligible_before_cutoff": int(n_eligible),
        "n_skus_selected": n_selected,
        "selected_units_before_cutoff": float(selected_before),
        "units_before_cutoff": float(before_all),
        "selected_unit_share_before_cutoff": float(selected_before) / float(before_all),
        "selected_units_full_period": float(selected_full),
        "all_clean_units": float(all_units),
        "selected_unit_share_full_period": float(selected_full) / float(all_units),
        "date_min": date_min.isoformat(),
        "date_max": date_max.isoformat(),
        "week_min": week_min.isoformat(),
        "week_max": week_max.isoformat(),
        "n_panel_rows": n_panel,
        "n_supervised_rows": int(n_supervised),
        "n_weekly_supervised_rows": int(n_weekly),
        "n_weekly_unit_mismatches": weekly_mismatches,
        "quality": quality,
        "skus": sku_rows,
    }
