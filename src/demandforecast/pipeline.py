"""Load the raw parquet into DuckDB and build the SQL feature tables."""

import json
from datetime import date, datetime

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
            cancel_match_days INTEGER
        )
        """
    )
    con.execute(
        "INSERT INTO config VALUES (?, ?, ?, ?, ?, ?)",
        [
            cfg.country,
            cfg.top_n,
            cfg.min_sale_days,
            cfg.horizon,
            cfg.min_history_days,
            cfg.cancel_match_days,
        ],
    )


def _jsonable(value):
    if isinstance(value, (date, datetime)):
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
        stats = _validate_and_collect(con, cfg)
        assert_supervised_time(con)
    finally:
        con.close()
    out = reports_dir() / "build_stats.json"
    out.write_text(json.dumps(stats, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    _log(
        f"Selected {stats['n_skus_selected']} SKUs, "
        f"{stats['n_supervised_rows']:,} supervised rows, "
        f"panel {stats['date_min']} to {stats['date_max']}"
    )
    return stats


def _validate_and_collect(con, cfg: Config) -> dict:
    n_panel, n_keys = con.execute(
        """
        SELECT
            count(*),
            (SELECT count(*) FROM (SELECT sku, date FROM panel GROUP BY 1, 2))
        FROM panel
        """
    ).fetchone()
    if n_panel != n_keys:
        raise RuntimeError("Panel has duplicate sku-date keys")
    n_calendar, span = con.execute(
        """
        SELECT count(*), date_diff('day', min(date), max(date)) + 1
        FROM calendar
        """
    ).fetchone()
    if n_calendar != span:
        raise RuntimeError(f"Calendar is not a continuous daily spine ({n_calendar} vs {span})")

    n_selected = con.execute("SELECT count(*) FROM selected_skus").fetchone()[0]
    if n_selected < cfg.top_n:
        raise RuntimeError(
            f"Only {n_selected} SKUs have at least {cfg.min_sale_days} sale days; "
            "lower min_sale_days or top_n."
        )
    n_supervised = con.execute("SELECT count(*) FROM supervised").fetchone()[0]
    if n_supervised == 0:
        raise RuntimeError("Supervised table is empty")

    n_raw = con.execute("SELECT count(*) FROM raw_invoices").fetchone()[0]
    n_clean = con.execute("SELECT count(*) FROM clean_lines").fetchone()[0]
    n_eligible = con.execute(
        """
        SELECT count(*) FROM (
            SELECT sku
            FROM daily_sku
            GROUP BY sku
            HAVING count(*) >= (SELECT min_sale_days FROM config)
        )
        """
    ).fetchone()[0]
    selected_units, all_units = con.execute(
        """
        SELECT
            (SELECT sum(total_units) FROM selected_skus),
            (SELECT sum(units) FROM daily_sku)
        """
    ).fetchone()
    date_min, date_max = con.execute("SELECT min(date), max(date) FROM panel").fetchone()
    skus = con.execute(
        """
        SELECT
            s.sku,
            c.description,
            s.total_units,
            s.sale_days,
            b.first_sale,
            b.last_sale
        FROM selected_skus s
        LEFT JOIN sku_catalog c USING (sku)
        JOIN (
            SELECT sku, min(date) AS first_sale, max(date) AS last_sale
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
        "n_raw_rows": int(n_raw),
        "n_clean_lines": int(n_clean),
        "n_skus_eligible": int(n_eligible),
        "n_skus_selected": int(n_selected),
        "selected_units": float(selected_units),
        "all_clean_units": float(all_units),
        "selected_unit_share": float(selected_units) / float(all_units),
        "date_min": date_min.isoformat(),
        "date_max": date_max.isoformat(),
        "n_panel_rows": int(n_panel),
        "n_supervised_rows": int(n_supervised),
        "skus": sku_rows,
    }
