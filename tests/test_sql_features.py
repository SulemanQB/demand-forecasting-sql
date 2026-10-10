"""SQL cleaning, window features, and the as-of leakage rule on a toy panel."""

import numpy as np
import pandas as pd
import pytest

from demandforecast.config import FEATURE_COLUMNS, TARGET_COLUMN, Config
from demandforecast.pipeline import create_config_table
from demandforecast.sql_runner import assert_supervised_time, run_sql_file

pytest.importorskip("duckdb")
import duckdb


def _toy_config(**overrides) -> Config:
    values = dict(horizon=4, min_history_days=29, top_n=10, min_sale_days=1, cancel_match_days=21)
    values.update(overrides)
    return Config(**values)


def _connect():
    return duckdb.connect()


def test_clean_filters_cancellations_and_non_products():
    con = _connect()
    rows = [
        _line("1001", "11111", "MUG", 5, "2020-01-01", 2.0, 9, "United Kingdom"),
        _line("1002", "11111", "MUG", 8, "2020-01-02", 2.0, 9, "United Kingdom"),
        _line("C1002", "11111", "MUG", -8, "2020-01-04", 2.0, 9, "United Kingdom"),
        _line("1003", "POST", "POSTAGE", 1, "2020-01-01", 5.0, 9, "United Kingdom"),
        _line("1004", "22222", "FR", 4, "2020-01-01", 2.0, 3, "France"),
        _line("1005", "33333", "FREE", 4, "2020-01-01", 0.0, 9, "United Kingdom"),
        _line("1006", "44444", "BOWL", 3, "2020-01-03", 1.5, None, "United Kingdom"),
        _line("C1006", "44444", "BOWL", -3, "2020-01-05", 1.5, None, "United Kingdom"),
        _line("1007", "gift_0001", "GIFT", 1, "2020-01-01", 10.0, 9, "United Kingdom"),
        _line("1008", "55555", "CANDLE", 6, "2020-01-06", 3.0, 7, "United Kingdom"),
        _line("1009", "85099b", "HEART", 2, "2020-01-02", 1.7, 8, "United Kingdom"),
        _line("C999", "66666", "ODD", 4, "2020-01-02", 1.0, 8, "United Kingdom"),
        _line("1010", "77777", "VASE", 9, "2020-01-01", 4.0, 4, "United Kingdom"),
        _line("C1010", "77777", "VASE", -9, "2020-02-01", 4.0, 4, "United Kingdom"),
    ]
    frame = pd.DataFrame(rows)
    frame["customer_id"] = frame["customer_id"].astype("Int64")
    con.register("raw_df", frame)
    con.execute(
        """
        CREATE TABLE raw_invoices AS
        SELECT
            invoice::VARCHAR AS invoice,
            stock_code::VARCHAR AS stock_code,
            description::VARCHAR AS description,
            quantity::BIGINT AS quantity,
            cast(invoice_date AS TIMESTAMP) AS invoice_date,
            price::DOUBLE AS price,
            try_cast(customer_id AS BIGINT) AS customer_id,
            country::VARCHAR AS country
        FROM raw_df
        """
    )
    create_config_table(con, _toy_config())
    run_sql_file(con, "01_clean.sql")

    codes = [row[0] for row in con.execute("SELECT DISTINCT stock_code FROM clean_lines ORDER BY 1").fetchall()]
    assert codes == ["11111", "44444", "55555", "77777", "85099B"]
    units = dict(con.execute("SELECT stock_code, sum(quantity) FROM clean_lines GROUP BY 1").fetchall())
    assert units["11111"] == 5
    assert units["77777"] == 9
    assert units["85099B"] == 2
    zeros = con.execute(
        "SELECT count(*) FROM panel WHERE sku = '11111' AND units = 0"
    ).fetchone()[0]
    assert zeros >= 1
    heart = con.execute(
        "SELECT description FROM sku_catalog WHERE sku = '85099B'"
    ).fetchone()[0]
    assert heart == "HEART"


def test_window_features_and_naive_date():
    con = _connect()
    _load_numeric_panel(con)
    create_config_table(con, _toy_config())
    run_sql_file(con, "02_features.sql")
    assert_supervised_time(con)

    row = con.execute(
        """
        SELECT * FROM supervised
        WHERE sku = 'A' AND asof_date = DATE '2020-01-29' AND horizon = 2
        """
    ).fetchdf().iloc[0]
    assert row["units_asof"] == pytest.approx(28)
    assert row["units_lag_7"] == pytest.approx(21)
    assert row["units_lag_14"] == pytest.approx(14)
    assert row["units_lag_21"] == pytest.approx(7)
    assert row["units_lag_28"] == pytest.approx(0)
    assert pd.isna(row["units_lag_364"])
    assert row["roll_mean_7"] == pytest.approx(25)
    assert row["roll_mean_28"] == pytest.approx(14.5)
    assert row["roll_std_7"] == pytest.approx(float(np.std(np.arange(22, 29), ddof=1)))
    assert row["same_weekday_mean_4"] == pytest.approx(10.5)
    assert row["level_ratio_7_28"] == pytest.approx(25 / 14.5)
    assert row["price_asof"] == pytest.approx(4.0)
    assert row["price_ratio"] == pytest.approx(1.0)
    assert row["history_days"] == 29
    assert row["target_units"] == pytest.approx(30)
    assert row["seasonal_naive_w7"] == pytest.approx(23)
    assert pd.Timestamp(row["naive_w7_date"]) == pd.Timestamp("2020-01-24")
    assert row["target_dow"] == 5
    assert row["target_is_weekend"] == 0
    christmas = (pd.Timestamp("2020-12-25") - pd.Timestamp("2020-01-31")).days
    assert row["days_to_christmas"] == christmas
    assert pd.isna(row["units_same_day_last_year"])

    later = con.execute(
        """
        SELECT price_asof FROM asof_features
        WHERE sku = 'A' AND asof_date = DATE '2020-02-05'
        """
    ).fetchone()[0]
    assert later == pytest.approx(8.0)

    columns = set(con.execute("SELECT * FROM supervised LIMIT 1").fetchdf().columns)
    assert set(FEATURE_COLUMNS).issubset(columns)
    assert TARGET_COLUMN in columns


def test_features_do_not_change_when_the_future_changes():
    con = _connect()
    _load_numeric_panel(con)
    create_config_table(con, _toy_config())
    run_sql_file(con, "02_features.sql")
    before = _snapshot(con)
    changed = con.execute(
        """
        SELECT target_units FROM supervised
        WHERE sku = 'A' AND asof_date = DATE '2020-01-31' AND horizon = 1
        """
    ).fetchone()[0]
    assert changed == pytest.approx(31)

    con.execute("UPDATE panel SET units = 999 WHERE date > DATE '2020-01-31'")
    run_sql_file(con, "02_features.sql")
    after = _snapshot(con)
    left = before.drop(columns=["target_units"])
    right = after.drop(columns=["target_units"])
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-8, atol=1e-8)

    updated = con.execute(
        """
        SELECT units_asof, target_units FROM supervised
        WHERE sku = 'A' AND asof_date = DATE '2020-01-31' AND horizon = 1
        """
    ).fetchone()
    assert updated[0] == pytest.approx(30)
    assert updated[1] == pytest.approx(999)


def _snapshot(con) -> pd.DataFrame:
    frame = con.execute(
        "SELECT * FROM supervised WHERE asof_date <= DATE '2020-01-31'"
    ).fetchdf()
    for column in ("asof_date", "target_date", "naive_w7_date", "same_day_last_year_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame.sort_values(["asof_date", "horizon"]).reset_index(drop=True)


def _load_numeric_panel(con) -> None:
    start = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(40):
        if i == 10:
            price = 4.0
        elif i >= 30:
            price = 8.0
        else:
            price = np.nan
        rows.append(
            {
                "sku": "A",
                "date": start + pd.Timedelta(days=i),
                "units": float(i),
                "avg_price": price,
            }
        )
    frame = pd.DataFrame(rows)
    con.register("panel_df", frame)
    con.execute(
        """
        CREATE TABLE panel AS
        SELECT
            sku::VARCHAR AS sku,
            cast(date AS DATE) AS date,
            units::DOUBLE AS units,
            avg_price::DOUBLE AS avg_price
        FROM panel_df
        """
    )
    con.execute("CREATE TABLE calendar AS SELECT DISTINCT date FROM panel ORDER BY 1")


def _line(invoice, stock, description, quantity, day, price, customer, country):
    return {
        "invoice": invoice,
        "stock_code": stock,
        "description": description,
        "quantity": quantity,
        "invoice_date": pd.Timestamp(day),
        "price": price,
        "customer_id": customer,
        "country": country,
    }
