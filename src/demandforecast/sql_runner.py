"""Execute project SQL files and check that supervised rows respect the origin."""

from pathlib import Path

from demandforecast.paths import sql_dir


def split_sql(text: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("--"):
            continue
        buffer.append(line)
        if line.strip().endswith(";"):
            statement = _strip_semicolon("\n".join(buffer))
            if statement:
                statements.append(statement)
            buffer = []
    tail = _strip_semicolon("\n".join(buffer))
    if tail:
        statements.append(tail)
    return statements


def _strip_semicolon(statement: str) -> str:
    statement = statement.strip()
    if statement.endswith(";"):
        statement = statement[:-1].strip()
    return statement


def run_sql_file(con, filename: str | Path) -> None:
    path = Path(filename)
    if not path.is_file():
        path = sql_dir() / filename
    statements = split_sql(path.read_text(encoding="utf-8"))
    if not statements:
        raise RuntimeError(f"No SQL statements in {path}")
    for statement in statements:
        con.execute(statement)


def assert_supervised_time(con) -> None:
    """Fail if any supervised feature can see the target day or a later day."""
    bad_time = con.execute(
        """
        SELECT count(*) FROM supervised
        WHERE asof_date >= target_date
           OR horizon <> date_diff('day', asof_date, target_date)
           OR horizon < 1
           OR naive_w7_date > asof_date
           OR same_day_last_year_date > asof_date
           OR naive_w7_date <> target_date - cast(7 * ceil(horizon / 7.0) AS INTEGER)
           OR same_day_last_year_date <> target_date - 364
        """
    ).fetchone()[0]
    if bad_time:
        raise RuntimeError(f"{bad_time} supervised rows break the as-of time rule")

    null_naive = con.execute(
        "SELECT count(*) FROM supervised WHERE seasonal_naive_w7 IS NULL"
    ).fetchone()[0]
    if null_naive:
        raise RuntimeError(f"{null_naive} supervised rows have a null weekly seasonal naive")

    mismatch = con.execute(
        """
        SELECT count(*) FROM supervised s
        JOIN panel p ON p.sku = s.sku AND p.date = s.naive_w7_date
        WHERE abs(s.seasonal_naive_w7 - p.units) > 1e-6
        """
    ).fetchone()[0]
    if mismatch:
        raise RuntimeError(f"{mismatch} weekly seasonal-naive values disagree with the panel")

    mismatch_year = con.execute(
        """
        SELECT count(*) FROM supervised s
        JOIN panel p ON p.sku = s.sku AND p.date = s.same_day_last_year_date
        WHERE s.units_same_day_last_year IS NULL
           OR abs(s.units_same_day_last_year - p.units) > 1e-6
        """
    ).fetchone()[0]
    if mismatch_year:
        raise RuntimeError(
            f"{mismatch_year} same-day-last-year values disagree with the panel"
        )
