"""Download UCI Online Retail II and write a local parquet file. Nothing is committed."""

import gc
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from demandforecast.config import (
    DATASET_SHA256,
    DATASET_URL,
    SHEET_EARLY,
    SHEET_LATE,
    SHEET_OVERLAP_CUT,
)
from demandforecast.paths import raw_dir, raw_parquet_path

OVERLAP_CUT = pd.Timestamp(SHEET_OVERLAP_CUT)
COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def _log(message: str) -> None:
    print(message, flush=True)


def normalize_stock_code(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.str.upper()


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in COLUMN_MAP if column not in df.columns]
    if missing:
        raise ValueError(f"Workbook is missing columns {missing}; found {list(df.columns)}")
    out = df.loc[:, list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()
    out["invoice"] = out["invoice"].astype("string").str.strip()
    out["stock_code"] = normalize_stock_code(out["stock_code"])
    out["description"] = out["description"].astype("string")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["invoice_date"] = pd.to_datetime(out["invoice_date"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    out["country"] = out["country"].astype("string").str.strip()
    out = out.dropna(subset=["invoice", "stock_code", "quantity", "invoice_date", "price", "country"])
    out["quantity"] = out["quantity"].astype("int64")
    out["price"] = out["price"].astype("float64")
    return out.reset_index(drop=True)


def combine_sheets(early: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    """Drop the overlapping December 2010 rows from the first sheet and keep the second."""
    kept = early.loc[early["invoice_date"] < OVERLAP_CUT]
    return pd.concat([kept, late], ignore_index=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_zip(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "demandforecast/1.0"})
    _log(f"Downloading {DATASET_URL}")
    with urllib.request.urlopen(request, timeout=180) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    digest = _sha256(dest)
    if digest != DATASET_SHA256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {dest.name}: {digest}")
    _log(f"SHA-256 ok ({digest})")


def _extract_xlsx(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".xlsx")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one xlsx in the zip, found {members}")
        archive.extract(members[0], zip_path.parent)
        return zip_path.parent / members[0]


def _sheet_to_parquet(xlsx: Path, sheet: str, dest: Path, *, early: bool) -> None:
    frame = pd.read_excel(xlsx, sheet_name=sheet, engine="calamine")
    frame = normalize_frame(frame)
    if early:
        frame = frame.loc[frame["invoice_date"] < OVERLAP_CUT].copy()
    frame.to_parquet(dest, index=False)
    _log(f"Normalized {sheet}: {len(frame):,} rows")
    del frame
    gc.collect()


def _write_parquet(xlsx: Path, dest: Path) -> None:
    early_path = dest.with_name("_early.parquet")
    late_path = dest.with_name("_late.parquet")
    try:
        _sheet_to_parquet(xlsx, SHEET_EARLY, early_path, early=True)
        _sheet_to_parquet(xlsx, SHEET_LATE, late_path, early=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        early_sql = str(early_path).replace("'", "''")
        late_sql = str(late_path).replace("'", "''")
        dest_sql = str(dest).replace("'", "''")
        duckdb.execute(
            f"""
            COPY (
                SELECT
                    invoice::VARCHAR AS invoice,
                    stock_code::VARCHAR AS stock_code,
                    description::VARCHAR AS description,
                    quantity::BIGINT AS quantity,
                    invoice_date::TIMESTAMP AS invoice_date,
                    price::DOUBLE AS price,
                    try_cast(customer_id AS BIGINT) AS customer_id,
                    country::VARCHAR AS country
                FROM (
                    SELECT * FROM read_parquet('{early_sql}')
                    UNION ALL
                    SELECT * FROM read_parquet('{late_sql}')
                )
            ) TO '{dest_sql}' (FORMAT PARQUET)
            """
        )
    finally:
        early_path.unlink(missing_ok=True)
        late_path.unlink(missing_ok=True)
    n_rows = duckdb.execute("SELECT count(*) FROM read_parquet(?)", [str(dest)]).fetchone()[0]
    if n_rows < 900_000 or n_rows > 1_200_000:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Unexpected row count after download: {n_rows}")
    _log(f"Wrote {n_rows:,} rows to {dest}")


def download_raw(force: bool = False) -> Path:
    dest = raw_parquet_path()
    if dest.exists() and not force:
        _log(f"Using cached raw parquet at {dest}")
        return dest
    raw = raw_dir()
    zip_path = raw / "online_retail_II.zip"
    xlsx_path: Path | None = None
    try:
        _download_zip(zip_path)
        xlsx_path = _extract_xlsx(zip_path)
        _write_parquet(xlsx_path, dest)
    finally:
        zip_path.unlink(missing_ok=True)
        if xlsx_path is not None:
            xlsx_path.unlink(missing_ok=True)
    return dest
