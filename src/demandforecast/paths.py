"""Locate the repository root and local data paths."""

from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), *Path.cwd().parents, *here.parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "sql" / "02_features.sql").is_file() and (
            candidate / "src" / "demandforecast"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the project root (sql/02_features.sql).")


def sql_dir() -> Path:
    return project_root() / "sql"


def data_dir() -> Path:
    path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    path = data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_parquet_path() -> Path:
    return raw_dir() / "online_retail_ii.parquet"


def duckdb_path() -> Path:
    return data_dir() / "forecast.duckdb"


def reports_dir() -> Path:
    path = project_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path
