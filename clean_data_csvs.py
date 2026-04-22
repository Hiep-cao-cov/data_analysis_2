"""
Batch-clean CSV files under ./data (or a folder you pass).

- Column names: strip whitespace, lowercase (matches app validate_dataframe / normalize_dataframe_columns).
- Column ``year`` (after rename): strip cell text, coerce to integer calendar year
  (handles 2024, 2024.0, " 2025 ", "2024-01-01", etc.).

Usage (from project root):
  python clean_data_csvs.py
  python clean_data_csvs.py path/to/folder
  python clean_data_csvs.py --dry-run

Backs up each file as ``*.csv.bak`` before overwrite unless --no-backup.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index([str(c).strip().lower() for c in df.columns])
    return df


def clean_year_series(series: pd.Series) -> pd.Series:
    """Strip values and return integer calendar years (nullable Int64)."""
    s = series.copy()
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.astype(str).str.strip()
        s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    num = pd.to_numeric(s, errors="coerce")
    dt = pd.to_datetime(s, errors="coerce")
    year_from_dt = dt.dt.year
    merged = num.where(num.notna(), year_from_dt)
    out = np.round(merged).astype("Int64")
    return out


def clean_one_csv(path: Path, dry_run: bool, backup: bool) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        return False, f"read error: {e}"

    if df.empty:
        return True, "empty file, skipped"

    df = normalize_column_names(df)

    # Resolve duplicate column names after lowercasing (rare)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        return False, f"duplicate column names after normalize: {dupes} — fix manually"

    if "year" in df.columns:
        df["year"] = clean_year_series(df["year"])
        if df["year"].isna().any():
            bad = int(df["year"].isna().sum())
            return False, f"'year' has {bad} value(s) that could not be parsed"

    if dry_run:
        return True, f"would write ({len(df)} rows)"

    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    df.to_csv(path, index=False, encoding="utf-8")
    return True, "updated"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize CSV headers and year column.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Folder containing CSV files (default: ./data)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak copies")
    args = parser.parse_args()
    root = Path(args.folder)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        print(f"No CSV files in {root}")
        return 0

    ok_all = True
    for p in csvs:
        ok, msg = clean_one_csv(p, dry_run=args.dry_run, backup=not args.no_backup)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {p.name}: {msg}")
        if not ok:
            ok_all = False

    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
