"""CSV reading and in-memory cleaning for market data (headers, types, nulls)."""
from __future__ import annotations

import io
import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from config import MATERIAL_CONFIG

# Encodings to try (Excel/Windows exports often use BOM or cp1252).
_READ_ENCODINGS: tuple = ("utf-8-sig", "utf-8", "cp1252")

# String tokens treated as null before numeric parse
_MESSY_NULL_TOKENS = frozenset(
    {
        "",
        "nan",
        "none",
        "n/a",
        "na",
        "null",
        "#n/a",
        "#na",
        "<na>",
    }
)

# All metric columns that should be coerced; supplier-like columns are filled with 0 after coerce.
STRICT_NUMERIC: tuple = (
    "demand",
    "volume",
    "sow",
    "year",
    "min",
    "base",
    "max",
    "covestro",
    "tosoh",
    "wanhua",
    "kmc",
    "basf",
    "sabic",
    "huntsman",
    "mcns",
    "hanwha",
    "other",
)
ALLOW_NEGATIVE: tuple = (
    "ppd",
    "pocket price",
    "vn_pp",
    "tw_pp",
    "seap_pp",
    "apac_pp",
)


def normalize_dataframe_columns(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.columns = pd.Index([str(c).strip().lower() for c in df.columns])


def _supplier_column_names(material: str | None) -> frozenset:
    if not material:
        return frozenset()
    m = material.lower()
    if m in MATERIAL_CONFIG:
        names = list(MATERIAL_CONFIG[m].get("suppliers") or [])
        names.append("other")
        return frozenset(names)
    return frozenset()


def coerce_messy_numeric(series: pd.Series) -> pd.Series:
    """Parse numbers from strings: spaces, comma decimals, %, '', None/NaN tokens."""
    t = series.astype(str).str.strip()
    lower = t.str.lower()
    t = t.where(~lower.isin(_MESSY_NULL_TOKENS), other=np.nan)
    t = t.replace({r"^\s*$": np.nan}, regex=True)
    t = t.str.replace(",", ".", regex=False)
    t = t.str.replace("%", "", regex=False)
    s = pd.to_numeric(t, errors="coerce")
    return s


def preprocess_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    material: str | None = None,
) -> int:
    """
    In-place: normalize headers, coerce numerics, fill supplier 0, optional medians, drop bad rows.
    Returns the number of rows removed for missing required fields.
    """
    normalize_dataframe_columns(df)
    if len(df) == 0:
        return 0

    suppliers = _supplier_column_names(material)
    required_present = [c for c in required_columns if c in df.columns]
    required_set = set(required_present)

    if "customer" in df.columns:
        c = df["customer"].astype(str).str.strip().str.upper()
        c = c.where(~c.str.lower().isin(_MESSY_NULL_TOKENS), other=np.nan)
        c = c.replace(r"^\s*$", np.nan, regex=True)
        df["customer"] = c

    all_metric = [c for c in STRICT_NUMERIC + ALLOW_NEGATIVE if c in df.columns]
    for col in all_metric:
        if col in suppliers:
            continue
        df[col] = coerce_messy_numeric(df[col])

    for col in suppliers:
        if col in df.columns:
            df[col] = coerce_messy_numeric(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in all_metric:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    fill_messages: List[str] = []
    for col in df.columns:
        if col in required_set or col in suppliers or col == "customer":
            continue
        if col in STRICT_NUMERIC or col in ALLOW_NEGATIVE:
            if df[col].isna().any():
                m = df[col].median()
                if pd.isna(m):
                    m = 0.0
                n = int(df[col].isna().sum())
                df[col] = df[col].fillna(m)
                if n:
                    fill_messages.append(f"{col}×{n} (median {m:.4g})")
    if fill_messages:
        st.info(
            "Filled missing **optional** numeric cells for charting: "
            + "; ".join(fill_messages[:8])
            + ("; …" if len(fill_messages) > 8 else "")
        )

    n_before = len(df)
    if required_present:
        df.dropna(subset=required_present, inplace=True)
    dropped = n_before - len(df)
    if dropped:
        st.warning(
            f"Removed **{dropped}** row(s) with empty or invalid value(s) in required field(s): "
            f"{', '.join(required_present)}."
        )
    return dropped


def read_csv_flexible(
    path_or_buffer: str | os.PathLike | io.IOBase | bytes,
    **kwargs,
) -> pd.DataFrame:
    """
    Read CSV with common encodings. For file-like objects, re-reads from bytes so the stream
    is reset for callers; uses low_memory=False for stable dtypes.
    """
    opts = {**kwargs, "low_memory": False, "keep_default_na": True}
    if isinstance(path_or_buffer, (str, os.PathLike)):
        for enc in _READ_ENCODINGS:
            try:
                return pd.read_csv(path_or_buffer, encoding=enc, **opts)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(
            path_or_buffer, encoding="utf-8", errors="replace", **opts
        )

    if isinstance(path_or_buffer, bytes):
        for enc in _READ_ENCODINGS:
            try:
                return pd.read_csv(io.BytesIO(path_or_buffer), encoding=enc, **opts)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(
            io.BytesIO(path_or_buffer), encoding="utf-8", errors="replace", **opts
        )

    raw = path_or_buffer.read()
    if hasattr(path_or_buffer, "seek"):
        try:
            path_or_buffer.seek(0)
        except OSError:
            pass
    for enc in _READ_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, **opts)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace", **opts)
