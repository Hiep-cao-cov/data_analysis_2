"""DataFrame validation and cleaning hooks for chart pipelines."""

import pandas as pd
import streamlit as st
from config import MATERIAL_CONFIG
from csv_utils import (
    ALLOW_NEGATIVE,
    normalize_dataframe_columns,
    preprocess_dataframe,
    STRICT_NUMERIC,
)

__all__ = ["validate_dataframe"]


def validate_dataframe(
    df,
    required_columns,
    material=None,
    country=None,
    chart_type=None,
    files_uploaded=False,
):
    """
    Validates DataFrame structure, data types, and value ranges.
    Mutates ``df`` in place during preprocessing.
    """
    if not files_uploaded:
        return False

    if df is None or df.empty:
        return False

    normalize_dataframe_columns(df)
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False

    preprocess_dataframe(df, required_columns, material=material)
    if df.empty:
        st.error(
            "No valid data rows remain after cleaning. Fix empty or invalid required fields "
            "(e.g. customer, demand, prices, or business plan min/base/max) and re-upload."
        )
        return False

    mat_key = material.lower() if material else None
    mat_config = MATERIAL_CONFIG.get(mat_key, {})

    if material:
        if "demand" in required_columns:
            expected_suppliers = mat_config.get("suppliers", [])
            available_suppliers = [col for col in expected_suppliers if col in df.columns]
            if not available_suppliers:
                st.error(f"No valid supplier columns found for {material}.")
                return False

        if country and "pocket price" in required_columns:
            expected_prices = mat_config.get("price_columns", {}).get(country, [])
            missing_prices = [p for p in expected_prices if p not in df.columns and p != "pocket price"]
            if missing_prices:
                st.warning(
                    f"Note: Some benchmark price columns are missing for {country}: {missing_prices}"
                )

    strictly_positive = list(STRICT_NUMERIC)
    allow_negative = list(ALLOW_NEGATIVE)
    all_numeric_to_check = [c for c in (strictly_positive + allow_negative) if c in df.columns]

    for col in all_numeric_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(
                f"Column '{col}' is still not numeric after cleaning. "
                "Check for text or symbols that are not valid numbers."
            )
            return False

        if df[col].isnull().any():
            st.error(
                f"Column '{col}' still has missing values after cleaning. "
                "This is unexpected; please report the file layout."
            )
            return False

        if col in strictly_positive and (df[col] < 0).any():
            st.error(
                f"Column '{col}' contains negative values, which is not allowed for this metric."
            )
            return False

    return True
