"""Shared year resolution for bubble charts (aligns with drawchat year helpers)."""

import pandas as pd
import streamlit as st
import drawchat


def normalize_bubble_year(y):
    """Calendar year as int for filters, selectbox, and session keys."""
    if y is None:
        return None
    v = pd.to_numeric(y, errors="coerce")
    if pd.isna(v):
        return None
    return int(round(float(v)))


def calendar_years_for_customer(df, customer_name, customer_col="customer", year_col="year"):
    """Sorted unique calendar years for ``customer_name`` (for demand / price-volume year pickers)."""
    if df is None or df.empty or year_col not in df.columns or not customer_name:
        return []
    sub = df[df[customer_col] == customer_name]
    if sub.empty:
        return []
    ycal = drawchat._series_calendar_year(sub[year_col]).dropna()
    if ycal.empty:
        return []
    return sorted({int(x) for x in ycal.unique()})


def filter_dataframe_by_calendar_years(df, selected_years, year_col="year"):
    """Keep only rows whose ``year`` resolves to one of ``selected_years`` (calendar ints)."""
    if df is None or df.empty or year_col not in df.columns:
        return df
    if selected_years is None:
        return df
    if len(selected_years) == 0:
        return df.iloc[0:0].copy()
    ycal = drawchat._series_calendar_year(df[year_col])
    want = {int(x) for x in selected_years}
    return df.loc[ycal.isin(want)].copy()


def resolve_year_for_bubble_charts(df, year_filter):
    """
    Same logic as former ``plot_bubble_chart`` / ``plot_bubble_chart_centered`` year branches.
    Returns ``(df_filtered, year_filter_effective)``.
    """
    if "year" not in df.columns and year_filter is not None:
        st.warning("Year column not found. Ignoring year filter.")
        year_filter = None

    if year_filter is not None and "year" in df.columns:
        df_try = drawchat.filter_dataframe_by_year(df, year_filter)
        if not df_try.empty:
            df_filtered = df_try
        else:
            st.warning(f"No data available for year {year_filter}. Defaulting to first available year.")
            years = sorted(
                df["year"].dropna().unique(),
                key=lambda x: pd.to_numeric(x, errors="coerce"),
            )
            year_filter = years[0] if years else None
            st.session_state.chart_settings["bubble_year"] = year_filter
            df_filtered = (
                drawchat.filter_dataframe_by_year(df, year_filter) if year_filter is not None else df
            )
    elif "year" in df.columns:
        y_cal = drawchat._series_calendar_year(df["year"]).dropna()
        if not y_cal.empty:
            year_filter = int(y_cal.max())
            df_filtered = drawchat.filter_dataframe_by_year(df, year_filter)
        else:
            df_filtered = df
    else:
        df_filtered = df

    return df_filtered, year_filter
