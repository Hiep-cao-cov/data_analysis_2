"""Y-axis and numeric range helpers for chart sliders (Streamlit + data)."""

import streamlit as st


def get_price_range(
    df,
    chart_type,
    material,
    country,
    customer_name=None,
    selected_price_columns=None,
):
    """Get the price range from dataframe for Y-axis slider."""
    if df.empty:
        return 0.0, 100.0
    try:
        df_filtered = (
            df[df["customer"] == customer_name]
            if customer_name and chart_type not in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"]
            else df
        )
        if chart_type == "Customer Bubble Chart (Centered)":
            price_columns = ["ppd"]
        else:
            price_columns = (
                ["pocket price", "vn_pp", "apac_pp"]
                if material == "TDI" and country == "Vietnam"
                else (
                    ["pocket price", "tw_pp", "apac_pp"]
                    if material == "TDI" and country == "Taiwan"
                    else ["pocket price", "vn_pp", "seap_pp", "apac_pp"]
                )
            )
        if selected_price_columns and chart_type == "Account price vs Volume":
            price_columns = [col for col in selected_price_columns if col in df_filtered.columns and col in price_columns]
        else:
            price_columns = [col for col in price_columns if col in df_filtered.columns]
        if not price_columns:
            st.warning(f"No valid price columns for {material} in {country}. Using default range.")
            return 0.0, 100.0
        min_price = float(df_filtered[price_columns].min().min())
        max_price = float(df_filtered[price_columns].max().max())
        padding = (max_price - min_price) * 0.1 if max_price > min_price else 10.0
        if chart_type == "Customer Bubble Chart (Centered)":
            padding = max(0.005, padding)
            return min_price - padding, max_price + padding
        return max(0.0, min_price - padding), max_price + padding
    except (ValueError, TypeError) as e:
        st.error(f"Error in get_price_range: {str(e)}")
        return 0.0, 100.0


def get_demand_range(df, chart_type, customer_name=None):
    """Get the demand/value range for Y-axis slider (used by extended UIs; keep for API parity)."""
    if df.empty:
        return 0.0, 100.0
    try:
        df_filtered = (
            df[df["customer"] == customer_name]
            if customer_name and chart_type not in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"]
            else df
        )
        if chart_type == "Customer Demand" and "demand" in df_filtered.columns:
            min_val = float(df_filtered["demand"].min())
            max_val = float(df_filtered["demand"].max())
        elif chart_type == "Account price vs Volume" and "demand" in df_filtered.columns:
            min_val = float(df_filtered["demand"].min())
            max_val = float(df_filtered["demand"].max())
        elif chart_type == "Business plan" and all(col in df_filtered.columns for col in ["min", "base", "max"]):
            total_val = df_filtered[["min", "base", "max"]].sum(axis=1)
            min_val = float(total_val.min())
            max_val = float(total_val.max())
        elif chart_type == "Customer Bubble Chart (Centered)" and "volume" in df_filtered.columns:
            min_val = float(df_filtered["volume"].min())
            max_val = float(df_filtered["volume"].max())
        else:
            return 0.0, 100.0
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
        return max(0.0, min_val - padding), max_val + padding
    except (ValueError, TypeError) as e:
        st.error(f"Error in get_demand_range: {str(e)}")
        return 0.0, 100.0
