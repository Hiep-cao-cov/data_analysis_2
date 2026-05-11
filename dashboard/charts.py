"""Streamlit-facing chart builders: validate, then delegate to drawchat."""

import pandas as pd
import streamlit as st
import drawchat
from config import REQUIRED_COLUMNS, SUPPLIERS

from dashboard.plotly_legend import update_legend_horizontal_bottom, update_legend_vertical_right
from dashboard.validation import validate_dataframe
from dashboard.year_filters import filter_dataframe_by_calendar_years, resolve_year_for_bubble_charts


def plot_customer_demand(
    df,
    customer_name,
    material,
    is_taiwan,
    title_fontsize,
    axis_label_fontsize,
    tick_fontsize,
    legend_fontsize,
    legend_title_fontsize,
    percentage_label_fontsize,
    customer_name_font_size,
    demand_label_font_size,
    y_min,
    y_max,
    demand_power_alpha=1.0,
    selected_years=None,
):
    """Plot customer demand chart with legend at bottom."""
    if not validate_dataframe(
        df,
        REQUIRED_COLUMNS["demand_charts"],
        material=material,
        chart_type="Customer Demand",
        files_uploaded=True,
    ):
        return None
    if selected_years is not None and len(selected_years) == 0:
        st.warning("Select at least one year to show on the chart.")
        return None
    if selected_years is not None:
        df = filter_dataframe_by_calendar_years(df, selected_years)
        if df.empty:
            st.warning("No rows left for the selected year(s).")
            return None
    suppliers = SUPPLIERS[material.lower()]
    available_suppliers = drawchat.demand_chart_volume_columns(
        df.columns, material=material, suppliers_fallback=suppliers
    )
    if not available_suppliers:
        st.error(f"No valid supplier columns for {material}")
        return None
    df_filtered = df[df["customer"] == customer_name]
    max_demand = 0.0
    if not df_filtered.empty:
        for _, row in df_filtered.iterrows():
            row_sum = sum(float(row[c]) if pd.notna(row[c]) else 0.0 for c in available_suppliers)
            max_demand = max(max_demand, row_sum)
    try:
        fig = drawchat.plot_customer_demand(
            df,
            customer_name,
            "customer",
            available_suppliers,
            "year",
            (0, max_demand * 1.4),
            title_fontsize,
            axis_label_fontsize,
            tick_fontsize,
            legend_fontsize,
            legend_title_fontsize,
            percentage_label_fontsize,
            customer_name_font_size,
            demand_label_font_size,
            y_min,
            y_max,
            demand_power_alpha,
            material=material,
        )
        update_legend_vertical_right(fig, legend_fontsize)
        return fig
    except Exception as e:
        st.error(f"Error generating Customer Demand chart: {str(e)}")
        return None


def plot_price_volume(
    df,
    customer_name,
    material,
    is_taiwan,
    title_fontsize,
    axis_label_fontsize,
    tick_fontsize,
    legend_fontsize,
    legend_title_fontsize,
    percentage_label_fontsize,
    customer_name_font_size,
    demand_label_font_size,
    y_min,
    y_max,
    y_demand_min,
    y_demand_max,
    selected_price_columns,
    price_annotation_fontsize,
    annotation_spacing,
    selected_years=None,
):
    """Plot price vs volume chart with selected price columns and legend at bottom."""
    if not validate_dataframe(
        df,
        REQUIRED_COLUMNS["price_charts"],
        material=material,
        chart_type="Account price vs Volume",
        files_uploaded=True,
    ):
        return None
    if selected_years is not None and len(selected_years) == 0:
        st.warning("Select at least one year to show on the chart.")
        return None
    if selected_years is not None:
        df = filter_dataframe_by_calendar_years(df, selected_years)
        if df.empty:
            st.warning("No rows left for the selected year(s).")
            return None
    df_filtered = df[df["customer"] == customer_name]
    max_demand = df_filtered["demand"].max() if not df_filtered.empty else 0
    max_price = df_filtered["pocket price"].max() if not df_filtered.empty else 0
    price_config = {
        "TDI": (["pocket price", "vn_pp", "apac_pp"], ["red", "purple", "green"])
        if not is_taiwan
        else (["pocket price", "tw_pp", "apac_pp"], ["red", "purple", "green"]),
        "PMDI": (["pocket price", "vn_pp", "seap_pp", "apac_pp"], ["red", "green", "blue", "purple"]),
    }
    all_price_columns, price_colors = price_config[material]
    price_columns = [col for col in selected_price_columns if col in df.columns and col in all_price_columns]
    if not price_columns:
        st.error(f"No valid selected price columns for {material}")
        return None
    color_map = dict(zip(all_price_columns, price_colors))
    selected_colors = [color_map[col] for col in price_columns]
    try:
        fig = drawchat.plot_customer_demand_with_price(
            df,
            customer_name,
            "customer",
            SUPPLIERS[material.lower()],
            "year",
            (0, max_demand * 2),
            (0.5, max_price * 1.5),
            price_columns,
            selected_colors,
            title_fontsize,
            axis_label_fontsize,
            tick_fontsize,
            legend_fontsize,
            legend_title_fontsize,
            percentage_label_fontsize,
            price_annotation_fontsize,
            annotation_spacing,
            customer_name_font_size,
            demand_label_font_size,
            y_min,
            y_max,
            y_demand_min,
            y_demand_max,
        )
        if fig:
            if y_demand_min is not None and y_demand_max is not None:
                fig.update_layout(
                    yaxis=dict(range=[y_demand_min, y_demand_max], autorange=False),
                    yaxis3=dict(range=[y_demand_min, y_demand_max], autorange=False),
                )
            if y_min is not None and y_max is not None:
                fig.update_layout(yaxis2=dict(range=[y_min, y_max], autorange=False))
            update_legend_horizontal_bottom(fig, legend_fontsize)
        return fig
    except Exception as e:
        st.error(f"Error generating Account price vs Volume chart: {str(e)}")
        return None


def plot_bubble_chart(
    df,
    customer_name,
    material,
    is_taiwan,
    title_fontsize,
    axis_label_fontsize,
    tick_fontsize,
    legend_fontsize,
    bubble_scale,
    alpha,
    customer_name_font_size,
    demand_label_font_size,
    y_min,
    y_max,
    year_filter,
):
    """Non-centered bubble: pocket price vs a supplier share column."""
    if not validate_dataframe(
        df,
        REQUIRED_COLUMNS["price_charts"],
        material=material,
        chart_type="Customer bubble Chart",
        files_uploaded=True,
    ):
        return None

    df_filtered, year_filter = resolve_year_for_bubble_charts(df, year_filter)

    settings_info = f"📊 Bubble Scale: {bubble_scale:.1f} | Transparency: {alpha:.1f} | "
    settings_info += f"Customer Name Font Size: {customer_name_font_size} | "
    settings_info += f"Volume Label Font Size: {demand_label_font_size}"
    if y_min is not None and y_max is not None:
        settings_info += f" | Y-axis: {y_min:.1f} - {y_max:.1f}"
    if year_filter is not None:
        settings_info += f" | Year: {year_filter}"
    st.info(settings_info)
    try:
        suppliers = SUPPLIERS.get(material.lower(), [])
        demand_column = (
            "covestro" if "covestro" in df_filtered.columns else next(
                (col for col in suppliers if col in df_filtered.columns), None
            )
        )
        if demand_column is None:
            st.error(f"No valid supplier columns found for {material} bubble chart.")
            return None
        chart_figure, _, _, _ = drawchat.plot_customer_bubble_clean_with_median(
            df_filtered,
            "customer",
            demand_column,
            "pocket price",
            year_filter,
            bubble_scale,
            alpha,
            title_fontsize,
            axis_label_fontsize,
            tick_fontsize,
            legend_fontsize,
            customer_name_font_size,
            demand_label_font_size,
            y_min,
            y_max,
        )
        update_legend_horizontal_bottom(chart_figure, legend_fontsize)
        return chart_figure
    except ValueError as e:
        st.error(f"Error generating bubble chart: {str(e)}")
        return None


def plot_bubble_chart_centered(
    df,
    material,
    title_fontsize,
    axis_label_fontsize,
    tick_fontsize,
    legend_fontsize,
    bubble_scale,
    alpha,
    customer_name_font_size,
    demand_label_font_size,
    y_min,
    y_max,
    year_filter,
    volume_min,
    volume_max,
    hidden_customers=None,
):
    """Centered PPD / SOW bubble from PPD file."""
    if df is None or df.empty:
        st.error(
            "No PPD data is loaded for Customer Bubble Chart (Centered). "
            "Upload the PPD CSV or load from Memory (e.g. VN_TDI_PPD.csv). "
            "Required columns: customer, year, sow, ppd, volume (numeric, no blank cells)."
        )
        return None
    if not validate_dataframe(
        df,
        REQUIRED_COLUMNS["bubble_centered"],
        chart_type="Customer Bubble Chart (Centered)",
        files_uploaded=True,
    ):
        return None

    df_filtered, year_filter = resolve_year_for_bubble_charts(df, year_filter)

    settings_info = f"📊 Bubble Scale: {bubble_scale:.1f} | Transparency: {alpha:.1f} | "
    settings_info += f"Customer Name Font Size: {customer_name_font_size} | "
    settings_info += f"Volume Label Font Size: {demand_label_font_size} | "
    settings_info += f"Volume range: {volume_min:.0f}–{volume_max:.0f} mt"
    if y_min is not None and y_max is not None:
        settings_info += f" | Y-axis: {y_min:.1f} - {y_max:.1f}"
    if year_filter is not None:
        settings_info += f" | Year: {year_filter}"
    st.info(settings_info)
    try:
        chart_figure = drawchat.plot_customer_bubble_centered(
            df_filtered,
            customer_column="customer",
            sow_column="sow",
            ppd_column="ppd",
            volume_column="volume",
            year_filter=year_filter,
            bubble_scale=bubble_scale,
            alpha=alpha,
            title_fontsize=title_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_fontsize=tick_fontsize,
            customer_name_font_size=customer_name_font_size,
            volume_label_font_size=demand_label_font_size,
            volume_min=volume_min,
            volume_max=volume_max,
            y_min=y_min,
            y_max=y_max,
            exclude_customers=hidden_customers,
        )
        update_legend_horizontal_bottom(chart_figure, legend_fontsize)
        return chart_figure
    except ValueError as e:
        st.error(f"Error generating centered bubble chart: {str(e)}")
        return None


def plot_business_plan(
    df,
    customer_name,
    material,
    is_taiwan,
    title_fontsize,
    axis_label_fontsize,
    tick_fontsize,
    legend_fontsize,
    percentage_label_fontsize,
):
    """Business plan min/base/max chart."""
    if not validate_dataframe(
        df,
        REQUIRED_COLUMNS["business_plan"],
        material=material,
        chart_type="Business plan",
        files_uploaded=True,
    ):
        return None
    try:
        return drawchat.plot_customer_business_plan(
            df,
            customer_name,
            is_taiwan,
            title_fontsize,
            axis_label_fontsize,
            tick_fontsize,
            legend_fontsize,
            percentage_label_fontsize,
        )
    except Exception as e:
        st.error(f"Error generating Business plan chart: {str(e)}")
        return None
