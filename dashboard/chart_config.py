"""Session-backed chart setting defaults and per-chart config dicts for the main view."""

import streamlit as st

BUBBLE_CHART_NAMES = ("Customer bubble Chart", "Customer Bubble Chart (Centered)")


def get_chart_config(
    chart_type,
    customer_name_font_size,
    demand_label_font_size,
    legend_font_size,
    y_min,
    y_max,
    bubble_y_min,
    bubble_y_max,
    bubble_scale,
    bubble_alpha,
    demand_power_alpha,
    price_volume_y_min,
    price_volume_y_max,
    y_demand_min,
    y_demand_max,
    **kwargs,
):
    """Return kwargs dict passed into chart builders (extra kwargs ignored for forward compatibility)."""
    base = {
        "title_fontsize": 22,
        "axis_label_fontsize": 16,
        "tick_fontsize": 12,
        "legend_fontsize": legend_font_size,
    }

    if chart_type in BUBBLE_CHART_NAMES:
        base.update(
            {
                "bubble_scale": bubble_scale,
                "alpha": bubble_alpha,
                "y_min": bubble_y_min,
                "y_max": bubble_y_max,
                "customer_name_font_size": customer_name_font_size,
                "demand_label_font_size": demand_label_font_size,
            }
        )
    elif chart_type == "Account price vs Volume":
        base.update(
            {
                "y_min": price_volume_y_min,
                "y_max": price_volume_y_max,
                "y_demand_min": y_demand_min,
                "y_demand_max": y_demand_max,
                "price_annotation_fontsize": 12,
                "annotation_spacing": 0.1,
                "legend_title_fontsize": 14,
                "percentage_label_fontsize": 12,
                "customer_name_font_size": customer_name_font_size,
                "demand_label_font_size": demand_label_font_size,
            }
        )
    elif chart_type == "Customer Demand":
        base.update(
            {
                "y_min": y_min,
                "y_max": y_max,
                "demand_power_alpha": demand_power_alpha,
                "legend_title_fontsize": 14,
                "percentage_label_fontsize": 12,
                "customer_name_font_size": customer_name_font_size,
                "demand_label_font_size": demand_label_font_size,
            }
        )
    else:
        base["percentage_label_fontsize"] = 12

    return base


def reset_axis_ranges(chart_type, customer_name):
    """Reset axis-related UI defaults when chart type or account changes."""
    if st.session_state.get("previous_chart_type") != chart_type or st.session_state.get("previous_customer") != customer_name:
        st.session_state.chart_settings.update(
            {
                "customer_name_font_size": 12,
                "demand_label_font_size": 14,
                "legend_font_size": 14,
                "y_min": None,
                "y_max": None,
                "bubble_y_min": None,
                "bubble_y_max": None,
                "price_volume_y_min": None,
                "price_volume_y_max": None,
                "y_demand_min": None,
                "y_demand_max": None,
                "bubble_scale": 5.0,
                "bubble_alpha": 0.7,
                "demand_power_alpha": 1.0,
                "use_custom_y_range": False,
                "use_custom_bubble_y_range": False,
                "use_custom_price_volume_y_range": False,
                "use_custom_y_demand_range": False,
            }
        )
    st.session_state.previous_chart_type = chart_type
    st.session_state.previous_customer = customer_name
