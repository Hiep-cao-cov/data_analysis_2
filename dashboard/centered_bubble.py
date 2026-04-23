"""Widget keys and hidden-customer set for the centered bubble chart."""

import hashlib

import streamlit as st
import drawchat


def centered_bubble_checkbox_widget_key(year_filter, cust_str):
    h = hashlib.md5(f"{year_filter}|{cust_str}".encode("utf-8")).hexdigest()[:16]
    return f"cbubble_chk_{year_filter}_{h}"


def hidden_customers_from_checkboxes(df, year_filter, volume_min, volume_max):
    """Unchecked checkboxes exclude customers from the centered bubble chart."""
    df_eff = df
    if (
        df is not None
        and not df.empty
        and year_filter is not None
        and "year" in df.columns
    ):
        df_eff = drawchat.filter_dataframe_by_year(df, year_filter)
    names = drawchat.list_customers_for_centered_bubble(
        df_eff,
        year_filter=None,
        volume_min=volume_min,
        volume_max=volume_max,
    )
    hidden = set()
    for name in names:
        k = centered_bubble_checkbox_widget_key(year_filter, str(name))
        if not st.session_state.get(k, True):
            hidden.add(str(name))
    return hidden


# Backward-compatible name used in earlier app.py
hidden_customers_centered_bubble_from_widgets = hidden_customers_from_checkboxes
