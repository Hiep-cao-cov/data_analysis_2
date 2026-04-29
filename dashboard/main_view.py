"""Main Streamlit layout: header, data load, visualization / data / settings tabs."""

import io
import os
from datetime import datetime

import drawchat
import pandas as pd
import streamlit as st
from config import CHART_TYPES, MATERIAL_CONFIG, MATERIALS, REQUIRED_COLUMNS
from csv_utils import ALLOW_NEGATIVE, STRICT_NUMERIC, coerce_messy_numeric, normalize_dataframe_columns

from dashboard.centered_bubble import (
    centered_bubble_checkbox_widget_key,
    hidden_customers_centered_bubble_from_widgets,
)
from dashboard.chart_config import get_chart_config, reset_axis_ranges
from dashboard.charts import (
    plot_bubble_chart,
    plot_bubble_chart_centered,
    plot_business_plan,
    plot_customer_demand,
    plot_price_volume,
)
from dashboard.data import get_dataframe, load_country_data
from dashboard.ranges import get_price_range
from dashboard.styles import MAIN_PAGE_STYLE
from dashboard.year_filters import normalize_bubble_year


def _expected_filename(country: str, material: str, table_key: str) -> str:
    if country == "Vietnam" and material == "PMDI":
        mapping = {
            "df_mdi": "VN_MDI_FINAL.csv",
            "df_mdi_bp": "VN_MDI_BP.csv",
            "df_ppd": "VN_MDI_PPD.csv",
        }
    elif country == "Vietnam" and material == "TDI":
        mapping = {
            "df_tdi": "VN_TDI_FINAL.csv",
            "df_tdi_bp": "VN_TDI_BP.csv",
            "df_ppd": "VN_TDI_PPD.csv",
        }
    else:
        mapping = {
            "df_tdi": "TW_TDI_FINAL.csv",
            "df_tdi_bp": "TW_TDI_BP.csv",
            "df_ppd": "TW_TDI_PPD.csv",
        }
    return mapping.get(table_key, f"{table_key}.csv")


def _required_columns_for_table(table_key: str) -> list[str]:
    if table_key in {"df_mdi", "df_tdi"}:
        return REQUIRED_COLUMNS["price_charts"]
    if table_key in {"df_mdi_bp", "df_tdi_bp"}:
        return REQUIRED_COLUMNS["business_plan"]
    if table_key == "df_ppd":
        return REQUIRED_COLUMNS["bubble_centered"]
    return []


def _validate_edited_dataframe(df: pd.DataFrame, table_key: str) -> list[str]:
    errors: list[str] = []
    probe = df.copy()
    normalize_dataframe_columns(probe)

    required_columns = _required_columns_for_table(table_key)
    missing = [c for c in required_columns if c not in probe.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")

    if "customer" in probe.columns:
        empty_customers = probe["customer"].astype(str).str.strip().eq("").sum()
        if empty_customers:
            errors.append(f"'customer' has {int(empty_customers)} empty cell(s).")

    numeric_candidates = [c for c in STRICT_NUMERIC + ALLOW_NEGATIVE if c in probe.columns]
    for col in numeric_candidates:
        source = probe[col]
        parsed = coerce_messy_numeric(source)
        bad_mask = source.notna() & source.astype(str).str.strip().ne("") & parsed.isna()
        bad_count = int(bad_mask.sum())
        if bad_count:
            bad_rows = list((probe.index[bad_mask] + 1).astype(int)[:5])
            errors.append(
                f"Column '{col}' has {bad_count} invalid value(s). Example row(s): {bad_rows}"
            )
    return errors


def _mark_editor_changed(table_key: str) -> None:
    st.session_state[f"editor_changed_{table_key}"] = True


def main_app(country, material, show_upload_section):
    """Main app body: session init, header, data_dict, and tabbed UI. Uses ``st.session_state.dataframes`` for charts."""
    del show_upload_section  # reserved for future use (upload mode lives in ``bootstrap``)

    if "chart_settings" not in st.session_state:
        st.session_state.chart_settings = {
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
            "bubble_year": None,
            "auto_generate_chart": False,
        }

    if "previous_chart_type" not in st.session_state:
        st.session_state.previous_chart_type = None
    if "previous_customer" not in st.session_state:
        st.session_state.previous_customer = None

    st.markdown(MAIN_PAGE_STYLE, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.title(f"📊 {material} Market Dashboard")
        st.caption(f"Analysis Region: {country} | Collaborative Intelligence")
    with col_h2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    st.divider()
    chart_type = st.sidebar.radio("Select Visualization", CHART_TYPES, key="chart_type_select")
    data_dict = load_country_data(st.session_state.dataframes, country, material, chart_type)
    df_current = get_dataframe(chart_type, material, data_dict, country)
    normalize_dataframe_columns(df_current)

    tab_vis, tab_data, tab_settings = st.tabs(["📈 Visualization", "📝 Data Editor", "⚙️ Layout Settings"])

    with tab_vis:
        st.markdown("##### Visualization Workspace")
        is_bubble_chart = "bubble" in chart_type.lower()
        selected_bubble_year = None
        if not df_current.empty and is_bubble_chart and "year" in df_current.columns:
            _y_opts = []
            for x in df_current["year"].dropna().unique():
                yi = normalize_bubble_year(x)
                if yi is not None:
                    _y_opts.append(yi)
            _y_opts = sorted(set(_y_opts), reverse=True)
            _y_sig = tuple(_y_opts)
            if _y_sig:
                if st.session_state.get("_bubble_year_opts_sig") != _y_sig:
                    st.session_state["_bubble_year_opts_sig"] = _y_sig
                    st.session_state.pop("bubble_year_select", None)
                selected_bubble_year = st.selectbox(
                    "Data Year",
                    _y_opts,
                    key="bubble_year_select",
                    help="Bubble charts use only rows for the selected calendar year (years are not mixed on one plot).",
                )
        selected_bubble_year = normalize_bubble_year(selected_bubble_year)

        col_plot, col_ctrl = st.columns([3, 1])

        with col_ctrl:
            st.markdown("##### Filter & Controls")
            customer_name = None
            with st.container(border=True):
                st.markdown("**Primary Filters**")

                if not df_current.empty:
                    if not is_bubble_chart:
                        if "customer" in df_current.columns:
                            customers = sorted(df_current["customer"].unique())
                            customer_name = st.selectbox("Select Account", customers, key="customer_select")
                        else:
                            st.warning("No 'customer' column found.")

                    if is_bubble_chart and "year" not in df_current.columns:
                        st.error("Column 'year' is required for Bubble Charts.")

            with st.container(border=True):
                st.markdown("**Advanced Controls**")
                if (
                    chart_type == "Customer Bubble Chart (Centered)"
                    and "year" in df_current.columns
                    and not df_current.empty
                ):
                    yr_vis = selected_bubble_year
                    if yr_vis is None:
                        y_cal = drawchat._series_calendar_year(df_current["year"]).dropna()
                        if not y_cal.empty:
                            yr_vis = int(y_cal.max())
                    df_year = (
                        drawchat.filter_dataframe_by_year(df_current, yr_vis)
                        if yr_vis is not None and "year" in df_current.columns
                        else df_current
                    )
                    vmin_data, vmax_data = drawchat.get_centered_bubble_volume_bounds(
                        df_current, year_filter=yr_vis
                    )
                    sig = (round(vmin_data, 6), round(vmax_data, 6))
                    sig_store = f"_centered_bubble_vol_bounds_sig_{yr_vis}"
                    if st.session_state.get(sig_store) != sig:
                        st.session_state[sig_store] = sig
                        sk_reset = f"centered_bubble_vol_range_{yr_vis}"
                        if sk_reset in st.session_state:
                            del st.session_state[sk_reset]

                    vol_slider_key = f"centered_bubble_vol_range_{yr_vis}"
                    _span = float(vmax_data - vmin_data)
                    step = 100.0 if _span >= 100 else max(1.0, _span / 50) if _span > 0 else 1.0
                    st.caption("Include bubbles only for customers whose total volume this year is in this range (mt).")
                    st.slider(
                        "Volume range (mt)",
                        min_value=vmin_data,
                        max_value=vmax_data,
                        value=(vmin_data, vmax_data),
                        step=step,
                        key=vol_slider_key,
                    )
                    vol_lo, vol_hi = st.session_state[vol_slider_key]

                    bubble_customers = drawchat.list_customers_for_centered_bubble(
                        df_year,
                        year_filter=None,
                        volume_min=vol_lo,
                        volume_max=vol_hi,
                    )
                    st.divider()
                    with st.expander("Customer Visibility", expanded=True):
                        if yr_vis is not None:
                            st.caption(
                                f"Accounts in **{yr_vis}** (volume filter applied). "
                                "Order matches bubble hover (CSV row order for this year)."
                            )
                        if not bubble_customers:
                            st.warning("No customers in the selected volume range for this year.")
                        for rank_i, cust in enumerate(bubble_customers, start=1):
                            st.checkbox(
                                f"{rank_i}. {cust}",
                                value=True,
                                key=centered_bubble_checkbox_widget_key(yr_vis, str(cust)),
                            )

                if chart_type == "Account price vs Volume" and not df_current.empty:
                    price_options = (
                        MATERIAL_CONFIG.get(material.lower(), {}).get("price_columns", {}).get(country, [])
                    )
                    available_prices = [p for p in price_options if p in df_current.columns]
                    st.multiselect(
                        "Price Series",
                        options=available_prices,
                        key="price_columns_select",
                        default=[available_prices[0]] if available_prices else [],
                    )

            st.divider()
            st.caption("Apply current filters and refresh the chart.")
            generate_chart = st.button("🚀 Generate Chart", type="primary", use_container_width=True)

        with col_plot:
            should_plot = generate_chart or st.session_state.chart_settings.get("auto_generate_chart")

            if should_plot and not df_current.empty:
                with st.spinner("Processing Visualization..."):
                    config = get_chart_config(chart_type, **st.session_state.chart_settings)
                    is_taiwan = country == "Taiwan"
                    chart_fig = None

                    if chart_type == "Customer Demand" and customer_name:
                        chart_fig = plot_customer_demand(df_current, customer_name, material, is_taiwan, **config)

                    elif chart_type == "Account price vs Volume" and customer_name:
                        sel_prices = st.session_state.get("price_columns_select", ["pocket price"])
                        chart_fig = plot_price_volume(
                            df_current,
                            customer_name,
                            material,
                            is_taiwan,
                            selected_price_columns=sel_prices,
                            **config,
                        )

                    elif chart_type == "Customer bubble Chart":
                        chart_fig = plot_bubble_chart(
                            df_current,
                            None,
                            material,
                            is_taiwan,
                            year_filter=selected_bubble_year,
                            **config,
                        )

                    elif chart_type == "Customer Bubble Chart (Centered)":
                        yr = normalize_bubble_year(selected_bubble_year)
                        if yr is None and "year" in df_current.columns:
                            y_cal = drawchat._series_calendar_year(df_current["year"]).dropna()
                            if not y_cal.empty:
                                yr = int(y_cal.max())
                        vol_key = f"centered_bubble_vol_range_{yr}"
                        vol_pair = st.session_state.get(vol_key)
                        if vol_pair and len(vol_pair) == 2:
                            v_lo, v_hi = float(vol_pair[0]), float(vol_pair[1])
                        else:
                            v_lo, v_hi = drawchat.get_centered_bubble_volume_bounds(
                                df_current, year_filter=yr
                            )
                        hidden_bubbles = hidden_customers_centered_bubble_from_widgets(
                            df_current, yr, v_lo, v_hi
                        )
                        chart_fig = plot_bubble_chart_centered(
                            df_current,
                            material,
                            year_filter=yr,
                            volume_min=v_lo,
                            volume_max=v_hi,
                            hidden_customers=hidden_bubbles,
                            **config,
                        )

                    elif chart_type == "Business plan" and customer_name:
                        chart_fig = plot_business_plan(df_current, customer_name, material, is_taiwan, **config)

                    if chart_fig:
                        st.plotly_chart(chart_fig, use_container_width=True)
                        buffer = io.StringIO()
                        chart_fig.write_html(buffer, include_plotlyjs="cdn")
                        st.download_button(
                            "📥 Save as HTML",
                            buffer.getvalue(),
                            f"{chart_type}.html",
                            "text/html",
                        )
            else:
                st.info("💡 **Ready to Analyze:** Please select an account and click **Generate Chart**.")

    with tab_data:
        st.subheader("Data Management")
        st.caption(
            "Review and edit source tables. Data stays in memory until session ends or new upload replaces it."
        )
        if st.session_state.dataframes:
            table_key = st.selectbox("Select Table to Edit", list(st.session_state.dataframes.keys()))
            editor_key = f"editor_{table_key}"
            edited_df = st.data_editor(
                st.session_state.dataframes[table_key],
                num_rows="dynamic",
                key=editor_key,
                on_change=_mark_editor_changed,
                args=(table_key,),
            )

            validation_errors = _validate_edited_dataframe(edited_df, table_key)
            has_changes = not edited_df.equals(st.session_state.dataframes[table_key])

            if st.session_state.get(f"editor_changed_{table_key}") and validation_errors:
                st.warning("Please fix invalid cells before applying changes or generating charts.")
                for msg in validation_errors[:8]:
                    st.write(f"- {msg}")

            c_apply, c_save = st.columns([1, 1])
            with c_apply:
                if st.button("Apply Changes to Session", type="primary", use_container_width=True):
                    if validation_errors:
                        st.error("Cannot apply changes. Data contains invalid values.")
                    elif has_changes:
                        st.session_state.dataframes[table_key] = edited_df.copy()
                        st.session_state.data_edited = True
                        st.session_state.chart_settings["auto_generate_chart"] = False
                        st.success("Changes applied. You can now generate visualization.")
                    else:
                        st.info("No changes detected.")

            with c_save:
                export_df = edited_df.copy() if has_changes else st.session_state.dataframes[table_key].copy()
                base_name = _expected_filename(country, material, table_key)
                name_without_ext = os.path.splitext(base_name)[0]
                modified_date = datetime.now().strftime("%Y%m%d-%H%M%S")
                output_name = f"{name_without_ext}-{modified_date}.csv"

                if validation_errors:
                    st.button("Save Edited CSV to Disk", use_container_width=True, disabled=True)
                    st.caption("Fix validation errors to enable CSV save.")
                else:
                    st.download_button(
                        "Save Edited CSV to Disk",
                        data=export_df.to_csv(index=False, encoding="utf-8-sig"),
                        file_name=output_name,
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.caption("Save location is managed by your browser download settings.")

    with tab_settings:
        st.subheader("Visual & Axis Refinement")
        st.caption("Adjust visual presentation only. Chart business logic and calculations remain unchanged.")
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("##### 🔡 Fonts")
            st.caption("Control text readability in labels and legends.")
            st.slider(
                "Customer Font",
                8,
                20,
                value=st.session_state.chart_settings["customer_name_font_size"],
                key="s_font_cust",
            )
            st.slider(
                "Legend Font",
                8,
                20,
                value=st.session_state.chart_settings["legend_font_size"],
                key="s_font_leg",
            )

        with s2:
            st.markdown("##### 🫧 Bubbles")
            st.caption("Tune bubble visibility and overlap behavior.")
            st.slider(
                "Size Scale",
                1.0,
                50.0,
                value=st.session_state.chart_settings["bubble_scale"],
                key="s_bub_scale",
            )
            st.slider(
                "Transparency",
                0.1,
                1.0,
                value=st.session_state.chart_settings["bubble_alpha"],
                key="s_bub_alpha",
            )
            if chart_type == "Customer Demand":
                st.slider(
                    "Demand Visual Power",
                    0.3,
                    1.0,
                    value=float(st.session_state.chart_settings.get("demand_power_alpha", 1.0)),
                    step=0.05,
                    key="s_demand_power_alpha",
                    help="1.0 = real scale. Lower values visually enlarge smaller stacks.",
                )

        with s3:
            st.markdown("##### 📏 Y-Axis Range")
            st.caption("Configure price-axis view for the price-volume chart.")
            if chart_type == "Account price vs Volume":
                r_min, r_max = get_price_range(df_current, chart_type, material, country, customer_name)
                st.session_state.chart_settings["price_volume_y_min"] = st.slider(
                    "Price Min ($/kg)",
                    0.0,
                    float(r_max * 1.5),
                    value=float(st.session_state.chart_settings["price_volume_y_min"] or r_min),
                    step=0.1,
                    key="s_p_min",
                )
                st.session_state.chart_settings["price_volume_y_max"] = st.slider(
                    "Price Max ($/kg)",
                    0.0,
                    float(r_max * 2.0),
                    value=float(st.session_state.chart_settings["price_volume_y_max"] or r_max),
                    step=0.1,
                    key="s_p_max",
                )
            else:
                st.info("Select 'Account price vs Volume' to adjust Price Y-axis.")

        st.session_state.chart_settings.update(
            {
                "customer_name_font_size": st.session_state.s_font_cust,
                "legend_font_size": st.session_state.s_font_leg,
                "bubble_scale": st.session_state.s_bub_scale,
                "bubble_alpha": st.session_state.s_bub_alpha,
                "demand_power_alpha": st.session_state.get(
                    "s_demand_power_alpha", st.session_state.chart_settings.get("demand_power_alpha", 1.0)
                ),
            }
        )

    reset_axis_ranges(chart_type, customer_name)
