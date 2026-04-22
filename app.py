import hashlib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import drawchat
import io
import os
from eda import load_and_extract_dataframes
from config import REQUIRED_COLUMNS, SUPPLIERS, CHART_TYPES, COUNTRIES, MATERIAL_CONFIG, MATERIALS

# This could live in config.py or at the top of your script
#========================== Configuration Dictionaries ==========================#
# ======================== End Configuration Dictionaries =========================#

def normalize_dataframe_columns(df):
    """Strip and lowercase headers so files match config (e.g. ``Year``, `` sow `` → ``year``, ``sow``)."""
    if df is None or df.empty:
        return
    df.columns = pd.Index([str(c).strip().lower() for c in df.columns])


def _normalize_bubble_year(y):
    """Calendar year as int for filters, widget keys, and selectbox values (handles float/numpy scalars)."""
    if y is None:
        return None
    v = pd.to_numeric(y, errors="coerce")
    if pd.isna(v):
        return None
    return int(round(float(v)))


def _coerce_bubble_metric_columns(df: pd.DataFrame, cols: tuple) -> None:
    """Parse year / sow / ppd / volume from messy CSV (spaces, comma decimals, %, any cell dtype)."""
    for col in cols:
        if col not in df.columns:
            continue
        t = df[col].astype(str).str.strip()
        t = t.where(~t.str.lower().isin(['nan', 'nat', 'none', '<na>']), other='')
        t = t.str.replace(',', '.', regex=False)
        t = t.str.replace('%', '', regex=False)
        df[col] = pd.to_numeric(t, errors='coerce')


def validate_dataframe(df, required_columns, material=None, country=None, chart_type=None, files_uploaded=False):
    """
    Validates DataFrame structure, data types, and value ranges.
    
    Args:
        df (pd.DataFrame): The dataframe to validate.
        required_columns (list): Mandatory columns for the specific chart type.
        material (str): 'PMDI' or 'TDI'.
        country (str): 'Vietnam' or 'Taiwan'.
        chart_type (str): The name of the chart being generated.
        files_uploaded (bool): State of file upload to prevent premature errors.
    """
    # 1. Initial Checks
    if not files_uploaded:
        return False  # Silently return False if user hasn't uploaded files yet
    
    if df is None or df.empty:
        return False

    normalize_dataframe_columns(df)
    
    # 2. Structural Validation
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False

    # PPD / centered bubble: coerce metrics (handles object, string, category; comma decimals; stray %)
    if chart_type == "Customer Bubble Chart (Centered)":
        _coerce_bubble_metric_columns(df, ('year', 'sow', 'ppd', 'volume'))

    # 3. Dynamic Config Lookup (using MATERIAL_CONFIG)
    mat_key = material.lower() if material else None
    mat_config = MATERIAL_CONFIG.get(mat_key, {})

    # 4. Supplier/Price Column Check
    if material:
        # Check Suppliers for Demand Charts
        if 'demand' in required_columns:
            expected_suppliers = mat_config.get('suppliers', [])
            available_suppliers = [col for col in expected_suppliers if col in df.columns]
            if not available_suppliers:
                st.error(f"No valid supplier columns found for {material}.")
                return False
        
        # Check Price Benchmarks for Price Charts
        if country and 'pocket price' in required_columns:
            expected_prices = mat_config.get('price_columns', {}).get(country, [])
            missing_prices = [p for p in expected_prices if p not in df.columns and p != 'pocket price']
            if missing_prices:
                st.warning(f"Note: Some benchmark price columns are missing for {country}: {missing_prices}")

    # 5. Type and Value Range Validation
    # Columns that MUST be >= 0 (Volume, Demand, SOW, Years, etc.)
    strictly_positive = [
        'demand', 'volume', 'sow', 'year', 'min', 'base', 'max', 'covestro', 
        'tosoh', 'wanhua', 'kmc', 'basf', 'sabic', 'huntsman', 'mcns', 'hanwha'
    ]
    
    # Columns that can be negative (Premium/Discount metrics)
    allow_negative = ['ppd', 'pocket price', 'vn_pp', 'tw_pp', 'seap_pp', 'apac_pp']

    all_numeric_to_check = [c for c in (strictly_positive + allow_negative) if c in df.columns]

    for col in all_numeric_to_check:
        # Null check
        if df[col].isnull().any():
            st.error(f"Column '{col}' contains missing (NaN) values.")
            return False
            
        # Data Type check
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column '{col}' must be numeric. Please check for non-numeric characters.")
            return False
            
        # Range check (Only for strictly positive columns)
        if col in strictly_positive and (df[col] < 0).any():
            st.error(f"Column '{col}' contains negative values, which is not allowed for this metric.")
            return False

    return True

#--------------test code starts here----------------#
# Note: load_country_data / get_dataframe are not @st.cache_data — they use session_state and
# mutable DataFrames; caching returned stale empty frames after upload or in-memory load.

def load_country_data(dataframes, country, material, chart_type):
    """Load data into data_dict from provided DataFrames"""
    files_uploaded = (
        any(st.session_state.uploaded_files.values()) or
        st.session_state.get('find_in_memory', False) or
        st.session_state.get('upload_complete', False)
    )
    data_dict = {}
    
    if country == "Vietnam":
        if material == "PMDI":
            data_dict['mdi'] = dataframes.get('df_mdi', pd.DataFrame()) if validate_dataframe(dataframes.get('df_mdi', pd.DataFrame()), REQUIRED_COLUMNS['price_charts'], material=material, country=country, files_uploaded=files_uploaded) else pd.DataFrame()
            data_dict['mdi_bp'] = dataframes.get('df_mdi_bp', pd.DataFrame()) if validate_dataframe(dataframes.get('df_mdi_bp', pd.DataFrame()), REQUIRED_COLUMNS['business_plan'], material=material, country=country, files_uploaded=files_uploaded) else pd.DataFrame()
            data_dict['vn_ppd_2024'] = dataframes.get('df_ppd', pd.DataFrame()) if validate_dataframe(dataframes.get('df_ppd', pd.DataFrame()), REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)", files_uploaded=files_uploaded) else pd.DataFrame()
        else:
            data_dict['tdi'] = dataframes.get('df_tdi', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi', pd.DataFrame()), REQUIRED_COLUMNS['price_charts'], material=material, country=country, files_uploaded=files_uploaded) else pd.DataFrame()
            data_dict['tdi_bp'] = dataframes.get('df_tdi_bp', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi_bp', pd.DataFrame()), REQUIRED_COLUMNS['business_plan'], material=material, country=country, files_uploaded=files_uploaded) else pd.DataFrame()
            data_dict['vn_ppd_2024'] = dataframes.get('df_ppd', pd.DataFrame()) if validate_dataframe(dataframes.get('df_ppd', pd.DataFrame()), REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)", files_uploaded=files_uploaded) else pd.DataFrame()
    else:  # Taiwan
        data_dict['tw_tdi'] = dataframes.get('df_tdi', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi', pd.DataFrame()), REQUIRED_COLUMNS['price_charts'], material=material, country=country, files_uploaded=files_uploaded) else pd.DataFrame()
        data_dict['tw_tdi_bp'] = dataframes.get('df_tdi_bp', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi_bp', pd.DataFrame()), REQUIRED_COLUMNS['business_plan'], material=material, country=country, files_uploaded=files_uploaded) else pd.DataFrame()
        data_dict['tw_ppd_2024'] = dataframes.get('df_ppd', pd.DataFrame()) if validate_dataframe(dataframes.get('df_ppd', pd.DataFrame()), REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)", files_uploaded=files_uploaded) else pd.DataFrame()
    
    return data_dict


def get_dataframe(chart_type, material, data_dict, country):
    """Select appropriate dataframe based on chart type, material, and country"""
    files_uploaded = (
        any(st.session_state.uploaded_files.values()) or
        st.session_state.get('find_in_memory', False) or
        st.session_state.get('upload_complete', False)
    )
    if not files_uploaded:
        return pd.DataFrame()
    
    # Map chart types to their data_dict keys
    mapping = {
        "Customer Bubble Chart (Centered)": 'vn_ppd_2024' if country == "Vietnam" else 'tw_ppd_2024',
        "Business plan": 'mdi_bp' if material == "PMDI" else 'tdi_bp' if 'tdi_bp' in data_dict else 'tw_tdi_bp',
        "Customer Demand": 'mdi' if material == "PMDI" else 'tdi' if 'tdi' in data_dict else 'tw_tdi',
        "Account price vs Volume": 'mdi' if material == "PMDI" else 'tdi' if 'tdi' in data_dict else 'tw_tdi',
        "Customer bubble Chart": 'mdi' if material == "PMDI" else 'tdi' if 'tdi' in data_dict else 'tw_tdi'
    }
    
    key = mapping.get(chart_type)
    
    # Logic Change: If the key is missing from data_dict, it means that specific file wasn't uploaded
    if key not in data_dict or data_dict[key].empty:
        if chart_type == "Customer Bubble Chart (Centered)":
            st.info("Load or upload the **PPD** CSV for this country/material (e.g. VN_TDI_PPD.csv). Validation must pass: columns **customer, year, sow, ppd, volume** with numeric values and no blanks.")
        elif "bp" in (key or ""):
            st.info(f"Please upload the **Business Plan** CSV to view the {chart_type}.")
        else:
            st.info(f"Please upload the **Main Data** CSV to view the {chart_type}.")
        return pd.DataFrame()
        
    return data_dict[key]


def get_price_range(df, chart_type, material, country, customer_name=None, selected_price_columns=None):
    """Get the price range from dataframe for Y-axis slider"""
    if df.empty:
        return 0.0, 100.0
    try:
        df_filtered = df[df['customer'] == customer_name] if customer_name and chart_type not in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"] else df
        if chart_type == "Customer Bubble Chart (Centered)":
            price_columns = ['ppd']
        else:
            price_columns = (
                ['pocket price', 'vn_pp', 'apac_pp'] if material == 'TDI' and country == 'Vietnam' else
                ['pocket price', 'tw_pp', 'apac_pp'] if material == 'TDI' and country == 'Taiwan' else
                ['pocket price', 'vn_pp', 'seap_pp', 'apac_pp']
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
    """Get the demand/value range for Y-axis slider"""
    if df.empty:
        return 0.0, 100.0
    try:
        df_filtered = df[df['customer'] == customer_name] if customer_name and chart_type not in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"] else df
        if chart_type == "Customer Demand" and 'demand' in df_filtered.columns:
            min_val = float(df_filtered['demand'].min())
            max_val = float(df_filtered['demand'].max())
        elif chart_type == "Account price vs Volume" and 'demand' in df_filtered.columns:
            min_val = float(df_filtered['demand'].min())
            max_val = float(df_filtered['demand'].max())
        elif chart_type == "Business plan" and all(col in df_filtered.columns for col in ['min', 'base', 'max']):
            total_val = df_filtered[['min', 'base', 'max']].sum(axis=1)
            min_val = float(total_val.min())
            max_val = float(total_val.max())
        elif chart_type == "Customer Bubble Chart (Centered)" and 'volume' in df_filtered.columns:
            min_val = float(df_filtered['volume'].min())
            max_val = float(df_filtered['volume'].max())
        else:
            return 0.0, 100.0
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
        return max(0.0, min_val - padding), max_val + padding
    except (ValueError, TypeError) as e:
        st.error(f"Error in get_demand_range: {str(e)}")
        return 0.0, 100.0

def plot_customer_demand(df, customer_name, material, is_taiwan, title_fontsize, 
                         axis_label_fontsize, tick_fontsize, legend_fontsize, 
                         legend_title_fontsize, percentage_label_fontsize, 
                         customer_name_font_size, demand_label_font_size, y_min, y_max, demand_power_alpha=1.0):
    """Plot customer demand chart with legend at bottom"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['demand_charts'], material=material, chart_type="Customer Demand", files_uploaded=True):
        return None
    suppliers = SUPPLIERS[material.lower()]
    available_suppliers = [col for col in suppliers if col in df.columns]
    if not available_suppliers:
        st.error(f"No valid supplier columns for {material}")
        return None
    df_filtered = df[df['customer'] == customer_name]
    max_demand = df_filtered['demand'].max() if not df_filtered.empty else 0
    try:
        fig = drawchat.plot_customer_demand(
            df, customer_name, 'customer', available_suppliers, 'year', (0, max_demand * 1.4),
            title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, 
            legend_title_fontsize, percentage_label_fontsize, customer_name_font_size, 
            demand_label_font_size, y_min, y_max, demand_power_alpha
        )
        if fig:
            fig.update_layout(
                legend=dict(
                    orientation="v",
                    yanchor='top',
                    y=0.99,
                    xanchor='left',
                    x=1.02,
                    font=dict(size=legend_fontsize)
                )
            )
        return fig
    except Exception as e:
        st.error(f"Error generating Customer Demand chart: {str(e)}")
        return None

def plot_price_volume(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, legend_title_fontsize, percentage_label_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max, y_demand_min, y_demand_max, selected_price_columns, price_annotation_fontsize, annotation_spacing):
    """Plot price vs volume chart with selected price columns and legend at bottom"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['price_charts'], material=material, chart_type="Account price vs Volume", files_uploaded=True):
        return None
    df_filtered = df[df['customer'] == customer_name]
    max_demand = df_filtered['demand'].max() if not df_filtered.empty else 0
    max_price = df_filtered['pocket price'].max() if not df_filtered.empty else 0
    price_config = {
        'TDI': (
            ['pocket price', 'vn_pp', 'apac_pp'], ['red', 'purple', 'green']
        ) if not is_taiwan else (
            ['pocket price', 'tw_pp', 'apac_pp'], ['red', 'purple', 'green']
        ),
        'PMDI': (
            ['pocket price', 'vn_pp', 'seap_pp', 'apac_pp'], ['red', 'green', 'blue', 'purple']
        )
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
            df, customer_name, 'customer', SUPPLIERS[material.lower()], 'year',
            (0, max_demand * 2), (0.5, max_price * 1.5), price_columns, selected_colors,
            title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, 
            legend_title_fontsize, percentage_label_fontsize, price_annotation_fontsize, 
            annotation_spacing, customer_name_font_size, demand_label_font_size, 
            y_min, y_max, y_demand_min, y_demand_max
        )
        if fig:
            # NEW: Force custom ranges to override autorange bug
            if y_demand_min is not None and y_demand_max is not None:
                fig.update_layout(
                    yaxis=dict(range=[y_demand_min, y_demand_max], autorange=False),
                    yaxis3=dict(range=[y_demand_min, y_demand_max], autorange=False)
                )
            if y_min is not None and y_max is not None:
                fig.update_layout(yaxis2=dict(range=[y_min, y_max], autorange=False))
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=legend_fontsize)
                )
            )
        return fig
    except Exception as e:
        st.error(f"Error generating Account price vs Volume chart: {str(e)}")
        return None

def plot_bubble_chart(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, bubble_scale, alpha, customer_name_font_size, demand_label_font_size, y_min, y_max, year_filter):
    """Plot bubble chart with fixed size and legend at bottom"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['price_charts'], material=material, chart_type="Customer bubble Chart", files_uploaded=True):
        return None
    if 'year' not in df.columns and year_filter is not None:
        st.warning("Year column not found. Ignoring year filter.")
        year_filter = None
    if year_filter is not None and 'year' in df.columns:
        df_try = drawchat.filter_dataframe_by_year(df, year_filter)
        if not df_try.empty:
            df_filtered = df_try
        else:
            st.warning(f"No data available for year {year_filter}. Defaulting to first available year.")
            years = sorted(
                df['year'].dropna().unique(),
                key=lambda x: pd.to_numeric(x, errors='coerce'),
            )
            year_filter = years[0] if years else None
            st.session_state.chart_settings['bubble_year'] = year_filter
            df_filtered = drawchat.filter_dataframe_by_year(df, year_filter) if year_filter is not None else df
    elif 'year' in df.columns:
        y_cal = drawchat._series_calendar_year(df['year']).dropna()
        if not y_cal.empty:
            year_filter = int(y_cal.max())
            df_filtered = drawchat.filter_dataframe_by_year(df, year_filter)
        else:
            df_filtered = df
    else:
        df_filtered = df
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
        demand_column = 'covestro' if 'covestro' in df_filtered.columns else next(
            (col for col in suppliers if col in df_filtered.columns),
            None
        )
        if demand_column is None:
            st.error(f"No valid supplier columns found for {material} bubble chart.")
            return None
        chart_figure, _, _, _ = drawchat.plot_customer_bubble_clean_with_median(
            df_filtered, 'customer', demand_column, 'pocket price', year_filter, bubble_scale, alpha,
            title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize,
            customer_name_font_size, demand_label_font_size, y_min, y_max
        )
        if chart_figure:
            chart_figure.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=legend_fontsize)
                )
            )
        return chart_figure
    except ValueError as e:
        st.error(f"Error generating bubble chart: {str(e)}")
        return None 

def plot_bubble_chart_centered(df, material, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, bubble_scale, alpha, customer_name_font_size, demand_label_font_size, y_min, y_max, year_filter, volume_min, volume_max, hidden_customers=None):
    """Plot centered bubble chart with SOW and PPD axes and legend at bottom"""
    if df is None or df.empty:
        st.error(
            "No PPD data is loaded for Customer Bubble Chart (Centered). "
            "Upload the PPD CSV or load from Memory (e.g. VN_TDI_PPD.csv). "
            "Required columns: customer, year, sow, ppd, volume (numeric, no blank cells)."
        )
        return None
    if not validate_dataframe(df, REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)", files_uploaded=True):
        return None
    if 'year' not in df.columns and year_filter is not None:
        st.warning("Year column not found. Ignoring year filter.")
        year_filter = None
    if year_filter is not None and 'year' in df.columns:
        df_try = drawchat.filter_dataframe_by_year(df, year_filter)
        if not df_try.empty:
            df_filtered = df_try
        else:
            st.warning(f"No data available for year {year_filter}. Defaulting to first available year.")
            years = sorted(
                df['year'].dropna().unique(),
                key=lambda x: pd.to_numeric(x, errors='coerce'),
            )
            year_filter = years[0] if years else None
            st.session_state.chart_settings['bubble_year'] = year_filter
            df_filtered = drawchat.filter_dataframe_by_year(df, year_filter) if year_filter is not None else df
    elif 'year' in df.columns:
        y_cal = drawchat._series_calendar_year(df['year']).dropna()
        if not y_cal.empty:
            year_filter = int(y_cal.max())
            df_filtered = drawchat.filter_dataframe_by_year(df, year_filter)
        else:
            df_filtered = df
    else:
        df_filtered = df
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
            customer_column='customer',
            sow_column='sow',
            ppd_column='ppd',
            volume_column='volume',
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
        if chart_figure:
            chart_figure.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=legend_fontsize)
                )
            )
        return chart_figure
    except ValueError as e:
        st.error(f"Error generating centered bubble chart: {str(e)}")
        return None

def plot_business_plan(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, percentage_label_fontsize):
    """Plot business plan chart"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['business_plan'], material=material, chart_type="Business plan", files_uploaded=True):
        return None
    try:
        return drawchat.plot_customer_business_plan(
            df, customer_name, is_taiwan, title_fontsize, axis_label_fontsize, 
            tick_fontsize, legend_fontsize, percentage_label_fontsize
        )
    except Exception as e:
        st.error(f"Error generating Business plan chart: {str(e)}")
        return None

def setup_page():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="PMDI and TDI Visualization", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.sidebar.title("🎯 Navigation")

def load_data_from_memory(country, material):
    """Load predefined CSV files from local data folder based on country/material selection."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_map = {
        ("Vietnam", "PMDI"): {
            "main": "VN_MDI_FINAL.csv",
            "bp": "VN_MDI_BP.csv",
            "ppd": "VN_MDI_PPD.csv"
        },
        ("Vietnam", "TDI"): {
            "main": "VN_TDI_FINAL.csv",
            "bp": "VN_TDI_BP.csv",
            "ppd": "VN_TDI_PPD.csv"
        },
        ("Taiwan", "TDI"): {
            "main": "TW_TDI_FINAL.csv",
            "bp": "TW_TDI_BP.csv",
            "ppd": "TW_TDI_PPD.csv"
        }
    }

    selected = file_map.get((country, material))
    if not selected:
        return {}, False

    table_main = 'df_mdi' if material == 'PMDI' else 'df_tdi'
    table_bp = 'df_mdi_bp' if material == 'PMDI' else 'df_tdi_bp'

    dataframes = {}
    required_loaded = True
    for table_key, file_key in [(table_main, "main"), (table_bp, "bp")]:
        file_path = os.path.join(data_dir, selected[file_key])
        if not os.path.exists(file_path):
            st.error(f"Required memory file not found: {selected[file_key]}")
            required_loaded = False
            continue
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if df.empty:
                st.error(f"Required memory file is empty: {selected[file_key]}")
                required_loaded = False
                continue
            dataframes[table_key] = df
        except pd.errors.EmptyDataError:
            st.error(f"Failed to parse memory file (empty/no columns): {selected[file_key]}")
            required_loaded = False
        except Exception as e:
            st.error(f"Error reading memory file {selected[file_key]}: {str(e)}")
            required_loaded = False

    # Optional PPD file: load if available, otherwise skip silently.
    ppd_path = os.path.join(data_dir, selected["ppd"])
    if os.path.exists(ppd_path):
        try:
            ppd_df = pd.read_csv(ppd_path, encoding='utf-8')
            if not ppd_df.empty:
                dataframes['df_ppd'] = ppd_df
            else:
                st.warning(f"Optional PPD memory file is empty and will be skipped: {selected['ppd']}")
        except pd.errors.EmptyDataError:
            st.warning(f"Optional PPD memory file has no columns and will be skipped: {selected['ppd']}")
        except Exception as e:
            st.warning(f"Optional PPD memory file could not be read and will be skipped: {selected['ppd']} ({str(e)})")

    return dataframes, required_loaded

def reset_axis_ranges(chart_type, customer_name):
    """Reset all chart settings based on chart type and customer change"""
    if (st.session_state.get('previous_chart_type') != chart_type or 
        st.session_state.get('previous_customer') != customer_name):
        # Reset all chart settings to defaults
        st.session_state.chart_settings.update({
            'customer_name_font_size': 12,
            'demand_label_font_size': 14,
            'legend_font_size': 14,
            'y_min': None,
            'y_max': None,
            'bubble_y_min': None,
            'bubble_y_max': None,
            'price_volume_y_min': None,
            'price_volume_y_max': None,
            'y_demand_min': None,
            'y_demand_max': None,
            'bubble_scale': 5.0,
            'bubble_alpha': 0.7,
            'demand_power_alpha': 1.0,
            'use_custom_y_range': False,
            'use_custom_bubble_y_range': False,
            'use_custom_price_volume_y_range': False,
            'use_custom_y_demand_range': False,
        })
    st.session_state.previous_chart_type = chart_type
    st.session_state.previous_customer = customer_name

def get_chart_config(chart_type, customer_name_font_size, demand_label_font_size, legend_font_size, 
                     y_min, y_max, bubble_y_min, bubble_y_max, bubble_scale, bubble_alpha, demand_power_alpha,
                     price_volume_y_min, price_volume_y_max, y_demand_min, y_demand_max, **kwargs):
    """
    Return chart configuration dictionary tailored to chart type.
    The **kwargs at the end prevents crashes if extra keys are passed.
    """
    # 1. Base settings shared by all charts
    base_config = {
        'title_fontsize': 22,
        'axis_label_fontsize': 16,
        'tick_fontsize': 12,
        'legend_fontsize': legend_font_size,
    }
    
    # 2. Add specific settings based on the chart type
    if chart_type == "Customer bubble Chart":
        base_config.update({
            'bubble_scale': bubble_scale,
            'alpha': bubble_alpha,
            'y_min': bubble_y_min,
            'y_max': bubble_y_max,
            'customer_name_font_size': customer_name_font_size,
            'demand_label_font_size': demand_label_font_size,
        })
        
    elif chart_type == "Customer Bubble Chart (Centered)":
        base_config.update({
            'bubble_scale': bubble_scale,
            'alpha': bubble_alpha,
            'y_min': bubble_y_min,
            'y_max': bubble_y_max,
            'customer_name_font_size': customer_name_font_size,
            'demand_label_font_size': demand_label_font_size,
        })
        
    elif chart_type == "Account price vs Volume":
        base_config.update({
            'y_min': price_volume_y_min,
            'y_max': price_volume_y_max,
            'y_demand_min': y_demand_min,
            'y_demand_max': y_demand_max,
            'price_annotation_fontsize': 12,
            'annotation_spacing': 0.1,
            'legend_title_fontsize': 14,
            'percentage_label_fontsize': 12,
            'customer_name_font_size': customer_name_font_size,
            'demand_label_font_size': demand_label_font_size,
        })
        
    elif chart_type == "Customer Demand":
        base_config.update({
            'y_min': y_min,
            'y_max': y_max,
            'demand_power_alpha': demand_power_alpha,
            'legend_title_fontsize': 14,
            'percentage_label_fontsize': 12,
            'customer_name_font_size': customer_name_font_size,
            'demand_label_font_size': demand_label_font_size,
        })
        
    else:  # Business plan
        base_config.update({
            'percentage_label_fontsize': 12,
        })
    
    return base_config


def _centered_bubble_checkbox_widget_key(year_filter, cust_str):
    h = hashlib.md5(f"{year_filter}|{cust_str}".encode("utf-8")).hexdigest()[:16]
    return f"cbubble_chk_{year_filter}_{h}"


def hidden_customers_centered_bubble_from_widgets(df, year_filter, volume_min, volume_max):
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
        k = _centered_bubble_checkbox_widget_key(year_filter, str(name))
        if not st.session_state.get(k, True):
            hidden.add(str(name))
    return hidden


def main_app(dataframes, country, material, show_upload_section):
    """
    Main application logic with professional UI, Tabbed navigation, 
    and robust error handling for data selection.
    """
    # --- 1. INITIALIZE SESSION STATE (Must be first to prevent AttributeErrors) ---
    if 'chart_settings' not in st.session_state:
        st.session_state.chart_settings = {
            'customer_name_font_size': 12,
            'demand_label_font_size': 14,
            'legend_font_size': 14,
            'y_min': None, 'y_max': None,
            'bubble_y_min': None, 'bubble_y_max': None,
            'price_volume_y_min': None, 'price_volume_y_max': None,
            'y_demand_min': None, 'y_demand_max': None,
            'bubble_scale': 5.0,
            'bubble_alpha': 0.7,
            'demand_power_alpha': 1.0,
            'bubble_year': None,
            'auto_generate_chart': False
        }

    # Tracking for state changes
    if 'previous_chart_type' not in st.session_state:
        st.session_state.previous_chart_type = None
    if 'previous_customer' not in st.session_state:
        st.session_state.previous_customer = None

    # --- 2. GLOBAL STYLING ---
    st.markdown("""
        <style>
        .block-container {padding-top: 1.5rem; max-width: 98%;}
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 5px 5px 0px 0px;
            padding: 10px 20px;
        }
        /* Centered bubble: customer name list — smaller type; grey when hidden (unchecked); tight row spacing */
        [data-testid="stExpander"] [data-testid="stCheckbox"] {
            min-height: unset;
            margin-top: -0.35rem !important;
            margin-bottom: -0.35rem !important;
        }
        /* Show names only; whole row stays clickable via label */
        [data-testid="stExpander"] [data-testid="stCheckbox"] label > div:first-child {
            display: none !important;
        }
        [data-testid="stExpander"] [data-testid="stCheckbox"] label[data-testid="stWidgetLabel"] p {
            font-size: 0.78rem !important;
            font-weight: 400 !important;
            line-height: 1.35 !important;
            margin: 0 !important;
        }
        [data-testid="stExpander"] [data-testid="stCheckbox"]:has(input:not(:checked)) label[data-testid="stWidgetLabel"] p {
            color: #9ca3af !important;
        }
        [data-testid="stExpander"] [data-testid="stCheckbox"]:has(input:checked) label[data-testid="stWidgetLabel"] p {
            color: #111827 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- 3. TOP NAVIGATION & HEADER ---
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.title(f"📊 {material} Market Dashboard")
        st.caption(f"Analysis Region: {country} | Collaborative Intelligence")
    with col_h2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    st.divider()# Determine current view
    # --- 4. DATA LOADING ---
    chart_type = st.sidebar.radio("Select Visualization", CHART_TYPES, key="chart_type_select")
    # Fetch data into data_dict based on your existing load function
    data_dict = load_country_data(st.session_state.dataframes, country, material, chart_type)
    df_current = get_dataframe(chart_type, material, data_dict, country)
    normalize_dataframe_columns(df_current)

    # --- 5. MAIN WORKSPACE ---
    tab_vis, tab_data, tab_settings = st.tabs(["📈 Visualization", "📝 Data Editor", "⚙️ Layout Settings"])

    # TAB 1: VISUALIZATION
    with tab_vis:
        is_bubble_chart = "bubble" in chart_type.lower()
        selected_bubble_year = None
        # Run year widget BEFORE columns: Streamlit renders col_plot (left) before col_ctrl (right),
        # so the chart must not read bubble_year_select until after the selectbox runs — place it here.
        if not df_current.empty and is_bubble_chart and 'year' in df_current.columns:
            _y_opts = []
            for x in df_current["year"].dropna().unique():
                yi = _normalize_bubble_year(x)
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
                    help="Bubble charts use only rows for this calendar year (2024 vs 2025 are never mixed).",
                )
        selected_bubble_year = _normalize_bubble_year(selected_bubble_year)

        col_plot, col_ctrl = st.columns([3, 1])
        
        with col_ctrl:
            st.subheader("Filters")
            customer_name = None
            
            if not df_current.empty:
                
                # 1. Customer Dropdown: Only shows for individual account charts
                if not is_bubble_chart:
                    if 'customer' in df_current.columns:
                        customers = sorted(df_current['customer'].unique())
                        customer_name = st.selectbox("Select Account", customers, key="customer_select")
                    else:
                        st.warning("No 'customer' column found.")
           
                    
                if is_bubble_chart and 'year' not in df_current.columns:
                    st.error("Column 'year' is required for Bubble Charts.")

                if chart_type == "Customer Bubble Chart (Centered)" and 'year' in df_current.columns and not df_current.empty:
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
                    with st.expander("Customers", expanded=True):
                        if yr_vis is not None:
                            st.caption(
                                f"Accounts in **{yr_vis}** (volume filter applied). "
                                "Order and numbers match bubble hover (same order as rows in the CSV for this year)."
                            )
                        if not bubble_customers:
                            st.warning("No customers in the selected volume range for this year.")
                        for rank_i, cust in enumerate(bubble_customers, start=1):
                            st.checkbox(
                                f"{rank_i}. {cust}",
                                value=True,
                                key=_centered_bubble_checkbox_widget_key(yr_vis, str(cust)),
                            )
                        
                # 3. Price Series Filter
                if chart_type == "Account price vs Volume":
                    price_options = MATERIAL_CONFIG.get(material.lower(), {}).get('price_columns', {}).get(country, [])
                    available_prices = [p for p in price_options if p in df_current.columns]
                    st.multiselect("Price Series", options=available_prices, key="price_columns_select", default=[available_prices[0]] if available_prices else [])

            st.divider()
            generate_chart = st.button("🚀 Generate Chart", type="primary", use_container_width=True)

        with col_plot:
            # Render if button clicked OR if data was just edited/uploaded
            should_plot = generate_chart or st.session_state.chart_settings.get('auto_generate_chart')
            
            if should_plot and not df_current.empty:
                with st.spinner("Processing Visualization..."):
                    # Use existing config function
                    config = get_chart_config(chart_type, **st.session_state.chart_settings)
                    is_taiwan = (country == "Taiwan")
                    chart_fig = None

                    # Route to existing plotting library
                    if chart_type == "Customer Demand" and customer_name:
                        chart_fig = plot_customer_demand(df_current, customer_name, material, is_taiwan, **config)
                    
                    elif chart_type == "Account price vs Volume" and customer_name:
                        sel_prices = st.session_state.get('price_columns_select', ['pocket price'])
                        chart_fig = plot_price_volume(df_current, customer_name, material, is_taiwan, selected_price_columns=sel_prices, **config)
                    
                    elif chart_type == "Customer bubble Chart":
                        chart_fig = plot_bubble_chart(
                            df_current, None, material, is_taiwan, year_filter=selected_bubble_year, **config
                        )
                    
                    elif chart_type == "Customer Bubble Chart (Centered)":
                        yr = _normalize_bubble_year(selected_bubble_year)
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
                        # HTML Export
                        buffer = io.StringIO()
                        chart_fig.write_html(buffer, include_plotlyjs='cdn')
                        st.download_button("📥 Save as HTML", buffer.getvalue(), f"{chart_type}.html", "text/html")
            else:
                st.info("💡 **Ready to Analyze:** Please select an account and click **Generate Chart**.")

    # TAB 2: DATA EDITOR
    with tab_data:
        st.subheader("Data Management")
        if st.session_state.dataframes:
            table_key = st.selectbox("Select Table to Edit", list(st.session_state.dataframes.keys()))
            edited_df = st.data_editor(
                st.session_state.dataframes[table_key],
                num_rows="dynamic",
                key=f"editor_{table_key}"
            )
            if not edited_df.equals(st.session_state.dataframes[table_key]):
                st.session_state.dataframes[table_key] = edited_df
                # Preserve edited values across reruns and refresh chart immediately.
                st.session_state.data_edited = True
                st.session_state.chart_settings['auto_generate_chart'] = True

    # TAB 3: SETTINGS
    # --- TAB 3: SETTINGS ---
    with tab_settings:
        st.subheader("Visual & Axis Refinement")
        s1, s2, s3 = st.columns(3)
        
        with s1:
            st.markdown("##### 🔡 Fonts")
            st.slider("Customer Font", 8, 20, value=st.session_state.chart_settings['customer_name_font_size'], key="s_font_cust")
            st.slider("Legend Font", 8, 20, value=st.session_state.chart_settings['legend_font_size'], key="s_font_leg")
        
        with s2:
            st.markdown("##### 🫧 Bubbles")
            st.slider("Size Scale", 1.0, 50.0, value=st.session_state.chart_settings['bubble_scale'], key="s_bub_scale")
            st.slider("Transparency", 0.1, 1.0, value=st.session_state.chart_settings['bubble_alpha'], key="s_bub_alpha")
            if chart_type == "Customer Demand":
                st.slider(
                    "Demand Visual Power",
                    0.3,
                    1.0,
                    value=float(st.session_state.chart_settings.get('demand_power_alpha', 1.0)),
                    step=0.05,
                    key="s_demand_power_alpha",
                    help="1.0 = real scale. Lower values visually enlarge smaller stacks."
                )

        with s3:
            st.markdown("##### 📏 Y-Axis Range")
            # Logic for Account price vs Volume specific sliders
            if chart_type == "Account price vs Volume":
                # Get dynamic range from data
                r_min, r_max = get_price_range(df_current, chart_type, material, country, customer_name)
                
                st.session_state.chart_settings['price_volume_y_min'] = st.slider(
                    "Price Min ($/kg)", 
                    0.0, float(r_max * 1.5), 
                    value=float(st.session_state.chart_settings['price_volume_y_min'] or r_min),
                    step=0.1, key="s_p_min"
                )
                st.session_state.chart_settings['price_volume_y_max'] = st.slider(
                    "Price Max ($/kg)", 
                    0.0, float(r_max * 2.0), 
                    value=float(st.session_state.chart_settings['price_volume_y_max'] or r_max),
                    step=0.1, key="s_p_max"
                )
            else:
                st.info("Select 'Account price vs Volume' to adjust Price Y-axis.")

        # --- SYNC ALL SETTINGS BACK TO SESSION STATE ---
        st.session_state.chart_settings.update({
            'customer_name_font_size': st.session_state.s_font_cust,
            'legend_font_size': st.session_state.s_font_leg,
            'bubble_scale': st.session_state.s_bub_scale,
            'bubble_alpha': st.session_state.s_bub_alpha,
            'demand_power_alpha': st.session_state.get('s_demand_power_alpha', st.session_state.chart_settings.get('demand_power_alpha', 1.0))
        })

    # --- 7. STATE CLEANUP ---
    reset_axis_ranges(chart_type, customer_name)
    
def main():
    """Main entry point for the Streamlit app"""
    st.sidebar.header("🌎 Country and Material")
    country = st.sidebar.selectbox("Select Country", COUNTRIES, key="country_select")
    material_options = ["TDI"] if country == "Taiwan" else MATERIALS
    material = st.sidebar.selectbox("Select Material", material_options, key="material_select")
    st.sidebar.markdown("### Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        options=["Memory (preloaded files)", "Manual CSV Upload"],
        key="data_source_mode"
    )
    find_in_memory = (data_source == "Memory (preloaded files)")
    st.session_state.find_in_memory = find_in_memory
    
    # Initialize session state for uploaded files
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {
            'main_file': None,
            'bp_file': None,
            'ppd_file': None
        }
    if 'upload_complete' not in st.session_state:
        st.session_state.upload_complete = False
    if 'data_edited' not in st.session_state:
        st.session_state.data_edited = False
    if 'chart_settings' not in st.session_state:
        st.session_state.chart_settings = {
            'customer_name_font_size': 12,
            'demand_label_font_size': 14,
            'legend_font_size': 14,
            'y_min': None, 'y_max': None,
            'bubble_y_min': None, 'bubble_y_max': None,
            'price_volume_y_min': None, 'price_volume_y_max': None,
            'y_demand_min': None, 'y_demand_max': None,
            'bubble_scale': 5.0,
            'bubble_alpha': 0.7,
            'demand_power_alpha': 1.0,
            'bubble_year': None,
            'auto_generate_chart': False
        }
    
    # Reset uploaded files if country or material changes
    if (st.session_state.get('previous_country_main') != country or
        st.session_state.get('previous_material_main') != material or
        st.session_state.get('previous_data_source_main') != data_source):
        st.session_state.uploaded_files = {
            'main_file': None,
            'bp_file': None,
            'ppd_file': None
        }
        st.session_state.upload_complete = False
        st.session_state.dataframes = {}
        st.session_state.data_edited = False
    st.session_state.previous_country_main = country
    st.session_state.previous_material_main = material
    st.session_state.previous_data_source_main = data_source
    
    # Upload section visibility follows selected data source mode.
    show_upload_section = (data_source == "Manual CSV Upload")

    dataframes = {}
    all_uploaded = False

    if find_in_memory:
        dataframes, all_uploaded = load_data_from_memory(country, material)
        if all_uploaded:
            st.session_state.upload_complete = True
            if not st.session_state.data_edited:
                st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
            st.session_state.chart_settings['auto_generate_chart'] = True
            st.success("Loaded data from memory files successfully.")
    else:
        # Load data from uploaded files
        _, _, _, all_uploaded = load_and_extract_dataframes(country, material, show_upload_section, st.session_state.uploaded_files)

        # Populate dataframes from uploaded files
        for file_key, file_name in [('main_file', 'df_mdi' if material == 'PMDI' else 'df_tdi'),
                                    ('bp_file', 'df_mdi_bp' if material == 'PMDI' else 'df_tdi_bp'),
                                    ('ppd_file', 'df_ppd')]:
            if st.session_state.uploaded_files[file_key]:
                try:
                    file_obj = st.session_state.uploaded_files[file_key]
                    file_obj.seek(0)
                    if file_obj.size == 0:
                        st.error(f"Uploaded file for {file_name} is empty. Please upload a valid CSV file.")
                        continue
                    df = pd.read_csv(file_obj, encoding='utf-8')
                    if df.empty:
                        st.error(f"Uploaded file for {file_name} contains no data. Please upload a valid CSV file.")
                        continue
                    dataframes[file_name] = df
                except pd.errors.EmptyDataError:
                    st.error(f"Failed to parse {file_name}: File is empty or has no columns. Please upload a valid CSV file.")
                except Exception as e:
                    st.error(f"Error reading {file_name}: {str(e)}. Please ensure the file is a valid CSV.")

        # Update upload complete status and dataframes
        required_keys = ['df_mdi' if material == 'PMDI' else 'df_tdi',
                         'df_mdi_bp' if material == 'PMDI' else 'df_tdi_bp',
                         'df_ppd']
        if all(key in dataframes and not dataframes[key].empty for key in required_keys):
            st.session_state.upload_complete = True
            if not st.session_state.data_edited:
                st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
            st.session_state.chart_settings['auto_generate_chart'] = True
    
    # Always call main_app to ensure UI renders
    if dataframes and all_uploaded:
        st.success("All files uploaded successfully! Ready to generate charts.")
    main_app(dataframes, country, material, show_upload_section)

if __name__ == "__main__":
    main()