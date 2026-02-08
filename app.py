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
        st.error(f"The data for {chart_type} is empty or was not loaded correctly.")
        return False
    
    # 2. Structural Validation
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False

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

@st.cache_data

def load_country_data(dataframes, country, material, chart_type):
    """Load data into data_dict from provided DataFrames"""
    files_uploaded = any(st.session_state.uploaded_files.values())
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

@st.cache_data
def get_dataframe(chart_type, material, data_dict, country):
    """Select appropriate dataframe based on chart type, material, and country"""
    files_uploaded = any(st.session_state.uploaded_files.values())
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
        # Check if the specific file for this chart was actually uploaded
        expected_file_label = "Business Plan" if "bp" in (key or "") else "Main Data"
        st.info(f"Please upload the **{expected_file_label}** CSV to view the {chart_type}.")
        return pd.DataFrame()
        
    return data_dict[key]

@st.cache_data
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

@st.cache_data
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

def plot_customer_demand(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, legend_title_fontsize, percentage_label_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max):
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
            demand_label_font_size, y_min, y_max
        )
        if fig:
            fig.update_layout(
                legend=dict(
                    orientation="v",
                    yanchor='middle',
                    y=0.5,
                    xanchor='right',
                    x=-0.15,
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
        fig = drawchat.plot_customer_demand_with_price_1(
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
    if year_filter is not None and 'year' in df.columns and not df[df['year'] == year_filter].empty:
        df_filtered = df[df['year'] == year_filter]
    else:
        if year_filter is not None and 'year' in df.columns:
            st.warning(f"No data available for year {year_filter}. Defaulting to first available year.")
            years = sorted(df['year'].unique())
            year_filter = years[0] if years else None
            st.session_state.chart_settings['bubble_year'] = year_filter
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
        chart_figure, _, _, _ = drawchat.plot_customer_bubble_clean_with_median(
            df_filtered, 'customer', 'covestro', 'pocket price', year_filter, bubble_scale, alpha,
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

def plot_bubble_chart_centered(df, material, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, bubble_scale, alpha, customer_name_font_size, demand_label_font_size, y_min, y_max, year_filter, min_volume_threshold):
    """Plot centered bubble chart with SOW and PPD axes and legend at bottom"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)", files_uploaded=True):
        return None
    if 'year' not in df.columns and year_filter is not None:
        st.warning("Year column not found. Ignoring year filter.")
        year_filter = None
    if year_filter is not None and 'year' in df.columns and not df[df['year'] == year_filter].empty:
        df_filtered = df[df['year'] == year_filter]
    else:
        if year_filter is not None and 'year' in df.columns:
            st.warning(f"No data available for year {year_filter}. Defaulting to first available year.")
            years = sorted(df['year'].unique())
            year_filter = years[0] if years else None
            st.session_state.chart_settings['bubble_year'] = year_filter
        df_filtered = df
    settings_info = f"📊 Bubble Scale: {bubble_scale:.1f} | Transparency: {alpha:.1f} | "
    settings_info += f"Customer Name Font Size: {customer_name_font_size} | "
    settings_info += f"Volume Label Font Size: {demand_label_font_size} | "
    settings_info += f"Min Volume Threshold: {min_volume_threshold}"
    if y_min is not None and y_max is not None:
        settings_info += f" | Y-axis: {y_min:.1f} - {y_max:.1f}"
    if year_filter is not None:
        settings_info += f" | Year: {year_filter}"
    st.info(settings_info)
    try:
        chart_figure = drawchat.plot_customer_bubble_centered(
            df_filtered, 'customer', 'sow', 'ppd', 'volume', year_filter, bubble_scale, alpha,
            title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize,
            customer_name_font_size, demand_label_font_size, min_volume_threshold, y_min, y_max
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

def reset_axis_ranges(chart_type, customer_name):
    """Reset all chart settings based on chart type and customer change"""
    if (st.session_state.get('previous_chart_type') != chart_type or 
        st.session_state.get('previous_customer') != customer_name):
        # Reset all chart settings to defaults
        st.session_state.chart_settings.update({
            'customer_name_font_size': 12,
            'demand_label_font_size': 14,
            'legend_font_size': 12,
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
            'use_custom_y_range': False,
            'use_custom_bubble_y_range': False,
            'use_custom_price_volume_y_range': False,
            'use_custom_y_demand_range': False,
            'min_volume_threshold': 50
        })
    st.session_state.previous_chart_type = chart_type
    st.session_state.previous_customer = customer_name

def get_chart_config(chart_type, customer_name_font_size, demand_label_font_size, legend_font_size, 
                     y_min, y_max, bubble_y_min, bubble_y_max, bubble_scale, bubble_alpha, 
                     price_volume_y_min, price_volume_y_max, y_demand_min, y_demand_max, **kwargs):
    """
    Return chart configuration dictionary tailored to chart type.
    The **kwargs at the end prevents crashes if extra keys are passed.
    """
    # 1. Base settings shared by all charts
    base_config = {
        'title_fontsize': 20,
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
            'min_volume_threshold': kwargs.get('min_volume_threshold', 50)
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
            'legend_font_size': 12,
            'y_min': None, 'y_max': None,
            'bubble_y_min': None, 'bubble_y_max': None,
            'price_volume_y_min': None, 'price_volume_y_max': None,
            'y_demand_min': None, 'y_demand_max': None,
            'bubble_scale': 5.0,
            'bubble_alpha': 0.7,
            'bubble_year': None,
            'min_volume_threshold': 50,
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

    # --- 5. MAIN WORKSPACE ---
    tab_vis, tab_data, tab_settings = st.tabs(["📈 Visualization", "📝 Data Editor", "⚙️ Layout Settings"])

    # TAB 1: VISUALIZATION
    with tab_vis:
        col_plot, col_ctrl = st.columns([3, 1])
        
        with col_ctrl:
            st.subheader("Filters")
            customer_name = None
            
            # Helper to check if we are in any bubble chart view
            is_bubble_chart = "bubble" in chart_type.lower()
            
            if not df_current.empty:
                
                # 1. Customer Dropdown: Only shows for individual account charts
                if not is_bubble_chart:
                    if 'customer' in df_current.columns:
                        customers = sorted(df_current['customer'].unique())
                        customer_name = st.selectbox("Select Account", customers, key="customer_select")
                    else:
                        st.warning("No 'customer' column found.")
           
                    
                # 2. Year Selection: Show for ALL bubble charts
                if is_bubble_chart:
                    if 'year' in df_current.columns:
                        years = sorted(df_current['year'].unique(), reverse=True)
                        st.selectbox("Data Year", years, key="bubble_year_select")
                    else:
                        st.error("Column 'year' is required for Bubble Charts.")
                        
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
                        yr = st.session_state.get('bubble_year_select')
                        chart_fig = plot_bubble_chart(df_current, None, material, is_taiwan, year_filter=yr, **config)
                    
                    elif chart_type == "Customer Bubble Chart (Centered)":
                        yr = st.session_state.get('bubble_year_select')
                        chart_fig = plot_bubble_chart_centered(
                            df_current, 
                            material, 
                            year_filter=yr,
                            **config
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
            st.session_state.dataframes[table_key] = st.data_editor(
                st.session_state.dataframes[table_key],
                num_rows="dynamic",
                key=f"editor_{table_key}"
            )

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
            'bubble_alpha': st.session_state.s_bub_alpha
        })

    # --- 7. STATE CLEANUP ---
    reset_axis_ranges(chart_type, customer_name)
    
def main():
    """Main entry point for the Streamlit app"""
    st.sidebar.header("🌎 Country and Material")
    country = st.sidebar.selectbox("Select Country", COUNTRIES, key="country_select")
    material_options = ["TDI"] if country == "Taiwan" else MATERIALS
    material = st.sidebar.selectbox("Select Material", material_options, key="material_select")
    
    # Initialize session state for uploaded files
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {
            'main_file': None,
            'bp_file': None,
            'ppd_file': None
        }
    if 'upload_complete' not in st.session_state:
        st.session_state.upload_complete = False
    
    # Reset uploaded files if country or material changes
    if (st.session_state.get('previous_country_main') != country or 
        st.session_state.get('previous_material_main') != material):
        st.session_state.uploaded_files = {
            'main_file': None,
            'bp_file': None,
            'ppd_file': None
        }
        st.session_state.upload_complete = False
        st.session_state.dataframes = {}
    st.session_state.previous_country_main = country
    st.session_state.previous_material_main = material
    
    # Sidebar toggle for upload section
    show_upload_section = st.sidebar.checkbox("Show CSV Upload Section", value=not st.session_state.upload_complete, key="show_upload_section")
    
    # Load data from uploaded files
    df_main, df_bp, df_ppd, all_uploaded = load_and_extract_dataframes(country, material, show_upload_section, st.session_state.uploaded_files)
    
    # Populate dataframes from uploaded files
    dataframes = {}
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
        st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
        st.session_state.chart_settings['auto_generate_chart'] = True
    
    # Always call main_app to ensure UI renders
    if dataframes and all_uploaded:
        st.success("All files uploaded successfully! Ready to generate charts.")
    main_app(dataframes, country, material, show_upload_section)

if __name__ == "__main__":
    main()