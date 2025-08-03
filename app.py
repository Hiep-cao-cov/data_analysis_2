import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import drawchat
import io
from EDA import load_and_extract_dataframes

# Constants
REQUIRED_COLUMNS = {
    'demand_charts': ['customer', 'demand'],
    'price_charts': ['customer', 'demand', 'pocket price'],
    'business_plan': ['customer', 'year', 'min', 'base', 'max'],
    'bubble_centered': ['customer', 'year', 'sow', 'ppd', 'volume']
}

SUPPLIERS = {
    'pmdi': ['covestro', 'tosoh', 'wanhua', 'kmc', 'basf', 'sabic', 'huntsman', 'other'],
    'tdi': ['covestro', 'mcns', 'wanhua', 'basf', 'hanwha', 'sabic', 'other'],
    'covestro': ['covestro']
}

CHART_TYPES = ["Customer Demand", "Account price vs Volume", "Business plan", "Customer bubble Chart", "Customer Bubble Chart (Centered)"]
COUNTRIES = ["Vietnam", "Taiwan"]
MATERIALS = ["PMDI", "TDI"]

def validate_dataframe(df, required_columns, material=None, country=None, chart_type=None):
    """Validate DataFrame has required columns, data types, and valid ranges"""
    if df.empty:
        st.error("DataFrame is empty")
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return False
    
    for col in required_columns:
        if df[col].isnull().any():
            st.error(f"Column '{col}' contains missing values")
            return False
    
    numeric_cols = ['demand', 'pocket price', 'year', 'covestro', 'tosoh', 'wanhua', 'kmc', 'basf', 
                    'sabic', 'huntsman', 'mcns', 'hanwha', 'other', 'vn_pp', 'tw_pp', 'seap_pp', 
                    'apac_pp', 'sow', 'volume']
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column '{col}' must be numeric")
            return False
        if col in df.columns and (df[col] < 0).any():
            st.error(f"Column '{col}' contains negative values, which are not allowed")
            return False
    
    if chart_type == "Customer Bubble Chart (Centered)" and 'ppd' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['ppd']):
            st.error("Column 'ppd' must be numeric")
            return False
    
    if set(['min', 'base', 'max']).issubset(required_columns):
        for col in ['min', 'base', 'max']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.error(f"Column '{col}' must be numeric")
                return False
    
    if material and material in ['PMDI', 'TDI'] and 'demand' in required_columns:
        expected_suppliers = SUPPLIERS[material.lower()]
        available_suppliers = [col for col in expected_suppliers if col in df.columns]
        missing_suppliers = [col for col in expected_suppliers if col not in df.columns]
        if missing_suppliers:
            st.warning(f"Missing supplier columns for {material}: {missing_suppliers}. Using available suppliers.")
        if not available_suppliers:
            st.error(f"No valid supplier columns for {material}")
            return False
    
    if 'pocket price' in required_columns and country is not None:
        if material == 'TDI':
            expected_price_cols = ['pocket price', 'vn_pp', 'apac_pp'] if country == 'Vietnam' else ['pocket price', 'tw_pp', 'apac_pp']
        else:
            expected_price_cols = ['pocket price', 'vn_pp', 'seap_pp', 'apac_pp']
        missing_price_cols = [col for col in expected_price_cols if col not in df.columns and col != 'pocket price']
        if missing_price_cols:
            st.warning(f"Missing price columns for {material} in {country}: {missing_price_cols}")
    
    return True

@st.cache_data
def load_country_data(dataframes, country, material, chart_type):
    """Load data into data_dict from provided DataFrames"""
    data_dict = {}
    
    if country == "Vietnam":
        if material == "PMDI":
            data_dict['mdi'] = dataframes.get('df_mdi', pd.DataFrame()) if validate_dataframe(dataframes.get('df_mdi', pd.DataFrame()), REQUIRED_COLUMNS['price_charts'], material=material, country=country) else pd.DataFrame()
            data_dict['mdi_bp'] = dataframes.get('df_mdi_bp', pd.DataFrame()) if validate_dataframe(dataframes.get('df_mdi_bp', pd.DataFrame()), REQUIRED_COLUMNS['business_plan'], material=material, country=country) else pd.DataFrame()
            data_dict['vn_ppd_2024'] = dataframes.get('df_ppd', pd.DataFrame()) if validate_dataframe(dataframes.get('df_ppd', pd.DataFrame()), REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)") else pd.DataFrame()
        else:
            data_dict['tdi'] = dataframes.get('df_tdi', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi', pd.DataFrame()), REQUIRED_COLUMNS['price_charts'], material=material, country=country) else pd.DataFrame()
            data_dict['tdi_bp'] = dataframes.get('df_tdi_bp', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi_bp', pd.DataFrame()), REQUIRED_COLUMNS['business_plan'], material=material, country=country) else pd.DataFrame()
            data_dict['vn_ppd_2024'] = dataframes.get('df_ppd', pd.DataFrame()) if validate_dataframe(dataframes.get('df_ppd', pd.DataFrame()), REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)") else pd.DataFrame()
    else:  # Taiwan, only TDI
        data_dict['tw_tdi'] = dataframes.get('df_tdi', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi', pd.DataFrame()), REQUIRED_COLUMNS['price_charts'], material=material, country=country) else pd.DataFrame()
        data_dict['tw_tdi_bp'] = dataframes.get('df_tdi_bp', pd.DataFrame()) if validate_dataframe(dataframes.get('df_tdi_bp', pd.DataFrame()), REQUIRED_COLUMNS['business_plan'], material=material, country=country) else pd.DataFrame()
        data_dict['tw_ppd_2024'] = dataframes.get('df_ppd', pd.DataFrame()) if validate_dataframe(dataframes.get('df_ppd', pd.DataFrame()), REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)") else pd.DataFrame()
    
    return data_dict

@st.cache_data
def get_dataframe(chart_type, material, data_dict, country):
    """Select appropriate dataframe based on chart type, material, and country"""
    if chart_type == "Customer Bubble Chart (Centered)":
        key = 'vn_ppd_2024' if country == "Vietnam" else 'tw_ppd_2024'
    elif material == "PMDI" and chart_type == "Business plan":
        key = 'mdi_bp'
    elif material == "PMDI":
        key = 'mdi'
    elif material == "TDI" and chart_type == "Business plan":
        key = 'tdi_bp' if 'tdi_bp' in data_dict else 'tw_tdi_bp'
    else:
        key = 'tdi' if 'tdi' in data_dict else 'tw_tdi'
    
    if key not in data_dict or data_dict[key].empty:
        st.error(f"No data available for {material} and {chart_type} in {country}")
        return pd.DataFrame()
    return data_dict[key]

@st.cache_data
def get_price_range(df, chart_type, material, country, customer_name=None, selected_price_columns=None):
    """Get the price range from dataframe for Y-axis slider"""
    if df.empty:
        return 0.0, 100.0
    try:
        df_filtered = df[df['customer'] == customer_name] if customer_name and chart_type != "Customer bubble Chart" and chart_type != "Customer Bubble Chart (Centered)" else df
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
        df_filtered = df[df['customer'] == customer_name] if customer_name and chart_type != "Customer bubble Chart" and chart_type != "Customer Bubble Chart (Centered)" else df
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

def plot_customer_demand(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, legend_title_fontsize, value_label_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max):
    """Plot customer demand chart"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['demand_charts'], material=material, chart_type="Customer Demand"):
        return None
    suppliers = SUPPLIERS[material.lower()]
    available_suppliers = [col for col in suppliers if col in df.columns]
    if not available_suppliers:
        st.error(f"No valid supplier columns for {material}")
        return None
    df_filtered = df[df['customer'] == customer_name]
    max_demand = df_filtered['demand'].max() if not df_filtered.empty else 0
    return drawchat.plot_customer_demand(
        df, customer_name, 'customer', available_suppliers, 'year', (0, max_demand * 1.4),
        title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, 
        legend_title_fontsize, value_label_fontsize, customer_name_font_size, 
        demand_label_font_size, y_min, y_max
    )

def plot_price_volume(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, legend_title_fontsize, price_annotation_fontsize, annotation_spacing, value_label_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max, y_demand_min, y_demand_max, selected_price_columns):
    """Plot price vs volume chart with selected price columns"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['price_charts'], material=material, chart_type="Account price vs Volume"):
        return None
    df_filtered = df[df['customer'] == customer_name]
    max_demand = df_filtered['demand'].max() if not df_filtered.empty else 0
    max_price = df_filtered['pocket price'].max() if not df_filtered.empty else 0
    price_config = {
        'TDI': (
            ['pocket price', 'vn_pp', 'apac_pp'], ['red', 'green', 'blue']
        ) if not is_taiwan else (
            ['pocket price', 'tw_pp', 'apac_pp'], ['red', 'green', 'blue']
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
    return drawchat.plot_customer_demand_with_price(
        df, customer_name, 'customer', SUPPLIERS['covestro'], 'year',
        (0, max_demand * 2), (0.5, max_price * 1.5), price_columns, selected_colors,
        title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, 
        legend_title_fontsize, value_label_fontsize, price_annotation_fontsize, 
        annotation_spacing, customer_name_font_size, demand_label_font_size, 
        y_min, y_max, y_demand_min, y_demand_max
    )

def plot_bubble_chart(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, bubble_scale, alpha, customer_name_font_size, demand_label_font_size, y_min, y_max, year_filter):
    """Plot bubble chart with fixed size"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['price_charts'], material=material, chart_type="Customer bubble Chart"):
        return None
    if 'year' not in df.columns and year_filter is not None:
        st.warning("Year column not found. Ignoring year filter.")
        year_filter = None
    settings_info = f"📊 Bubble Scale: {bubble_scale:.1f} | Transparency: {alpha:.1f} | "
    settings_info += f"Customer Name Font Size: {customer_name_font_size} | "
    settings_info += f"Demand Label Font Size: {demand_label_font_size}"
    if y_min is not None and y_max is not None:
        settings_info += f" | Y-axis: {y_min:.1f} - {y_max:.1f}"
    if year_filter is not None:
        settings_info += f" | Year: {year_filter}"
    st.info(settings_info)
    try:
        chart_figure, _, _, _ = drawchat.plot_customer_bubble_clean_with_median(
            df, 'customer', 'covestro', 'pocket price', year_filter, bubble_scale, alpha,
            title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize,
            customer_name_font_size, demand_label_font_size, y_min, y_max
        )
        return chart_figure
    except ValueError as e:
        st.error(f"Error generating bubble chart: {str(e)}")
        return None

def plot_bubble_chart_centered(df, material, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, bubble_scale, alpha, customer_name_font_size, demand_label_font_size, y_min, y_max, year_filter, min_volume_threshold=50):
    """Plot centered bubble chart with SOW and PPD axes"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['bubble_centered'], chart_type="Customer Bubble Chart (Centered)"):
        return None
    if 'year' not in df.columns and year_filter is not None:
        st.warning("Year column not found. Ignoring year filter.")
        year_filter = None
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
            df, 'customer', 'sow', 'ppd', 'volume', year_filter, bubble_scale, alpha,
            title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize,
            customer_name_font_size, demand_label_font_size, min_volume_threshold, y_min, y_max
        )
        return chart_figure
    except ValueError as e:
        st.error(f"Error generating centered bubble chart: {str(e)}")
        return None

def plot_business_plan(df, customer_name, material, is_taiwan, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, value_label_fontsize):
    """Plot business plan chart"""
    if not validate_dataframe(df, REQUIRED_COLUMNS['business_plan'], material=material, chart_type="Business plan"):
        return None
    return drawchat.plot_customer_business_plan(
        df, customer_name, is_taiwan, title_fontsize, axis_label_fontsize, 
        tick_fontsize, legend_fontsize, value_label_fontsize
    )

def setup_page():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="PMDI and TDI Visualization", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.sidebar.title("🎯 Navigation")

def reset_axis_ranges(chart_type, customer_name):
    """Reset Y-axis ranges based on chart type and customer change"""
    if (st.session_state.get('previous_chart_type') != chart_type or 
        st.session_state.get('previous_customer') != customer_name):
        if chart_type == "Customer bubble Chart" or chart_type == "Customer Bubble Chart (Centered)":
            st.session_state.chart_settings['bubble_y_min'] = None
            st.session_state.chart_settings['bubble_y_max'] = None
            st.session_state.chart_settings['use_custom_bubble_y_range'] = False
        elif chart_type == "Account price vs Volume":
            st.session_state.chart_settings['price_volume_y_min'] = None
            st.session_state.chart_settings['price_volume_y_max'] = None
            st.session_state.chart_settings['use_custom_price_volume_y_range'] = False
            st.session_state.chart_settings['y_demand_min'] = None
            st.session_state.chart_settings['y_demand_max'] = None
            st.session_state.chart_settings['use_custom_y_demand_range'] = False
        elif chart_type == "Customer Demand":
            st.session_state.chart_settings['y_min'] = None
            st.session_state.chart_settings['y_max'] = None
            st.session_state.chart_settings['use_custom_y_range'] = False
    st.session_state.previous_chart_type = chart_type
    st.session_state.previous_customer = customer_name

def get_chart_config(chart_type, customer_name_font_size, demand_label_font_size, y_min, y_max, bubble_y_min, bubble_y_max, bubble_scale, bubble_alpha, price_volume_y_min, price_volume_y_max, y_demand_min, y_demand_max):
    """Return chart configuration dictionary"""
    config = {
        'title_fontsize': 20,
        'axis_label_fontsize': 16,
        'tick_fontsize': 12,
        'legend_fontsize': 12,
        'legend_title_fontsize': 14,
        'value_label_fontsize': 12,
        'price_annotation_fontsize': 12,
        'annotation_spacing': 0.1,
        'customer_name_font_size': customer_name_font_size,
        'demand_label_font_size': demand_label_font_size,
        'bubble_scale': bubble_scale,
        'alpha': bubble_alpha,
        'min_volume_threshold': 50
    }
    if chart_type == "Customer bubble Chart" or chart_type == "Customer Bubble Chart (Centered)":
        config['y_min'] = bubble_y_min
        config['y_max'] = bubble_y_max
    elif chart_type == "Account price vs Volume":
        config['y_min'] = price_volume_y_min
        config['y_max'] = price_volume_y_max
        config['y_demand_min'] = y_demand_min
        config['y_demand_max'] = y_demand_max
    else:
        config['y_min'] = y_min
        config['y_max'] = y_max
    return config

def main_app(dataframes, country, material):
    """Main application logic with DataFrames for the selected country and material"""
    setup_page()
    
    # Initialize session state for DataFrames and chart settings
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
    
    if 'chart_settings' not in st.session_state:
        st.session_state.chart_settings = {
            'customer_name_font_size': 12,
            'demand_label_font_size': 14,
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
            'bubble_year': None,
            'selected_price_columns': None,
            'min_volume_threshold': 50
        }
    
    # Clear cache and reset DataFrames button
    col_reset, col_clear = st.columns([1, 1])
    with col_reset:
        if st.button("Reset DataFrames to Original"):
            st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
            st.success("DataFrames reset to original values!")
    with col_clear:
        if st.button("Clear Cache and Reload"):
            load_country_data.clear()
            get_dataframe.clear()
            get_price_range.clear()
            get_demand_range.clear()
            st.rerun()
    
    st.title(f"📊 {material} Market Visualization Dashboard - {country}")
    st.markdown("---")
    
    # DataFrame editing section
    with st.expander("📝 Edit DataFrames", expanded=False):
        st.subheader("Modify DataFrames")
        tabs = st.tabs(list(dataframes.keys()))
        
        for i, key in enumerate(dataframes.keys()):
            with tabs[i]:
                st.session_state.dataframes[key] = st.data_editor(
                    st.session_state.dataframes[key],
                    num_rows="dynamic",
                    key=f"editor_{key}"
                )
    
    # Sidebar controls
    st.sidebar.header("🌍 Chart Controls")
    chart_type = st.sidebar.radio("Select Chart Type", CHART_TYPES, key="chart_type_select")
    is_taiwan = country == "Taiwan"
    
    # Reset selected_price_columns when country or material changes
    if (st.session_state.get('previous_country') != country or 
        st.session_state.get('previous_material') != material):
        st.session_state.chart_settings['selected_price_columns'] = ['pocket price']
    st.session_state.previous_country = country
    st.session_state.previous_material = material
    
    # Load data from edited DataFrames
    with st.spinner("Loading data..."):
        data_dict = load_country_data(st.session_state.dataframes, country, material, chart_type)
    
    # Define columns
    col1, col2 = st.columns([3, 1])
    
    # Customer selection
    df = get_dataframe(chart_type, material, data_dict, country)
    customer_name = None
    with col1:
        if chart_type not in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"] and not df.empty and 'customer' in df.columns:
            customers = sorted(df['customer'].unique())
            with st.container():
                st.markdown(
                    """
                    <style>
                    .customer-selectbox {
                        max-width: 50%;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                customer_name = st.selectbox(
                    f"Select Customer for {material}", 
                    customers,
                    key="customer_select"
                )
    
    # Reset axis ranges
    reset_axis_ranges(chart_type, customer_name)
    
    with col2:
        st.info(f"""
        **Current Selection:**
        - Country: {country}
        - Material: {material}
        - Chart: {chart_type}
        - Customer: {customer_name if customer_name else 'Not applicable'}
        - Year: {st.session_state.chart_settings['bubble_year'] if chart_type in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"] and st.session_state.chart_settings['bubble_year'] is not None else 'Not applicable'}
        """)
        
        with st.container():
            st.subheader("🎨 Chart Settings")
            
            if st.button("Reset Chart Settings"):
                st.session_state.chart_settings = {
                    'customer_name_font_size': 12,
                    'demand_label_font_size': 14,
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
                    'bubble_year': None,
                    'selected_price_columns': None,
                    'min_volume_threshold': 50
                }
                st.success("Chart settings reset to default!")
            
            with st.expander("🔤 Font Settings"):
                st.session_state.chart_settings['customer_name_font_size'] = st.slider(
                    "Customer Name Font Size",
                    min_value=8,
                    max_value=20,
                    value=st.session_state.chart_settings['customer_name_font_size'],
                    step=1,
                    key="customer_name_font_size"
                )
                
                st.session_state.chart_settings['demand_label_font_size'] = st.slider(
                    "Demand Label Font Size",
                    min_value=8,
                    max_value=24,
                    value=st.session_state.chart_settings['demand_label_font_size'],
                    step=1,
                    key="demand_label_font_size"
                )
            
            if chart_type in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"]:
                with st.expander("🔵 Bubble Chart Specific Settings"):
                    st.session_state.chart_settings['bubble_scale'] = st.slider(
                        "Bubble Size Scale", 
                        min_value=1.0, 
                        max_value=50.0,
                        value=st.session_state.chart_settings['bubble_scale'],
                        step=0.1,
                        key="bubble_scale"
                    ) 
                    
                    st.session_state.chart_settings['bubble_alpha'] = st.slider(
                        "Bubble Transparency", 
                        min_value=0.1, 
                        max_value=1.0, 
                        value=st.session_state.chart_settings['bubble_alpha'],
                        step=0.1,
                        key="bubble_alpha"
                    )
                    
                    if chart_type == "Customer Bubble Chart (Centered)":
                        st.session_state.chart_settings['min_volume_threshold'] = st.slider(
                            "Minimum Volume Threshold",
                            min_value=0,
                            max_value=1000,
                            value=st.session_state.chart_settings['min_volume_threshold'],
                            step=10,
                            key="min_volume_threshold"
                        )
                    
                    if st.session_state.chart_settings['bubble_scale'] > 10.0:
                        st.warning("⚠️ Large bubble sizes may cause overlap. Consider reducing scale.")
            
            if chart_type == "Account price vs Volume":
                with st.expander("📈 Price Columns Selection"):
                    price_options = (
                        ['pocket price', 'vn_pp', 'apac_pp'] if material == 'TDI' and country == 'Vietnam' else
                        ['pocket price', 'tw_pp', 'apac_pp'] if material == 'TDI' and country == 'Taiwan' else
                        ['pocket price', 'vn_pp', 'seap_pp', 'apac_pp']
                    )
                    price_options = [col for col in price_options if col in df.columns]
                    if not price_options:
                        st.warning(f"No valid price columns available for {material} in {country}")
                        default_price_columns = []
                    else:
                        default_price_columns = ['pocket price'] if 'pocket price' in price_options else [price_options[0]]
                    st.session_state.chart_settings['selected_price_columns'] = st.multiselect(
                        "Select Price Columns to Display",
                        options=price_options,
                        default=st.session_state.chart_settings['selected_price_columns'] or default_price_columns,
                        key="price_columns_select"
                    )
            
            if chart_type in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"]:
                with st.expander("📏 Bubble Chart Y-Axis Range Control (Price Axis, $/kg)"):
                    st.session_state.chart_settings['use_custom_bubble_y_range'] = st.checkbox(
                        "Custom Bubble Chart Y-axis Range",
                        value=st.session_state.chart_settings.get('use_custom_bubble_y_range', False),
                        key="custom_bubble_y_range"
                    )
                    if st.session_state.chart_settings['use_custom_bubble_y_range']:
                        range_min, range_max = get_price_range(df, chart_type, material, country, customer_name)
                        slider_step = 0.01 if chart_type == "Customer Bubble Chart (Centered)" else 0.1
                        format_str = ".2f" if chart_type == "Customer Bubble Chart (Centered)" else ".1f"
                        
                        if st.session_state.chart_settings['bubble_y_min'] is None:
                            st.session_state.chart_settings['bubble_y_min'] = float(range_min)
                        if st.session_state.chart_settings['bubble_y_max'] is None:
                            st.session_state.chart_settings['bubble_y_max'] = float(range_max)
                        
                        st.session_state.chart_settings['bubble_y_min'] = st.slider(
                            "Bubble Chart Y-axis Minimum",
                            min_value=-1.0 if chart_type == "Customer Bubble Chart (Centered)" else 0.0,
                            max_value=float(range_max),
                            value=st.session_state.chart_settings['bubble_y_min'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="bubble_y_min"
                        )
                        
                        st.session_state.chart_settings['bubble_y_max'] = st.slider(
                            "Bubble Chart Y-axis Maximum",
                            min_value=float(st.session_state.chart_settings['bubble_y_min'] + slider_step),
                            max_value=float(range_max * 2),
                            value=st.session_state.chart_settings['bubble_y_max'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="bubble_y_max"
                        )
                        
                        if st.session_state.chart_settings['bubble_y_min'] >= st.session_state.chart_settings['bubble_y_max']:
                            st.error("⚠️ Bubble Chart Y-axis minimum must be less than maximum!")
                            st.session_state.chart_settings['bubble_y_min'] = None
                            st.session_state.chart_settings['bubble_y_max'] = None
                        else:
                            st.info(f"Custom Bubble Chart Y-axis range: {st.session_state.chart_settings['bubble_y_min']:.2f} - {st.session_state.chart_settings['bubble_y_max']:.2f}")
                    else:
                        range_min, range_max = get_price_range(df, chart_type, material, country, customer_name)
                        st.info(f"Auto Bubble Chart Y-axis range: {range_min:.2f} - {range_max:.2f}")
                        st.session_state.chart_settings['bubble_y_min'] = None
                        st.session_state.chart_settings['bubble_y_max'] = None
            
            elif chart_type == "Account price vs Volume":
                with st.expander("📏 Price vs Volume Y-Axis Range Control (Price Axis, $/kg)"):
                    st.session_state.chart_settings['use_custom_price_volume_y_range'] = st.checkbox(
                        "Custom Price vs Volume Y-axis Range",
                        value=st.session_state.chart_settings.get('use_custom_price_volume_y_range', False),
                        key="custom_price_volume_y_range"
                    )
                    if st.session_state.chart_settings['use_custom_price_volume_y_range']:
                        range_min, range_max = get_price_range(df, chart_type, material, country, customer_name, st.session_state.chart_settings['selected_price_columns'])
                        slider_step = 0.1
                        format_str = ".1f"
                        
                        if st.session_state.chart_settings['price_volume_y_min'] is None:
                            st.session_state.chart_settings['price_volume_y_min'] = float(range_min)
                        if st.session_state.chart_settings['price_volume_y_max'] is None:
                            st.session_state.chart_settings['price_volume_y_max'] = float(range_max)
                        
                        st.session_state.chart_settings['price_volume_y_min'] = st.slider(
                            "Price vs Volume Y-axis Minimum",
                            min_value=0.0,
                            max_value=float(range_max),
                            value=st.session_state.chart_settings['price_volume_y_min'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="price_volume_y_min"
                        )
                        
                        st.session_state.chart_settings['price_volume_y_max'] = st.slider(
                            "Price vs Volume Y-axis Maximum",
                            min_value=float(st.session_state.chart_settings['price_volume_y_min'] + slider_step),
                            max_value=float(range_max * 2),
                            value=st.session_state.chart_settings['price_volume_y_max'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="price_volume_y_max"
                        )
                        
                        if st.session_state.chart_settings['price_volume_y_min'] >= st.session_state.chart_settings['price_volume_y_max']:
                            st.error("⚠️ Price vs Volume Y-axis minimum must be less than maximum!")
                            st.session_state.chart_settings['price_volume_y_min'] = None
                            st.session_state.chart_settings['price_volume_y_max'] = None
                        else:
                            st.info(f"Custom Price vs Volume Y-axis range: {st.session_state.chart_settings['price_volume_y_min']:.1f} - {st.session_state.chart_settings['price_volume_y_max']:.1f}")
                    else:
                        range_min, range_max = get_price_range(df, chart_type, material, country, customer_name, st.session_state.chart_settings['selected_price_columns'])
                        st.info(f"Auto Price vs Volume Y-axis range: {range_min:.1f} - {range_max:.1f}")
                        st.session_state.chart_settings['price_volume_y_min'] = None
                        st.session_state.chart_settings['price_volume_y_max'] = None
                
                with st.expander("📏 Demand Y-Axis Range Control (Demand Axis, mt)"):
                    st.session_state.chart_settings['use_custom_y_demand_range'] = st.checkbox(
                        "Custom Demand Y-axis Range",
                        value=st.session_state.chart_settings.get('use_custom_y_demand_range', False),
                        key="custom_y_demand_range"
                    )
                    if st.session_state.chart_settings['use_custom_y_demand_range']:
                        range_min, range_max = get_demand_range(df, chart_type, customer_name)
                        slider_step = 1.0
                        format_str = ".0f"
                        
                        if st.session_state.chart_settings['y_demand_min'] is None:
                            st.session_state.chart_settings['y_demand_min'] = float(range_min)
                        if st.session_state.chart_settings['y_demand_max'] is None:
                            st.session_state.chart_settings['y_demand_max'] = float(range_max)
                        
                        st.session_state.chart_settings['y_demand_min'] = st.slider(
                            "Demand Y-axis Minimum",
                            min_value=0.0,
                            max_value=float(range_max),
                            value=st.session_state.chart_settings['y_demand_min'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="y_demand_min"
                        )
                        
                        st.session_state.chart_settings['y_demand_max'] = st.slider(
                            "Demand Y-axis Maximum",
                            min_value=float(st.session_state.chart_settings['y_demand_min'] + slider_step),
                            max_value=float(range_max * 2),
                            value=st.session_state.chart_settings['y_demand_max'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="y_demand_max"
                        )
                        
                        if st.session_state.chart_settings['y_demand_min'] >= st.session_state.chart_settings['y_demand_max']:
                            st.error("⚠️ Demand Y-axis minimum must be less than maximum!")
                            st.session_state.chart_settings['y_demand_min'] = None
                            st.session_state.chart_settings['y_demand_max'] = None
                        else:
                            st.info(f"Custom Demand Y-axis range: {st.session_state.chart_settings['y_demand_min']:.0f} - {st.session_state.chart_settings['y_demand_max']:.0f}")
                    else:
                        range_min, range_max = get_demand_range(df, chart_type, customer_name)
                        st.info(f"Auto Demand Y-axis range: {range_min:.0f} - {range_max:.0f}")
                        st.session_state.chart_settings['y_demand_min'] = None
                        st.session_state.chart_settings['y_demand_max'] = None
            
            elif chart_type == "Customer Demand":
                with st.expander("📏 Y-Axis Range Control (Demand Axis, mt)"):
                    st.session_state.chart_settings['use_custom_y_range'] = st.checkbox(
                        "Custom Y-axis Range",
                        value=st.session_state.chart_settings.get('use_custom_y_range', False),
                        key="custom_y_range"
                    )
                    if st.session_state.chart_settings['use_custom_y_range']:
                        range_min, range_max = get_demand_range(df, chart_type, customer_name)
                        slider_step = 1.0
                        format_str = ".0f"
                        
                        if st.session_state.chart_settings['y_min'] is None:
                            st.session_state.chart_settings['y_min'] = float(range_min)
                        if st.session_state.chart_settings['y_max'] is None:
                            st.session_state.chart_settings['y_max'] = float(range_max)
                        
                        st.session_state.chart_settings['y_min'] = st.slider(
                            "Y-axis Minimum",
                            min_value=0.0,
                            max_value=float(range_max),
                            value=st.session_state.chart_settings['y_min'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="y_min"
                        )
                        
                        st.session_state.chart_settings['y_max'] = st.slider(
                            "Y-axis Maximum",
                            min_value=float(st.session_state.chart_settings['y_min'] + slider_step),
                            max_value=float(range_max * 2),
                            value=st.session_state.chart_settings['y_max'],
                            step=slider_step,
                            format=f"%{format_str}",
                            key="y_max"
                        )
                        
                        if st.session_state.chart_settings['y_min'] >= st.session_state.chart_settings['y_max']:
                            st.error("⚠️ Y-axis minimum must be less than maximum!")
                            st.session_state.chart_settings['y_min'] = None
                            st.session_state.chart_settings['y_max'] = None
                        else:
                            st.info(f"Custom Y-axis range: {st.session_state.chart_settings['y_min']:.0f} - {st.session_state.chart_settings['y_max']:.0f}")
                    else:
                        range_min, range_max = get_demand_range(df, chart_type, customer_name)
                        st.info(f"Auto Y-axis range: {range_min:.0f} - {range_max:.0f}")
                        st.session_state.chart_settings['y_min'] = None
                        st.session_state.chart_settings['y_max'] = None
    
    with col1:
        st.subheader(f"{chart_type} Visualization")
        
        if not df.empty and (chart_type in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"] or 'customer' in df.columns):
            chart_config = get_chart_config(
                chart_type,
                st.session_state.chart_settings['customer_name_font_size'],
                st.session_state.chart_settings['demand_label_font_size'],
                st.session_state.chart_settings['y_min'],
                st.session_state.chart_settings['y_max'],
                st.session_state.chart_settings['bubble_y_min'],
                st.session_state.chart_settings['bubble_y_max'],
                st.session_state.chart_settings['bubble_scale'],
                st.session_state.chart_settings['bubble_alpha'],
                st.session_state.chart_settings['price_volume_y_min'],
                st.session_state.chart_settings['price_volume_y_max'],
                st.session_state.chart_settings['y_demand_min'],
                st.session_state.chart_settings['y_demand_max']
            )
            
            year_filter = None
            if chart_type in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"] and 'year' in df.columns:
                available_years = sorted(df['year'].unique())
                if available_years:
                    if st.session_state.chart_settings['bubble_year'] not in available_years:
                        st.session_state.chart_settings['bubble_year'] = available_years[-1]
                    year_filter = st.selectbox(
                        "Select Year for Bubble Chart",
                        available_years,
                        index=available_years.index(st.session_state.chart_settings['bubble_year']),
                        key="bubble_year_select"
                    )
                    st.session_state.chart_settings['bubble_year'] = year_filter
                else:
                    st.warning("No valid years found in the data. Using all data.")
            
            if st.button(f"🎯 Generate {chart_type} Chart", type="primary", key="generate_chart"):
                with st.spinner("Generating chart..."):
                    if chart_type == "Customer Demand":
                        chart_figure = plot_customer_demand(
                            df, customer_name, material, is_taiwan,
                            chart_config['title_fontsize'], chart_config['axis_label_fontsize'],
                            chart_config['tick_fontsize'], chart_config['legend_fontsize'],
                            chart_config['legend_title_fontsize'], chart_config['value_label_fontsize'],
                            chart_config['customer_name_font_size'], chart_config['demand_label_font_size'],
                            chart_config['y_min'], chart_config['y_max']
                        )
                    elif chart_type == "Account price vs Volume":
                        chart_figure = plot_price_volume(
                            df, customer_name, material, is_taiwan,
                            chart_config['title_fontsize'], chart_config['axis_label_fontsize'],
                            chart_config['tick_fontsize'], chart_config['legend_fontsize'],
                            chart_config['legend_title_fontsize'], chart_config['price_annotation_fontsize'],
                            chart_config['annotation_spacing'], chart_config['value_label_fontsize'],
                            chart_config['customer_name_font_size'], chart_config['demand_label_font_size'],
                            chart_config['y_min'], chart_config['y_max'],
                            chart_config['y_demand_min'], chart_config['y_demand_max'],
                            st.session_state.chart_settings['selected_price_columns']
                        )
                    elif chart_type == "Customer bubble Chart":
                        chart_figure = plot_bubble_chart(
                            df, None, material, is_taiwan,
                            chart_config['title_fontsize'], chart_config['axis_label_fontsize'],
                            chart_config['tick_fontsize'], chart_config['legend_fontsize'],
                            chart_config['bubble_scale'], chart_config['alpha'],
                            chart_config['customer_name_font_size'], chart_config['demand_label_font_size'],
                            chart_config['y_min'], chart_config['y_max'],
                            year_filter
                        )
                    elif chart_type == "Customer Bubble Chart (Centered)":
                        chart_figure = plot_bubble_chart_centered(
                            df, material,
                            chart_config['title_fontsize'], chart_config['axis_label_fontsize'],
                            chart_config['tick_fontsize'], chart_config['legend_fontsize'],
                            chart_config['bubble_scale'], chart_config['alpha'],
                            chart_config['customer_name_font_size'], chart_config['demand_label_font_size'],
                            chart_config['y_min'], chart_config['y_max'],
                            year_filter,
                            chart_config['min_volume_threshold']
                        )
                    elif chart_type == "Business plan":
                        chart_figure = plot_business_plan(
                            df, customer_name, material, is_taiwan,
                            chart_config['title_fontsize'], chart_config['axis_label_fontsize'],
                            chart_config['tick_fontsize'], chart_config['legend_fontsize'],
                            chart_config['value_label_fontsize']
                        )
                     
                    if chart_figure:
                        st.plotly_chart(chart_figure, use_container_width=True)
                        st.success("Chart generated successfully! 🎉")
                        
                        # Save chart as HTML
                        filename = f"{chart_type.replace(' ', '_')}_{country}_{material}"
                        if customer_name and chart_type not in ["Customer bubble Chart", "Customer Bubble Chart (Centered)"]:
                            filename += f"_{customer_name}"
                        if year_filter:
                            filename += f"_{year_filter}"
                        filename = filename.replace(" ", "_").replace("(", "").replace(")", "") + ".html"
                        
                        buffer = io.StringIO()
                        chart_figure.write_html(buffer, full_html=True)
                        buffer.seek(0)
                        html_content = buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Save Chart as HTML",
                            data=html_content,
                            file_name=filename,
                            mime="text/html",
                            type="primary",
                            key="save_chart_html"
                        )
                    else:
                        st.error("Failed to generate chart due to invalid data or settings.")
        else:
            st.error("❌ No data available to display. Please check your input DataFrames.")
            
            if st.checkbox("Show data information"):
                st.write("Available data keys:", list(data_dict.keys()))
                for key, data in data_dict.items():
                    if not data.empty:
                        st.write(f"{key}: {data.shape[0]} rows, {data.shape[1]} columns")
                        if 'customer' in data.columns:
                            st.write(f"Customers in {key}: {len(data['customer'].unique())}")
                        if 'year' in data.columns:
                            st.write(f"Years in {key}: {sorted(data['year'].unique())}")

def main():
    """Main entry point to control app flow"""
    # Initialize session state
    if 'all_files_uploaded' not in st.session_state:
        st.session_state.all_files_uploaded = False
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = None
    if 'selected_material' not in st.session_state:
        st.session_state.selected_material = None
    
    # Country and material selection
    st.title("📊 PMDI and TDI Market Visualization Dashboard")
    st.markdown("---")
    with st.container():
        st.subheader("🌍 Select Country and Material")
        col_country, col_material = st.columns(2)
        with col_country:
            country = st.selectbox("Choose a country to upload data for", COUNTRIES, key="country_select")
        with col_material:
            # Disable PMDI for Taiwan
            material_options = ["TDI"] if country == "Taiwan" else MATERIALS
            material = st.selectbox("Choose a material", material_options, key="material_select")
        
        # Reset upload status if country or material changes
        if (st.session_state.selected_country != country or 
            st.session_state.selected_material != material):
            st.session_state.all_files_uploaded = False
            st.session_state.dataframes = {}
            st.session_state.selected_country = country
            st.session_state.selected_material = material
    
    # If all files are not uploaded, show upload interface
    if not st.session_state.all_files_uploaded:
        st.subheader(f"📂 Upload CSV Files for {material} in {country}")
        st.info(f"Please upload all required CSV files for {material} in {country} to proceed. Files are processed in-memory and not stored on the server.")
        
        # Load and validate DataFrames
        df_main, df_bp, df_ppd, all_uploaded = load_and_extract_dataframes(country, material)
        dataframes = {
            'df_mdi' if material == "PMDI" else 'df_tdi': df_main,
            'df_mdi_bp' if material == "PMDI" else 'df_tdi_bp': df_bp,
            'df_ppd': df_ppd
        }
        
        # Update session state if all files are valid
        if all_uploaded:
            st.session_state.all_files_uploaded = True
            st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
            st.rerun()  # Rerun to switch to main app
        else:
            st.warning(f"Please upload and validate all required CSV files for {material} in {country} to proceed.")
    else:
        # Show main app with DataFrame editing and chart visualization
        main_app(st.session_state.dataframes, country, material)

if __name__ == "__main__":
    main()