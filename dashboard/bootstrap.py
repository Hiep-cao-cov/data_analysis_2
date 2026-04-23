"""Entrypoint: page config, sidebar, data source, and ``main_app`` dispatch."""

import pandas as pd
import streamlit as st
from config import COUNTRIES, MATERIALS
from csv_utils import read_csv_flexible
from eda import load_and_extract_dataframes

from dashboard.data import load_data_from_memory
from dashboard.main_view import main_app


def setup_page():
    """Optional: call once for ``st.set_page_config`` (not used by default run)."""
    st.set_page_config(
        page_title="PMDI and TDI Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("🎯 Navigation")


def main():
    st.sidebar.header("🌎 Country and Material")
    country = st.sidebar.selectbox("Select Country", COUNTRIES, key="country_select")
    material_options = ["TDI"] if country == "Taiwan" else MATERIALS
    material = st.sidebar.selectbox("Select Material", material_options, key="material_select")
    st.sidebar.markdown("### Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        options=["Memory (preloaded files)", "Manual CSV Upload"],
        key="data_source_mode",
    )
    find_in_memory = data_source == "Memory (preloaded files)"
    st.session_state.find_in_memory = find_in_memory

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {
            "main_file": None,
            "bp_file": None,
            "ppd_file": None,
        }
    if "upload_complete" not in st.session_state:
        st.session_state.upload_complete = False
    if "data_edited" not in st.session_state:
        st.session_state.data_edited = False
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

    if (
        st.session_state.get("previous_country_main") != country
        or st.session_state.get("previous_material_main") != material
        or st.session_state.get("previous_data_source_main") != data_source
    ):
        st.session_state.uploaded_files = {
            "main_file": None,
            "bp_file": None,
            "ppd_file": None,
        }
        st.session_state.upload_complete = False
        st.session_state.dataframes = {}
        st.session_state.data_edited = False
    st.session_state.previous_country_main = country
    st.session_state.previous_material_main = material
    st.session_state.previous_data_source_main = data_source

    show_upload_section = data_source == "Manual CSV Upload"

    dataframes = {}
    all_uploaded = False

    if find_in_memory:
        dataframes, all_uploaded = load_data_from_memory(country, material)
        if all_uploaded:
            st.session_state.upload_complete = True
            if not st.session_state.data_edited:
                st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
            st.session_state.chart_settings["auto_generate_chart"] = True
            st.success("Loaded data from memory files successfully.")
    else:
        _, _, _, all_uploaded = load_and_extract_dataframes(
            country, material, show_upload_section, st.session_state.uploaded_files
        )

        for file_key, file_name in [
            ("main_file", "df_mdi" if material == "PMDI" else "df_tdi"),
            ("bp_file", "df_mdi_bp" if material == "PMDI" else "df_tdi_bp"),
            ("ppd_file", "df_ppd"),
        ]:
            if st.session_state.uploaded_files[file_key]:
                try:
                    file_obj = st.session_state.uploaded_files[file_key]
                    file_obj.seek(0)
                    if file_obj.size == 0:
                        st.error(
                            f"Uploaded file for {file_name} is empty. Please upload a valid CSV file."
                        )
                        continue
                    df = read_csv_flexible(file_obj)
                    if df.empty:
                        st.error(
                            f"Uploaded file for {file_name} contains no data. Please upload a valid CSV file."
                        )
                        continue
                    dataframes[file_name] = df
                except pd.errors.EmptyDataError:
                    st.error(
                        f"Failed to parse {file_name}: File is empty or has no columns. Please upload a valid CSV file."
                    )
                except Exception as e:
                    st.error(
                        f"Error reading {file_name}: {str(e)}. Please ensure the file is a valid CSV file."
                    )

        required_keys = [
            "df_mdi" if material == "PMDI" else "df_tdi",
            "df_mdi_bp" if material == "PMDI" else "df_tdi_bp",
            "df_ppd",
        ]
        if all(key in dataframes and not dataframes[key].empty for key in required_keys):
            st.session_state.upload_complete = True
            if not st.session_state.data_edited:
                st.session_state.dataframes = {k: v.copy() for k, v in dataframes.items()}
            st.session_state.chart_settings["auto_generate_chart"] = True

    if dataframes and all_uploaded:
        st.success("All files uploaded successfully! Ready to generate charts.")
    main_app(country, material, show_upload_section)
