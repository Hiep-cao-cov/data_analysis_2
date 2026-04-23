"""Load data from session dataframes, disk (memory mode), and frame selection by chart type."""

import os

import pandas as pd
import streamlit as st
from config import REQUIRED_COLUMNS
from csv_utils import read_csv_flexible

from dashboard.validation import validate_dataframe

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


def _data_file_path(expected_name: str):
    """
    Resolve a CSV under ``data/`` by expected name.

    On **Linux** (e.g. Render), ``VN_TDI_FINAL.csv`` and ``VN_TDI_final.csv`` are different
    files. Windows often hides that. This matches the requested name case-insensitively so
    memory mode works the same locally and on the server.
    """
    path = os.path.join(_DATA_DIR, expected_name)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(_DATA_DIR):
        return None
    want = expected_name.lower()
    for entry in os.listdir(_DATA_DIR):
        if entry.lower() == want:
            return os.path.join(_DATA_DIR, entry)
    return None

# Predefined CSVs per (country, material) for "Memory" mode
MEMORY_FILE_MAP = {
    ("Vietnam", "PMDI"): {
        "main": "VN_MDI_FINAL.csv",
        "bp": "VN_MDI_BP.csv",
        "ppd": "VN_MDI_PPD.csv",
    },
    ("Vietnam", "TDI"): {
        "main": "VN_TDI_FINAL.csv",
        "bp": "VN_TDI_BP.csv",
        "ppd": "VN_TDI_PPD.csv",
    },
    ("Taiwan", "TDI"): {
        "main": "TW_TDI_FINAL.csv",
        "bp": "TW_TDI_BP.csv",
        "ppd": "TW_TDI_PPD.csv",
    },
}


def _files_uploaded_flag():
    return (
        any(st.session_state.uploaded_files.values())
        or st.session_state.get("find_in_memory", False)
        or st.session_state.get("upload_complete", False)
    )


def load_country_data(dataframes, country, material, chart_type):
    """Load and validate per-chart DataFrames from session ``dataframes`` into ``data_dict`` keys."""
    del chart_type  # API compatibility; each slot uses a fixed required-column set
    files_uploaded = _files_uploaded_flag()
    data_dict = {}

    if country == "Vietnam" and material == "PMDI":
        _add_if_valid(
            data_dict, "mdi", dataframes.get("df_mdi", pd.DataFrame()), REQUIRED_COLUMNS["price_charts"],
            material, country, None, files_uploaded
        )
        _add_if_valid(
            data_dict, "mdi_bp", dataframes.get("df_mdi_bp", pd.DataFrame()), REQUIRED_COLUMNS["business_plan"],
            material, country, None, files_uploaded
        )
        _add_if_valid(
            data_dict, "vn_ppd_2024", dataframes.get("df_ppd", pd.DataFrame()), REQUIRED_COLUMNS["bubble_centered"],
            material, country, "Customer Bubble Chart (Centered)", files_uploaded
        )
    elif country == "Vietnam" and material == "TDI":
        _add_if_valid(
            data_dict, "tdi", dataframes.get("df_tdi", pd.DataFrame()), REQUIRED_COLUMNS["price_charts"],
            material, country, None, files_uploaded
        )
        _add_if_valid(
            data_dict, "tdi_bp", dataframes.get("df_tdi_bp", pd.DataFrame()), REQUIRED_COLUMNS["business_plan"],
            material, country, None, files_uploaded
        )
        _add_if_valid(
            data_dict, "vn_ppd_2024", dataframes.get("df_ppd", pd.DataFrame()), REQUIRED_COLUMNS["bubble_centered"],
            material, country, "Customer Bubble Chart (Centered)", files_uploaded
        )
    elif country == "Taiwan":
        _add_if_valid(
            data_dict, "tw_tdi", dataframes.get("df_tdi", pd.DataFrame()), REQUIRED_COLUMNS["price_charts"],
            material, country, None, files_uploaded
        )
        _add_if_valid(
            data_dict, "tw_tdi_bp", dataframes.get("df_tdi_bp", pd.DataFrame()), REQUIRED_COLUMNS["business_plan"],
            material, country, None, files_uploaded
        )
        _add_if_valid(
            data_dict, "tw_ppd_2024", dataframes.get("df_ppd", pd.DataFrame()), REQUIRED_COLUMNS["bubble_centered"],
            material, country, "Customer Bubble Chart (Centered)", files_uploaded
        )

    return data_dict


def _add_if_valid(
    out,
    key,
    df,
    required_columns,
    material,
    country,
    chart_type,
    files_uploaded,
):
    if validate_dataframe(
        df,
        required_columns,
        material=material,
        country=country,
        chart_type=chart_type,
        files_uploaded=files_uploaded,
    ):
        out[key] = df
    else:
        out[key] = pd.DataFrame()


def get_dataframe(chart_type, material, data_dict, country):
    """Return the working DataFrame for the selected visualization."""
    if not _files_uploaded_flag():
        return pd.DataFrame()

    mapping = {
        "Customer Bubble Chart (Centered)": "vn_ppd_2024" if country == "Vietnam" else "tw_ppd_2024",
        "Business plan": (
            "mdi_bp" if material == "PMDI" else "tdi_bp" if "tdi_bp" in data_dict else "tw_tdi_bp"
        ),
        "Customer Demand": "mdi" if material == "PMDI" else "tdi" if "tdi" in data_dict else "tw_tdi",
        "Account price vs Volume": "mdi" if material == "PMDI" else "tdi" if "tdi" in data_dict else "tw_tdi",
        "Customer bubble Chart": "mdi" if material == "PMDI" else "tdi" if "tdi" in data_dict else "tw_tdi",
    }

    key = mapping.get(chart_type)
    if key not in data_dict or data_dict[key].empty:
        if chart_type == "Customer Bubble Chart (Centered)":
            st.info(
                "Load or upload the **PPD** CSV for this country/material (e.g. VN_TDI_PPD.csv). "
                "Validation must pass: columns **customer, year, sow, ppd, volume** with numeric values and no blanks."
            )
        elif "bp" in (key or ""):
            st.info(f"Please upload the **Business Plan** CSV to view the {chart_type}.")
        else:
            st.info(f"Please upload the **Main Data** CSV to view the {chart_type}.")
        return pd.DataFrame()

    return data_dict[key]


def load_data_from_memory(country, material):
    """Load predefined CSVs from the project ``data/`` folder (Memory data source)."""
    selected = MEMORY_FILE_MAP.get((country, material))
    if not selected:
        return {}, False

    table_main = "df_mdi" if material == "PMDI" else "df_tdi"
    table_bp = "df_mdi_bp" if material == "PMDI" else "df_tdi_bp"

    dataframes = {}
    required_loaded = True
    for table_key, file_key in [(table_main, "main"), (table_bp, "bp")]:
        logical_name = selected[file_key]
        file_path = _data_file_path(logical_name)
        if not file_path:
            st.error(
                f"Required memory file not found: {logical_name}. "
                f"Ensure it exists in the ``data/`` folder on the server (check **filename case** in Git; Linux is case-sensitive)."
            )
            required_loaded = False
            continue
        try:
            df = read_csv_flexible(file_path)
            if df.empty:
                st.error(f"Required memory file is empty: {logical_name}")
                required_loaded = False
                continue
            dataframes[table_key] = df
        except pd.errors.EmptyDataError:
            st.error(f"Failed to parse memory file (empty/no columns): {logical_name}")
            required_loaded = False
        except Exception as e:
            st.error(f"Error reading memory file {logical_name}: {str(e)}")
            required_loaded = False

    ppd_path = _data_file_path(selected["ppd"])
    if ppd_path and os.path.isfile(ppd_path):
        try:
            ppd_df = read_csv_flexible(ppd_path)
            if not ppd_df.empty:
                dataframes["df_ppd"] = ppd_df
            else:
                st.warning(f"Optional PPD memory file is empty and will be skipped: {selected['ppd']}")
        except pd.errors.EmptyDataError:
            st.warning(f"Optional PPD memory file has no columns and will be skipped: {selected['ppd']}")
        except Exception as e:
            st.warning(
                f"Optional PPD memory file could not be read and will be skipped: {selected['ppd']} ({str(e)})"
            )

    return dataframes, required_loaded
