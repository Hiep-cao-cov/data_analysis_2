import streamlit as st
import pandas as pd
from config import REQUIRED_COLUMNS, MATERIALS

def load_and_extract_dataframes(country, material, show_upload_section, uploaded_files):
    """Load and extract DataFrames for the given country and material, using session state for persistence"""
    df_main = pd.DataFrame()
    df_bp = pd.DataFrame()
    df_ppd = pd.DataFrame()
    all_uploaded = False

    if show_upload_section and not st.session_state.get('upload_complete', False):
        with st.container():
            st.header("📤 Upload CSV Files")
            if material == "PMDI":
                main_label = "Main Data (PMDI)"
                bp_label = "Business Plan (PMDI)"
                ppd_label = "PPD Data"
            else:
                main_label = "Main Data (TDI)"
                bp_label = "Business Plan (TDI)"
                ppd_label = "PPD Data"
            
            # Create tabs for file uploads
            tabs = st.tabs([main_label, bp_label, ppd_label])
            
            # File uploaders in tabs
            with tabs[0]:
                new_main_file = st.file_uploader(f"Upload {main_label} CSV", type=["csv"], key="main_file")
                if new_main_file is not None:
                    new_main_file.seek(0)
                    if new_main_file.size == 0:
                        st.error(f"{main_label} CSV is empty. Please upload a valid CSV file.")
                    else:
                        try:
                            df = pd.read_csv(new_main_file, encoding='utf-8')
                            if df.empty:
                                st.error(f"{main_label} CSV contains no data. Please upload a valid CSV file.")
                            else:
                                uploaded_files['main_file'] = new_main_file
                        except pd.errors.EmptyDataError:
                            st.error(f"Failed to parse {main_label} CSV: File is empty or has no columns.")
                        except Exception as e:
                            st.error(f"Error reading {main_label} CSV: {str(e)}. Please ensure the file is a valid CSV.")
            
            with tabs[1]:
                new_bp_file = st.file_uploader(f"Upload {bp_label} CSV", type=["csv"], key="bp_file")
                if new_bp_file is not None:
                    new_bp_file.seek(0)
                    if new_bp_file.size == 0:
                        st.error(f"{bp_label} CSV is empty. Please upload a valid CSV file.")
                    else:
                        try:
                            df = pd.read_csv(new_bp_file, encoding='utf-8')
                            if df.empty:
                                st.error(f"{bp_label} CSV contains no data. Please upload a valid CSV file.")
                            else:
                                uploaded_files['bp_file'] = new_bp_file
                        except pd.errors.EmptyDataError:
                            st.error(f"Failed to parse {bp_label} CSV: File is empty or has no columns.")
                        except Exception as e:
                            st.error(f"Error reading {bp_label} CSV: {str(e)}. Please ensure the file is a valid CSV.")
            
            with tabs[2]:
                new_ppd_file = st.file_uploader(f"Upload {ppd_label} CSV", type=["csv"], key="ppd_file")
                if new_ppd_file is not None:
                    new_ppd_file.seek(0)
                    if new_ppd_file.size == 0:
                        st.error(f"{ppd_label} CSV is empty. Please upload a valid CSV file.")
                    else:
                        try:
                            df = pd.read_csv(new_ppd_file, encoding='utf-8')
                            if df.empty:
                                st.error(f"{ppd_label} CSV contains no data. Please upload a valid CSV file.")
                            else:
                                uploaded_files['ppd_file'] = new_ppd_file
                        except pd.errors.EmptyDataError:
                            st.error(f"Failed to parse {ppd_label} CSV: File is empty or has no columns.")
                        except Exception as e:
                            st.error(f"Error reading {ppd_label} CSV: {str(e)}. Please ensure the file is a valid CSV.")
    
    # Read uploaded files into DataFrames
    for file_key, df_name in [('main_file', 'df_main'), ('bp_file', 'df_bp'), ('ppd_file', 'df_ppd')]:
        if uploaded_files.get(file_key):
            try:
                uploaded_files[file_key].seek(0)
                df = pd.read_csv(uploaded_files[file_key], encoding='utf-8')
                locals()[df_name] = df
            except pd.errors.EmptyDataError:
                st.error(f"Failed to parse {file_key}: File is empty or has no columns.")
            except Exception as e:
                st.error(f"Error reading {file_key}: {str(e)}.")
    
    if all(uploaded_files.get(key) for key in ['main_file', 'bp_file', 'ppd_file']):
        all_uploaded = not df_main.empty and not df_bp.empty and not df_ppd.empty

    return df_main, df_bp, df_ppd, all_uploaded