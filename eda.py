import streamlit as st
import pandas as pd

def load_and_extract_dataframes(country, material):
    """
    Load and extract DataFrames from user-uploaded CSV files for the selected country and material.
    Returns three DataFrames and a boolean indicating if all files are uploaded and valid.
    """
    # Define expected column requirements for validation
    REQUIRED_COLUMNS = {
        'price_charts': ['customer', 'demand', 'pocket price'],
        'business_plan': ['customer', 'year', 'min', 'base', 'max'],
        'bubble_centered': ['customer', 'year', 'sow', 'ppd', 'volume']
    }
    
    # Initialize empty DataFrames and upload status
    df_main = pd.DataFrame()
    df_bp = pd.DataFrame()
    df_ppd = pd.DataFrame()
    upload_status = {
        'main': False,
        'bp': False,
        'ppd': False
    }
    
    # Create container for upload interface
    with st.container():
        st.markdown(f"### 📂 Upload Required CSV Files for {material} in {country}")
        st.info("Upload all required CSV files to proceed. Files are processed in-memory and not stored on the server.")
        
        # Define tab names based on material and country
        if material == "PMDI":
            tab_names = ["MDI", "MDI BP", "PPD"]
            main_label = f"MDI ({country})"
            bp_label = f"MDI BP ({country})"
        else:
            tab_names = ["TDI", "TDI BP", "PPD"]
            main_label = f"TDI ({country})"
            bp_label = f"TDI BP ({country})"
        
        # Create tabs for file uploads
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            uploaded_file = st.file_uploader(f"Upload {main_label} CSV", type=["csv"], key=f"upload_main_{country.lower()}_{material.lower()}")
            if uploaded_file:
                try:
                    df_main = pd.read_csv(uploaded_file)
                    if all(col in df_main.columns for col in REQUIRED_COLUMNS['price_charts']):
                        st.success(f"{main_label} CSV uploaded successfully!")
                        upload_status['main'] = True
                    else:
                        st.error(f"{main_label} CSV missing required columns: {REQUIRED_COLUMNS['price_charts']}")
                        df_main = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error reading {main_label} CSV: {str(e)}")
                    df_main = pd.DataFrame()
        
        with tabs[1]:
            uploaded_file = st.file_uploader(f"Upload {bp_label} CSV", type=["csv"], key=f"upload_bp_{country.lower()}_{material.lower()}")
            if uploaded_file:
                try:
                    df_bp = pd.read_csv(uploaded_file)
                    if all(col in df_bp.columns for col in REQUIRED_COLUMNS['business_plan']):
                        st.success(f"{bp_label} CSV uploaded successfully!")
                        upload_status['bp'] = True
                    else:
                        st.error(f"{bp_label} CSV missing required columns: {REQUIRED_COLUMNS['business_plan']}")
                        df_bp = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error reading {bp_label} CSV: {str(e)}")
                    df_bp = pd.DataFrame()
        
        with tabs[2]:
            uploaded_file = st.file_uploader(f"Upload PPD ({country}) CSV", type=["csv"], key=f"upload_ppd_{country.lower()}")
            if uploaded_file:
                try:
                    df_ppd = pd.read_csv(uploaded_file)
                    if all(col in df_ppd.columns for col in REQUIRED_COLUMNS['bubble_centered']):
                        st.success(f"PPD ({country}) CSV uploaded successfully!")
                        upload_status['ppd'] = True
                    else:
                        st.error(f"PPD ({country}) CSV missing required columns: {REQUIRED_COLUMNS['bubble_centered']}")
                        df_ppd = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error reading PPD ({country}) CSV: {str(e)}")
                    df_ppd = pd.DataFrame()
    
    # Check if all files are uploaded and valid
    all_uploaded = all(upload_status.values())
    
    return df_main, df_bp, df_ppd, all_uploaded