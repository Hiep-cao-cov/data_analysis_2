"""Page CSS injected into Streamlit (centered-bubble list styling, tabs, width)."""

MAIN_PAGE_STYLE = """
    <style>
    .block-container {padding-top: 1.2rem; max-width: 98%;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f7ff;
        border: 1px solid #dbe7ff;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f0ff !important;
        border-bottom-color: #e8f0ff !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7faff 0%, #f9fafb 100%);
        border-right: 1px solid #e5e7eb;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        font-weight: 600;
    }
    div[data-testid="stDownloadButton"] > button {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        font-weight: 600;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    [data-testid="stExpander"] [data-testid="stCheckbox"] {
        min-height: unset;
        margin-top: -0.35rem !important;
        margin-bottom: -0.35rem !important;
    }
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
"""
