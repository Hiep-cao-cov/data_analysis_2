"""Page CSS injected into Streamlit (centered-bubble list styling, tabs, width)."""

MAIN_PAGE_STYLE = """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 98%;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
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
