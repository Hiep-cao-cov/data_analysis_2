# PMDI/TDI Market Dashboard

Interactive Streamlit dashboard for PMDI/TDI market analysis with customer-level demand, price, business plan, and bubble visualizations.

## Features

- Support for `Vietnam` and `Taiwan` regions.
- Material selection:
  - `Vietnam`: `PMDI`, `TDI`
  - `Taiwan`: `TDI`
- Data source modes:
  - `Memory (preloaded files)` from `data/`
  - `Manual CSV Upload`
- Chart types:
  - `Customer Demand`
  - `Account price vs Volume`
  - `Business plan`
  - `Customer bubble Chart`
  - `Customer Bubble Chart (Centered)`
- In-app data editor and chart export to HTML.

## Project Structure

- `app.py`: Streamlit entry point and UI logic.
- `drawchat.py`: Plotly chart rendering functions.
- `config.py`: Chart options, required columns, suppliers, and material config.
- `eda.py`: CSV upload and dataframe extraction helpers.
- `data/`: Preloaded CSV files for memory mode.

## Requirements

- Python 3.10+ (recommended)
- Dependencies from `requirements.txt`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Data Source Modes

### 1) Memory (preloaded files)
The app auto-loads files from `data/` based on selected country/material:

- `Vietnam + PMDI`
  - Main: `VN_MDI_FINAL.csv`
  - Business Plan: `VN_MDI_BP.csv`
  - PPD (optional): `VN_MDI_PPD.csv`
- `Vietnam + TDI`
  - Main: `VN_TDI_FINAL.csv`
  - Business Plan: `VN_TDI_BP.csv`
  - PPD (optional): `VN_TDI_PPD.csv`
- `Taiwan + TDI`
  - Main: `TW_TDI_FINAL.csv`
  - Business Plan: `TW_TDI_BP.csv`
  - PPD (optional): `TW_TDI_PPD.csv`

> Note: PPD files are optional in memory mode. Main and Business Plan are required.

### 2) Manual CSV Upload
Upload files directly in the app:
- Main data CSV
- Business plan CSV
- PPD CSV

## Minimum Column Requirements

Configured in `config.py`:

- Demand charts: `customer`, `demand`
- Price charts: `customer`, `demand`, `pocket price`
- Business plan: `customer`, `year`, `min`, `base`, `max`
- Centered bubble: `customer`, `year`, `sow`, `ppd`, `volume`

## Notes

- Some visual modes (for readability) may apply display scaling for chart rendering. Hover values continue to show original values where implemented.
- Charts include UI controls for fonts, axis ranges, bubble scaling, and other layout options.

## Troubleshooting

- If a chart is empty:
  - Verify required CSV columns.
  - Verify selected country/material matches the dataset.
  - Check that uploaded files are not empty.
- If memory mode fails:
  - Confirm required files exist in `data/` with exact names.
- If labels overlap:
  - Use chart settings (font sizes, scale controls) to improve readability.

## License

Internal project use.
