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

## Authentication Setup (Required)

The app now requires sign-in before users can access dashboard content.

Set multi-user credentials (environment variables or Streamlit secrets):

- `APP_LOGIN_CREDENTIALS` in format `user1:pass1,user2:pass2`

For your requested 4 users, example `.env`:

```env
APP_LOGIN_CREDENTIALS=hiep:12345,binh:12345,chuong:12345,bella:12345
```

If `APP_LOGIN_CREDENTIALS` is not set, the app uses these built-in defaults:
- `hiep:12345`
- `binh:12345`
- `chuong:12345`
- `bella:12345`

## Deploy on Render (GitHub)

Typical failures come from **wrong port/host** (Streamlit defaults to `localhost` and a fixed port; Render injects **`PORT`** and expects **`0.0.0.0`**).

1. **Root directory** — If this project is a **subfolder** of your Git repo, set **Root Directory** in the Render service to that folder (the one that contains `app.py`, `dashboard/`, and `requirements.txt`).
2. **Build command:** `pip install --upgrade pip && pip install -r requirements.txt`
3. **Start command:**
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
   The repo includes `render.yaml` and `Procfile` with this command; you can also use **New → Blueprint** and point at `render.yaml`.
4. **Python** — Set **Environment** `PYTHON_VERSION` to `3.11.0` (or match `render.yaml`) if the build picks the wrong runtime.

If the build still fails, open the **Logs** tab and check for missing packages (add to `requirements.txt`) or path errors (`ModuleNotFoundError: dashboard` usually means the wrong root directory).

- **`Invalid requirement: '\x00#...` when pip runs** — `requirements.txt` was saved as **UTF-16** (e.g. Windows “Unicode” in some editors). Re-save the file as **UTF-8** (VS Code: bottom status bar → “UTF-8”), remove `#` comment lines if needed, commit, and redeploy. This repo’s `requirements.txt` is plain UTF-8 with only package lines.
- **Memory (preload) works for PMDI but not TDI on Render** — Linux is **case-sensitive** for file names. If Git has `VN_TDI_final.csv` but the app looks for `VN_TDI_FINAL.csv`, it fails on the server (Windows often still opens the file). Rename the CSVs in the repo to match the names in `dashboard/data.py` (`VN_TDI_FINAL.csv`, `VN_TDI_BP.csv`, `VN_TDI_PPD.csv`) or rely on the app’s case-insensitive lookup under `data/`.

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
