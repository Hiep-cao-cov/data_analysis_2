"""
PMDI / TDI market dashboard — thin entry that delegates to the ``dashboard`` package.

Run: ``streamlit run app.py`` from the project root (directory containing ``data/``).
"""

from dashboard.bootstrap import main

if __name__ == "__main__":
    main()
