"""
PMDI / TDI market dashboard — thin entry that delegates to the ``dashboard`` package.

Run: ``streamlit run app.py`` from the project root (directory containing ``data/``).
"""

from dashboard.bootstrap import main

# Streamlit runs this file on every session; do not guard with ``if __name__ == "__main__"``
# (that can prevent ``main()`` from running in some hosted environments).
main()
