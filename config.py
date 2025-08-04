# Configuration file for PMDI and TDI Market Visualization Dashboard

# Required columns for different chart types
REQUIRED_COLUMNS = {
    'demand_charts': ['customer', 'demand'],
    'price_charts': ['customer', 'demand', 'pocket price'],
    'business_plan': ['customer', 'year', 'min', 'base', 'max'],
    'bubble_centered': ['customer', 'year', 'sow', 'ppd', 'volume']
}

# Supplier lists for PMDI and TDI
SUPPLIERS = {
    'pmdi': ['covestro', 'tosoh', 'wanhua', 'kmc', 'basf', 'sabic', 'huntsman', 'other'],
    'tdi': ['covestro', 'mcns', 'wanhua', 'basf', 'hanwha', 'sabic', 'other'],
    'covestro': ['covestro']
}

# Available chart types
CHART_TYPES = [
    "Customer Demand",
    "Account price vs Volume",
    "Business plan",
    "Customer bubble Chart",
    "Customer Bubble Chart (Centered)"
]

# Supported countries and materials
COUNTRIES = ["Vietnam", "Taiwan"]
MATERIALS = ["PMDI", "TDI"]