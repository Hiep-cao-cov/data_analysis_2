import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def demand_chart_volume_columns(all_columns, material=None, suppliers_fallback=None):
    """
    Supplier volume columns for the Customer Demand stacked chart.

    When ``material`` is set, columns are taken by position on the main CSV layout:
    indices 4:-3 for PMDI/MDI (exclude trailing vn_pp, seap_pp, apac_pp) and 4:-2 for TDI
    (exclude vn_pp/tw_pp and apac_pp). Otherwise uses ``suppliers_fallback`` names that exist
    in ``all_columns``.
    """
    cols = list(all_columns)
    if material:
        key = (material or "").strip().upper().replace(" ", "")
        if key in ("PMDI", "MDI"):
            if len(cols) < 8:
                return []
            chosen = cols[4:-3]
        elif key == "TDI":
            if len(cols) < 7:
                return []
            chosen = cols[4:-2]
        else:
            chosen = []
        return [c for c in chosen if c in cols]
    return [s for s in (suppliers_fallback or []) if s in cols]


def plot_customer_demand(df, customer_name, customer_column, suppliers, year_column, y_range, 
                         title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, 
                         legend_title_fontsize, percentage_label_fontsize, customer_name_font_size, 
                         demand_label_font_size, y_min=None, y_max=None, demand_power_alpha=1.0,
                         material=None):
    
    df_filtered = df[df[customer_column] == customer_name].copy()
    if df_filtered.empty:
        raise ValueError(f"No data for customer {customer_name}")

    volume_cols = demand_chart_volume_columns(
        df_filtered.columns, material=material, suppliers_fallback=suppliers
    )
    if not volume_cols:
        raise ValueError(
            f"No supplier volume columns for customer {customer_name} "
            f"(material={material!r}, layout or suppliers list)."
        )

    # --- 1. SORT SUPPLIERS BY TOTAL VOLUME ---
    # We sum the volume for each supplier across all years to determine the order
    supplier_totals = {s: df_filtered[s].sum() for s in volume_cols}
    # Sort: Smallest total volume first (will be at the bottom of the stack)
    sorted_suppliers = sorted(supplier_totals.keys(), key=lambda x: supplier_totals[x])

    total_supply_by_year = {}
    for _, row in df_filtered.iterrows():
        y = row[year_column]
        total_supply_by_year[y] = sum(
            (row[s] if pd.notna(row[s]) else 0) for s in sorted_suppliers
        )

    fig = go.Figure()

    # Visual scaling for display only; hover/text still show real values.
    alpha = float(demand_power_alpha) if demand_power_alpha is not None else 1.0
    alpha = min(1.0, max(0.3, alpha))

    # --- 2. ADD SORTED SUPPLIERS ---
    # Keep existing sort/stack order logic, but pin Covestro to blue for consistency.
    covestro_blue = px.colors.qualitative.Plotly[0]
    scaled_total_by_year = {year: 0.0 for year in df_filtered[year_column].tolist()}
    for i, supplier in enumerate(sorted_suppliers):
        values = []
        real_values = []
        text_labels = []
        
        for _, row in df_filtered.iterrows():
            year = row[year_column]
            value = row[supplier] if pd.notna(row[supplier]) else 0
            total_supply = total_supply_by_year.get(year, 0)
            
            if total_supply > 0:
                percentage = (value / total_supply) * 100
                scaled_value = np.power(max(value, 0), alpha)
                values.append(scaled_value)
                scaled_total_by_year[year] = scaled_total_by_year.get(year, 0.0) + scaled_value
                real_values.append(value)
                text_labels.append(f"{percentage:.1f}%")
            else:
                values.append(0)
                real_values.append(0)
                text_labels.append("")
        
        supplier_color = (
            covestro_blue
            if str(supplier).strip().lower() == "covestro"
            else px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        )
        fig.add_trace(go.Bar(
            x=df_filtered[year_column],
            y=values,
            name=supplier.capitalize(),
            text=text_labels,
            textposition='auto', 
            cliponaxis=False,
            # Make bars wider for better readability.
            width=0.7,
            textfont=dict(size=percentage_label_fontsize, color='black',family='Arial Black'),
            marker=dict(color=supplier_color),
            customdata=real_values,
            hoverlabel=dict(
                bgcolor=supplier_color,
                bordercolor=supplier_color,
                font=dict(color='white')
            ),
            hovertemplate=(
                    '<b>Supplier:</b> ' + supplier.capitalize() + '<br>' +
                    '<b>Volume:</b> %{customdata:.0f} mt<extra></extra>'
                ),
            legendrank=i + 1 
        ))

    # --- 3. ADD TOTAL DEMAND OUTLINE ---
    # Use "demand" column for all years when available.
    if "demand" in df_filtered.columns:
        demand_values = [float(row["demand"]) if pd.notna(row["demand"]) else 0.0 for _, row in df_filtered.iterrows()]
        demand_values_scaled = [np.power(max(v, 0.0), alpha) for v in demand_values]
    else:
        # Fallback for legacy files without demand column.
        demand_values = [total_supply_by_year.get(row[year_column], 0.0) for _, row in df_filtered.iterrows()]
        demand_values_scaled = [scaled_total_by_year.get(year, 0.0) for year in df_filtered[year_column].tolist()]
    fig.add_trace(go.Bar(
        x=df_filtered[year_column],
        y=demand_values_scaled,
        name='Total Demand',
        width=0.7, # Match the supplier bar width
        marker=dict(color='rgba(0,0,0,0)', line=dict(color='red', width=2)),
        text=[f"{val:.0f} mt" for val in demand_values],
        textposition='outside',
        textfont=dict(size=demand_label_font_size+8, color='red'),
        yaxis='y2',
        hoverinfo='skip',
        
    ))

    max_val = max(demand_values_scaled) if demand_values_scaled else 100
    final_y_range = [y_min if y_min is not None else 0, 
                     y_max if y_max is not None else max_val * 1.2]

    # --- 4. LAYOUT WITH THINNER BARS & LEFT LEGEND ---
    is_visual_scaled = alpha < 0.999
    y_axis_title = f"Volume (mt) [visual scale, alpha={alpha:.2f}]" if is_visual_scaled else "Volume (mt)"
    title_suffix = (
        f" (Visual power={alpha:.2f} | WARNING: Y-axis is visually scaled, not absolute mt)"
        if is_visual_scaled else ""
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"{customer_name} Volume Demand",
            font=dict(size=title_fontsize, family='Arial Black'),
            x=0.5, xanchor='center'
        ),
        xaxis=dict(type='category', tickfont=dict(size=tick_fontsize + 3, family='Arial Black')),
        yaxis=dict(
            title=dict(text=y_axis_title, font=dict(size=axis_label_fontsize)),
            range=final_y_range,
            tickfont=dict(size=tick_fontsize)
        ),
        yaxis2=dict(
            overlaying='y',
            side='right',
            showticklabels=False,
            matches='y',
            range=final_y_range,
        ),
        barmode='stack',
        bargap=0.2, # Smaller gap to make bars look wider
        
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor='left',   
            y=0.99,
            yanchor='top',
            traceorder="reversed", 
            font=dict(size=legend_fontsize),
            title=dict(text='Suppliers', font=dict(size=legend_title_fontsize))
        ),
        margin=dict(l=90, r=170, t=100, b=100), 
        width=1100, 
        height=700
    )
    
    return fig



def plot_customer_bubble_clean_with_median(df, customer_column, demand_column, price_column, year_filter, bubble_scale, alpha, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max, min_demand_threshold=10):
    """Enhanced interactive bubble chart using Plotly with hover information and interactive features"""
    if 'year' in df.columns:
        if year_filter is not None:
            df_filtered = filter_dataframe_by_year(df, year_filter)
        else:
            ymx = _series_calendar_year(df['year']).dropna()
            df_filtered = filter_dataframe_by_year(df, int(ymx.max())) if not ymx.empty else df.iloc[0:0].copy()
    else:
        df_filtered = df.copy()
    
    customer_data = df_filtered.groupby(customer_column).agg({
        demand_column: 'sum',
        price_column: 'mean'
    }).reset_index().dropna(subset=[demand_column, price_column])
    
    if customer_data.empty:
        raise ValueError("No valid data after filtering and aggregation")
    
    customer_data = customer_data[customer_data[demand_column] > min_demand_threshold].copy()
    
    if customer_data.empty:
        raise ValueError(f"No customers with demand > {min_demand_threshold} after filtering")
    
    print(f"Filtered out customers with demand <= {min_demand_threshold}. Remaining customers: {len(customer_data)}")
    
    avg_price = customer_data[price_column].mean()
    median_price = customer_data[price_column].median()
    customer_data = customer_data.sort_values(price_column, ascending=False)
    # Unique hover order 1..n by demand (dense rank gave duplicate numbers when volumes tied).
    _demand_order = customer_data.sort_values(
        [demand_column, customer_column],
        ascending=[False, True],
        kind="mergesort",
    )
    customer_data["rank"] = pd.Series(
        range(1, len(_demand_order) + 1), index=_demand_order.index
    )
    
    n_customers = len(customer_data)
    
    # Generate colors
    if n_customers <= 10: 
        colors = px.colors.qualitative.Set1[:n_customers]
    elif n_customers <= 20:
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        colors = colors[:n_customers]
    else:
        colors = px.colors.qualitative.Set3 * ((n_customers // len(px.colors.qualitative.Set3)) + 1)
        colors = colors[:n_customers]
    
    # Bubble size calculation
    min_demand = customer_data[demand_column].min()
    max_demand = customer_data[demand_column].max()
   
    def calculate_bubble_size_log(demand, min_demand, max_demand, bubble_scale):
        if demand <= 0:
            return 20 * bubble_scale
        log_demand = np.log10(demand + 1)
        log_min = np.log10(min_demand + 1)
        log_max = np.log10(max_demand + 1)
        
        if log_max == log_min:
            normalized = 0.5
        else:
            normalized = (log_demand - log_min) / (log_max - log_min)
        
        min_size = 40 * bubble_scale
        max_size = 180 * bubble_scale
        bubble_size = min_size + (normalized * (max_size - min_size))
        return bubble_size
    
    def calculate_bubble_size_sqrt(demand, min_demand, max_demand, bubble_scale):
        if demand <= 0:
            return 20 * bubble_scale
        sqrt_demand = np.sqrt(demand)
        sqrt_min = np.sqrt(min_demand)
        sqrt_max = np.sqrt(max_demand)
        
        if sqrt_max == sqrt_min:
            normalized = 0.5
        else:
            normalized = (sqrt_demand - sqrt_min) / (sqrt_max - sqrt_min)
        
        min_size = 30 * bubble_scale
        max_size = 120 * bubble_scale
        bubble_size = min_size + (normalized * (max_size - min_size))
        return bubble_size
    
    use_log_scaling = True
    bubble_sizes = []
    for _, row in customer_data.iterrows():
        demand = row[demand_column]
        bubble_size = calculate_bubble_size_log(demand, min_demand, max_demand, bubble_scale) if use_log_scaling else calculate_bubble_size_sqrt(demand, min_demand, max_demand, bubble_scale)
        bubble_sizes.append(bubble_size)
    
    # Create the figure
    fig = go.Figure()
    
    # Customer names under bubbles: only top-demand accounts to reduce clutter.
    top_label_count = min(12, n_customers)
    top_label_customers = set(
        customer_data.nlargest(top_label_count, demand_column)[customer_column].astype(str).tolist()
    )
    customer_text = [
        name if str(name) in top_label_customers else ""
        for name in customer_data[customer_column]
    ]
    demand_center_text = [f"{d:.0f}" for d in customer_data[demand_column]]

    # Add bubble scatter plot
    fig.add_trace(go.Scatter(
        x=list(range(n_customers)),
        y=customer_data[price_column],
        mode='markers+text',
        marker=dict(
            size=[size/8 for size in bubble_sizes],
            color=colors,
            opacity=max(0.35, alpha),
            line=dict(width=1.5, color='rgba(40,40,40,0.75)'),
            sizemode='diameter'
        ),
        text=customer_text,
        textposition='bottom center',
        textfont=dict(
            size=customer_name_font_size,
            color='black'
        ),
        customdata=customer_data[[demand_column, 'rank', customer_column]],
        hovertemplate=(
            '<b>%{customdata[1]:.0f}. %{customdata[2]}</b><br>' +
            f'{price_column.replace("_", " ").title()}: %{{y:.2f}} $/kg<br>' +
            f'{demand_column.replace("_", " ").title()}: %{{customdata[0]:.0f}} mt<br>' +
            '<extra></extra>'
        ),
        name='Customers',
        showlegend=False
    ))
    
    # Add demand labels inside bubbles
    fig.add_trace(go.Scatter(
        x=list(range(n_customers)),
        y=customer_data[price_column],
        mode='text',
        text=demand_center_text,
        textfont=dict(
            size=demand_label_font_size,
            color='black',
            family='Arial Black'
        ),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add average and median guide lines
    fig.add_hline(
        y=avg_price,
        line_dash="dashdot",
        line_color="#D32F2F",
        line_width=2,
        annotation_text=f"Average: {avg_price:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#D32F2F"),
        annotation_bgcolor="white",
        annotation_bordercolor="#D32F2F",
        annotation_borderwidth=1
    )
    
    # Add median price line
    fig.add_hline(
        y=median_price,
        line_dash="dot",
        line_color="#2E7D32",
        line_width=2,
        annotation_text=f"Median: {median_price:.2f}",
        annotation_position="bottom right" if abs(avg_price - median_price) < 0.5 else "top right",
        annotation_font=dict(size=12, color="#2E7D32"),
        annotation_bgcolor="white",
        annotation_bordercolor="#2E7D32",
        annotation_borderwidth=1
    )

    # Add simple bubble size legend (small / medium / large demand)
    legend_demands = [
        float(min_demand),
        float((min_demand + max_demand) / 2),
        float(max_demand)
    ]
    legend_sizes = [
        calculate_bubble_size_log(d, min_demand, max_demand, bubble_scale) / 8
        for d in legend_demands
    ]
    legend_labels = ['Small', 'Medium', 'Large']
    for size, demand_val, label in zip(legend_sizes, legend_demands, legend_labels):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=size,
                color='rgba(120,120,120,0.45)',
                line=dict(width=1, color='rgba(80,80,80,0.8)')
            ),
            name=f"{label}: {demand_val:.0f} mt",
            legendgroup='size_guide'
        ))
    
    # Customize layout
    scaling_method = "Log" if use_log_scaling else "Sqrt"
    title_text = f"Pocket Prices ({year_filter}"
    if y_min is not None and y_max is not None:
        title_text += f', Y-range: {y_min:.1f}-{y_max:.1f}'
    title_text += ')'
    
    if bubble_scale > 10.0:
        title_text += ' - Overlap Mode'
    
    # Set Y-axis range
    if y_min is not None and y_max is not None:
        y_range = [y_min, y_max]
    else:
        price_min = min(avg_price, median_price, customer_data[price_column].min())
        price_max = max(avg_price, median_price, customer_data[price_column].max())
        range_magnitude = price_max - price_min
        padding = max(0.5, range_magnitude * 0.1)
        y_range = [max(0, price_min - padding), price_max + padding]
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=title_fontsize, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text='Customers',
                font=dict(size=axis_label_fontsize)
            ),
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, n_customers - 0.5]
        ),
        yaxis=dict(
            title=dict(
                text=f'{price_column.replace("_", " ").title()} ($/kg)',
                font=dict(size=axis_label_fontsize)
            ),
            tickfont=dict(size=tick_fontsize),
            gridcolor='rgba(120,120,120,0.25)',
            gridwidth=0.8,
            griddash='dot',
            range=y_range
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#D0D5DD",
            font_size=14,
            font_family="Arial",
            font_color="#101828"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1250,
        height=860,
        margin=dict(l=70, r=70, t=90, b=70),
        legend=dict(
            orientation='h',
            x=0.5,
            xanchor='center',
            y=1.02,
            yanchor='bottom',
            font=dict(size=legend_fontsize),
            title=dict(text='Bubble Size Guide')
        ),
        hovermode='closest'
    )
    
    return fig, avg_price, median_price, customer_data


def _series_calendar_year(s: pd.Series) -> pd.Series:
    """Integer calendar year from a ``year`` column (datetimes or numeric/strings)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.year.astype('Int64')
    x = pd.to_numeric(s, errors='coerce')
    return np.round(x).astype('Int64')


def filter_dataframe_by_year(df, year_filter):
    """Rows for one calendar year only (never mix years in the same frame)."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    if 'year' not in df.columns:
        return df.copy()
    y_tgt = pd.to_numeric(year_filter, errors='coerce')
    if pd.isna(y_tgt):
        return df.iloc[0:0].copy()
    y_tgt_i = int(round(float(y_tgt)))
    y_col = _series_calendar_year(df['year'])
    return df.loc[y_col == y_tgt_i].copy()


def _filtered_df_centered_bubble_source(df, year_filter):
    """Rows for one calendar year in **original row order** (CSV order), before groupby."""
    if df is None or df.empty:
        return pd.DataFrame()
    if 'year' not in df.columns:
        df_filtered = df.copy()
    elif year_filter is not None:
        df_filtered = filter_dataframe_by_year(df, year_filter)
    else:
        y_cal = _series_calendar_year(df['year'])
        if y_cal.dropna().empty:
            return pd.DataFrame()
        y_default = int(y_cal.max())
        df_filtered = df.loc[y_cal == y_default].copy()
    if df_filtered.empty:
        return pd.DataFrame()
    if 'year' in df_filtered.columns:
        y_uni = _series_calendar_year(df_filtered['year']).dropna().unique()
        if len(y_uni) > 1 and year_filter is not None:
            yt = int(round(float(pd.to_numeric(year_filter, errors='coerce'))))
            y_cal = _series_calendar_year(df_filtered['year'])
            df_filtered = df_filtered.loc[y_cal == yt].copy()
        elif len(y_uni) > 1:
            y_cal = _series_calendar_year(df_filtered['year'])
            y_keep = int(y_cal.max())
            df_filtered = df_filtered.loc[y_cal == y_keep].copy()
    if df_filtered.empty:
        return pd.DataFrame()
    return df_filtered


def _csv_first_appearance_customer_order(df_filtered, customer_column='customer'):
    """Customer names in order of first row appearance in ``df_filtered`` (file/CSV order)."""
    if df_filtered is None or df_filtered.empty or customer_column not in df_filtered.columns:
        return []
    seen = set()
    order = []
    for val in df_filtered[customer_column].tolist():
        if pd.isna(val):
            continue
        c = str(val).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        order.append(c)
    return order


def _aggregate_customers_centered_bubble(
    df,
    customer_column='customer',
    sow_column='sow',
    ppd_column='ppd',
    volume_column='volume',
    year_filter=None,
):
    """Per-customer SOW/PPD means and volume sum for the centered bubble chart."""
    df_filtered = _filtered_df_centered_bubble_source(df, year_filter)
    if df_filtered.empty:
        return pd.DataFrame()
    return df_filtered.groupby(customer_column).agg({
        sow_column: 'mean',
        ppd_column: 'mean',
        volume_column: 'sum'
    }).reset_index().dropna(subset=[sow_column, ppd_column, volume_column])


def get_centered_bubble_volume_bounds(
    df,
    customer_column='customer',
    sow_column='sow',
    ppd_column='ppd',
    volume_column='volume',
    year_filter=None,
):
    """Min/max aggregated customer volume in the dataset (for range slider limits)."""
    customer_data = _aggregate_customers_centered_bubble(
        df, customer_column, sow_column, ppd_column, volume_column, year_filter
    )
    if customer_data.empty:
        return 0.0, 1.0
    lo = float(customer_data[volume_column].min())
    hi = float(customer_data[volume_column].max())
    if lo > hi:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    return lo, hi


def list_customers_for_centered_bubble(
    df,
    customer_column='customer',
    sow_column='sow',
    ppd_column='ppd',
    volume_column='volume',
    year_filter=None,
    volume_min=0.0,
    volume_max=None,
):
    """Customer names in **CSV / file row order** (first appearance that year), after volume range — same as hover rank."""
    if volume_max is None:
        volume_max = float('inf')
    if df is None or df.empty:
        return []
    df_src = _filtered_df_centered_bubble_source(df, year_filter)
    csv_order = _csv_first_appearance_customer_order(df_src, customer_column)
    customer_data = _aggregate_customers_centered_bubble(
        df, customer_column, sow_column, ppd_column, volume_column, year_filter
    )
    if customer_data.empty:
        return []
    customer_data = customer_data[
        (customer_data[volume_column] >= volume_min) &
        (customer_data[volume_column] <= volume_max)
    ]
    if customer_data.empty:
        return []
    valid = set(customer_data[customer_column].astype(str).str.strip())
    ordered = [c for c in csv_order if c in valid]
    ordered.extend(sorted(valid - set(ordered)))
    return ordered


def plot_customer_bubble_centered(df, customer_column, sow_column, ppd_column, volume_column, year_filter, bubble_scale=1.0, alpha=0.6, 
                                 title_fontsize=20, axis_label_fontsize=16, tick_fontsize=12, customer_name_font_size=12, 
                                 volume_label_font_size=10, volume_min=0.0, volume_max=None, y_min=None, y_max=None, material=None,
                                 exclude_customers=None):
    """Enhanced interactive bubble chart using Plotly with centered axes (sow=50, ppd=0)"""
    if volume_max is None:
        raise ValueError("volume_max is required for the centered bubble chart")
    if 'year' in df.columns:
        if year_filter is not None:
            df = filter_dataframe_by_year(df, year_filter)
        else:
            ymx = _series_calendar_year(df['year']).dropna()
            if not ymx.empty:
                y_sel = int(ymx.max())
                df = filter_dataframe_by_year(df, y_sel)
                year_filter = y_sel
    customer_data = _aggregate_customers_centered_bubble(
        df, customer_column, sow_column, ppd_column, volume_column, year_filter
    )

    if customer_data.empty:
        raise ValueError("No valid data after filtering and aggregation")

    # Marker sizes use the full market for this year (min→smallest bubble, max→largest). The volume
    # slider and “hide customer” only change who is drawn, not the size scale — so 600 vs 660 mt
    # stay close unless the whole dataset spans a wide volume range.
    ref_min = float(customer_data[volume_column].min())
    ref_max = float(customer_data[volume_column].max())

    customer_data = customer_data[
        (customer_data[volume_column] >= volume_min) &
        (customer_data[volume_column] <= volume_max)
    ].copy()

    if customer_data.empty:
        raise ValueError(
            f"No customers in volume range {volume_min:g}–{volume_max:g} mt (after aggregation)."
        )

    # Rank 1..n in **CSV row order** (first appearance in file for this year), same as the right-hand list.
    # Assign before exclude_customers so hover numbers stay stable when some bubbles are hidden.
    csv_order = _csv_first_appearance_customer_order(df, customer_column)
    valid = set(customer_data[customer_column].astype(str).str.strip())
    ordered_keys = [c for c in csv_order if c in valid]
    ordered_keys.extend(sorted(valid - set(ordered_keys)))
    customer_data = (
        customer_data.assign(_k=customer_data[customer_column].astype(str).str.strip())
        .set_index("_k")
        .loc[ordered_keys]
        .reset_index(drop=True)
    )
    customer_data["rank"] = np.arange(1, len(customer_data) + 1)

    if exclude_customers:
        ex = {str(x) for x in exclude_customers}
        customer_data = customer_data[~customer_data[customer_column].astype(str).isin(ex)].copy()
    if customer_data.empty:
        raise ValueError(
            "No bubbles to show — every customer is hidden or filtered out. "
            "Show at least one customer in the list on the right."
        )
    
    print(f"Volume filter {volume_min:g}–{volume_max:g} mt. Customers shown: {len(customer_data)}")
    
    customer_data['sow_shifted'] = customer_data[sow_column] - 50
    
    n_customers = len(customer_data)
    
    # Generate colors
    if n_customers <= 10:
        colors = px.colors.qualitative.Set1[:n_customers]
    elif n_customers <= 20:
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        colors = colors[:n_customers]
    else:
        colors = px.colors.qualitative.Set3 * ((n_customers // len(px.colors.qualitative.Set3)) + 1)
        colors = colors[:n_customers]
    
    def calculate_bubble_size_log(volume, min_volume, max_volume, bubble_scale):
        if volume <= 0:
            return 20 * bubble_scale
        log_volume = np.log10(volume + 1)
        log_min = np.log10(min_volume + 1)
        log_max = np.log10(max_volume + 1)
        
        if log_max == log_min:
            normalized = 0.5
        else:
            normalized = (log_volume - log_min) / (log_max - log_min)
        
        min_size = 40 * bubble_scale
        max_size = 180 * bubble_scale
        return min_size + (normalized * (max_size - min_size))
    
    def calculate_bubble_size_sqrt(volume, min_volume, max_volume, bubble_scale):
        if volume <= 0:
            return 20 * bubble_scale
        sqrt_volume = np.sqrt(volume)
        sqrt_min = np.sqrt(min_volume)
        sqrt_max = np.sqrt(max_volume)
        
        if sqrt_max == sqrt_min:
            normalized = 0.5
        else:
            normalized = (sqrt_volume - sqrt_min) / (sqrt_max - sqrt_min)
        
        min_size = 30 * bubble_scale
        max_size = 120 * bubble_scale
        return min_size + (normalized * (max_size - min_size))
    
    use_log_scaling = True
    bubble_sizes = []
    for _, row in customer_data.iterrows():
        volume = row[volume_column]
        bubble_size = (
            calculate_bubble_size_log(volume, ref_min, ref_max, bubble_scale)
            if use_log_scaling
            else calculate_bubble_size_sqrt(volume, ref_min, ref_max, bubble_scale)
        )
        bubble_sizes.append(bubble_size / 8)
    
    # Create the figure
    fig = go.Figure()

    # Customer names above bubbles: only top-volume accounts (keeps the top of the chart readable).
    top_label_count = min(12, n_customers)
    top_label_customers = set(
        customer_data.nlargest(top_label_count, volume_column)[customer_column].astype(str).tolist()
    )
    customer_text = [
        name if str(name) in top_label_customers else ""
        for name in customer_data[customer_column]
    ]
    # Volume inside every bubble (previously only the top 12 showed a center number).
    volume_center_text = [f"{v:.0f}" for v in customer_data[volume_column]]
    
    # Add bubble scatter plot
    fig.add_trace(go.Scatter(
        x=customer_data['sow_shifted'],
        y=customer_data[ppd_column],
        mode='markers+text',
        marker=dict(
            size=bubble_sizes,
            color=colors,
            opacity=max(0.35, alpha),
            line=dict(width=1.5, color='rgba(40,40,40,0.75)'),
            sizemode='diameter'
        ),
        text=customer_text,
        textposition='top center',
        textfont=dict(
            size=customer_name_font_size,
            color='black',
            family='Arial'
        ),
        customdata=customer_data[[sow_column, volume_column, 'rank', customer_column]],
        hovertemplate=(
            '<b>%{customdata[2]:.0f}. %{customdata[3]}</b><br>' +
            f'SOW: %{{customdata[0]:.2f}}<br>' +
            f'PPD: %{{y:.6f}}<br>' +
            f'Volume: %{{customdata[1]:.0f}} mt<br>' +
            '<extra></extra>'
        ),
        name='Customers',
        showlegend=False
    ))
    
    # Add volume labels inside bubbles
    fig.add_trace(go.Scatter(
        x=customer_data['sow_shifted'],
        y=customer_data[ppd_column],
        mode='text',
        text=volume_center_text,
        textfont=dict(
            size=volume_label_font_size,
            color='black',
            family='Arial Black'
        ),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Calculate ranges for axes
    sow_shifted_min = customer_data['sow_shifted'].min()
    sow_shifted_max = customer_data['sow_shifted'].max()
    x_range = [min(-50, sow_shifted_min - 5), max(50, sow_shifted_max + 5)]
    
    if y_min is not None and y_max is not None:
        y_range = [y_min, y_max]
    else:
        ppd_min = customer_data[ppd_column].min()
        ppd_max = customer_data[ppd_column].max()
        range_magnitude = ppd_max - ppd_min
        padding = max(0.005, range_magnitude * 0.1)
        y_range = [ppd_min - padding, ppd_max + padding]
    
    # Customize layout
    scaling_method = "Log" if use_log_scaling else "Sqrt"
    title_text = f'Customer Bubble Chart {year_filter} (Volume: {volume_min:.0f}–{volume_max:.0f} mt)'
    if y_min is not None and y_max is not None:
        title_text += f', Y-range: {y_min:.1f}-{y_max:.1f}'
    
    # Add simple bubble size legend (small / medium / large volume) — same ref as markers
    legend_volumes = [
        ref_min,
        float((ref_min + ref_max) / 2),
        ref_max,
    ]
    legend_sizes = [
        calculate_bubble_size_log(v, ref_min, ref_max, bubble_scale) / 8
        for v in legend_volumes
    ]
    legend_labels = ['Small', 'Medium', 'Large']
    for size, vol_val, label in zip(legend_sizes, legend_volumes, legend_labels):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=size,
                color='rgba(120,120,120,0.45)',
                line=dict(width=1, color='rgba(80,80,80,0.8)')
            ),
            name=f"{label}: {vol_val:.0f} mt",
            legendgroup='size_guide_centered'
        ))

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=title_fontsize, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text='SOW (<50 left, >50 right)',
                font=dict(size=axis_label_fontsize)
            ),
            tickfont=dict(size=tick_fontsize),
            gridcolor='rgba(120,120,120,0.25)',
            gridwidth=0.8,
            griddash='dot',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            range=x_range,
            tickvals=[-50, -25, 0, 25, 50],
            ticktext=['0', '25', '50', '75', '100']
        ),
        yaxis=dict(
            title=dict(
                text='PPD (<0 down, >0 up)',
                font=dict(size=axis_label_fontsize)
            ),
            tickfont=dict(size=tick_fontsize),
            gridcolor='rgba(120,120,120,0.25)',
            gridwidth=0.8,
            griddash='dot',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            range=y_range
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#D0D5DD",
            font_size=14,
            font_family="Arial",
            font_color="#101828"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1250,
        height=860,
        margin=dict(l=70, r=70, t=90, b=70),
        legend=dict(
            orientation='h',
            x=0.5,
            xanchor='center',
            y=1.02,
            yanchor='bottom',
            font=dict(size=12),
            title=dict(text='Bubble Size Guide')
        ),
        hovermode='closest'
    )
    
    return fig

def plot_customer_demand_with_price(df, customer_name, customer_column, suppliers, year_column, 
                                   demand_range, price_range, price_columns, price_colors, 
                                   title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, 
                                   legend_title_fontsize, value_label_fontsize, price_annotation_fontsize, 
                                   annotation_spacing, customer_name_font_size, demand_label_font_size, 
                                   y_min=None, y_max=None, y_demand_min=None, y_demand_max=None):
    """Plot price vs volume chart using Plotly with Covestro volume and demand outline overlay"""
    df_filtered = df[df[customer_column] == customer_name]
    if df_filtered.empty:
        raise ValueError(f"No data for customer {customer_name}")
    
    fig = go.Figure()
    
    # Calculate total DEMAND for each year
    total_demand_by_year = {}
    for _, row in df_filtered.iterrows():
        year = row[year_column]
        total_demand = row['demand']
        total_demand_by_year[year] = total_demand
    
    # Add single bar for Covestro
    covestro_supplier = 'covestro'
    if covestro_supplier in df_filtered.columns:
        values = []
        text_labels = []
        
        for _, row in df_filtered.iterrows():
            year = row[year_column]
            value = row[covestro_supplier] if pd.notna(row[covestro_supplier]) else 0
            total_demand = total_demand_by_year[year]
            
            if total_demand > 0:
                percentage = (value / total_demand) * 100
                values.append(value)
                text_labels.append(f"{value:.0f} mt\n")
            else:
                values.append(0)
                text_labels.append("0 mt\n0%")
        
        fig.add_trace(go.Bar(
            x=df_filtered[year_column],
            y=values,
            name="Covestro",
            text=text_labels,
            textposition='inside',
            textfont=dict(size=value_label_fontsize*1.4),
            marker=dict(color=px.colors.qualitative.Plotly[0]),
            hovertemplate=(
                'Supplier: <b>Covestro</b><br>' +
                'Volume  : <b>%{y:.0f}</b> mt<extra></extra>'
            )
        ))
    
    # Add demand outline bars using tertiary y-axis
    if 'demand' in df_filtered.columns:
        demand_values = df_filtered['demand'].tolist()
    else:
        demand_values = [total_demand_by_year[year] for year in df_filtered[year_column]]
    
    fig.add_trace(go.Bar(
        x=df_filtered[year_column],
        y=demand_values,
        name='Total Demand',
        marker=dict(
            color='rgba(0,0,0,0)',  # Transparent fill
            line=dict(color='blue', width=3)  # Thick blue outline
        ),
        text=[f"{val:.0f} mt" for val in demand_values],
        textposition='outside',
        textfont=dict(size=demand_label_font_size, color='red',family='Arial Black'),
        hoverinfo='skip',
        showlegend=True,
        yaxis='y3'
    ))
    
    # Smart label staggering to reduce overlap when price lines are close.
    num_points = len(df_filtered)
    text_positions = {col: ['top center'] * num_points for col in price_columns}
    text_values = {
        col: df_filtered[col].round(2).astype(str).tolist()
        for col in price_columns
    }
    stagger_positions = [
        'top center', 'top right', 'middle right', 'bottom right',
        'bottom center', 'bottom left', 'middle left', 'top left'
    ]
    close_threshold = max(float(annotation_spacing), 0.01)

    for idx in range(num_points):
        points = []
        for col in price_columns:
            val = df_filtered.iloc[idx][col]
            if pd.notna(val):
                points.append((col, float(val)))
        if len(points) <= 1:
            continue

        points.sort(key=lambda x: x[1], reverse=True)
        start = 0
        while start < len(points):
            end = start
            while end + 1 < len(points) and abs(points[end][1] - points[end + 1][1]) <= close_threshold:
                end += 1

            cluster = points[start:end + 1]
            if len(cluster) > 1:
                for j, (col, _) in enumerate(cluster):
                    if j < len(stagger_positions):
                        text_positions[col][idx] = stagger_positions[j]
                    else:
                        text_values[col][idx] = ''
            start = end + 1

    # Price lines
    for i, price_col in enumerate(price_columns):
        price_label = price_col.replace('_', ' ').title()
        series_color = price_colors[i]
        fig.add_trace(go.Scatter(
            x=df_filtered[year_column],
            y=df_filtered[price_col],
            name=price_label,
            mode='lines+markers+text',
            yaxis='y2',
            text=text_values[price_col],
            textposition=text_positions[price_col],
            textfont=dict(size=price_annotation_fontsize*1.4, color=price_colors[i]),
            line=dict(color=price_colors[i], width=2),
            hoverlabel=dict(
                bgcolor=series_color,
                bordercolor=series_color,
                font=dict(color='white')
            ),
            hovertemplate=(
                'Year : <b>%{x}</b><br>' +
                f'{price_label}: <b>%{{y:.2f}}</b> USD/kg<extra></extra>'
            )
        ))
    
    # Set axis ranges
    if y_demand_min is not None and y_demand_max is not None:
        demand_range = [y_demand_min, y_demand_max]
    else:
        max_demand = max(demand_values) if demand_values else 0
        demand_range = [0, max(demand_range[1], max_demand * 1.1)]
    
    if y_min is not None and y_max is not None:
        price_range = [y_min, y_max]
    else:
        price_range = [price_range[0], price_range[1]]
    
    fig.update_layout(
        title=dict(
            text=f"{customer_name} Demand and Price",
            font=dict(size=title_fontsize, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Year', font=dict(size=axis_label_fontsize)),
            tickfont=dict(size=tick_fontsize + 3, family='Arial Black'),
            type='category'
        ),
        yaxis=dict(
            title=dict(text='Volume (mt)', font=dict(size=axis_label_fontsize)),
            tickfont=dict(size=tick_fontsize),
            range=demand_range,
            autorange=False,
            matches='y3'
        ),
        yaxis2=dict(
            title=dict(text='Price ($/kg)', font=dict(size=axis_label_fontsize)),
            tickfont=dict(size=tick_fontsize),
            range=price_range,
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            tickfont=dict(size=tick_fontsize),
            range=demand_range,
            overlaying='y',
            side='left',
            showticklabels=False,
            autorange=False
        ),
        barmode='group',  # Changed from 'stack' to 'group' since only one supplier bar
        legend=dict(
            title=dict(text='Metrics', font=dict(size=legend_title_fontsize)),
            font=dict(size=legend_fontsize)
        ),
        hoverlabel=dict(
            bgcolor="lightblue",
            bordercolor="darkblue",
            font=dict(size=16, family="Verdana", color="black")
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600
    )
    return fig


def plot_customer_business_plan(df, customer_name, is_taiwan, title_fontsize, axis_label_fontsize, 
                                tick_fontsize, legend_fontsize, value_label_fontsize):
    """Plot business plan chart using Plotly as a Stacked Bar Chart"""
    df_filtered = df[df['customer'] == customer_name]
    if df_filtered.empty:
        raise ValueError(f"No data for customer {customer_name}")
    
    fig = go.Figure()
    
    # We keep the same columns and colors
    for col, color in zip(['min', 'base', 'max'], ['blue', 'green', 'red']):
        fig.add_trace(go.Bar(
            x=df_filtered['year'],
            y=df_filtered[col],
            name=col.capitalize(),
            text=df_filtered[col].round(0).astype(str),
            textposition='inside', # 'inside' often looks better in stacked charts
            textfont=dict(size=value_label_fontsize + 3, color='white',family='Arial Black'),
            marker=dict(color=color),
            hoverinfo='skip',
        ))
    
    fig.update_layout(
        title=dict(
            text=f"{customer_name} Business Plan",
            font=dict(size=title_fontsize, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Year', font=dict(size=axis_label_fontsize)),
            tickfont=dict(size=tick_fontsize + 3, family='Arial Black'),
            type='category'
        ),
        yaxis=dict(
            title=dict(text='Demand (mt)', font=dict(size=axis_label_fontsize)),
            tickfont=dict(size=tick_fontsize)
        ),
        # CHANGE MADE HERE: group -> stack
        barmode='stack', 
        legend=dict(
            title=dict(text='Plan Type', font=dict(size=legend_fontsize)),
            font=dict(size=legend_fontsize)
        ),
        hoverlabel=dict(
            bgcolor="lightblue",
            bordercolor="darkblue",
            font=dict(size=16, family="Verdana", color="black")
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600
    )
    return fig
