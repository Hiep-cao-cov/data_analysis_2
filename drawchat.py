import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_customer_demand(df, customer_name, customer_column, suppliers, year_column, y_range, 
                         title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, 
                         legend_title_fontsize, percentage_label_fontsize, customer_name_font_size, 
                         demand_label_font_size, y_min=None, y_max=None):
    
    df_filtered = df[df[customer_column] == customer_name].copy()
    if df_filtered.empty:
        raise ValueError(f"No data for customer {customer_name}")

    # --- 1. SORT SUPPLIERS BY TOTAL VOLUME ---
    # We sum the volume for each supplier across all years to determine the order
    supplier_totals = {s: df_filtered[s].sum() for s in suppliers if s in df_filtered.columns}
    # Sort: Smallest total volume first (will be at the bottom of the stack)
    sorted_suppliers = sorted(supplier_totals.keys(), key=lambda x: supplier_totals[x])

    fig = go.Figure()
    total_supply_by_year = {row[year_column]: row['demand'] for _, row in df_filtered.iterrows()}

    # --- 2. ADD SORTED SUPPLIERS ---
    for i, supplier in enumerate(sorted_suppliers):
        values = []
        text_labels = []
        
        for _, row in df_filtered.iterrows():
            year = row[year_column]
            value = row[supplier] if pd.notna(row[supplier]) else 0
            total_supply = total_supply_by_year.get(year, 0)
            
            if total_supply > 0:
                percentage = (value / total_supply) * 100
                values.append(value)
                text_labels.append(f"{percentage:.1f}%")
            else:
                values.append(0)
                text_labels.append("")
        
        fig.add_trace(go.Bar(
            x=df_filtered[year_column],
            y=values,
            name=supplier.capitalize(),
            text=text_labels,
            textposition='auto', 
            cliponaxis=False,
            # Adjust bar width indirectly via layout bargap, or directly here:
            width=0.4, # 0.4 makes the bars thinner (default is ~0.8)
            textfont=dict(size=percentage_label_fontsize, color='black',family='Arial Black'),
            marker=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
            hovertemplate=(
                    '<b>Supplier:</b> ' + supplier.capitalize() + '<br>' +
                    '<b>Volume:</b> %{y:.0f} mt<extra></extra>'
                ),
            legendrank=i + 1 
        ))

    # --- 3. ADD TOTAL DEMAND OUTLINE ---
    demand_values = df_filtered['demand'].tolist()
    fig.add_trace(go.Bar(
        x=df_filtered[year_column],
        y=demand_values,
        name='Total Demand',
        width=0.4, # Match the supplier bar width
        marker=dict(color='rgba(0,0,0,0)', line=dict(color='red', width=2)),
        text=[f"{val:.0f} mt" for val in demand_values],
        textposition='outside',
        textfont=dict(size=demand_label_font_size+8, color='red'),
        yaxis='y2',
        hoverinfo='skip',
        
    ))

    max_val = max(demand_values) if demand_values else 100
    final_y_range = [y_min if y_min is not None else 0, 
                     y_max if y_max is not None else max_val * 1.2]

    # --- 4. LAYOUT WITH THINNER BARS & LEFT LEGEND ---
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"{customer_name} Volume: Sorted by Size",
            font=dict(size=customer_name_font_size, family='Arial Black'),
            x=0.5, xanchor='center'
        ),
        xaxis=dict(type='category', tickfont=dict(size=tick_fontsize)),
        yaxis=dict(range=final_y_range, tickfont=dict(size=tick_fontsize)),
        yaxis2=dict(overlaying='y', side='right', range=final_y_range, showticklabels=False),
        barmode='stack',
        bargap=0.5, # Increases space between year groups (makes bars look thinner)
        
        legend=dict(
            orientation="v",
            x=-0.3,           
            xanchor='right',   
            y=0.5,
            yanchor='middle',
            traceorder="reversed", 
            font=dict(size=legend_fontsize),
            title=dict(text='Suppliers', font=dict(size=legend_title_fontsize))
        ),
        margin=dict(l=250, r=50, t=100, b=100), 
        width=1100, 
        height=700
    )
    
    return fig



def plot_customer_bubble_clean_with_median(df, customer_column, demand_column, price_column, year_filter, bubble_scale, alpha, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max, min_demand_threshold=10):
    """Enhanced interactive bubble chart using Plotly with hover information and interactive features"""
    df_filtered = df[df['year'] == year_filter].copy() if year_filter else df.copy()
    
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
    
    # Add bubble scatter plot
    fig.add_trace(go.Scatter(
        x=list(range(n_customers)),
        y=customer_data[price_column],
        mode='markers+text',
        marker=dict(
            size=[size/8 for size in bubble_sizes],
            color=colors,
            opacity=alpha,
            line=dict(width=2, color='black'),
            sizemode='diameter'
        ),
        text=customer_data[customer_column],
        textposition='bottom center',
        textfont=dict(
            size=customer_name_font_size,
            color='black'
        ),
        customdata=customer_data[demand_column],
        hovertemplate=(
            '<b>%{text}</b><br>' +
            f'{price_column.replace("_", " ").title()}: %{{y:.2f}} $/kg<br>' +
            f'{demand_column.replace("_", " ").title()}: %{{customdata:.0f}} mt<br>' +
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
        text=[f'{demand:.0f}' for demand in customer_data[demand_column]],
        textfont=dict(
            size=demand_label_font_size,
            color='black',
            family='Arial Black'
        ),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add average price line
    fig.add_hline(
        y=avg_price,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Average: {avg_price:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color="red"),
        annotation_bgcolor="white",
        annotation_bordercolor="red",
        annotation_borderwidth=1
    )
    
    # Add median price line
    fig.add_hline(
        y=median_price,
        line_dash="dash",
        line_color="green",
        line_width=3,
        annotation_text=f"Median: {median_price:.2f}",
        annotation_position="bottom right" if abs(avg_price - median_price) < 0.5 else "top right",
        annotation_font=dict(size=12, color="green"),
        annotation_bgcolor="white",
        annotation_bordercolor="green",
        annotation_borderwidth=1
    )
    
    # Customize layout
    scaling_method = "Log" if use_log_scaling else "Sqrt"
    title_text = f'Pocket Prices {year_filter} (Scale: {bubble_scale:.1f}, {scaling_method} scaling, Min: {min_demand_threshold})'
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
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dot',
            range=y_range
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=16,
            font_family="Arial",
            font_color="blue"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=900,
        margin=dict(l=80, r=80, t=100, b=80),
        hovermode='closest'
    )
    
    return fig, avg_price, median_price, customer_data

def plot_customer_bubble_centered(df, customer_column, sow_column, ppd_column, volume_column, year_filter, bubble_scale=1.0, alpha=0.6, 
                                 title_fontsize=20, axis_label_fontsize=16, tick_fontsize=12, customer_name_font_size=12, 
                                 volume_label_font_size=10, min_volume_threshold=10, y_min=None, y_max=None, material=None):
    """Enhanced interactive bubble chart using Plotly with centered axes (sow=50, ppd=0)"""
    df_filtered = df[df['year'] == year_filter].copy() if year_filter else df.copy()
    
    customer_data = df_filtered.groupby(customer_column).agg({
        sow_column: 'mean',
        ppd_column: 'mean',
        volume_column: 'sum'
    }).reset_index().dropna(subset=[sow_column, ppd_column, volume_column])
    
    if customer_data.empty:
        raise ValueError("No valid data after filtering and aggregation")
    
    customer_data = customer_data[customer_data[volume_column] > min_volume_threshold].copy()
    
    if customer_data.empty:
        raise ValueError(f"No customers with volume > {min_volume_threshold} after filtering")
    
    print(f"Filtered out customers with volume <= {min_volume_threshold}. Remaining customers: {len(customer_data)}")
    
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
    
    # Bubble size calculation
    min_volume = customer_data[volume_column].min()
    max_volume = customer_data[volume_column].max()
    
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
        bubble_size = calculate_bubble_size_log(volume, min_volume, max_volume, bubble_scale) if use_log_scaling else calculate_bubble_size_sqrt(volume, min_volume, max_volume, bubble_scale)
        bubble_sizes.append(bubble_size / 8)
    
    # Create the figure
    fig = go.Figure()
    
    # Add bubble scatter plot
    fig.add_trace(go.Scatter(
        x=customer_data['sow_shifted'],
        y=customer_data[ppd_column],
        mode='markers+text',
        marker=dict(
            size=bubble_sizes,
            color=colors,
            opacity=alpha,
            line=dict(width=2, color='black'),
            sizemode='diameter'
        ),
        text=customer_data[customer_column],
        textposition='top center',
        textfont=dict(
            size=customer_name_font_size,
            color='black',
            family='Arial'
        ),
        customdata=customer_data[[sow_column, volume_column]],
        hovertemplate=(
            '<b>%{text}</b><br>' +
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
        text=[f'{volume:.0f}' for volume in customer_data[volume_column]],
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
    title_text = f'Customer Bubble Chart {year_filter} (Scale: {bubble_scale:.1f}, {scaling_method} scaling, Min Volume: {min_volume_threshold})'
    if y_min is not None and y_max is not None:
        title_text += f', Y-range: {y_min:.1f}-{y_max:.1f}'
    
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
            gridcolor='lightgray',
            gridwidth=1,
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
            gridcolor='lightgray',
            gridwidth=1,
            griddash='dot',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            range=y_range
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=16,
            font_family="Arial",
            font_color="blue"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=900,
        margin=dict(l=80, r=80, t=100, b=80),
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
    
    # Price lines (unchanged)
    for i, price_col in enumerate(price_columns):
        fig.add_trace(go.Scatter(
            x=df_filtered[year_column],
            y=df_filtered[price_col],
            name=price_col.replace('_', ' ').title(),
            mode='lines+markers+text',
            yaxis='y2',
            text=df_filtered[price_col].round(2).astype(str),
            textposition='top center',
            textfont=dict(size=price_annotation_fontsize*1.4, color=price_colors[i]),
            line=dict(color=price_colors[i], width=2),
            hovertemplate=(
                'Year : <b>%{x}</b><br>' +
                'Price: <b>%{y:.2f}</b> USD/kg<extra></extra>'
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
            font=dict(size=customer_name_font_size, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Year', font=dict(size=axis_label_fontsize)),
            tickfont=dict(size=tick_fontsize),
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

import plotly.graph_objects as go

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
            marker=dict(color=color)
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
            tickfont=dict(size=tick_fontsize),
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
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600
    )
    return fig