import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_customer_demand(df, customer_name, customer_column, suppliers, year_column, demand_ylim, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, legend_title_fontsize, value_label_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max, show_percentages=True):
    """
    Plot a combined chart with stacked bars for supplier volumes for all years (2022-2025).
    Overlay transparent bars for 'demand' column values for 2022-2024.
    For 2025, overlay a transparent 'demand' bar with 'demand: <value>' label.
    Support custom Y-axis range with dynamic tick steps for volume (demand) axis.
    Handle missing supplier columns by plotting only 'demand' bars.
    Support showing percentages instead of values for supplier stack labels.
    """
    required_columns = [customer_column, year_column, 'demand']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if customer_name not in df[customer_column].values:
        available_customers = sorted(df[customer_column].unique())
        raise ValueError(f"Customer '{customer_name}' not found in column '{customer_column}'. "
                        f"Available customers: {available_customers}")
    
    # Check for supplier columns
    available_suppliers = [sup for sup in suppliers if sup in df.columns]
    if not available_suppliers:
        print("Warning: No supplier columns found. Plotting only 'demand' bars.")
    
    customer_df = df[df[customer_column] == customer_name].copy()
    
    if customer_df.empty:
        raise ValueError(f"No data found for customer '{customer_name}' in column '{customer_column}'")
    
    customer_df = customer_df.sort_values(year_column)
    
    def generate_colors(n):
        """Generate n distinct colors"""
        if n <= 10:
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            return base_colors[:n]
        else:
            colormap = plt.cm.get_cmap('tab20')
            return [colormap(i/n) for i in range(n)]
    
    years = sorted(customer_df[year_column].unique())
    n_years = len(years)
    
    if n_years == 0:
        raise ValueError(f"No data found for customer '{customer_name}'")
    
    bar_width = 0.6
    x = np.arange(n_years)
    
    fig, ax1 = plt.subplots(figsize=(12, 9))
    
    max_demand = 0
    
    if available_suppliers:
        # Plot stacked bars for suppliers
        supplier_colors = generate_colors(len(available_suppliers))
        bottom = np.zeros(n_years)
        
        for j, supplier in enumerate(available_suppliers):
            values = np.array([customer_df[customer_df[year_column] == year][supplier].iloc[0] 
                              if year in customer_df[year_column].values and not customer_df[customer_df[year_column] == year].empty
                              else 0 for year in years])
            
            bars = ax1.bar(x, values, bar_width, bottom=bottom, 
                          label=supplier.replace('_', ' ').title(), color=supplier_colors[j])
            
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    label_y = bottom[i] + value / 2
                    year_data = customer_df[customer_df[year_column] == years[i]]
                    total_height = year_data[available_suppliers].sum(axis=1).iloc[0] if not year_data.empty else 0
                    show_label = value > total_height * 0.05
                    
                    if show_label:
                        font_size = value_label_fontsize if value_label_fontsize is not None else max(6, min(10, 80//len(available_suppliers)))
                        
                        # Calculate percentage if requested
                        if show_percentages and total_height > 0:
                            percentage = (value / total_height) * 100
                            label_text = f'{percentage:.1f}%'
                        else:
                            label_text = f'{value:.0f}'
                        
                        ax1.text(bar.get_x() + bar.get_width()/2, label_y, 
                                label_text,
                                ha='center', va='center', 
                                fontsize=font_size,
                                fontweight='bold',
                                color='white' if value > total_height * 0.15 else 'black',
                                bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='black' if value > total_height * 0.15 else 'white',
                                        alpha=0.7, edgecolor='none'))
            
            bottom += values
            max_demand = max(max_demand, bottom.max())
    else:
        # Plot 'demand' bars only
        demand_values = np.array([customer_df[customer_df[year_column] == year]['demand'].iloc[0] 
                                 if year in customer_df[year_column].values and not customer_df[customer_df[year_column] == year].empty
                                 else 0 for year in years])
        
        bars = ax1.bar(x, demand_values, bar_width, 
                      label='Total Demand', color='#1f77b4')
        
        for i, (bar, value) in enumerate(zip(bars, demand_values)):
            if value > 0:
                # For demand-only bars, always show values (not percentages since there's only one bar)
                ax1.text(bar.get_x() + bar.get_width()/2, value / 2, 
                        f'{value:.0f}',
                        ha='center', va='center', 
                        fontsize=value_label_fontsize,
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))
        
        max_demand = max(max_demand, demand_values.max() if demand_values.size > 0 else 0)
    
    # Overlay transparent demand bars for 2022-2024
    demand_values = np.array([customer_df[customer_df[year_column] == year]['demand'].iloc[0] 
                             if year in customer_df[year_column].values and not customer_df[customer_df[year_column] == year].empty
                             else 0 for year in years])
    
    for i, year in enumerate(years):
        if year in [2022, 2023, 2024]:
            demand_value = demand_values[i]
            if pd.notna(demand_value) and demand_value > 0:
                demand_bar = ax1.bar(x[i], demand_value, bar_width, 
                                    facecolor='none', edgecolor='blue', linewidth=0.3, 
                                    label='Demand (2022-2024)' if i == 0 else None)
                ax1.text(x[i], demand_value + (demand_value * 0.05), 
                        f'{demand_value:.0f}',
                        ha='center', va='bottom', fontsize=demand_label_font_size, fontweight='bold',
                        color='blue')
                max_demand = max(max_demand, demand_value)
    
    # Overlay 2025 demand bar
    if 2025 in years:
        idx_2025 = years.index(2025)
        year_data = customer_df[customer_df[year_column] == 2025]
        if not year_data.empty:
            demand_value = year_data['demand'].iloc[0]
            if pd.notna(demand_value):
                demand_bar = ax1.bar(x[idx_2025], demand_value, bar_width, 
                                    facecolor='none', edgecolor='red', linewidth=2, 
                                    label='2025 Demand')
                ax1.text(x[idx_2025], demand_value + (demand_value * 0.05), 
                        f'{demand_value:.0f}',
                        ha='center', va='bottom', fontsize=demand_label_font_size, fontweight='bold',
                        color='red')
                max_demand = max(max_demand, demand_value)
    
    ax1.set_xlabel(year_column.title(), fontsize=axis_label_fontsize)
    ax1.set_ylabel('Demand (mt)', color='black', fontsize=axis_label_fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45 if len(str(years[0])) > 4 else 0, fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
    ax1.grid(True, alpha=0.3)
    
    if y_min is not None and y_max is not None:
        ax1.set_ylim(y_min, y_max)
        tick_step = _calculate_dynamic_tick_step(y_max - y_min)
        y_ticks = np.arange(y_min, y_max + tick_step, tick_step)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
    else:
        max_demand = max(max_demand, 100)
        range_magnitude = max_demand * 1.4  # Aligned with main.py
        ax1.set_ylim(0, range_magnitude)
        tick_step = _calculate_dynamic_tick_step(range_magnitude)
        y_ticks = np.arange(0, range_magnitude + tick_step, tick_step)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
    
    handles, labels = ax1.get_legend_handles_labels()
    ncols = min(6, max(2, len(handles) // 3))
    
    ax1.legend(handles, labels, 
              title='Legend', 
              loc='center right', 
              bbox_to_anchor=(1, 0.9),
              frameon=True, 
              fancybox=True, 
              shadow=True, 
              ncol=ncols, 
              fontsize=legend_fontsize,
              title_fontsize=legend_title_fontsize)
    
    # Update title to indicate percentage mode
    title_suffix = " (% of total demand)" if show_percentages else " total demand (mt)"
    plt.title(f'Demand Analysis for {customer_name}{title_suffix}', 
              fontsize=title_fontsize, fontweight='bold', pad=20)
    
    fig.subplots_adjust(bottom=0.18)
    return fig

def plot_customer_demand_with_price(df, customer_name, customer_column, suppliers, year_column, demand_ylim, price_ylim, price_columns, price_colors, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, legend_title_fontsize, value_label_fontsize, price_annotation_fontsize, annotation_spacing, customer_name_font_size, demand_label_font_size, y_min, y_max, y_demand_min, y_demand_max):
    """
    Plot a combined chart with stacked bars for supplier volumes and line charts for prices.
    Support custom Y-axis range with dynamic tick steps for both demand (ax1) and price (ax2) axes.
    Handle missing supplier columns by plotting only 'demand' bars.
    """
    required_columns = [customer_column, year_column, 'demand']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if customer_name not in df[customer_column].values:
        available_customers = sorted(df[customer_column].unique())
        raise ValueError(f"Customer '{customer_name}' not found in column '{customer_column}'. "
                        f"Available customers: {available_customers}")
    
    # Check for supplier columns
    available_suppliers = [sup for sup in suppliers if sup in df.columns]
    if not available_suppliers:
        print("Warning: No supplier columns found. Plotting only 'demand' bars.")
    
    customer_df = df[df[customer_column] == customer_name].copy()
    
    if customer_df.empty:
        raise ValueError(f"No data found for customer '{customer_name}' in column '{customer_column}'")
    
    customer_df = customer_df.sort_values(year_column)
    
    if price_columns:
        missing_price_cols = [col for col in price_columns if col not in df.columns]
        if missing_price_cols:
            print(f"Warning: Price columns {missing_price_cols} not found in dataframe. Skipping these columns.")
            price_columns = [col for col in price_columns if col in df.columns]
        
        if not price_columns:
            print("Warning: No valid price columns found. Only demand chart will be displayed.")
            price_columns = None
    
    def generate_colors(n):
        """Generate n distinct colors"""
        if n <= 10:
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            return base_colors[:n]
        else:
            colormap = plt.cm.get_cmap('tab20')
            return [colormap(i/n) for i in range(n)]
    
    years = sorted(customer_df[year_column].unique())
    n_years = len(years)
    
    if n_years == 0:
        raise ValueError(f"No data found for customer '{customer_name}'")
    
    bar_width = 0.6
    x = np.arange(n_years)
    
    fig, ax1 = plt.subplots(figsize=(12, 9))
    if price_columns:
        ax2 = ax1.twinx()
    
    max_demand = 0
    
    if available_suppliers:
        # Plot stacked bars for suppliers
        supplier_colors = generate_colors(len(available_suppliers))
        bottom = np.zeros(n_years)
        
        for j, supplier in enumerate(available_suppliers):
            values = np.array([customer_df[customer_df[year_column] == year][supplier].iloc[0] 
                              if year in customer_df[year_column].values and not customer_df[customer_df[year_column] == year].empty
                              else 0 for year in years])
            
            bars = ax1.bar(x, values, bar_width, bottom=bottom, 
                          label=supplier.replace('_', ' ').title(), color=supplier_colors[j])
            
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    label_y = bottom[i] + value / 2
                    year_data = customer_df[customer_df[year_column] == years[i]]
                    total_height = year_data[available_suppliers].sum(axis=1).iloc[0] if not year_data.empty else 0
                    show_label = value > total_height * 0.05
                    
                    if show_label:
                        font_size = value_label_fontsize if value_label_fontsize is not None else max(6, min(10, 80//len(available_suppliers)))
                        
                        ax1.text(bar.get_x() + bar.get_width()/2, label_y, 
                                f'{value:.0f}',
                                ha='center', va='center', 
                                fontsize=font_size,
                                fontweight='bold',
                                color='white' if value > total_height * 0.15 else 'black',
                                bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='black' if value > total_height * 0.15 else 'white',
                                        alpha=0.7, edgecolor='none'))
            
            bottom += values
            max_demand = max(max_demand, bottom.max())
    else:
        # Plot 'demand' bars only
        demand_values = np.array([customer_df[customer_df[year_column] == year]['demand'].iloc[0] 
                                 if year in customer_df[year_column].values and not customer_df[customer_df[year_column] == year].empty
                                 else 0 for year in years])
        
        bars = ax1.bar(x, demand_values, bar_width, 
                      label='Total Demand', color='#1f77b4')
        
        for i, (bar, value) in enumerate(zip(bars, demand_values)):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, value / 2, 
                        f'{value:.0f}',
                        ha='center', va='center', 
                        fontsize=value_label_fontsize,
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))
        
        max_demand = max(max_demand, demand_values.max() if demand_values.size > 0 else 0)
    
    # Overlay transparent demand bars for 2022-2024
    demand_values = np.array([customer_df[customer_df[year_column] == year]['demand'].iloc[0] 
                             if year in customer_df[year_column].values and not customer_df[customer_df[year_column] == year].empty
                             else 0 for year in years])
    
    for i, year in enumerate(years):
        if year in [2022, 2023, 2024]:
            demand_value = demand_values[i]
            if pd.notna(demand_value) and demand_value > 0:
                demand_bar = ax1.bar(x[i], demand_value, bar_width, 
                                    facecolor='none', edgecolor='blue', linewidth=0.3, 
                                    label='Demand (2022-2024)' if i == 0 else None)
                ax1.text(x[i], demand_value + (demand_value * 0.05), 
                        f'{demand_value:.0f}',
                        ha='center', va='bottom', fontsize=demand_label_font_size, fontweight='bold',
                        color='blue')
                max_demand = max(max_demand, demand_value)
    
    # Overlay 2025 demand bar
    if 2025 in years:
        idx_2025 = years.index(2025)
        year_data = customer_df[customer_df[year_column] == 2025]
        if not year_data.empty:
            demand_value = year_data['demand'].iloc[0]
            if pd.notna(demand_value):
                demand_bar = ax1.bar(x[idx_2025], demand_value, bar_width, 
                                    facecolor='none', edgecolor='red', linewidth=2, 
                                    label='2025 Demand')
                ax1.text(x[idx_2025], demand_value + (demand_value * 0.05), 
                        f'{demand_value:.0f}',
                        ha='center', va='bottom', fontsize=demand_label_font_size, fontweight='bold',
                        color='red')
                max_demand = max(max_demand, demand_value)
    
    if price_columns:
        if price_colors is None:
            price_line_colors = ['#FF0000', '#00FF00', '#008000', '#FF00FF', '#00FFFF', 
                                '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A']
        else:
            price_line_colors = price_colors * ((len(price_columns) // len(price_colors)) + 1)
        
        price_line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        price_markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', '+', 'x']
        
        max_price = 0
        for idx, price_col in enumerate(price_columns):
            price_values = np.array([customer_df[customer_df[year_column] == year][price_col].iloc[0] 
                                    if year in customer_df[year_column].values and not customer_df[customer_df[year_column] == year].empty
                                    else np.nan for year in years])
            
            valid_indices = ~np.isnan(price_values)
            valid_years = x[valid_indices]
            valid_prices = price_values[valid_indices]
            
            if len(valid_prices) > 0:
                ax2.plot(valid_years, valid_prices, 
                        color=price_line_colors[idx], 
                        linestyle=price_line_styles[idx], 
                        marker=price_markers[idx], 
                        markersize=8, 
                        linewidth=2, 
                        label=price_col.replace('_', ' ').title())
                
                for i, price in enumerate(price_values):
                    if not np.isnan(price):
                        y_position = price
                        offset = annotation_spacing * (idx % 2 * 2 - 1)
                        ax2.annotate(f'{price:.2f}', 
                                    (x[i], y_position), 
                                    xytext=(0, offset), textcoords='offset points',
                                    ha='center', va='center', fontsize=price_annotation_fontsize,
                                    fontweight='bold', color=price_line_colors[idx],
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                max_price = max(max_price, np.nanmax(price_values))
    
        ax2.set_ylabel('Price ($/kg)', color='black', fontsize=axis_label_fontsize)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
        
        if y_min is not None and y_max is not None:
            ax2.set_ylim(y_min, y_max)
            tick_step = _calculate_dynamic_tick_step(y_max - y_min, True)
            y_ticks = np.arange(y_min, y_max + tick_step, tick_step)
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
        else:
            max_price = max(max_price, 1)
            range_magnitude = max_price * 1.1
            ax2.set_ylim(0, range_magnitude)
            tick_step = _calculate_dynamic_tick_step(range_magnitude, True)
            y_ticks = np.arange(0, range_magnitude + tick_step, tick_step)
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
    
    ax1.set_xlabel(year_column.title(), fontsize=axis_label_fontsize)
    ax1.set_ylabel('Demand (mt)', color='black', fontsize=axis_label_fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45 if len(str(years[0])) > 4 else 0, fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
    ax1.grid(True, alpha=0.3)
    
    if y_demand_min is not None and y_demand_max is not None:
        ax1.set_ylim(y_demand_min, y_demand_max)
        tick_step = _calculate_dynamic_tick_step(y_demand_max - y_demand_min)
        y_ticks = np.arange(y_demand_min, y_demand_max + tick_step, tick_step)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
    else:
        max_demand = max(max_demand, 100)
        range_magnitude = max_demand * 1.4  # Aligned with main.py
        ax1.set_ylim(0, range_magnitude)
        tick_step = _calculate_dynamic_tick_step(range_magnitude)
        y_ticks = np.arange(0, range_magnitude + tick_step, tick_step)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    if price_columns:
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles, all_labels = handles1 + handles2, labels1 + labels2
    else:
        all_handles, all_labels = handles1, labels1
    
    total_items = len(all_handles)
    ncols = 1 if total_items <= 4 else 2 if total_items <= 8 else 3
    
    legend = ax1.legend(all_handles, all_labels, 
                       title='Legend', 
                       loc='upper right',
                       frameon=True, 
                       fancybox=True, 
                       shadow=True, 
                       fontsize=legend_fontsize,
                       title_fontsize=legend_title_fontsize,
                       ncol=ncols,
                       columnspacing=0.8,
                       handletextpad=0.5,
                       borderaxespad=0.5,
                       framealpha=0.95,
                       facecolor='white',
                       edgecolor='gray')
    
    legend.get_frame().set_linewidth(1.2)
    
    plt.title(f'COV sale volumes & Pocket price to {customer_name}', 
              fontsize=title_fontsize, fontweight='bold', pad=20)
    
    return fig

def plot_customer_business_plan(dataframe, customer_name, show_percentages, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, value_label_fontsize):
    """
    Plot business plan with default Y-axis range and dynamic tick steps for volume (value) axis.
    """
    customer_data = dataframe[dataframe['customer'] == customer_name].copy()
    
    if customer_data.empty:
        print(f"No data found for customer: {customer_name}")
        return None
    
    customer_data = customer_data.sort_values('year')
    
    customer_data = customer_data.fillna(0)
    
    years = customer_data['year'].astype(str)
    min_values = customer_data['min']
    base_values = customer_data['base'] 
    max_values = customer_data['max']
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    bar_width = 0.7
    
    bars1 = ax.bar(years, min_values, bar_width, label='Min', color='#009fe4', alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax.bar(years, base_values, bar_width, bottom=min_values, label='Base', color='#00bb7e', alpha=0.8, edgecolor='white', linewidth=1)
    bars3 = ax.bar(years, max_values, bar_width, bottom=min_values + base_values, label='Max', color='#ff7f41', alpha=0.8, edgecolor='white', linewidth=1)
    
    for i, year in enumerate(years):
        min_val = min_values.iloc[i]
        base_val = base_values.iloc[i]
        max_val = max_values.iloc[i]
        total = min_val + base_val + max_val
        
        min_height_for_text = max(total * 0.05, 50)
        
        if min_val > min_height_for_text:
            text = f'{min_val:.0f}' if not show_percentages else f'{(min_val / total * 100):.1f}%' if total > 0 else ''
            ax.text(i, min_val/2, text,
                    ha='center', va='center', fontweight='bold', 
                    color='white', fontsize=value_label_fontsize, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.3))
        
        if base_val > min_height_for_text:
            text = f'{base_val:.0f}' if not show_percentages else f'{(base_val / total * 100):.1f}%' if total > 0 else ''
            ax.text(i, min_val + base_val/2, text,
                    ha='center', va='center', fontweight='bold', 
                    color='white', fontsize=value_label_fontsize,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.3))
        
        if max_val > min_height_for_text:
            text = f'{max_val:.0f}' if not show_percentages else f'{(max_val / total * 100):.1f}%' if total > 0 else ''
            ax.text(i, min_val + base_val + max_val/2, text,
                    ha='center', va='center', fontweight='bold', 
                    color='white', fontsize=value_label_fontsize,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.3))
    
    ax.set_xlabel('Year', fontsize=axis_label_fontsize, fontweight='bold')
    ax.set_ylabel('Value (mt)', fontsize=axis_label_fontsize, fontweight='bold')
    ax.set_title(f'Business plan 2023-2027 of {customer_name}', fontsize=title_fontsize, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=legend_fontsize)
    
    plt.xticks(rotation=0, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    y_max = max([min_values.iloc[i] + base_values.iloc[i] + max_values.iloc[i] 
                for i in range(len(years)) if not pd.isna(min_values.iloc[i] + base_values.iloc[i] + max_values.iloc[i])], default=100)
    range_magnitude = y_max * 1.15
    ax.set_ylim(0, range_magnitude)
    tick_step = _calculate_dynamic_tick_step(range_magnitude)
    y_ticks = np.arange(0, range_magnitude + tick_step, tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
    
    plt.tight_layout()
    return fig

def _calculate_dynamic_tick_step(range_magnitude, is_price=False):
    """
    Calculate dynamic tick step based on range magnitude.
    For volume: Use larger steps for larger ranges (e.g., 100, 500, 1000).
    For price: Use smaller steps (e.g., 0.1, 0.5, 1.0).
    """
    if is_price:
        if range_magnitude <= 2:
            return 0.1
        elif range_magnitude <= 10:
            return 0.5
        elif range_magnitude <= 50:
            return 1.0
        else:
            return 5.0
    else:
        if range_magnitude <= 100:
            return 10
        elif range_magnitude <= 1000:
            return 100
        elif range_magnitude <= 5000:
            return 500
        elif range_magnitude <= 10000:
            return 1000
        else:
            return 2000



#####################################################
def plot_customer_bubble_clean_with_median(df, customer_column, demand_column, price_column, year_filter, bubble_scale, alpha, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max, min_demand_threshold=10):
    """
    Enhanced interactive bubble chart using Plotly with hover information and interactive features.
    Added features:
    - Filter out bubbles with demand below threshold
    - Improved bubble size scaling to reduce extreme size differences
    """
    # Data processing (same as original)
    df_filtered = df[df['year'] == year_filter].copy() if year_filter else df.copy()
    
    customer_data = df_filtered.groupby(customer_column).agg({
        demand_column: 'sum',
        price_column: 'mean'
    }).reset_index().dropna(subset=[demand_column, price_column])
    
    if customer_data.empty:
        raise ValueError("No valid data after filtering and aggregation")
    
    # 1. FILTER OUT SMALL/ZERO VALUES
    # Remove customers with demand <= threshold
    customer_data = customer_data[customer_data[demand_column] > min_demand_threshold].copy()
    
    if customer_data.empty:
        raise ValueError(f"No customers with demand > {min_demand_threshold} after filtering")
    
    print(f"Filtered out customers with demand <= {min_demand_threshold}. Remaining customers: {len(customer_data)}")
    
    avg_price = customer_data[price_column].mean()
    median_price = customer_data[price_column].median()
    customer_data = customer_data.sort_values(price_column, ascending=False)
    
    n_customers = len(customer_data)
    
    # Generate colors (convert to hex for Plotly)
    if n_customers <= 10:
        colors = px.colors.qualitative.Set1[:n_customers]
    elif n_customers <= 20:
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        colors = colors[:n_customers]
    else:
        colors = px.colors.qualitative.Set3 * ((n_customers // len(px.colors.qualitative.Set3)) + 1)
        colors = colors[:n_customers]
    
    # 2. IMPROVED BUBBLE SIZE CALCULATION
    min_demand = customer_data[demand_column].min()
    max_demand = customer_data[demand_column].max()
   
    
    # Use logarithmic scaling to reduce extreme differences
    def calculate_bubble_size_log(demand, min_demand, max_demand, bubble_scale):
        """Calculate bubble size using logarithmic scaling to reduce extreme differences"""
        if demand <= 0:
            return 20 * bubble_scale
        
        # Apply log transformation to reduce extreme differences
        log_demand = np.log10(demand + 1)  # +1 to avoid log(0)
        log_min = np.log10(min_demand + 1)
        log_max = np.log10(max_demand + 1)
        
        if log_max == log_min:
            normalized = 0.5
        else:
            normalized = (log_demand - log_min) / (log_max - log_min)
        
        # Define size range
        min_size = 40 * bubble_scale
        max_size = 180 * bubble_scale  # Reduced max size for better proportion
        
        bubble_size = min_size + (normalized * (max_size - min_size))
        return bubble_size
    
    # Alternative: Square root scaling (less aggressive than log)
    def calculate_bubble_size_sqrt(demand, min_demand, max_demand, bubble_scale):
        """Calculate bubble size using square root scaling"""
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
    
    # Choose scaling method (you can switch between them)
    use_log_scaling = True  # Set to False to use sqrt scaling
    
    bubble_sizes = []
    for _, row in customer_data.iterrows():
        demand = row[demand_column]
        
        if use_log_scaling:
            bubble_size = calculate_bubble_size_log(demand, min_demand, max_demand, bubble_scale)
        else:
            bubble_size = calculate_bubble_size_sqrt(demand, min_demand, max_demand, bubble_scale)
        
        bubble_sizes.append(bubble_size)
    
    # Create the figure
    fig = go.Figure()
    
    # Add bubble scatter plot
    fig.add_trace(go.Scatter(
        x=list(range(n_customers)),
        y=customer_data[price_column],
        mode='markers+text',
        marker=dict(
            size=[size/8 for size in bubble_sizes],  # Adjusted scaling for Plotly
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
    '''''
    # Add average price line
    fig.add_hline(
        y=avg_price,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Average: {avg_price:.2f}",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color="red",
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
        annotation_font_size=12,
        annotation_font_color="green",
        annotation_bgcolor="white",
        annotation_bordercolor="green",
        annotation_borderwidth=1
    )
    '''
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
        bgcolor="white",           # Background color of hover box
        bordercolor="black",       # Border color
        font_size=16,             # Font size (default is usually 12)
        font_family="Arial",      # Font family
        font_color="blue"        # Font color (default is grey)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=900,
        margin=dict(l=80, r=80, t=100, b=80),
        hovermode='closest'
    )
     
    return fig, avg_price, median_price, customer_data
#############################
def plot_customer_bubble_centered(df, customer_column, sow_column, ppd_column, volume_column, year_filter, bubble_scale=1.0, alpha=0.6, 
                                 title_fontsize=20, axis_label_fontsize=16, tick_fontsize=12, customer_name_font_size=12, 
                                 volume_label_font_size=10, min_volume_threshold=10, y_min=None, y_max=None):
    """
    Enhanced interactive bubble chart using Plotly with centered axes (sow=50, ppd=0).
    Features:
    - X-axis: sow < 50 to the left, > 50 to the right, origin at sow=50
    - Y-axis: ppd < 0 below, > 0 above, origin at ppd=0
    - Bubble size: volume with logarithmic or square root scaling
    - Filter out customers with volume below threshold
    - Colors and styling adapted from the provided function
    """
    # Data processing
    df_filtered = df[df['year'] == year_filter].copy() if year_filter else df.copy()
    
    customer_data = df_filtered.groupby(customer_column).agg({
        sow_column: 'mean',
        ppd_column: 'mean',
        volume_column: 'sum'
    }).reset_index().dropna(subset=[sow_column, ppd_column, volume_column])
    
    if customer_data.empty:
        raise ValueError("No valid data after filtering and aggregation")
    
    # Filter out customers with volume <= threshold
    customer_data = customer_data[customer_data[volume_column] > min_volume_threshold].copy()
    
    if customer_data.empty:
        raise ValueError(f"No customers with volume > {min_volume_threshold} after filtering")
    
    print(f"Filtered out customers with volume <= {min_volume_threshold}. Remaining customers: {len(customer_data)}")
    
    # Shift sow to center at 50
    customer_data['sow_shifted'] = customer_data[sow_column] - 50
    
    n_customers = len(customer_data)
    
    # Generate colors (convert to hex for Plotly)
    if n_customers <= 10:
        colors = px.colors.qualitative.Set1[:n_customers]
    elif n_customers <= 20:
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        colors = colors[:n_customers]
    else:
        colors = px.colors.qualitative.Set3 * ((n_customers // len(px.colors.qualitative.Set3)) + 1)
        colors = colors[:n_customers]
    
    # Bubble size calculation (adapted from provided function)
    min_volume = customer_data[volume_column].min()
    max_volume = customer_data[volume_column].max()
    
    def calculate_bubble_size_log(volume, min_volume, max_volume, bubble_scale):
        """Calculate bubble size using logarithmic scaling"""
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
        """Calculate bubble size using square root scaling"""
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
    
    use_log_scaling = True  # Set to False for sqrt scaling
    bubble_sizes = []
    for _, row in customer_data.iterrows():
        volume = row[volume_column]
        bubble_size = calculate_bubble_size_log(volume, min_volume, max_volume, bubble_scale) if use_log_scaling else calculate_bubble_size_sqrt(volume, min_volume, max_volume, bubble_scale)
        bubble_sizes.append(bubble_size / 8)  # Adjust for Plotly scaling
    
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
