import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_customer_demand(df, customer_name, customer_column, suppliers, year_column, demand_ylim, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, legend_title_fontsize, value_label_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max):
    """
    Plot a combined chart with stacked bars for supplier volumes for all years (2022-2025).
    Overlay transparent bars for 'demand' column values for 2022-2024.
    For 2025, overlay a transparent 'demand' bar with 'demand: <value>' label.
    Support custom Y-axis range with dynamic tick steps for volume (demand) axis.
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
    
    plt.title(f'Demand Analysis for {customer_name} total demand (mt)', 
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

def plot_customer_bubble_clean_with_median(df, customer_column, demand_column, price_column, year_filter, bubble_scale, alpha, title_fontsize, axis_label_fontsize, tick_fontsize, legend_fontsize, customer_name_font_size, demand_label_font_size, y_min, y_max):
    """
    Enhanced bubble chart with smaller bubble sizes, fixed figure size (12x9 inches).
    Use fixed font sizes for customer names and demand labels.
    """
    df_filtered = df[df['year'] == year_filter].copy() if year_filter else df.copy()
    
    customer_data = df_filtered.groupby(customer_column).agg({
        demand_column: 'sum',
        price_column: 'mean'
    }).reset_index().dropna(subset=[demand_column, price_column])
    
    if customer_data.empty:
        raise ValueError("No valid data after filtering and aggregation")
    
    avg_price = customer_data[price_column].mean()
    median_price = customer_data[price_column].median()
    customer_data = customer_data.sort_values(price_column, ascending=False)
    
    n_customers = len(customer_data)
    if n_customers <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_customers))
    elif n_customers <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_customers))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_customers))
    
    fig, ax = plt.subplots(figsize=(12, 9))
    x_positions = np.arange(n_customers) * 2.5
    
    min_demand = customer_data[demand_column].min()
    max_demand = customer_data[demand_column].max()
    demand_range = max_demand - min_demand
    
    base_size = 50 * bubble_scale  # Reduced from 500
    scale_multiplier = 5  # Reduced from 20
    scale_factor = scale_multiplier * 50  # Reduced from 300
    
    for i, (_, row) in enumerate(customer_data.iterrows()):
        customer_name = row[customer_column]
        price = row[price_column]
        demand = row[demand_column]
        
        if demand_range > 0:
            normalized_demand = (demand - min_demand) / demand_range
            bubble_size = base_size + (normalized_demand * scale_factor * 10)  # Reduced multiplier from 50 to 10
        else:
            bubble_size = base_size + (scale_factor * 5)  # Reduced from 25 to 5
        
        max_bubble_size = min(bubble_size, 5000) if bubble_scale <= 5.0 else bubble_size  # Reduced max size from 15000 to 5000
        max_bubble_size = max(max_bubble_size, base_size)
        
        ax.scatter(x_positions[i], price, 
                  s=max_bubble_size,
                  c=[colors[i]], 
                  alpha=alpha, 
                  edgecolors='black', 
                  linewidth=1)  # Reduced linewidth for smaller bubbles
        
        ax.annotate(customer_name, 
                   (x_positions[i], price), 
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', va='top', 
                   fontsize=customer_name_font_size, fontweight='bold',
                   rotation=45,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           alpha=0.9, edgecolor='gray'))
        
        ax.annotate(f'{demand:.0f}', 
                   (x_positions[i], price), 
                   ha='center', va='center', 
                   fontsize=demand_label_font_size, fontweight='bold', color='black')
    
    _add_reference_lines(ax, avg_price, median_price, max(x_positions))
    
    _customize_plot_with_y_range(ax, price_column, axis_label_fontsize, tick_fontsize, 
                                legend_fontsize, avg_price, median_price, y_min, y_max)
    
    title_text = f'Pocket Prices vs Average & Median {year_filter} (Scale: {bubble_scale:.1f}'
    if y_min is not None and y_max is not None:
        title_text += f', Y-range: {y_min:.1f}-{y_max:.1f}'
    title_text += ')'
    
    if bubble_scale > 10.0:
        title_text += ' - Overlap Mode'
    
    plt.title(title_text, fontsize=title_fontsize, fontweight='bold')
    plt.tight_layout()
    
    return fig, avg_price, median_price, customer_data

def _add_reference_lines(ax, avg_price, median_price, max_x_pos):
    """Add average and median reference lines with offset annotations"""
    ax.axhline(y=avg_price, color='red', linestyle='--', linewidth=3, alpha=0.8)
    ax.axhline(y=median_price, color='green', linestyle='--', linewidth=3, alpha=0.8)
    
    offset = 0.5
    
    if abs(avg_price - median_price) < 0.5:
        avg_y_offset = offset
        median_y_offset = -offset
    else:
        avg_y_offset = 0
        median_y_offset = 0
    
    ax.text(max_x_pos * 0.85, avg_price + avg_y_offset, f'Average: {avg_price:.2f}', 
           fontsize=12, fontweight='bold', color='red',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    alpha=0.9, edgecolor='red'))
    
    ax.text(max_x_pos * 0.85, median_price + median_y_offset, f'Median: {median_price:.2f}', 
           fontsize=12, fontweight='bold', color='green',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    alpha=0.9, edgecolor='green'))

def _customize_plot_with_y_range(ax, price_column, axis_label_fontsize, tick_fontsize, legend_fontsize, avg_price, median_price, y_min, y_max):
    """Customize plot appearance with Y-axis range and dynamic tick steps"""
    ax.set_ylabel(f'{price_column.replace("_", " ").title()} ($/kg)', fontsize=axis_label_fontsize)
    ax.set_xlabel('Customers', fontsize=axis_label_fontsize)
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
        tick_step = _calculate_dynamic_tick_step(y_max - y_min, True)
        y_ticks = np.arange(y_min, y_max + tick_step, tick_step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
    else:
        price_min = min(avg_price, median_price)
        price_max = max(avg_price, median_price)
        range_magnitude = price_max - price_min
        padding = max(0.5, range_magnitude * 0.1)
        ax.set_ylim(max(0, price_min - padding), price_max + padding)
        tick_step = _calculate_dynamic_tick_step(price_max - price_min + 2 * padding, True)
        y_ticks = np.arange(max(0, price_min - padding), price_max + padding + tick_step, tick_step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
    
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, linestyle='--', 
                  label=f'Average: {avg_price:.2f}'),
        plt.Line2D([0], [0], color='green', lw=3, linestyle='--', 
                  label=f'Median: {median_price:.2f}')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize)