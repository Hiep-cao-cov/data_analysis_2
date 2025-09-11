import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px

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
        
        min_size = 80 * bubble_scale
        max_size = 120 * bubble_scale
        return min_size + (normalized * (max_size - min_size))
    
    use_log_scaling = True  # Set to False for sqrt scaling
    bubble_sizes = []
    for _, row in customer_data.iterrows():
        volume = row[volume_column]
        bubble_size = calculate_bubble_size_log(volume, min_volume, max_volume, bubble_scale) if use_log_scaling else calculate_bubble_size_sqrt(volume, min_volume, max_volume, bubble_scale)
        bubble_sizes.append(bubble_size / 2)  # Adjust for Plotly scaling
    
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
            zerolinecolor='red',
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

# Data
data = {
    'year': [2024] * 18,
    'customer': ['Headway', 'CASARREDO', 'CHIYA', 'ELAN', 'EVERGREEN', 'FAR East', 'HA THANH', 'HANG TAI', 'Inoac', 
                 'JIAYANG', 'JM', 'Nitori', 'TECH FINE', 'XIN HUI', 'XINRUI', 'TAN THANH', 'TAI SHENG', 'KHOI MINH'],
    'sow': [63.0952381, 80, 50.4084658, 77.29540865, 8.369408369, 71.70092539, 82.77310924, 53.46506011, 54.11014896, 
            70.08101852, 61.11111111, 51.72076807, 60.71428571, 29.67820621, 50.27161917, 72.14285714, 42.38095238, 61.55722326],
    'ppd': [-0.026007047, 0.003320846, 7.04501E-05385, -0.025412942, -0.002101393, 0.007664696, 0.027561914, 0.015473556, 
            -0.044648856, 0.003982253, -0.020012017, 0.012628546, 0.023283617, -0.004970064, -0.009283025, -0.002180069, 
            0.008128568, 0.004045556],
    'volume': [540, 100, 1018, 700, 120, 5400, 340, 1380, 700, 320, 60, 680, 240, 300, 120, 160, 100, 820]
}

df = pd.DataFrame(data)

# Plot the chart
fig = plot_customer_bubble_centered(
    df=df,
    customer_column='customer',
    sow_column='sow',
    ppd_column='ppd',
    volume_column='volume',
    year_filter=2024,
    bubble_scale=1.0,
    alpha=0.6,
    title_fontsize=20,
    axis_label_fontsize=16,
    tick_fontsize=12,
    customer_name_font_size=12,
    volume_label_font_size=10,
    min_volume_threshold=50,
    y_min=-0.05,
    y_max=0.05
)

# Show the plot
fig.show()