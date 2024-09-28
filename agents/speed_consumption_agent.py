import pandas as pd
from utils.database_utils import fetch_data_from_db
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Function to fetch baseline data
def fetch_baseline_data(vessel_name: str):
    """
    Fetch baseline data for a given vessel from the vessel_performance_model_data table.
    Separate the data into laden and ballast conditions.
    """
    baseline_query = f"""
    SELECT speed_kts, me_consumption_mt, load_type
    FROM vessel_performance_model_data
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch baseline data from the database
    baseline_data = fetch_data_from_db(baseline_query)
    print(f"Fetched baseline data for {vessel_name}: {baseline_data.shape}")  # Debugging output
    
    # Split baseline data into laden and ballast
    laden_baseline = baseline_data[baseline_data['load_type'].str.lower().isin(['scantling', 'design'])]
    ballast_baseline = baseline_data[baseline_data['load_type'].str.lower() == 'ballast']
    
    print(f"Laden baseline data: {laden_baseline.shape}")
    print(f"Ballast baseline data: {ballast_baseline.shape}")
    
    return laden_baseline, ballast_baseline

# Function to fetch operational data
def fetch_ops_data(vessel_name: str):
    """
    Fetch operational (ops) data for the vessel from the vessel_performance_summary table.
    Remove rows where beaufort_scale >= 4 and normalised_me_consumption <= 5.
    Split the data into laden and ballast based on load_type.
    """
    ops_query = f"""
    SELECT observed_speed, beaufort_scale, load_type, reportdate, vessel_name, normalised_me_consumption
    FROM vessel_performance_summary
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch ops data
    ops_data = fetch_data_from_db(ops_query)
    
    # Filter out rows where beaufort_scale >= 4 or normalised_me_consumption <= 5
    ops_data = ops_data[(ops_data['beaufort_scale'] < 4) & (ops_data['normalised_me_consumption'] > 5)]
    print(f"Ops data after filtering: {ops_data.shape}")  # Debugging output
    
    # Ensure that the data types are correct
    ops_data['reportdate'] = pd.to_datetime(ops_data['reportdate'], errors='coerce')
    ops_data['observed_speed'] = pd.to_numeric(ops_data['observed_speed'], errors='coerce')
    ops_data['normalised_me_consumption'] = pd.to_numeric(ops_data['normalised_me_consumption'], errors='coerce')
    
    # Split the data into laden and ballast based on load_type
    laden_ops_data = ops_data[ops_data['load_type'].str.lower() == 'laden']
    ballast_ops_data = ops_data[ops_data['load_type'].str.lower() == 'ballast']
    
    print(f"Laden ops data: {laden_ops_data.shape}")
    print(f"Ballast ops data: {ballast_ops_data.shape}")
    
    return laden_ops_data, ballast_ops_data

# Function to add baseline points to operational data
def add_baseline_points(ops_data, baseline_data, speeds=[8, 10, 14]):
    """
    Add baseline points for speeds 8, 10, and 14 to the ops data.
    """
    added_points = baseline_data[baseline_data['speed_kts'].isin(speeds)]
    print(f"Adding {added_points.shape[0]} baseline points.")  # Debugging output
    if not added_points.empty:
        ops_data = pd.concat([ops_data, added_points[['speed_kts', 'me_consumption_mt']]], ignore_index=True)
        ops_data.rename(columns={'speed_kts': 'observed_speed', 'me_consumption_mt': 'normalised_me_consumption'}, inplace=True)
    return ops_data

# Function to plot speed consumption data
def plot_speed_consumption(vessel_name, laden_ops, ballast_ops, laden_baseline, ballast_baseline):
    """
    Plot both ops data and baseline data with exponential best-fit curves for both.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Helper function to plot data and fit curves
    def plot_data(ax, ops_data, baseline_data, title, ops_color='cyan', baseline_color='red'):
        if not ops_data.empty:
            ops_data = ops_data.dropna(subset=['reportdate', 'observed_speed', 'normalised_me_consumption'])
            print(f"{title} ops data size after dropping NaNs: {ops_data.shape}")  # Debugging output
            if not ops_data.empty:
                dates = pd.to_datetime(ops_data['reportdate'])
                x = ops_data['observed_speed'].values
                y = ops_data['normalised_me_consumption'].values

                # Scatter plot for ops data with gradient color based on time progression
                scatter = ax.scatter(x, y, c=(dates - dates.min()).dt.days, cmap='viridis', s=50, alpha=0.8, label='Ops Data')

                # Exponential fit for ops data
                if len(x) > 1:
                    exp_coeffs = np.polyfit(x, np.log(y), 1)
                    exp_poly = np.poly1d(exp_coeffs)
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_smooth, np.exp(exp_poly(x_smooth)), color=ops_color, linestyle='-', label='Ops Best Fit')

                # Plot baseline data
                if not baseline_data.empty:
                    x_baseline = baseline_data['speed_kts'].values
                    y_baseline = baseline_data['me_consumption_mt'].values
                    ax.scatter(x_baseline, y_baseline, color=baseline_color, s=100, label='Baseline', zorder=5)

                    # Exponential fit for baseline data
                    if len(x_baseline) > 1:
                        exp_coeffs_base = np.polyfit(x_baseline, np.log(y_baseline), 1)
                        exp_poly_base = np.poly1d(exp_coeffs_base)
                        x_smooth_base = np.linspace(x_baseline.min(), x_baseline.max(), 100)
                        ax.plot(x_smooth_base, np.exp(exp_poly_base(x_smooth_base)), color='blue', linestyle='--', label='Baseline Best Fit')

                ax.legend(fontsize=8)
                ax.set_title(title)
                ax.set_xlabel('Speed (knots)')
                ax.set_ylabel('ME Consumption (mT/d)')
                plt.colorbar(scatter, ax=ax, label="Time Progression (days)")
        else:
            print(f"Warning: Operational data is empty for {title}.")

    # Plot laden data and baseline
    plot_data(ax1, laden_ops, laden_baseline, 'Laden Condition')

    # Plot ballast data and baseline
    plot_data(ax2, ballast_ops, ballast_baseline, 'Ballast Condition')

    plt.tight_layout()
    fig.suptitle(f"Speed vs Consumption - {vessel_name}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    return fig

# Main function to analyze speed consumption
def analyze_speed_consumption(vessel_name: str):
    """
    Main function to analyze speed consumption, fetch baseline and ops data,
    and generate a chart with both plotted.
    """
    # Fetch baseline data
    laden_baseline, ballast_baseline = fetch_baseline_data(vessel_name)
    
    # Fetch ops data
    laden_ops_data, ballast_ops_data = fetch_ops_data(vessel_name)

    # Add additional baseline points (speeds 8, 10, 14) to both laden and ballast ops data
    laden_ops_data = add_baseline_points(laden_ops_data, laden_baseline)
    ballast_ops_data = add_baseline_points(ballast_ops_data, ballast_baseline)
    
    # Generate the plot
    fig = plot_speed_consumption(vessel_name, laden_ops_data, ballast_ops_data, laden_baseline, ballast_baseline)

    return f"Speed consumption for {vessel_name} executed.", fig
