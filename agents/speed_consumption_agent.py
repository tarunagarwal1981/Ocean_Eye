import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from datetime import datetime
from utils.database_utils import fetch_data_from_db
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_baseline_data(vessel_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch baseline data for a given vessel and split into laden and ballast conditions.
    """
    query = f"""
    SELECT speed_kts, me_consumption_mt, load_type
    FROM vessel_performance_model_data
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    baseline_data = fetch_data_from_db(query)
    
    if baseline_data.empty:
        logger.warning(f"No baseline data found for vessel: {vessel_name}")
        return pd.DataFrame(), pd.DataFrame()
    
    laden_baseline = baseline_data[baseline_data['load_type'].isin(['Scantling', 'Design'])]
    ballast_baseline = baseline_data[baseline_data['load_type'] == 'Ballast']
    
    logger.info(f"Fetched baseline data: Laden ({len(laden_baseline)} rows), Ballast ({len(ballast_baseline)} rows)")
    return laden_baseline, ballast_baseline

def fetch_ops_data(vessel_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch operational data for a given vessel, apply filters, and split into laden and ballast conditions.
    """
    query = f"""
    SELECT observed_speed, beaufort_scale, load_type, reportdate, normalised_me_consumption
    FROM vessel_performance_summary
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    ops_data = fetch_data_from_db(query)
    
    if ops_data.empty:
        logger.warning(f"No operational data found for vessel: {vessel_name}")
        return pd.DataFrame(), pd.DataFrame()
    
    logger.info(f"Initial operational data fetched: {len(ops_data)} rows")
    
    # Apply filters
    ops_data = ops_data[
        (ops_data['beaufort_scale'] < 4) & 
        (ops_data['normalised_me_consumption'] > 5)
    ]
    
    logger.info(f"After applying filters: {len(ops_data)} rows")
    
    # Convert reportdate to datetime and handle potential errors
    ops_data['reportdate'] = pd.to_datetime(ops_data['reportdate'], errors='coerce')
    ops_data = ops_data.dropna(subset=['reportdate'])  # Remove rows with invalid dates
    
    logger.info(f"After cleaning dates: {len(ops_data)} rows")
    
    # Split data
    laden_ops = ops_data[ops_data['load_type'] == 'Laden']
    ballast_ops = ops_data[ops_data['load_type'] == 'Ballast']
    
    logger.info(f"Split data: Laden ({len(laden_ops)} rows), Ballast ({len(ballast_ops)} rows)")
    return laden_ops, ballast_ops

def add_baseline_points(ops_data: pd.DataFrame, baseline_data: pd.DataFrame, speeds: List[float] = [8, 10, 14]) -> pd.DataFrame:
    """
    Add baseline points for specified speeds to the operational data.
    """
    if baseline_data.empty:
        logger.warning("No baseline data available to add points")
        return ops_data
    
    baseline_points = baseline_data[baseline_data['speed_kts'].isin(speeds)]
    return pd.concat([ops_data, baseline_points.rename(columns={'speed_kts': 'observed_speed', 'me_consumption_mt': 'normalised_me_consumption'})], ignore_index=True)

def plot_speed_consumption(vessel_name: str, laden_ops: pd.DataFrame, ballast_ops: pd.DataFrame, 
                           laden_baseline: pd.DataFrame, ballast_baseline: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Plot speed consumption data for both laden and ballast conditions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='#000C20')
    fig.suptitle(f'Speed vs Consumption - {vessel_name}', fontsize=16, color='white')
    
    def plot_condition(ax, ops_data, baseline_data, condition):
        ax.set_facecolor('#000C20')
        
        if ops_data.empty:
            ax.text(0.5, 0.5, f"No {condition} data available", ha='center', va='center', color='white')
            logger.warning(f"No {condition} operational data available for plotting")
            return

        # Sort ops_data by date
        ops_data = ops_data.sort_values('reportdate')
        
        # Create a color map based on the date
        min_date = ops_data['reportdate'].min()
        max_date = ops_data['reportdate'].max()
        norm = plt.Normalize(min_date.timestamp(), max_date.timestamp())
        
        # Plot ops data with color gradient
        scatter = ax.scatter(ops_data['observed_speed'], ops_data['normalised_me_consumption'],
                             c=ops_data['reportdate'].apply(lambda x: x.timestamp()), 
                             cmap='plasma', norm=norm, s=50, alpha=0.8)
        
        # Create a custom colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Date', color='white')
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
        cbar.ax.tick_params(colors='white')
        
        # Plot baseline data
        if not baseline_data.empty:
            ax.scatter(baseline_data['speed_kts'], baseline_data['me_consumption_mt'], 
                       color='#FF00FF', s=100, label='Baseline')
        else:
            logger.warning(f"No baseline data available for {condition} condition")
        
        # Fit exponential curves
        def fit_exp(x, y):
            return np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
        
        ops_fit = fit_exp(ops_data['observed_speed'], ops_data['normalised_me_consumption'])
        ax.plot(ops_data['observed_speed'], np.exp(ops_fit[1]) * np.exp(ops_fit[0] * ops_data['observed_speed']), 
                '--', color='#00FFFF', linewidth=2, label='Ops Fit')
        
        if not baseline_data.empty:
            base_fit = fit_exp(baseline_data['speed_kts'], baseline_data['me_consumption_mt'])
            ax.plot(baseline_data['speed_kts'], np.exp(base_fit[1]) * np.exp(base_fit[0] * baseline_data['speed_kts']), 
                    '--', color='#FF00FF', linewidth=2, label='Baseline Fit')
        
        ax.set_title(f'{condition} Condition', color='white')
        ax.set_xlabel('Speed (knots)', color='white')
        ax.set_ylabel('ME Consumption (mt/day)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#000C20', edgecolor='white', labelcolor='white')
        
        # Set grid
        ax.grid(True, linestyle='--', alpha=0.3, color='#FFFFFF')
        
    plot_condition(ax1, laden_ops, laden_baseline, 'Laden')
    plot_condition(ax2, ballast_ops, ballast_baseline, 'Ballast')
    
    plt.tight_layout()
    return fig

def analyze_speed_consumption(vessel_name: str) -> Tuple[str, Optional[plt.Figure]]:
    """
    Analyze speed consumption for a given vessel and generate a plot.
    """
    try:
        laden_baseline, ballast_baseline = fetch_baseline_data(vessel_name)
        laden_ops, ballast_ops = fetch_ops_data(vessel_name)
        
        if laden_baseline.empty and ballast_baseline.empty:
            return f"No baseline data available for {vessel_name}", None
        
        if laden_ops.empty and ballast_ops.empty:
            return f"No operational data available for {vessel_name} after applying filters", None
        
        laden_ops = add_baseline_points(laden_ops, laden_baseline)
        ballast_ops = add_baseline_points(ballast_ops, ballast_baseline)
        
        fig = plot_speed_consumption(vessel_name, laden_ops, ballast_ops, laden_baseline, ballast_baseline)
        
        if fig is None:
            return f"Unable to generate plot for {vessel_name}", None
        
        return f"Speed consumption analysis completed for {vessel_name}", fig
    except Exception as e:
        logger.exception(f"Error in speed consumption analysis for {vessel_name}")
        return f"Error in speed consumption analysis for {vessel_name}: {str(e)}", None

# Example usage (for testing)
if __name__ == "__main__":
    vessel_name = "Example Vessel"
    message, figure = analyze_speed_consumption(vessel_name)
    print(message)
    if figure:
        plt.show()
    else:
        print("No figure generated.")
