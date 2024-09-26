import pandas as pd
from utils.database_utils import fetch_data_from_db
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def plot_speed_consumption(vessel_name, data):
    if data.empty:
        print("Warning: Input data is empty.")
        return None, {}
    
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago)]
    
    if filtered_data.empty:
        print("Warning: Filtered data is empty.")
        return None, {}
    
    laden_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'laden']
    ballast_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'ballast']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    stats = {
        'laden': {},
        'ballast': {},
        'overall': {}
    }
    
    for ax, condition_data, title, condition in [
        (ax1, laden_data, 'Laden Condition', 'laden'),
        (ax2, ballast_data, 'Ballast Condition', 'ballast')
    ]:
        if not condition_data.empty:
            dates = pd.to_datetime(condition_data['report_date'])
            x = condition_data['speed'].values
            y = condition_data['normalised_consumption'].values
            
            scatter = ax.scatter(x, y, c=(dates - dates.min()).dt.days, cmap='viridis', s=50, alpha=0.8)
            
            try:
                # First-order polynomial fit (catch errors)
                if len(x) > 1 and len(y) > 1:
                    coeffs = np.polyfit(x, y, 1)
                    poly = np.poly1d(coeffs)
                    
                    # Calculate R-squared
                    yhat = poly(x)
                    ybar = np.sum(y) / len(y)
                    ssreg = np.sum((yhat - ybar)**2)
                    sstot = np.sum((y - ybar)**2)
                    r_squared = ssreg / sstot
                    
                    # Plot polynomial fit
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_smooth, poly(x_smooth), 'r-', label=f'Polynomial Fit (R² = {r_squared:.3f})')
                    
                    # Collect statistics
                    stats[condition] = {
                        'speed_range': (x.min(), x.max()),
                        'consumption_range': (y.min(), y.max()),
                        'slope': coeffs[0],
                        'r_squared': r_squared
                    }
                else:
                    print(f"Not enough data points for {title}")
                    stats[condition] = {"error": "Insufficient data for fitting"}
                    
            except Exception as e:
                print(f"Error fitting polynomial curve for {title}: {str(e)}")
            
            ax.legend(fontsize=8)
            ax.set_title(title)
            ax.set_xlabel('Speed (knots)')
            ax.set_ylabel('ME Consumption (mT/d)')
            plt.colorbar(scatter, ax=ax, label="Time Progression (days)")
    
    # Overall statistics
    if len(laden_data) > 0 or len(ballast_data) > 0:
        all_speeds = np.concatenate([laden_data['speed'].values, ballast_data['speed'].values])
        all_consumptions = np.concatenate([laden_data['normalised_consumption'].values, ballast_data['normalised_consumption'].values])
        stats['overall'] = {
            'speed_range': (all_speeds.min(), all_speeds.max()),
            'consumption_range': (all_consumptions.min(), all_consumptions.max())
        }
    
    plt.tight_layout()
    fig.suptitle(f"Speed vs Consumption - {vessel_name}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    return fig, stats

def analyze_speed_consumption(vessel_name: str):
    # SQL query to fetch speed consumption data for the vessel
    query = f"""
    SELECT vessel_name, report_date, speed, normalised_consumption, loading_condition
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch data from the database
    data = fetch_data_from_db(query)
    
    # Check if data was fetched successfully
    if data.empty:
        return f"No speed consumption data available for {vessel_name}.", None
    
    # Call the plot_speed_consumption function to generate the chart
    fig, stats = plot_speed_consumption(vessel_name, data)
    
    # Return the analysis summary and the chart
    return f"Speed consumption for {vessel_name} executed.", fig
