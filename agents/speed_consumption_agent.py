import pandas as pd
from utils.database_utils import fetch_data_from_db
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Function to plot speed consumption with filtering and additional data points from baseline
def plot_speed_consumption(vessel_name, data, baseline_data):
    if data.empty:
        print("Warning: Input data is empty.")
        return None, {}
    
    # Apply filters: Normalised consumption > 5 and Beaufort scale < 5
    data = data[(data['normalised_consumption'] > 5) & (data['beaufort_scale'] < 5)]
    
    # Convert report_date to datetime and filter data from the last 6 months
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

    # Plot baseline data on the chart and include additional speed/consumption points
    for ax, condition_data, title, condition, baseline_condition in [
        (ax1, laden_data, 'Laden Condition', 'laden', baseline_data[baseline_data['load_type'].str.lower().isin(['scantling', 'design'])]),
        (ax2, ballast_data, 'Ballast Condition', 'ballast', baseline_data[baseline_data['load_type'].str.lower() == 'ballast'])
    ]:
        if not condition_data.empty:
            dates = pd.to_datetime(condition_data['report_date'])
            x = condition_data['speed'].values
            y = condition_data['normalised_consumption'].values
            
            scatter = ax.scatter(x, y, c=(dates - dates.min()).dt.days, cmap='viridis', s=50, alpha=0.8)

            # Add extra data points from baseline: 8, 10, 14 speed/consumption points
            extra_points = pd.DataFrame({
                'speed': [8, 10, 14],
                'normalised_consumption': [baseline_condition[baseline_condition['speed_kts'] == 8]['me_consumption_mt'].values[0],
                                           baseline_condition[baseline_condition['speed_kts'] == 10]['me_consumption_mt'].values[0],
                                           baseline_condition[baseline_condition['speed_kts'] == 14]['me_consumption_mt'].values[0]]
            })

            x = np.concatenate([x, extra_points['speed'].values])
            y = np.concatenate([y, extra_points['normalised_consumption'].values])

            # Plot baseline data points (in red)
            if not baseline_condition.empty:
                x_baseline = baseline_condition['speed_kts'].values
                y_baseline = baseline_condition['me_consumption_mt'].values
                ax.scatter(x_baseline, y_baseline, color='red', s=100, label='Baseline', zorder=5)  # Red dots for baseline data

                # Fit an exponential curve to the baseline data
                try:
                    exp_coeffs = np.polyfit(x_baseline, np.log(y_baseline), 1)
                    exp_poly = np.poly1d(exp_coeffs)
                    x_smooth_baseline = np.linspace(x_baseline.min(), x_baseline.max(), 100)
                    ax.plot(x_smooth_baseline, np.exp(exp_poly(x_smooth_baseline)), color='blue', linestyle='--', label='Baseline Fit', zorder=6)
                except Exception as e:
                    print(f"Error fitting exponential curve for {title} baseline: {str(e)}")
            
            # Fit an exponential curve to the modified data
            try:
                if len(x) > 1 and len(y) > 1:
                    exp_coeffs = np.polyfit(x, np.log(y), 1)
                    exp_poly = np.poly1d(exp_coeffs)
                    
                    # Calculate R-squared for the exponential fit
                    yhat = np.exp(exp_poly(x))
                    ybar = np.sum(y) / len(y)
                    ssreg = np.sum((yhat - ybar)**2)
                    sstot = np.sum((y - ybar)**2)
                    r_squared = ssreg / sstot
                    
                    # Plot exponential fit for the actual data
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_smooth, np.exp(exp_poly(x_smooth)), 'r-', label=f'Exponential Fit (R² = {r_squared:.3f})')
                    
                    # Collect statistics for the condition
                    stats[condition] = {
                        'speed_range': (x.min(), x.max()),
                        'consumption_range': (y.min(), y.max()),
                        'exp_fit_coeffs': exp_coeffs,
                        'r_squared': r_squared
                    }
                else:
                    print(f"Not enough data points for {title}")
                    stats[condition] = {"error": "Insufficient data for fitting"}
                    
            except Exception as e:
                print(f"Error fitting exponential curve for {title}: {str(e)}")
            
            ax.legend(fontsize=8)
            ax.set_title(title)
            ax.set_xlabel('Speed (knots)')
            ax.set_ylabel('ME Consumption (mT/d)')
            plt.colorbar(scatter, ax=ax, label="Time Progression (days)")
    
    # Overall statistics calculation
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


# Function to analyze the speed consumption for a vessel with added filtering and data points
def analyze_speed_consumption(vessel_name: str):
    # SQL query to fetch speed consumption data for the vessel
    query = f"""
    SELECT vessel_name, report_date, speed, normalised_consumption, loading_condition, beaufort_scale
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # SQL query to fetch baseline data from the vessel_performance_model_data table
    baseline_query = f"""
    SELECT speed_kts, me_consumption_mt, load_type
    FROM vessel_performance_model_data
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch data from the database
    data = fetch_data_from_db(query)
    baseline_data = fetch_data_from_db(baseline_query)
    
    # Check if data was fetched successfully
    if data.empty:
        return f"No speed consumption data available for {vessel_name}.", None
    
    # Call the plot_speed_consumption function to generate the chart with baseline data
    fig, stats = plot_speed_consumption(vessel_name, data, baseline_data)
    
    # Return the analysis summary and the chart
    return f"Speed consumption for {vessel_name} executed.", fig
