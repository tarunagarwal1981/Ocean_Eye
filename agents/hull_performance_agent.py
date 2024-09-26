import pandas as pd
import matplotlib.pyplot as plt
from utils.database_utils import fetch_data_from_db

def analyze_hull_performance(vessel_name: str):
    # SQL query to fetch hull performance data for the vessel
    query = f"""
    SELECT vessel_name, report_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch data from the database
    data = fetch_data_from_db(query)
    
    # Check if data was fetched successfully
    if data.empty:
        return f"No hull performance data available for {vessel_name}.", None, None, None
    
    # Convert report_date to datetime
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    
    if data['hull_roughness_power_loss'].isnull().all():
        return f"No valid hull roughness data available for {vessel_name}.", None, None, None
    
    # Plotting the hull roughness power loss over time
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['report_date'], data['hull_roughness_power_loss'], color='blue', marker='o', linestyle='-')
        ax.set_title(f'Hull Roughness Power Loss for {vessel_name}', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Hull Roughness Power Loss (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
    except Exception as e:
        return f"Error generating chart for {vessel_name}: {str(e)}", None, None, None
    
    # Compute power loss statistics
    avg_power_loss = data['hull_roughness_power_loss'].mean()
    hull_condition = "Good" if avg_power_loss < 15 else ("Average" if avg_power_loss < 25 else "Poor")
    
    # Always return 4 values (analysis summary, average power loss, hull condition, chart)
    return f"Hull performance for {vessel_name}: Average power loss is {avg_power_loss:.2f}%. Hull condition is {hull_condition}.", avg_power_loss, hull_condition, fig
