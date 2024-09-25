import pandas as pd
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
        return f"No hull performance data available for {vessel_name}.", None, None
    
    # Process the data to calculate average power loss and hull condition
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    
    if data['hull_roughness_power_loss'].isnull().all():
        return f"No valid hull roughness data available for {vessel_name}.", None, None
    
    # Compute power loss statistics
    avg_power_loss = data['hull_roughness_power_loss'].mean()
    hull_condition = "Good" if avg_power_loss < 15 else ("Average" if avg_power_loss < 25 else "Poor")
    
    # Return analysis summary
    return f"Hull performance for {vessel_name}: Average power loss is {avg_power_loss:.2f}%. Hull condition is {hull_condition}.", avg_power_loss, hull_condition
