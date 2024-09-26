import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.database_utils import fetch_data_from_db

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
    
    # Plotting speed vs. normalized consumption
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of speed vs. consumption
    ax.scatter(data['speed'], data['normalised_consumption'], color='green', alpha=0.6)
    
    # Add a trend line
    coeffs = np.polyfit(data['speed'], data['normalised_consumption'], 1)
    poly = np.poly1d(coeffs)
    x = np.linspace(min(data['speed']), max(data['speed']), 100)
    ax.plot(x, poly(x), color='red', linestyle='--', label='Trend Line')
    
    # Set plot labels and title
    ax.set_title(f'Speed vs. Normalized Consumption for {vessel_name}', fontsize=14)
    ax.set_xlabel('Speed (knots)', fontsize=12)
    ax.set_ylabel('Normalized Consumption (MT/day)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    
    # Return analysis summary and the figure
    return f"Speed consumption for {vessel_name} executed.", fig
