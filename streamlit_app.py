%pip install tabulate 
%pip install python-docx 
%pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch data from Databricks using Spark SQL
def fetch_table_data_as_pandas(sql_query):
    # Mocking the actual database connection in Databricks (as this is running on Streamlit)
    # In production, replace this with the actual spark.sql query on Databricks.
    # Example:
    # return spark.sql(sql_query).toPandas()

    # Sample dataframe simulating the output of the database fetch
    data = {
        'vessel_name': ['VESSEL_A', 'VESSEL_A', 'VESSEL_A', 'VESSEL_B', 'VESSEL_B'],
        'report_date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-01-15', '2024-03-15']),
        'hull_roughness_power_loss': [0.5, 1.2, 1.7, 0.3, 1.1]
    }
    df = pd.DataFrame(data)
    return df

# Function to generate the plot for hull roughness power loss
def plot_hull_roughness(vessel_name):
    # Convert vessel name to upper case to ensure case insensitivity
    vessel_name = vessel_name.upper()
    
    # Define the SQL query (for actual use on Databricks, fetch data from the actual table)
    data_query = f"SELECT * FROM reporting_layer.digital_desk.hull_performance WHERE vessel_name = '{vessel_name}'"
    
    # Fetch the data from the table (mocked for Streamlit here)
    data = fetch_table_data_as_pandas(data_query)
    
    # Filter data for the specific vessel
    data = data[data['vessel_name'] == vessel_name]

    if data.empty:
        st.error(f"No data available for vessel '{vessel_name}'. Please check the vessel name.")
        return
    
    # Extract the necessary columns
    dates = pd.to_datetime(data['report_date'])  # Ensure the dates are in datetime format
    power_loss = data['hull_roughness_power_loss']
    
    # Calculate difference in days (for trendline)
    x_numeric = (dates - dates.min()).dt.days
    
    # Plot the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8, label='Hull Roughness Power Loss')  # Reduced size of scatter dots
    
    # Add best-fit line (linear regression)
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    # Generate x-values for the best-fit line
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    ax.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='#00FF00', linewidth=2, linestyle='-', label='Best Fit Line')  # Neon green line
    
    # Set background color and labels
    ax.set_facecolor('#000C20')
    fig.patch.set_facecolor('#000C20')
    ax.set_xlabel('Dates', fontsize=12, color='white')
    ax.set_ylabel('Excess Power %', fontsize=12, color='white')
    ax.set_title(f'Hull Roughness Power Loss - {vessel_name}', fontsize=14, color='white')
    
    # Format x-axis to show only months
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    
    # Adjust tick parameters
    plt.xticks(color='white', fontsize=10)
    plt.yticks(color='white', fontsize=10)
    
    # Dynamically adjust axis limits using max() and min()
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(power_loss.min() - 0.05 * (power_loss.max() - power_loss.min()), power_loss.max() + 0.05 * (power_loss.max() - power_loss.min()))
    
    # Add legend for the best-fit line
    ax.legend(loc='upper left', fontsize=10, frameon=False, facecolor='none', edgecolor='none', labelcolor='white')
    
    # Return the figure to be displayed in Streamlit
    return fig

# Streamlit app UI

# Title of the Streamlit App
st.title("Hull Roughness Power Loss Analysis")

# User input for vessel name
vessel_name = st.text_input("Enter the vessel name:")

# Button to trigger the chart generation
if st.button("Generate Chart"):
    if vessel_name:
        chart = plot_hull_roughness(vessel_name)
        if chart:
            st.pyplot(chart)
    else:
        st.error("Please enter a vessel name.")
