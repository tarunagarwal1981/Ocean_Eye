# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import streamlit as st
from datetime import datetime, timedelta

# Function to fetch data from Koyeb PostgreSQL
def fetch_table_data_from_koyeb(vessel_name):
    # Koyeb PostgreSQL connection details
    koyeb_host = "ep-rapid-wind-a1jdywyi.ap-southeast-1.pg.koyeb.app"
    koyeb_database = "koyebdb"
    koyeb_user = "koyeb-adm"
    koyeb_password = "YBK7jd6wLaRD"
    koyeb_port = "5432"
    
    # Establish connection to the database
    conn = psycopg2.connect(
        host=koyeb_host,
        database=koyeb_database,
        user=koyeb_user,
        password=koyeb_password,
        port=koyeb_port
    )
    
    # Define the query to fetch data for the specified vessel
    query = f"""
    SELECT vessel_name, report_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch data
    data = pd.read_sql(query, conn)
    
    # Fetch the latest hull_rough_power_loss_pct_ed value
    latest_value_query = f"""
    SELECT hull_rough_power_loss_pct_ed
    FROM hull_performance_six_months
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    ORDER BY report_date DESC
    LIMIT 1
    """
    latest_value = pd.read_sql(latest_value_query, conn)
    
    # Close the connection
    conn.close()
    
    return data, latest_value

# Function to determine hull condition
def get_hull_condition(value):
    if value > 25:
        return "Poor"
    elif 15 <= value <= 25:
        return "Average"
    else:
        return "Good"

# Function to generate scatter plot for hull roughness power loss with best fit line
def plot_hull_roughness(vessel_name):
    # Fetch data from Koyeb PostgreSQL
    data, latest_value = fetch_table_data_from_koyeb(vessel_name)
    
    # Check if there's data for the vessel
    if data.empty:
        st.error(f"No data available for vessel '{vessel_name}'")
        return None, None
    
    # Convert event_date to datetime format
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    
    # Filter for data where hull_roughness_power_loss is not null
    filtered_data = data[data['hull_roughness_power_loss'].notnull()]
    
    if filtered_data.empty:
        st.error(f"No valid data available for vessel '{vessel_name}'.")
        return None, None
    
    # Extract the necessary columns
    dates = pd.to_datetime(filtered_data['report_date'])
    power_loss = filtered_data['hull_roughness_power_loss']
    
    # Calculate difference in days
    x_numeric = (dates - dates.min()).dt.days
    
    # Plot the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8)
    
    # Add best-fit line (linear regression)
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    # Generate x-values for the best fit line
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    ax.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='#00FF00', linewidth=2, linestyle='-', label='Best Fit Line')
    
    # Set background color
    ax.set_facecolor('#000C20')
    fig.patch.set_facecolor('#000C20')
    
    # Set axis labels and title
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
    
    # Add legend for the best fit line
    ax.legend(loc='upper left', fontsize=10, frameon=False, facecolor='none', edgecolor='none', labelcolor='white')
    
    # Return the figure and latest value for display in Streamlit
    return fig, latest_value

# Streamlit App UI

# Title of the Streamlit App
st.title("Hull Roughness Power Loss Analysis")

# User input for vessel name
vessel_name = st.text_input("Enter the vessel name:")

# Button to trigger the chart generation
if st.button("Generate Chart"):
    if vessel_name:
        chart, latest_value = plot_hull_roughness(vessel_name)
        if chart:
            st.pyplot(chart)
            
            if not latest_value.empty:
                excess_power = latest_value['hull_rough_power_loss_pct_ed'].values[0]
                st.write(f"Excess Power % = {excess_power:.2f}%")
                
                hull_condition = get_hull_condition(excess_power)
                st.write(f"Hull condition = {hull_condition}")
            else:
                st.error("Unable to retrieve the latest Excess Power % value.")
    else:
        st.error("Please enter a vessel name.")
