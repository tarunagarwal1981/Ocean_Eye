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
    SELECT vessel_name, event_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch data
    data = pd.read_sql(query, conn)
    
    # Close the connection
    conn.close()
    
    return data

# Function to generate scatter plot for hull roughness power loss with best fit line
def plot_hull_roughness(vessel_name):
    # Fetch data from Koyeb PostgreSQL
    data = fetch_table_data_from_koyeb(vessel_name)
    
    # Check if there's data for the vessel
    if data.empty:
        st.error(f"No data available for vessel '{vessel_name}'")
        return
    
    # Convert event_date to datetime format and filter the last 6 months
    data['event_date'] = pd.to_datetime(data['event_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    # Filter for data from the last 6 months
    filtered_data = data[(data['event_date'].dt.date >= six_months_ago) & (data['hull_roughness_power_loss'].notnull())]
    
    if filtered_data.empty:
        st.error(f"No data available for vessel '{vessel_name}' in the last 6 months.")
        return
    
    # Extract the necessary columns
    dates = pd.to_datetime(filtered_data['event_date'])  # Ensure the dates are in datetime format
    power_loss = filtered_data['hull_roughness_power_loss']
    
    # Calculate difference in days
    x_numeric = (dates - dates.min()).dt.days  # Convert dates to numeric (difference in days from the minimum date)
    
    # Plot the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8)  # Reduced size of scatter dots
    
    # Add best-fit line (linear regression)
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    # Generate x-values for the best fit line
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    ax.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='#00FF00', linewidth=2, linestyle='-', label='Best Fit Line')  # Neon green line
    
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
    
    # Return the figure for display in Streamlit
    return fig

# Streamlit App UI

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
