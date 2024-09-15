# speed_consumption.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Function to fetch data from Koyeb PostgreSQL
def fetch_speed_consumption_data(vessel_name):
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
    
    # Define the query to fetch speed and consumption data for the specified vessel
    query = f"""
    SELECT vessel_name, report_date, speed, normalised_consumption, loading_condition
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch data
    data = pd.read_sql(query, conn)
    
    # Close the connection
    conn.close()
    
    return data

# Function to generate a gradient color based on dates
def get_color_gradient(dates):
    norm = mcolors.Normalize(vmin=dates.min(), vmax=dates.max())
    cmap = cm.get_cmap('plasma')  # You can try 'viridis', 'plasma', 'magma', etc. for neon-like color schemes
    return cmap(norm(dates))

# Function to generate speed consumption chart with time-based color gradient
def plot_speed_consumption(vessel_name):
    # Fetch data from Koyeb PostgreSQL
    data = fetch_speed_consumption_data(vessel_name)
    
    # Check if there's data for the vessel
    if data.empty:
        st.error(f"No data available for vessel '{vessel_name}'")
        return
    
    # Convert report_date to datetime format and filter the last 6 months
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    # Filter for data from the last 6 months
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago) & (data['normalised_consumption'].notnull()) & (data['speed'].notnull())]

    if filtered_data.empty:
        st.error(f"No data available for vessel '{vessel_name}' in the last 6 months.")
        return
    
    # Filter data by loading condition: Laden and Ballast
    laden_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'laden']
    ballast_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'ballast']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Laden condition data with gradient color based on report_date
    if not laden_data.empty:
        color_laden = get_color_gradient(laden_data['report_date'].values)
        ax.scatter(laden_data['speed'], laden_data['normalised_consumption'], c=color_laden, label='Laden', alpha=0.8, edgecolors='white')

    # Plot Ballast condition data with gradient color based on report_date
    if not ballast_data.empty:
        color_ballast = get_color_gradient(ballast_data['report_date'].values)
        ax.scatter(ballast_data['speed'], ballast_data['normalised_consumption'], c=color_ballast, label='Ballast', alpha=0.8, edgecolors='white')

    # Set chart title and labels
    ax.set_title(f"Speed vs ME Consumption - {vessel_name} (Last 6 months)", fontsize=14, color='white')
    ax.set_xlabel('Speed (knots)', fontsize=12, color='white')
    ax.set_ylabel('ME Consumption (mT/d)', fontsize=12, color='white')

    # Set background color and grid
    ax.set_facecolor('#000C20')
    fig.patch.set_facecolor('#000C20')
    
    # Adjust tick parameters and add legend
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(loc='upper left', fontsize=10)

    # Dynamically adjust axis limits using max() and min()
    ax.set_xlim(filtered_data['speed'].min() - 1, filtered_data['speed'].max() + 1)
    ax.set_ylim(filtered_data['normalised_consumption'].min() - 0.05 * filtered_data['normalised_consumption'].ptp(), 
                filtered_data['normalised_consumption'].max() + 0.05 * filtered_data['normalised_consumption'].ptp())

    # Return the figure for display in Streamlit
    return fig

# Streamlit App UI

# Title of the Streamlit App
st.title("Speed vs ME Consumption Analysis")

# User input for vessel name
vessel_name = st.text_input("Enter the vessel name:")

# Button to trigger the chart generation
if st.button("Generate Chart"):
    if vessel_name:
        chart = plot_speed_consumption(vessel_name)
        if chart:
            st.pyplot(chart)
    else:
        st.error("Please enter a vessel name.")
