%pip install tabulate %pip install python-docx 
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch data from Databricks using Spark SQL
def fetch_table_data_as_pandas(sql_query):
    return spark.sql(sql_query).toPandas()

# Function to generate scatter plot for hull roughness power loss with best fit line
def plot_hull_roughness(vessel_name):
    # Convert vessel name to upper case to ensure case insensitivity
    vessel_name = vessel_name.upper()
    
    # Define SQL queries for tables
    data_query = f"SELECT * FROM reporting_layer.digital_desk.hull_performance WHERE vessel_name = '{vessel_name}'"
    
    # Fetch data from the table
    data = fetch_table_data_as_pandas(data_query)
    
    # Convert report_date to datetime format, ensuring correct parsing of the format with timezone information
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    
    # Ensure we are working with date only (ignoring time)
    data['report_date'] = data['report_date'].dt.date
    
    # Get today's date and calculate 6 months ago
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    # Filter for data from the last 6 months and non-null hull_roughness_power_loss
    filtered_data = data[(data['report_date'] >= six_months_ago) & (data['hull_roughness_power_loss'].notnull())]
    
    if filtered_data.empty:
        print(f"No data available for vessel '{vessel_name}' in the last 6 months.")
        return
    
    # Extract the necessary columns
    dates = pd.to_datetime(filtered_data['report_date'])  # Ensure the dates are in datetime format
    power_loss = filtered_data['hull_roughness_power_loss']
    
    # Calculate difference in days
    x_numeric = (dates - dates.min()).dt.days  # Convert dates to numeric (difference in days from the minimum date)
    
    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8)  # Reduced size of scatter dots
    
    # Add best-fit line (linear regression)
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    # Generate x-values for the best fit line
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    plt.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='#00FF00', linewidth=2, linestyle='-', label='Best Fit Line')  # Neon green line
    
    # Set background color (as in the previous code)
    plt.gca().set_facecolor('#000C20')
    plt.gcf().patch.set_facecolor('#000C20')
    
    # Set axis labels and title
    plt.xlabel('Dates', fontsize=12, color='white')
    plt.ylabel('Excess Power %', fontsize=12, color='white')
    plt.title(f'Hull Roughness Power Loss - {vessel_name}', fontsize=14, color='white')
    
    # Format x-axis to show only months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    
    # Adjust tick parameters
    plt.xticks(color='white', fontsize=10)
    plt.yticks(color='white', fontsize=10)
    
    # Dynamically adjust axis limits using max() and min()
    plt.xlim(dates.min(), dates.max())
    plt.ylim(power_loss.min() - 0.05 * (power_loss.max() - power_loss.min()), power_loss.max() + 0.05 * (power_loss.max() - power_loss.min()))
    
    # Add legend for the best fit line
    plt.legend(loc='upper left', fontsize=10, frameon=False, facecolor='none', edgecolor='none', labelcolor='white')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

    # Print the range of the data used for plotting
   # print(f"Data used for plotting: {len(filtered_data)} data points.")
   # print(f"Date range: {dates.min()} to {dates.max()}")
   # print(f"Power Loss range: {power_loss.min()} to {power_loss.max()}")

# Prompt for user input
vessel_name = input("Enter the vessel name you want to analyze: ").strip()

# Plot hull roughness power loss for the entered vessel name
plot_hull_roughness(vessel_name)
