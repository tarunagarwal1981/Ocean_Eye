import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_hull_roughness(vessel_name, data):
    if data.empty:
        return None
    
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = pd.Timestamp.now().date()
    six_months_ago = today - pd.Timedelta(days=180)
    
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago) & (data['hull_roughness_power_loss'].notnull())]
    
    if filtered_data.empty:
        return None
    
    dates = pd.to_datetime(filtered_data['report_date'])
    power_loss = filtered_data['hull_roughness_power_loss']
    
    x_numeric = (dates - dates.min()).dt.days
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8)
    
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    ax.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='lime', linewidth=2)
    
    ax.set_xlabel('Dates')
    ax.set_ylabel('Excess Power %')
    ax.set_title(f'Hull Roughness Power Loss - {vessel_name}')
    
    return fig

def plot_speed_consumption(vessel_name, data):
    if data.empty:
        return None, {}
    
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    six_months_ago = pd.Timestamp.now().date() - pd.Timedelta(days=180)
    
    filtered_data = data[data['report_date'].dt.date >= six_months_ago]
    
    if filtered_data.empty:
        return None, {}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(filtered_data['speed'], filtered_data['normalised_consumption'], alpha=0.8)
    
    ax.set_xlabel('Speed (knots)')
    ax.set_ylabel('Consumption (mT/d)')
    ax.set_title(f'Speed vs Consumption - {vessel_name}')
    
    return fig, {}
