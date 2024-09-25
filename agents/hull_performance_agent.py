import pandas as pd
from utils.database_utils import fetch_data_from_db
from utils.visualization_utils import plot_hull_roughness

def fetch_performance_data(vessel_name):
    query = f"""
    SELECT vessel_name, report_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def fetch_six_months_data(vessel_name):
    query = f"""
    SELECT vessel_name, hull_rough_power_loss_pct_ed
    FROM hull_performance_six_months
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def analyze_hull_performance(vessel_name):
    performance_data = fetch_performance_data(vessel_name)
    six_months_data = fetch_six_months_data(vessel_name)
    
    chart = plot_hull_roughness(vessel_name, performance_data)
    
    if not six_months_data.empty:
        power_loss_pct_ed = six_months_data['hull_rough_power_loss_pct_ed'].iloc[-1]
        hull_condition = get_hull_condition(power_loss_pct_ed)
    else:
        power_loss_pct_ed = None
        hull_condition = None
    
    return chart, power_loss_pct_ed, hull_condition

def get_hull_condition(power_loss_pct):
    if power_loss_pct > 25:
        return "Poor"
    elif 15 <= power_loss_pct <= 25:
        return "Average"
    else:
        return "Good"
