import pandas as pd
from utils.database_utils import fetch_data_from_db
from utils.visualization_utils import plot_speed_consumption

def fetch_speed_consumption_data(vessel_name):
    query = f"""
    SELECT vessel_name, report_date, speed, normalised_consumption, loading_condition
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def analyze_speed_consumption(vessel_name):
    speed_data = fetch_speed_consumption_data(vessel_name)
    if speed_data.empty:
        return None, {}
    chart, stats = plot_speed_consumption(vessel_name, speed_data)
    return chart, stats
