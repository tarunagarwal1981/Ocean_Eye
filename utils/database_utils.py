import urllib.parse
from sqlalchemy import create_engine, text
import streamlit as st

def get_db_engine():
    supabase_host = st.secrets["supabase"]["host"]
    supabase_database = st.secrets["supabase"]["database"]
    supabase_user = st.secrets["supabase"]["user"]
    supabase_password = st.secrets["supabase"]["password"]
    supabase_port = st.secrets["supabase"]["port"]

    encoded_password = urllib.parse.quote(supabase_password)
    db_url = f"postgresql+psycopg2://{supabase_user}:{encoded_password}@{supabase_host}:{supabase_port}/{supabase_database}"
    engine = create_engine(db_url)
    return engine

def fetch_hull_performance_data(vessel_name, engine, from_date=None, to_date=None):
    query = """
    SELECT vessel_name, report_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) LIKE :vessel_name
    """
    
    params = {'vessel_name': f'%{vessel_name.upper()}%'}
    
    if from_date and to_date:
        query += " AND report_date BETWEEN :from_date AND :to_date"
        params['from_date'] = from_date
        params['to_date'] = to_date
    
    with engine.connect() as connection:
        result = connection.execute(text(query), params)
        return result.fetchall()

def fetch_speed_consumption_data(vessel_name, engine, from_date=None, to_date=None):
    query = """
    SELECT vessel_name, report_date, speed, normalised_consumption, loading_condition
    FROM hull_performance
    WHERE UPPER(vessel_name) LIKE :vessel_name
    """
    
    params = {'vessel_name': f'%{vessel_name.upper()}%'}
    
    if from_date and to_date:
        query += " AND report_date BETWEEN :from_date AND :to_date"
        params['from_date'] = from_date
        params['to_date'] = to_date
    
    with engine.connect() as connection:
        result = connection.execute(text(query), params)
        return result.fetchall()
