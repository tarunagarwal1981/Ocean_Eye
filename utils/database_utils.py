# utils/database_utils.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """Create SQLAlchemy engine using Streamlit secrets."""
    # Construct database URL using secrets
    database_url = (
        f"postgresql://{st.secrets['supabase']['user']}:"
        f"{st.secrets['supabase']['password']}@"
        f"{st.secrets['supabase']['host']}:"
        f"{st.secrets['supabase']['port']}/"
        f"{st.secrets['supabase']['database']}"
    )
    
    # Create engine
    engine = create_engine(database_url)
    
    try:
        # Yield engine connection
        with engine.connect() as conn:
            yield conn
    finally:
        # Dispose engine
        engine.dispose()

def fetch_data_from_db(query: str) -> pd.DataFrame:
    """
    Fetch data from database using SQLAlchemy.
    
    Args:
        query (str): SQL query to execute
        
    Returns:
        pd.DataFrame: Query results as a pandas DataFrame
    """
    try:
        with get_db_connection() as conn:
            data = pd.read_sql(query, conn)
            return data
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return pd.DataFrame()

def test_connection() -> bool:
    """Test database connection."""
    try:
        with get_db_connection() as conn:
            result = pd.read_sql("SELECT 1;", conn)
            return True if not result.empty else False
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False
