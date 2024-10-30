# utils/database_utils.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
from typing import Optional
from urllib.parse import quote_plus

def get_db_engine():
    """Create and return SQLAlchemy engine."""
    try:
        # Database credentials
        host = "aws-0-ap-south-1.pooler.supabase.com"
        database = "postgres"
        user = "postgres.conrxbcvuogbzfysomov"
        password = "wXAryCC8@iwNvj#"
        port = "6543"
        
        # URL encode the password and user to handle special characters
        encoded_password = quote_plus(password)
        encoded_user = quote_plus(user)
        
        # Construct database URL
        database_url = f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/{database}"
        
        # Create SQLAlchemy engine
        engine = create_engine(database_url)
        return engine
    except Exception as e:
        st.error(f"Error creating database connection: {str(e)}")
        raise

@st.cache_resource
def get_cached_engine():
    """Get or create cached database engine."""
    return get_db_engine()

def fetch_data_from_db(query: str) -> pd.DataFrame:
    """
    Fetch data from database using SQLAlchemy.
    
    Args:
        query (str): SQL query to execute
        
    Returns:
        pd.DataFrame: Query results as a pandas DataFrame
    """
    try:
        engine = get_cached_engine()
        with engine.connect() as connection:
            return pd.read_sql(query, connection)
    except Exception as e:
        st.error(f"Database query error: {str(e)}")
        return pd.DataFrame()

def execute_query(query: str, params: Optional[dict] = None) -> bool:
    """
    Execute a database query (for INSERT, UPDATE, DELETE operations).
    
    Args:
        query (str): SQL query to execute
        params (dict, optional): Query parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        engine = get_cached_engine()
        with engine.connect() as connection:
            if params:
                connection.execute(query, params)
            else:
                connection.execute(query)
            return True
    except Exception as e:
        st.error(f"Database execution error: {str(e)}")
        return False

# Add a test function to verify connection
def test_connection():
    """Test database connection and return status."""
    try:
        engine = get_cached_engine()
        with engine.connect() as conn:
            result = conn.execute("SELECT 1").fetchone()
            return True
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False
