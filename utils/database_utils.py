# utils/database_utils.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
from typing import Optional
from urllib.parse import quote_plus

def get_db_engine():
    """Create and return SQLAlchemy engine using Supabase credentials from Streamlit secrets."""
    try:
        # Get database credentials from Streamlit secrets
        db_credentials = st.secrets["supabase"]
        
        # URL encode the password to handle special characters
        encoded_password = quote_plus(db_credentials['password'])
        encoded_user = quote_plus(db_credentials['user'])
        
        # Construct database URL for Supabase with encoded credentials
        database_url = (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['database']}"
        )
        
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
    Fetch data from Supabase using SQLAlchemy.
    
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
