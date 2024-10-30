# utils/database_utils.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from contextlib import contextmanager
from urllib.parse import quote_plus

@contextmanager
def get_db_connection():
    """Create SQLAlchemy engine using Streamlit secrets with proper URL encoding."""
    try:
        # Get credentials from secrets
        db_creds = st.secrets['supabase']
        
        # URL encode the password and username
        encoded_password = quote_plus(db_creds['password'])
        encoded_user = quote_plus(db_creds['user'])
        
        # Construct database URL
        database_url = (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{db_creds['host']}:{db_creds['port']}"
            f"/{db_creds['database']}"
        )
        
        # Create engine
        engine = create_engine(database_url)
        
        # Yield connection
        with engine.connect() as conn:
            yield conn
            
    finally:
        # Dispose engine if it was created
        if 'engine' in locals():
            engine.dispose()

def convert_to_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Safely convert a column to datetime format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to convert
        
    Returns:
        pd.Series: Converted datetime series or original series if conversion fails
    """
    try:
        return pd.to_datetime(df[column])
    except (ValueError, TypeError):
        return df[column]

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
            # Execute query
            data = pd.read_sql(query, conn)
            
            # Convert date columns if they exist
            date_columns = [
                col for col in data.columns 
                if any(date_term in col.lower() 
                      for date_term in ['date', 'time', 'timestamp'])
            ]
            
            # Convert each date column safely
            for col in date_columns:
                data[col] = convert_to_datetime(data, col)
                    
            return data
            
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        st.error(f"Query that failed: {query}")
        return pd.DataFrame()

def test_connection() -> bool:
    """Test database connection."""
    try:
        with get_db_connection() as conn:
            result = pd.read_sql("SELECT 1;", conn)
            if not result.empty:
                st.success("Database connection successful")
                return True
            return False
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False

# Optional: Add this function if you need to execute non-SELECT queries
def execute_query(query: str) -> bool:
    """
    Execute a database query (for INSERT, UPDATE, DELETE operations).
    
    Args:
        query (str): SQL query to execute
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            conn.execute(query)
            return True
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return False
