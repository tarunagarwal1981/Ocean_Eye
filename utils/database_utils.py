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
        
        # URL encode the password and username to handle special characters
        encoded_password = quote_plus(db_creds['password'])
        encoded_user = quote_plus(db_creds['user'])
        
        # Construct database URL with encoded credentials
        database_url = (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{db_creds['host']}:{db_creds['port']}"
            f"/{db_creds['database']}"
        )
        
        # Create engine with the encoded URL
        engine = create_engine(database_url)
        
        # Yield connection
        with engine.connect() as conn:
            yield conn
            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        raise
    finally:
        # Dispose engine if it was created
        if 'engine' in locals():
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
            # Execute query and fetch results
            data = pd.read_sql(query, conn)
            
            # Convert date columns if they exist
            date_columns = [col for col in data.columns if 'date' in col.lower()]
            for col in date_columns:
                try:
                    data[col] = pd.to_datetime(data[col], errors='ignore')
                except:
                    pass
                    
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

# Add a debug function to verify connection string
def debug_connection():
    """Debug connection string formation."""
    try:
        db_creds = st.secrets['supabase']
        encoded_password = quote_plus(db_creds['password'])
        encoded_user = quote_plus(db_creds['user'])
        
        # Print connection details (remove in production)
        connection_string = (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{db_creds['host']}:{db_creds['port']}"
            f"/{db_creds['database']}"
        )
        
        st.write("Connection string formed successfully")
        return True
        
    except Exception as e:
        st.error(f"Error forming connection string: {str(e)}")
        return False
