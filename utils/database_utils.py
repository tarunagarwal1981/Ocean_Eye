import os
import psycopg2
import pandas as pd

def fetch_data_from_db(query: str) -> pd.DataFrame:
    # Current code using environment variables or hardcoded values (e.g., os.getenv)
    conn = psycopg2.connect(
        host=os.getenv('SUPABASE_HOST'),
        database=os.getenv('SUPABASE_DB'),
        user=os.getenv('SUPABASE_USER'),
        password=os.getenv('SUPABASE_PASSWORD'),
        port=os.getenv('SUPABASE_PORT')
    )
    try:
        data = pd.read_sql(query, conn)
    finally:
        conn.close()
    return data
