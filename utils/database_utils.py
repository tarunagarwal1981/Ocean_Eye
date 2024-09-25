import pandas as pd
import psycopg2
import os

def fetch_data_from_db(query: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=os.getenv('SUPABASE_HOST'),
        database=os.getenv('SUPABASE_DB'),
        user=os.getenv('SUPABASE_USER'),
        password=os.getenv('SUPABASE_PASSWORD')
    )
    try:
        data = pd.read_sql(query, conn)
    finally:
        conn.close()
    return data
