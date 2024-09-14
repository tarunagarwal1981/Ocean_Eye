import psycopg2
import pandas as pd

def get_db_connection():
    return psycopg2.connect(
        host="ep-rapid-wind-a1jdywyi.ap-southeast-1.pg.koyeb.app",
        database="koyebdb",
        user="koyeb-adm",
        password="YBK7jd6wLaRD",
        port="5432"
    )

def fetch_data_from_db(query):
    conn = get_db_connection()
    try:
        data = pd.read_sql(query, conn)
    finally:
        conn.close()
    return data
