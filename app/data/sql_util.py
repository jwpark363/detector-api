import sqlite3
import pandas as pd
def read_sql(sql):
    conn = sqlite3.connect('./app/data/app.db')
    df = pd.read_sql(sql, conn)
    conn.close()
    return df