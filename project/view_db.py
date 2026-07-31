import sqlite3
import pandas as pd
import os

DB_FILE = "users.db"

def view_feedback():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query('SELECT * FROM feedback', conn)
        if df.empty:
            print("No feedback found in the database.")
        else:
            print("--- FEEDBACK IN DATABASE ---")
            print(df.to_string())
    finally:
        conn.close()

if __name__ == "__main__":
    view_feedback()