# read data from sql
import pandas as pd
import sqlite3

try:
    conn = sqlite3.connect("students.db")

    data = pd.read_sql("SELECT * FROM students", conn)
    print(data)
    conn.close()
except Exception as e:
    print("Database connection failed  ", e)
