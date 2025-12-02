# import csv file using pandas library
import pandas as pd
import os

file_path = f"{os.getcwd()}/SalesData.csv"

try:
    df = pd.read_csv(file_path)
    print("Data successfully loaded into a Pandas DataFrame:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)

    print("\nMean of SalesValue column:", df["SalesValue"].mean())
    print(df.to_string())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Importing csv from Url
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)
print(data)
