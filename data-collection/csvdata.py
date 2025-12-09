import csv
import os


file_path = f"{os.getcwd()}/SalesData.csv"
data = []
header = []
try:
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            data.append(row)

except FileNotFoundError:
    print("File Not Found", file_path)
except Exception as e:
    print(e)

print("Header :", header)
print(data[:10])
