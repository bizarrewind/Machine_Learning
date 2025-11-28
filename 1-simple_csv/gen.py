import csv
import random

matrix= [[random.randint(0,2000) for _ in range(20)] for _ in range (20)]

with open("data.csv","w",newline = "") as f:
    writer = csv.writer(f)
    writer.writerows(matrix)
   # for i in range (20):
   #     for j in range (20):
   #             f.write(f"{matrix[i][j]},")
   #     f.write("\n")

