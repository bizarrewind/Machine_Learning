# x = list(map(float, input("Enter x values separated by space: ").split()))
# y = list(map(float, input("Enter y values separated by space: ").split()))
# if len(x) != len(y):
#    print("Error: x and y must have the same number of values.")
#    exit()

import numpy as np

x = [60, 62, 67, 70, 71, 72, 75, 78]
y = [140, 155, 159, 179, 192, 200, 212, 215]

xbar = sum(x) / len(x)
x = [x[i] - xbar for i in range(len(x))]
print(x)

x = np.array(x)
y = np.array(y)

x_mean = np.mean(x)
y_mean = np.mean(y)

b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

a = y_mean - b * x_mean

print("\nRegression Line:")
print(f"y = {a:.2f} + {b:.2f} x")
