# multiple linear regression using

x1 = [60, 62, 67, 70, 71, 72, 75, 78]
x2 = [22, 25, 24, 20, 15, 14, 14, 11]
y = [140, 155, 159, 179, 192, 200, 212, 215]

x1bar = sum(x1) / len(x1)
x2bar = sum(x2) / len(x2)
ybar = sum(y) / len(y)

# normalizing values of x =  x - mean(x)
x1 = [x1[i] - x1bar for i in range(len(x1))]
x2 = [x2[i] - x2bar for i in range(len(x1))]

print(f"{'X1':<10} {'X2':<10} {'Y':<10}")
print("-" * 30)

# zip to conbime lists
for a, b, c in zip(x1, x2, y):
    print(f"{a:<10} {b:<10} {c:<10}")

x1sq = [i**2 for i in x1]
x2sq = [i**2 for i in x2]

x1x2 = [x1[i] * x2[i] for i in range(len(x1))]
x1y = [x1[i] * y[i] for i in range(len(x1))]
x2y = [x2[i] * y[i] for i in range(len(x2))]

den = sum(x1sq) * sum(x2sq) - ((sum(x1x2)) ** 2)
b1 = (sum(x2sq) * sum(x1y) - sum(x1x2) * sum(x2y)) / den
b2 = (sum(x1sq) * sum(x2y) - sum(x1x2) * sum(x1y)) / den

b0 = ybar - b1 * x1bar - b2 * x2bar
print(f" Equation : y = {b0:.2f} + {b1:.2f} X1 + {b2:.2f} X2 ")
