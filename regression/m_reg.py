import numpy as np

X = np.array([[1, 1, 4], [1, 2, 5], [1, 3, 8], [1, 4, 2]])
Y = np.array([[1], [6], [8], [12]])

xt = X.T

res = np.dot(np.linalg.inv(np.dot(xt, X)), np.dot(xt, Y))

print(f" Equation : y = {res[0][0]:.2f}+{res[1][0]:.2f} X1 + {res[2][0]:.2f} X2")
