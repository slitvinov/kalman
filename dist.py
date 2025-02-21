import numpy as np
import random

random.seed(12345)
n = 6
P = np.ones(n)
Q = np.array([[1 if abs(i - j) <= 2 else 0 for i in range(n)]
              for j in range(n)],
             dtype=float)
R = np.array([[1 if abs(i - j) <= 1 else 0 for i in range(n)]
              for j in range(n)],
             dtype=float)

x, = random.choices(range(n), P)
for i in range(10):
    z, = random.choices(range(n), R[x])
    P = R.T[z] * P
    P /= np.sum(P)
    xp = P @ range(n)
    print(x, z, xp)
    P = Q @ P
    x, = random.choices(range(n), Q[x])
