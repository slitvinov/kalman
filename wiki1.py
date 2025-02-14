import math
import matplotlib.pyplot as plt
import numpy as np
import scipy


def transpose(x):
    return x


def inverse(Z):
    return 1 / Z


def rand():
    return np.random.normal(0, 1, 1)


def sqrt(x):
    return scipy.linalg.sqrtm(x)

x = np.array([0], dtype=float)
xx = np.array([1000], dtype=float)

u = np.array([5], dtype=float)
B = np.array([[1]], dtype=float)
Q = np.array([[100]], dtype=float)
R = np.array([[10000]], dtype=float)

P = np.array([[1]], dtype=float)
H = np.array([[1]], dtype=float)
F = np.array([[1]], dtype=float)
I = np.array([[1]], dtype=float)
Trace = []

for t in range(100):
    x = F @ x + B @ u + sqrt(Q) @ rand() # state
    z = H @ x + sqrt(R) @ rand()         # observation

    xp = F @ xx + B @ u
    P = F @ P @ transpose(F) + Q

    y = z - H @ xp
    S = H @ P @ transpose(H) + R
    K = P @ transpose(H) @ inverse(S)
    
    xx = xp + K @ y
    P = P @ (I - K @ H) 
    Trace.append((x, z, xx))

x, z, xx = zip(*Trace)
plt.plot(x, '-b', z, 'o', xx, '-r')
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki1.png")
