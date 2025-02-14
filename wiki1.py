import matplotlib.pyplot as plt
import numpy as np
import scipy

dim = 1
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
    x = F @ x + B @ u + scipy.linalg.sqrtm(Q) @ np.random.normal(0, 1, dim) # state
    z = H @ x + scipy.linalg.sqrtm(R) @ np.random.normal(0, 1, dim)         # observation

    xp = F @ xx + B @ u
    P = F @ P @ np.transpose(F) + Q

    y = z - H @ xp
    S = H @ P @ np.transpose(H) + R
    K = P @ np.transpose(H) @ np.linalg.inv(S)

    xx = xp + K @ y
    P = P @ (np.eye(dim) - K @ H)
    Trace.append((x, z, xx))

x, z, xx = zip(*Trace)
plt.plot(x, "-b", z, "o", xx, "-r")
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki1.png")
