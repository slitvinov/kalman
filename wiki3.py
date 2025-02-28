import matplotlib.pyplot as plt
import numpy as np
import scipy

dim = 2
dt = 1
x = np.array([0, 0], dtype=float)
F = np.array([[1, dt], [0, 1]], dtype=float)
Q = np.array([[1 / 4 * dt**4, 1 / 2 * dt**3], [1 / 2 * dt**3, dt**2]],
             dtype=float)
R = np.eye(dim)
H = np.eye(dim)

xx = np.array([0, 0], dtype=float)
P = np.eye(dim)
Trace = []

for t in range(100):
    x = F @ x + scipy.linalg.sqrtm(Q) @ np.random.normal(0, 1, dim)
    z = H @ x + scipy.linalg.sqrtm(R) @ np.random.normal(0, 1, dim)

    xp = F @ xx
    P = F @ P @ np.transpose(F) + Q
    y = z - H @ xp
    S = H @ P @ np.transpose(H) + R
    K = P @ np.transpose(H) @ np.linalg.inv(S)

    xx = xp + K @ y
    P = P @ (np.eye(dim) - K @ H)
    Trace.append((x, z, xx))

x, z, xx = zip(*Trace)
x0, x1 = zip(*x)
z0, z1 = zip(*z)
xx0, xx1 = zip(*xx)

# position, velocity
plt.plot(x0, x1, "-b", z0, z1, "o", xx0, xx1, "-r")
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki3.svg")
plt.close()
