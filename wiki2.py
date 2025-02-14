import matplotlib.pyplot as plt
import numpy as np
import scipy

dim = 2
dt = 0.1
x = np.array([0, 0], dtype=float)
u = np.array([0, 0], dtype=float)
F = np.array([[1, dt], [0, 1]], dtype=float)
B = np.eye(dim)
Q = np.array([[1 / 4 * dt**4, 1 / 2 * dt**3], [1 / 2 * dt**3, dt**2]],
             dtype=float)
R = 0.1 * np.eye(dim)
H = np.eye(dim)

xx = np.array([0, 0], dtype=float)
P = np.eye(dim)
Trace = []

for t in range(100):
    x = F @ x + B @ u + scipy.linalg.sqrtm(Q) @ np.random.normal(0, 1, dim)
    z = H @ x + scipy.linalg.sqrtm(R) @ np.random.normal(0, 1, dim)

    xp = F @ xx + B @ u
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

# position
plt.plot(x0, "-b", z0, "o", xx0, "-r")
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki2.0.svg")
plt.close()

# velocity
plt.plot(x1, "-b", z1, "o", xx1, "-r")
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki2.1.svg")
