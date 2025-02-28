import matplotlib.pyplot as plt
import numpy as np
import scipy

rng = np.random.default_rng(seed=12345)
dim = 2
dt = 1
x = np.array([0, 0], dtype=float)
F = np.array([[1, dt], [0, 1]], dtype=float)
Q = np.array([[1 / 4 * dt**4, 1 / 2 * dt**3], [1 / 2 * dt**3, dt**2]],
             dtype=float)
R = np.eye(dim)

xx = np.array([0, 0], dtype=float)
P = np.eye(dim)
Trace = []

for t in range(100):
    x = F @ x + scipy.linalg.sqrtm(Q) @ rng.normal(0, 1, dim)
    z = x + scipy.linalg.sqrtm(R) @ rng.normal(0, 1, dim)

    xp = F @ xx
    P = F @ P @ np.transpose(F) + Q
    K = P @ np.linalg.inv(P + R)

    xx = xp + K @ (z - xp)
    P = P @ (np.eye(dim) - K)
    Trace.append((x, z, xx))

x, z, xx = zip(*Trace)
x0, x1 = zip(*x)
z0, z1 = zip(*z)
xx0, xx1 = zip(*xx)

# position, velocity
plt.plot(x0, x1, "-xb", z0, z1, "o", xx0, xx1, "-r")
plt.legend(["state", "observation", "estimate"])
plt.xlabel("position")
plt.ylabel("velocity")
plt.savefig("wiki3.svg")
plt.close()
