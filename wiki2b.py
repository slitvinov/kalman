import matplotlib
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=12345)
dim = 2
x = np.array([0, 0], dtype=float)
F = np.array([[1, 1], [0, 1]], dtype=float)

cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T

cR = np.eye(dim)
R = cR @ cR.T

H = np.eye(dim)

xx = np.array([0, 0], dtype=float)
P = np.eye(dim)
Trace = []

steps = 20
traj = []
for t in range(steps):
    x = F @ x + cQ @ rng.normal(0, 1, dim)
    z = H @ x + cR @ rng.normal(0, 1, dim)
    traj.append((x, z))

for x, z in traj:
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

plt.plot(x0, "-b", z0, "o", xx0, "-r")
plt.legend(["state", "observation", "estimate"])
plt.xlabel("time step")
plt.ylabel("position")
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.savefig("wiki2b.0.png")
plt.close()

plt.plot(x1, "-b", z1, "o", xx1, "-r")
plt.xlabel("time step")
plt.ylabel("velocity")
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki2b.1.png")
