import random
import math
import matplotlib.pyplot as plt


def transpose(x):
    return x


def inverse(Z):
    return 1 / Z


def rand():
    return random.gauss(0, 1)


def sqrt(x):
    return math.sqrt(x)

x = 0
xx = 1000

u = 5
B = 1
Q = 100
R = 10000

P = 1
H = 1
F = 1
I = 1
Trace = []

for t in range(100):
    x = F * x + B * u + sqrt(Q) * rand() # state
    z = H * x + sqrt(R) * rand()         # observation

    xp = F * xx + B * u
    P = F * P * transpose(F) + Q

    y = z - H * xp
    S = H * P * transpose(H) + R
    K = P * transpose(H) * inverse(S)
    
    xx = xp + K * y
    P = P * (I - K * H) 
    Trace.append((x, z, xx))

x, z, xx = zip(*Trace)
plt.plot(x, '-b', z, 'o', xx, '-r')
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki.png")
