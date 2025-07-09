import matplotlib.pyplot
import numpy as np

S = [0,1,2]

Q = [[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]]

pi0 = [0,0,1]

x = np.random.choice(S, p = pi0)
print(x)

X = []

for t in range(1000):
    x = np.random.choice(S, p = Q[x])
    X.append(x)

print(f"Hay {np.sum(X)} unos y {np.abs(np.sum(X)-len(X))} ceros")