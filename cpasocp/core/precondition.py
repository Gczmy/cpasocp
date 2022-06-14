import numpy as np


def precondition(L, scaling_factor):
    tau = [None] * L.shape[1]
    sigma = [None] * L.shape[0]
    q = np.count_nonzero(L, axis=1)
    for j in range(L.shape[1]):
        tau[j] = 1 / (scaling_factor * np.linalg.norm(L[:, j]) ** 2)
    for i in range(L.shape[0]):
        sigma[i] = scaling_factor / q[i]
    T = np.diagflat(tau)
    Sigma = np.diagflat(sigma)
    print(T)
    print(Sigma)
    return T, Sigma
