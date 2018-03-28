import numpy as np


def reinitialize_zero_columns(L):
    for i in range(L.shape[1]):
        if np.count_nonzero(L.T[i]) == 0:
            L.T[i] = np.random.randn(L.shape[0])
    return L


def discounted_r(rewards, gamma=0.01 ):
    r = list(map(lambda x: x * gamma, rewards))
    return r
