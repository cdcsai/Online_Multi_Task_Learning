import numpy as np


def reinitialize_zero_columns(L):
    for i in range(L.shape[1]):
        if np.count_nonzero(L.T[i]) == 0:
            L.T[i] = np.random.randn(L.shape[0])
    return L

def discounted_r(rewards, gamma=0.01 ):
    r = list(map(lambda x: x * gamma, rewards))
    return r

def hess_norm(L, s, alpha, hess):
    return np.dot(np.dot((alpha - np.dot(L, s)).T, hess), alpha - np.dot(L, s))


def loss(L, s, alpha, hess, mu=0.1):
    return mu * np.linalg.norm(s, ord=1) + hess_norm(L, s, alpha, hess)

def random_policy():
    return np.random.randn()


def stringer(string):
    if string[-1] == "y":
        string = string[:-1] + "x"
    else:
        string = string[:-1] + "y"
    return string
