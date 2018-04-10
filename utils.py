import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from numpy.random import randn
import numpy as np


def lipschitz_constant_(L, hess):
    return 2 * np.linalg.norm(L.T @ hess @ L, ord=2)


def reinitialize_zero_columns(L):
    for i in range(L.shape[1]):
        if np.count_nonzero(L.T[i]) == 0:
            L.T[i] = np.random.randn(L.shape[0])
    return L


def discounted_r(rewards, gamma=0.01 ):
    r = list(map(lambda x: x * gamma, rewards))
    return r


def hess_norm(L, s, alpha, hess):
    return (alpha - L @ s).T @ hess @ (alpha - L @ s)


def loss(L, s, alpha, hess, mu=0.1):
    l = mu * np.linalg.norm(s, ord=1) + hess_norm(L, s, alpha, hess)
    return l[0][0]


def stringer(string):
    if string[-1] == "y":
        string = string[:-1] + "x"
    else:
        string = string[:-1] + "y"
    return string


def cap_action(action):
    if action < -1:
        action = -1
    elif action > 1:
        action = 1
    return action


