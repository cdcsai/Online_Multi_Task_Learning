import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from numpy.random import randn
import numpy as np


def lipschitz_constant_(L, hess):
    return 2 * np.linalg.norm(L.T @ hess @ L, ord=2)


def minimize_(L, hess, alpha, k, lbda, t=1, ):
    s0 = np.zeros(k)
    n_iter = 1000
    prox_lasso = lambda s, mu : np.sign(s) * np.maximum(np.abs(s) - mu * t, np.zeros(s.size))
    lasso = lambda s, mu : mu * np.linalg.norm(s, ord=1)
    hess_norm = lambda s : (alpha - L @ s).T @ hess @ (alpha - L @ s)
    lipschitz_constant = lipschitz_constant_(L, hess)
    grad_loss = lambda s : -2 * L.T @ hess @ (alpha - L @ s)

    s_fista, objective_fista, = ista_fista(s0=s0,
                                           f=hess_norm,
                                           grad_f=grad_loss,
                                           g=lasso,
                                           prox_g=prox_lasso,
                                           step=1 / lipschitz_constant,
                                           mu=lbda,
                                           n_iter=n_iter)
    return s_fista


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


