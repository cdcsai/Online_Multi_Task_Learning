import numpy as np
from sporco.admm import bpdn


def loss(L, s, alpha, hess, mu):
    t1 = mu * np.linalg.norm(s, ord=1)
    t2 = np.dot(np.dot((alpha - np.dot(L, s)).T, hess), (alpha - np.dot(L, s)))
    return t1 + t2

def policy_learning(k, d, s, alpha, hess, dims, lmbda):
    A = np.zeros((k*d, k*d))
    b = np.zeros((k*d, 1))
    L = np.zeros((d, k))
    T = []

    while True:
        if t not in T:
            (T, R)  = get_random_trajectory()
        else:
            (T, R) = get_trajectories(alpha)
            A = A - np.kron(np.dot(s, s.T), hess)
            temp = np.kron(s.T, np.dot(alpha, hess))
            b = b - temp.reshape(-1, 1)

        L = reinitialize_zero_columns(L)

        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                                 'RelStopTol': 1e-3, 'AutoRho': {'RsdlTarget': 1.0}})

        b = bpdn.BPDN(L, alpha, lmbda, opt)
        b.solve()

        s = lasso_min(loss)

        A = A + np.kron(np.dot(s, s.T), hess)
        temp = np.kron(s.T, np.dot(alpha, hess))
        b = b + temp.reshape(-1, 1)
        L = np.reshape((dims))




def reinitialize_zero_columns(L):
    idx = []
    for i in range(L.shape[1]):
        if np.count_nonzero(L.T[i]) == 0:
            L.T[i] = np.random.randn(L.shape[0])
    return L


def lasso_min():
    pass

def get_random_trajectory():
    pass

def get_trajectories(alpha):
    pass