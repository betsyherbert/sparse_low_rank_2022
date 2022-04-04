import numpy as np
import random
from matplotlib.ticker import NullFormatter

def proj_op(u, v):  # projection operator of Gram-Schmidt process
    return (u.dot(v) / u.dot(u)) * u

def orthogonal_vectors(V):
    u1 = V[0]
    u2 = V[1] - proj_op(u1, V[1])
    u3 = V[2] - proj_op(u1, V[2]) - proj_op(u2, V[2])
    return u1, u2, u3

def make_connectivity_vectors(N, sigma, rho):
    vecs = np.random.randn(3, N) 
    x1, x2, x3 = orthogonal_vectors(vecs)    # generate orthogonal vectors, mean 0 and variance 1   
    m = (np.sqrt(sigma ** 2 - rho ** 2) * x1) + (rho * x3)
    n = (np.sqrt(sigma ** 2 - rho ** 2) * x2) + (rho * x3)
    return m, n

def sparsify(J, N, s):
    chi = 1 - np.random.binomial(1, s, (N, N))
    return J * chi

def sparsify_columns(J, N, C):
    chi = np.zeros((N, N))
    for i in range(N):
        indices = random.sample(range(N), C)  # select C connections per column
        chi[indices, i] = 1
    return J * chi

def remove_ticks_and_labels(ax):
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())