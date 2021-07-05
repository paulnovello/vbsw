
from sklearn.mixture import GaussianMixture

import os
import sys

syspath = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.insert(0, syspath)
import autograd.numpy as np
from vbsw_module.functions.df import taylor_w, variance_old

def uniform_sampler(boundaries, n):
    boundaries = np.array(boundaries)
    length = boundaries.shape[1]
    min_x = boundaries[0]
    max_x = boundaries[1]
    X = np.zeros((n, length))
    for i in range(length):
        X[:, i] = np.random.random_sample(n) * (max_x[i] - min_x[i]) + min_x[i]
    return X

def grid_sampler(boundaries, n):
    boundaries = np.array(boundaries)
    length = boundaries.shape[1]
    min_x = boundaries[0]
    max_x = boundaries[1]

    if length == 1:
        X= np.linspace(min_x, max_x, n)
        return X
    elif length == 2:
        n_s = int(np.sqrt(n))
        x0 = np.linspace(min_x[0], max_x[0], n_s)
        x1 = np.linspace(min_x[1], max_x[1], n_s)
        X = np.array(np.meshgrid(x0, x1)).T
        X = np.reshape(X, (n_s*n_s, 2))
        return X
    else:
        print("dim > 2 not yet implemented")


def tbs(boundaries, X, function_list, n, epsilon, ratio, n_comp):
    input_dim = X.shape[1]
    length = X.shape[0]
    d2 = np.zeros((length,))
    if len(function_list) > 1:
        for i in range(len(function_list)):
            d2_add = taylor_w(function_list[i], 2, X, epsilon)
            d2 += np.abs(d2_add)
    else:
        d2 = taylor_w(function_list[0], 2, X, epsilon)
        d2 = np.abs(d2)

    min_d2 = np.max(d2) / ratio
    d2[np.where(d2 < min_d2)] = 0
    d2 /= min_d2
    data_gm = []
    for i in range(length):
        for k in range(int(d2[i])):
            data_gm.append(X[i, :])

    data_gm = np.reshape(data_gm, (len(data_gm), X.shape[1]))
    gm = GaussianMixture(n_comp, n_init=5)
    gm.fit(data_gm)
    samples = gm.sample(n)[0]
    while np.prod([(np.min(samples[:, i]) > boundaries[0][i]) &
                   (np.max(samples[:, i]) < boundaries[1][i])
                   for i in range(input_dim)]) == 0:
        to_sample = np.where(np.sum([(samples[:, i] <= boundaries[0][i]) |
                                       (samples[:, i] >= boundaries[1][i])
                                       for i in range(input_dim)], axis=0) > 0
                               )[0]
        new_samples = gm.sample(to_sample.shape[0])[0]
        samples[to_sample, :] = new_samples

    return samples

def lvbs(boundaries, X, Y, n, n_var, ratio, n_comp):
    input_dim = X.shape[1]
    length = X.shape[0]
    d2 = np.zeros((length,))
    if len(Y.shape) == 2 & Y.shape[1] > 1:
        for i in range(Y.shape[1]):
            d2_add = variance_old(X, Y[:, i], n_var)
            d2 += np.abs(d2_add)
    else:
        d2 = variance_old(X, Y, n_var)
        d2 = np.abs(d2)

    min_d2 = np.max(d2) / ratio
    d2[np.where(d2 < min_d2)] = 0
    d2 /= min_d2
    data_gm = []
    for i in range(length):
        for k in range(int(d2[i])):
            data_gm.append(X[i, :])

    data_gm = np.reshape(data_gm, (len(data_gm), X.shape[1]))
    gm = GaussianMixture(n_comp, n_init=5)
    gm.fit(data_gm)
    samples = gm.sample(n)[0]
    while np.prod([(np.min(samples[:, i]) > boundaries[0][i]) &
                   (np.max(samples[:, i]) < boundaries[1][i])
                   for i in range(input_dim)]) == 0:
        to_sample = np.where(np.sum([(samples[:, i] <= boundaries[0][i]) |
                                       (samples[:, i] >= boundaries[1][i])
                                       for i in range(input_dim)], axis=0) > 0
                               )[0]
        new_samples = gm.sample(to_sample.shape[0])[0]
        samples[to_sample, :] = new_samples

    return samples

