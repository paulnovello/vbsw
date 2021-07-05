import autograd.numpy as np
from sklearn.neighbors import KDTree
from autograd import jacobian
from scipy.special import gamma
import time

import sys
import os
import pickle as pkl

syspath = os.path.dirname(os.path.realpath(__file__)) + '/../..'
sys.path.insert(0, syspath)

def dataset_weighting(x, y, ratio, N_stat, case_name):

    if (case_name is not None) & (case_name not in ['cifar10', 'mnist']):
        try:
            with open(syspath + "/data/w/w_" + str(N_stat) + "_" + case_name, "rb") as f:
                w = pkl.load(f)
        except FileNotFoundError:
            start = time.time()
            w = df(np.array(x), np.array(y), N_stat, normalization=False)
            print("variance took: " + str(time.time() - start))

            with open(syspath + "/data/w/w_" + str(N_stat) + "_" + case_name, "wb") as f:
                pkl.dump(w, f)
    else:
        w = df(np.array(x), np.array(y), N_stat, normalization=False)

    m = np.max(w)
    w = w / m * (ratio - 1)
    w += 1
    y_w = np.c_[y, w]
    dataset = (x, y_w)
    return dataset


def df(x_train, y_train, N_stat, normalization=False):
    tree = KDTree(x_train, leaf_size = 2)
    variance= []
    densities = []
    for k in range(x_train.shape[0]):
        to_var_x = tree.query(np.reshape(x_train[k], (1,-1)), N_stat, return_distance=True)
        to_var_y = y_train[to_var_x[1][0]]
        if normalization:
            dist = np.max(to_var_x[0][0])
            dim = x_train.shape[1]
            v_ball = np.pi**(dim/2) / gamma(dim/2 + 1) * dist**dim
            dens = N_stat / (x_train.shape[0]*v_ball)
            variance.append(np.sum(np.var(to_var_y, axis=0)))
            densities.append(dens)
        else:
            variance.append(np.sum(np.var(to_var_y, axis=0)))
    variance = np.array(variance)
    if normalization:
        densities = np.array(densities)
        return np.reshape(variance, (x_train.shape[0])), np.reshape(densities, (x_train.shape[0]))
    else:
        return np.reshape(variance, (x_train.shape[0]))



def taylor_w(fun, n, X, epsilon):
    # fun must be R^n -> R
    weights = []
    for x in X:
        weight=0
        for i in range(1,n+1):
            to_grad = fun
            for k in range(i):
                to_grad = jacobian(to_grad)
            weight += epsilon**i*1/fact(i)*np.sum(np.abs(to_grad(x)))
        weights.append(weight)
    return np.reshape(weights, (X.shape[0]))


def fact(n):
    if (n == 1) |(n==0):
        return 1
    else:
        return n*fact(n-1)

def variance_old(x_train, y_train, N_stat):
    tree = KDTree(x_train, leaf_size = 2)
    variance= []
    for k in range(x_train.shape[0]):
        to_var_x = tree.query(np.reshape(x_train[k], (1,-1)), N_stat, return_distance = False)[0]
        to_var_y = y_train[to_var_x]
        variance.append(np.sum(np.var(to_var_y, axis=0)))
    variance = np.array(variance)
    return np.reshape(variance, (x_train.shape[0]))