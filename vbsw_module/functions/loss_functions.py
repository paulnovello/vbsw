import tensorflow as tf
import numpy as np
import scipy
import sklearn


def loss_list(name):
    if name == "mse":
        return mse
    if name == "sup":
        return sup
    if name == "mae":
        return mae
    if name == "binary_crossentropy":
        return bin_cross
    if name == "binary_crossentropy_w":
        return bin_cross_w
    if name == "mse_w":
        return mse_w
    if name == "binary_accuracy":
        return bin_accuracy
    if name == "categorical_accuracy":
        return cat_accuracy
    if name == "categorical_crossentropy":
        return cat_cross
    if name == "categorical_crossentropy_max":
        return cat_cross_max
    if name == "categorical_crossentropy_w":
        return cat_cross_w
    if name == "Accuracy_without_w":
        return acc_w
    if name == "spearman_correlation":
        return spearman_correlation
    if name == "pearson_correlation":
        return pearson_correlation
    if name == "matthews_correlation":
        return matthews_correlation
    if name == "F1":
        return f1

def mse(x, y):
    return tf.keras.losses.MeanSquaredError()(x, y)

def mae(x, y):
    return tf.keras.losses.MeanAbsoluteError()(x, y)

def sup(x, y):
    return np.max((x - y)**2)

def mse_w(x, y):
    weights = x[:, :, -1]
    w = tf.reshape(weights, (weights.shape[0], 1, 1))
    return tf.keras.losses.MeanSquaredError()(x[:, :, :-1], y, sample_weight=w)

def bin_accuracy(x, pred):
    y = pred
    x = x[:, 0, 0]
    y = y[:, 0, 0]
    return tf.keras.metrics.binary_accuracy(x, y)

def cat_accuracy(x, pred):
    y = pred
    return np.mean(tf.keras.metrics.categorical_accuracy(x, y))

def recall(x, pred):
    y = pred
    y = np.round(y)
    if x[0, 0, 0] - int(x[0, 0, 0]) != 0:
        print("warning, reverse x and y to obtain recall")
    if x.shape[2] > 1:
        print("warning, recall only for 1d")
        pass
    else:
        x = x[:,0,0]
        y = y[:,0,0]
        y = y[np.where(x == 1)]
        return np.sum(y)/y.shape[0]

def acc_w(x, y):
    y = np.round(y)
    x = x[:, :, :-1]
    return np.sum(np.abs(x-y))/len(x)

def cat_cross(x, y):
    return tf.keras.losses.CategoricalCrossentropy()(x,y)

def bin_cross(x, y):
    return tf.keras.losses.BinaryCrossentropy()(x,y)

def cat_cross_max(x, y):
    x_t = tf.convert_to_tensor(x)
    y_t = tf.convert_to_tensor(y)
    return float(tf.keras.backend.max(tf.keras.backend.categorical_crossentropy(x_t,y_t)))

def cat_cross_w(x, y):
    weights = x[:, :, -1]
    w = tf.reshape(weights, (weights.shape[0], 1, 1))
    return tf.keras.losses.CategoricalCrossentropy()(x[:,:,:-1], y, sample_weight=w)

def bin_cross_w(x, y):
    weights = x[:, :, -1]
    w = tf.reshape(weights, (weights.shape[0], 1, 1))
    return tf.keras.losses.BinaryCrossentropy()(x[:, :, :-1], y, sample_weight=w)

def spearman_correlation(x, y):
    if x.shape[2] > 1:
        print("warning, spearman correlation only for 1d")
        pass
    else:
        x = x[:,0,0]
        y = y[:,0,0]
        return scipy.stats.spearmanr(x,y)[0]

def pearson_correlation(x, y):
    if x.shape[2] > 1:
        print("warning, pearson correlation only for 1d")
        pass
    else:
        x = x[:,0,0]
        y = y[:,0,0]
        return scipy.stats.pearsonr(x,y)[0]

def f1(x, y):
    if x.shape[2] > 1:
        print("warning, f1 only for 1d")
        pass
    else:
        p = bin_accuracy(x, y)
        r = recall(x, y)
        return 2*p*r / (p + r)

def matthews_correlation(x, y):
    x = x[:,0,0]
    y = np.round(y[:,0,0])
    return sklearn.metrics.matthews_corrcoef(x, y)