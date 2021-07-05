import autograd.numpy as np


def fun_list(name):
    if name == "gauss":
        return gauss
    elif name == "gauss_sinc":
        return gauss_sinc
    elif name == "strange":
        return strange
    elif name == "tanh":
        return tanh
    elif name == "runge":
        return runge
    elif name == "mich":
        return mich
    elif name == "u":
        return u
    elif name == "eta":
        return eta
    elif name == "Ueta":
        return u_eta
    elif name == "tau":
        return tau

def gauss(x, params=[0, 1]):
    mu = params[0]
    sigma = params[1]
    return 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gauss_sinc(x, params=None):
    epsilon = 0.025
    return 0.5 * gauss(x, 0.5, 0.2) + 2 * np.sin(100 * (x + epsilon)) / (100 * (x + epsilon))


def strange(x, params=None):
    return np.abs(-0.6 + gauss(x, 0.5, 0.2)
                  - 1 / 80 * gauss(x, 0.2, 0.01)
                  + 1 / 30 * gauss(x, 0.13, 0.01)
                  + 1 / 40 * gauss(x, 0.1, 0.01)
                  + 1 / 50 * gauss(x, 0.23, 0.01))


def tanh(x, params=[0.5, 10]):
    shift = params[0]
    coeff = params[1]
    return np.tanh((x - shift) * coeff) + 1


def runge(x, params=[0.5, 25]):
    shift = params[0]
    coeff = params[1]
    return 1 / (1 + coeff * (x - shift) ** 2)


def mich(x, params=[5]):

    m = params[0]
    res = 0
    if (np.array(x).shape[0] == 1) | (len(x.shape) == 1):
        n = x.shape[0]
        for i in range(1, n + 1):
            ind = i + 1
            res = res + np.sin(x[i - 1]) * np.sin(ind * x[i - 1] ** 2 / np.pi) ** (2 * m)
    else:
        n = x.shape[1]
        for i in range(1, n + 1):
            ind = i + 1
            res = res + np.sin(x[:, i - 1]) * np.sin(ind * x[:, i - 1] ** 2 / np.pi) ** (2 * m)
        res = np.reshape(res, (x.shape[0], 1))
    return res


def u(x, params=[-0.45, -0.45, 1]):
    if len(np.array(x).shape) == 1:
        U_0 = x[0]
        q_0 = x[1]
        t = x[2]
    else:
        U_0 = x[:, 0]
        q_0 = x[:, 1]
        t = x[:, 2]
    sig_a = params[0]
    sig_r = params[1]
    v = params[2]
    u = (sig_r * U_0 + q_0 * sig_a) * U_0 / (sig_r * U_0 + q_0 * sig_a * np.exp(v * t * (sig_r * U_0 + q_0 * sig_a)))
    if len(np.array([u]).shape) == 1:
        if (U_0 == 0):
            return 0
        elif (q_0 == 0):
            return U_0
        else:
            return u
    else:
        u[np.where(U_0 == 0)] = 0
        if len(np.array([U_0 == 0]).shape) > 1:
            u[np.where(q_0 == 0)] = U_0[np.where(q_0 == 0)]
        else:
            u[np.where(q_0 == 0)] = U_0
        return u


def eta(x, params=[-0.45, -0.45, 1]):
    if len(np.array(x).shape) == 1:
        U_0 = x[0]
        q_0 = x[1]
        t = x[2]
    else:
        U_0 = x[:, 0]
        q_0 = x[:, 1]
        t = x[:, 2]
    sig_a = params[0]
    sig_r = params[1]
    v = params[2]
    qu = (sig_r * U_0 + q_0 * sig_a) * q_0 / (q_0 * sig_a + sig_r * U_0 * np.exp(-v * t * (sig_r * U_0 + q_0 * sig_a)))
    if len(np.array([qu]).shape) == 1:
        if (U_0 == 0):
            return q_0
        elif (q_0 == 0):
            return 0
        else:
            return qu
    else:
        if len(np.array([q_0 == 0]).shape) > 1:
            qu[np.where(U_0 == 0)] = q_0[np.where(U_0 == 0)]
        else:
            qu[np.where(U_0 == 0)] = q_0
        qu[np.where(q_0 == 0)] = 0
        return qu


def u_eta(x, params=[-0.45, -0.45, 1]):
    return np.c_[u(x, params), eta(x, params)]


def tau(x, params):
    if len(np.array(x).shape) == 1:
        U_0 = x[0]
        q_0 = x[1]
        t = x[2]
        U_T = x[3]
    else:
        U_0 = x[:, 0]
        q_0 = x[:, 1]
        t = x[:, 2]
        U_T = x[:, 3]
    if (np.min(U_T) == 0) | ((np.min(U_0) == 0) & (np.min(q_0) == 0)):
        print("check U_T, U_0 and q_0")
        return
    sig_a = params[0]
    sig_r = params[1]
    sig_s = params[2]
    v = params[3]
    tau_max = params[4]
    aq = sig_a * q_0
    ur = U_0 * sig_r
    num = aq * np.exp((aq + ur) * t - sig_a / sig_s / v * np.log(U_T)) + np.exp(
        -np.log(U_T) * sig_a / sig_s / v) * ur - ur
    res = v * (-t + 1 / (aq + ur) * np.log(num / aq))
    if len(np.array([res]).shape) > 1:
        res[np.where(num > 0)] = tau_max
        res[np.where(res >= tau_max)] = tau_max
    else:
        if num / aq <= 0:
            res = tau_max
        else:
            res = v * (-t + 1 / (aq + ur) * np.log(num / aq))
            if res > tau_max:
                res = tau_max
    return np.reshape(res, (U_0.shape[0],1))
