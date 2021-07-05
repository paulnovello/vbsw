from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl
import os
import sys
import tensorflow as tf

syspath = os.path.dirname(os.path.realpath(__file__)) + '/..'
sys.path.insert(0, syspath)
from vbsw_module.models.fcnn_old import FCNN
from vbsw_module.functions.df import variance_old


case_name = "double_moon"
import datetime
if os.path.isdir(syspath + "/results/" + case_name):
    listdir = os.listdir(syspath + "/results/" + case_name)
    if len(listdir) > 0:
        if not os.path.isdir(syspath + "/results/" + case_name + "/old"):
            os.makedirs(syspath + "/results/" + case_name + "/old")
        today = str(datetime.datetime.today()).replace(" ", "")
        os.makedirs(syspath + "/results/" + case_name + "/old/" + today)
        for file in listdir:
            if file != "old":
                os.system("mv ~/scratch/results/" + case_name + "/" + file + " ~/scratch/results/" + case_name + "/old/" + today)



X, y = make_moons(300, noise=0.2, random_state=0)
Xt, yt = make_moons(500, noise=0.15, random_state=1)

X0 = X[np.where(y == 0)[0],]
X1 = X[np.where(y == 1)[0],]
Xt0 = Xt[np.where(yt == 0)[0],]
Xt1 = Xt[np.where(yt == 1)[0],]

y = np.reshape(y, (y.shape[0],1))

if len(sys.argv) == 2:
    N = int(sys.argv[1])
else:
    N = 50
batch_size = 100
epochs = 10000
optimizer = "sgd"
loss_function = "mse"
test_losses = ["mse", 'binary_accuracy']
saving_period = 10
test_set = [Xt, yt]
verbose = 1
verbose_period = 1000

list_dir = os.listdir(syspath + "/results/")
if "double_moon" not in list_dir:
    os.system("mkdir " + syspath + "/results/double_moon")
    
path_uni = syspath + "/results/double_moon/results_uni" +  str(os.getpid()) + str(time.time())[:10]
path_lvbs = syspath + "/results/double_moon/results_lvbs" +  str(os.getpid()) + str(time.time())[:10]

print("loop starts")
for l in range(N):
    tf.keras.backend.clear_session()
    NN = FCNN(2, 1, [4] , ["relu", "sigmoid"])
    NN.train(train_set=[X, y],
             batch_size=batch_size,
             epochs=epochs,
             optimizer=optimizer,
             loss_function=loss_function,
             test_losses=test_losses,
             saving_period=saving_period,
             test_set=test_set,
             verbose=verbose,
             verbose_period=verbose_period)

    x2 = np.linspace(-1.55, 2.45, 100)
    x1 = np.linspace(-1.6, 2.4, 100)
    x = np.array(np.meshgrid(x1, x2)).T
    x = np.reshape(x, (10000, 2))
    ypred = NN.model(x)

    im = np.zeros((100,100))
    k = 0
    for i in range(100):
        for j in range(100):
            im[j,i] = ypred[k]
            k += 1

    pic_list = os.listdir(syspath + "/figures/double_moon/decision_boundary")
    n_pic = len(pic_list)

    plt.contourf(x2, x1, im, levels=1, cmap="cividis", alpha=0.4)
    plt.plot(Xt0[:, 0], Xt0[:, 1], '.', color="blue")
    plt.plot(Xt1[:, 0], Xt1[:, 1], '.', color="red")
    plt.xlim(-2,3)
    plt.ylim(-2,3)
    plt.savefig(syspath + "/figures/double_moon/decision_boundary/unif" + str(n_pic//2 -1 + l) )
    plt.close()

    try:
        with open(path_uni, "rb") as f:
            results = pkl.load(f)

        results["data"].append(NN.results["data"])
        results["hyperparams"] = NN.results["hyperparams"]
        results["training"] = NN.results["training"]


        with open(path_uni, "wb") as f:
            pkl.dump(results, f)

    except FileNotFoundError:
        results = {}
        results["data"] = [NN.results["data"]]
        results["hyperparams"] = NN.results["hyperparams"]
        results["training"] = NN.results["training"]

        with open(path_uni, "wb") as f:
            pkl.dump(results, f)

    w = variance_old(X, y, 20)
    ratio = 100
    w = w/np.max(w)*(ratio-1)
    w += 1
    yw = np.c_[y, w]

    tf.keras.backend.clear_session()
    NN = FCNN(2, 1, [4], ["relu", "sigmoid"])
    NN.train(train_set=[X, yw],
             batch_size=batch_size,
             epochs=epochs,
             optimizer=optimizer,
             loss_function=loss_function + "_w",
             test_losses=test_losses,
             saving_period=saving_period,
             test_set=test_set,
             verbose=verbose,
             verbose_period=verbose_period)

    x2 = np.linspace(-1.55, 2.45, 100)
    x1 = np.linspace(-1.6, 2.4, 100)
    x = np.array(np.meshgrid(x1, x2)).T
    x = np.reshape(x, (10000, 2))
    ypred = NN.model(x)

    im = np.zeros((100,100))
    k = 0
    for i in range(100):
        for j in range(100):
            im[ j,i] = ypred[k]
            k += 1

    plt.contourf(x2, x1, im, levels=1, cmap="cividis", alpha=0.4)
    plt.plot(Xt0[:, 0], Xt0[:, 1], '.', color="blue")
    plt.plot(Xt1[:, 0], Xt1[:, 1], '.', color="red")
    plt.xlim(-2,3)
    plt.ylim(-2,3)
    plt.savefig(syspath + "/figures/double_moon/decision_boundary/lvbs" + str(n_pic//2 -1 + l))
    plt.close()
    
    try:
        with open(path_lvbs, "rb") as f:
            results = pkl.load(f)

        results["data"].append(NN.results["data"])
        results["hyperparams"] = NN.results["hyperparams"]
        results["training"] = NN.results["training"]


        with open(path_lvbs, "wb") as f:
            pkl.dump(results, f)

    except FileNotFoundError:
        results = {}
        results["data"] = [NN.results["data"]]
        results["hyperparams"] = NN.results["hyperparams"]
        results["training"] = NN.results["training"]

        with open(path_lvbs, "wb") as f:
            pkl.dump(results, f)
