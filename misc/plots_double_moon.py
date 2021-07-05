from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl
import os
import sys
import tensorflow as tf

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

syspath = os.path.dirname(os.path.realpath(__file__)) + '/..'
sys.path.insert(0, syspath)
from vbsw_module.models.fcnn_old import FCNN
from vbsw_module.functions.df import variance_old


#########DOUBLE MOON

X, y = make_moons(300, noise=0.2, random_state=0)
Xt, yt = make_moons(500, noise=0.2, random_state=1)

X0 = X[np.where(y == 0)[0],]
X1 = X[np.where(y == 1)[0],]
Xt0 = Xt[np.where(yt == 0)[0],]
Xt1 = Xt[np.where(yt == 1)[0],]

y = np.reshape(y, (y.shape[0], 1))

im = np.zeros((1000, 1000)) + 1
plt.imshow(im, extent=[-1.7, 2.3, -1, 2], cmap="Greys")
plt.plot(Xt0[:, 0], Xt0[:, 1], '.', color="royalblue")
plt.plot(Xt1[:, 0], Xt1[:, 1], '.', color="darkorange")
plt.xlim(-1.55, 2.45)
plt.ylim(-1.6, 2.4)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(syspath + "/figures/double_moon/data")
plt.close()

w = variance_old(Xt, yt, 20)
ratio = 100
w = w / np.max(w) * (ratio - 1)
w += 1

im = np.zeros((1000, 1000))
plt.imshow(im, extent=[-1.7, 2.3, -1, 2], cmap="Greys")
plt.scatter(Xt[:,0],Xt[:,1] , s=10, c=w, cmap = "bwr")
plt.xlim(-1.55, 2.45)
plt.ylim(-1.6, 2.4)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(syspath + "/figures/double_moon/weighted_data")
plt.close()

if len(sys.argv) == 1:
    N = sys.argv[1]
else:
    N = 1
batch_size = 100
epochs = 20000
optimizer = "sgd"
loss_function = "mse"
test_losses = ["mse", 'binary_accuracy']
saving_period = 10
test_set = [Xt, yt]
verbose = 1
verbose_period = 1000

print("loop starts")
for l in range(N):




    w = variance_old(X, y, 20)
    ratio = 100
    w = w / np.max(w) * (ratio - 1)
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

    x2 = np.linspace(-1.55, 2.45, 200)
    x1 = np.linspace(-1.6, 2.4, 200)
    x = np.array(np.meshgrid(x1, x2)).T
    x = np.reshape(x, (40000, 2))
    ypred = NN.model(x)

    im = np.zeros((200, 200))
    k = 0
    for i in range(200):
        for j in range(200):
            im[j, i] = ypred[k]
            k += 1

    plt.contourf(x2, x1, im, levels=1, cmap="cividis", alpha=0.4)
    #plt.imshow(im, cmap='cividis', extent=[-1.55, 2.45, -1.6, 2.4], alpha=0.4)
    plt.plot(Xt0[:, 0], Xt0[:, 1], '.', color="royalblue")
    plt.plot(Xt1[:, 0], Xt1[:, 1], '.', color="darkorange")
    plt.xlim(-1.55, 2.45)
    plt.ylim(-1.6, 2.4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(syspath + "/figures/double_moon/decision_boundary/vbsw" + str(l) + "")
    plt.close()

    tf.keras.backend.clear_session()
    NN = FCNN(2, 1, [4], ["relu", "sigmoid"])
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

    x2 = np.linspace(-1.55, 2.45, 200)
    x1 = np.linspace(-1.6, 2.4, 200)
    x = np.array(np.meshgrid(x1, x2)).T
    x = np.reshape(x, (40000, 2))
    ypred = NN.model(x)

    im = np.zeros((200, 200))
    k = 0
    for i in range(200):
        for j in range(200):
            im[j, i] = ypred[k]
            k += 1

    plt.contourf(x2, x1, im, levels=1, cmap="cividis", alpha=0.4)
    #plt.imshow(im, cmap='cividis', extent=[-1.55, 2.45, -1.6, 2.4], alpha=0.4)
    plt.plot(Xt0[:, 0], Xt0[:, 1], '.', color="royalblue")
    plt.plot(Xt1[:, 0], Xt1[:, 1], '.', color="darkorange")
    plt.xlim(-1.55, 2.45)
    plt.ylim(-1.6, 2.4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(syspath + "/figures/double_moon/decision_boundary/unif" + str(l) + "")
    plt.close()
