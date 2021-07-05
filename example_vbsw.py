import numpy as np
import sys
import os

syspath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, syspath)

from vbsw_module.algorithms.vbsw import vbsw
from vbsw_module.algorithms.training import training
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


### Load the data
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

### Preprocess the data
max_x = np.max(x_train, axis=0)
x_train /= max_x
x_test /= max_x

### Train a linear model with VBSW
model_vbsw = vbsw(training_set=(x_train, y_train),
                  test_set=(x_test, y_test),
                  ratio=57,
                  N_stat=35,
                  N_seeds=1,
                  N_layers=0,
                  N_units=0,
                  activation_hidden="",
                  activation_output="linear",
                  batch_size=x_train.shape[0],
                  epochs=1000,
                  optimizer="adam",
                  learning_rate=1e-3,
                  loss_function="mse",
                  test_losses=["mse"],
                  keep_best=True,
                  saving_period=100)

### Train a linear model without VBSW
model = training(training_set=(x_train, y_train),
                 test_set=(x_test, y_test),
                 N_layers=0,
                 N_units=0,
                 activation_hidden="",
                 activation_output="linear",
                 N_seeds=1,
                 batch_size=x_train.shape[0],
                 epochs=1000,
                 optimizer="adam",
                 learning_rate=1e-3,
                 loss_function="mse",
                 test_losses=["mse"],
                 keep_best=True,
                 saving_period=100)

x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
accuracy_vbsw = np.mean((model_vbsw.predict(x_test) - y_test)**2)
accuracy_base = np.mean((model.predict(x_test) - y_test)**2)
print("Accuracy without VBSW: " + str(accuracy_base) + "\n" +\
      "Accuracy with VBSW: " + str(accuracy_vbsw))