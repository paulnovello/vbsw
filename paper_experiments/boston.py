import numpy as np
import sys
import os
import time

syspath = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.insert(0, syspath)

from vbsw_module.algorithms.vbsw import vbsw
from vbsw_module.algorithms.training import training
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

case_name="boston_paper_results"
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
                  ratio=8,
                  N_stat=35,
                  N_seeds=int(sys.argv[-1]),
                  N_layers=0,
                  N_units=0,
                  activation_hidden="",
                  activation_output="linear",
                  batch_size=x_train.shape[0],
                  epochs=50000,
                  optimizer="adam",
                  learning_rate=1e-3,
                  loss_function="mse",
                  test_losses=["mse"],
                  keep_best=True,
                  saving_period=100,
                  case_name=case_name,
                  dataset="boston")

### Train a linear model without VBSW
save_dir = syspath + "/results/" + case_name
res_name = "res_" + str(os.getpid()) + str(time.time())[:10]
save_file = os.path.join(save_dir, res_name)

model = training(training_set=(x_train, y_train),
                 test_set=(x_test, y_test),
                 N_layers=0,
                 N_units=0,
                 activation_hidden="",
                 activation_output="linear",
                 N_seeds=int(sys.argv[-1]),
                 batch_size=x_train.shape[0],
                 epochs=50000,
                 optimizer="adam",
                 learning_rate=1e-3,
                 loss_function="mse",
                 test_losses=["mse"],
                 keep_best=True,
                 saving_period=100,
                 save_file=save_file)

