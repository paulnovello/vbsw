import numpy as np
import sys
import tensorflow as tf
import time
import os
from tensorflow.keras.datasets.mnist import load_data

syspath = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.insert(0, syspath)

from vbsw_module.algorithms.vbsw import vbsw_for_dl

case_name="mnist_paper_results"
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

### Load and preprocess the data
num_classes = 10
(x_train, y_train), (x_test, y_test) = load_data()
# Set numeric type to float32 from uint8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)


### Load the initial model
model_list = os.listdir(syspath + "/data/saved_models/models_mnist")
if len(model_list) ==1:
    print("Please train a model first using \"python main.py launch cifar10 -i\"")

for model_name in model_list:
    if model_name == "init":
        continue
    model_init = tf.keras.models.load_model(syspath + "/data/saved_models/models_mnist/" + model_name)
    accuracy_init = model_init.evaluate(x_test, y_test)[1]

    ### Apply VBSW for dl
    model_vbsw = vbsw_for_dl(model_init=model_init,
                             training_set=(x_train, y_train),
                             test_set=(x_test, y_test),
                             ratio=40,
                             N_stat=20,
                             N_seeds=int(sys.argv[-1]),
                             activation_output="softmax",
                             batch_size=25,
                             epochs=25,
                             optimizer="adam",
                             learning_rate=1e-3,
                             loss_function="categorical_crossentropy",
                             test_losses=["categorical_accuracy"],
                             keep_best=False,
                             case_name=case_name,
                             dataset="mnist")
