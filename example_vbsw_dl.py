import numpy as np
import sys
import tensorflow as tf
import time
import os
from tensorflow.keras.datasets.mnist import load_data

syspath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, syspath)

from vbsw_module.algorithms.vbsw import vbsw_for_dl

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
model_name = "mnist_lenet_70621591889666.h5"
model_init = tf.keras.models.load_model(syspath + "/data/saved_models/models_mnist/" + model_name)
accuracy_init = model_init.evaluate(x_test, y_test)[1]

### Apply VBSW for dl
model_vbsw = vbsw_for_dl(model_init=model_init,
                         training_set=(x_train, y_train),
                         test_set=(x_test, y_test),
                         ratio=40,
                         N_stat=20,
                         N_seeds=10,
                         activation_output="softmax",
                         batch_size=25,
                         epochs=3,
                         optimizer="adam",
                         learning_rate=1e-3,
                         loss_function="categorical_crossentropy",
                         test_losses=["categorical_accuracy"],
                         keep_best=True,
                         dataset="mnist")

accuracy_vbsw = np.mean(tf.keras.metrics.categorical_accuracy(model_vbsw.predict(x_test), y_test))
print("Initial accuracy: " + str(accuracy_init) + "\n" +\
      "Accuracy after VBSW: " + str(accuracy_vbsw))