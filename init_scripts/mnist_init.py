import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import time
import matplotlib.pyplot as plt

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data

num_classes=10

if sys.argv[0] == "mnist_init.py":
    syspath = os.path.dirname(os.path.realpath(__file__)) + '/..'

sys.path.insert(0, syspath)


(x_train, y_train), (x_test, y_test) = load_data()
# Set numeric type to float32 from uint8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)


tf.keras.backend.clear_session()
#Instantiate an empty model
model = Sequential()

# C1 Convolutional Layer
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding='same'))

# S2 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

# C3 Convolutional Layer
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

# S4 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# C5 Fully Connected Convolutional Layer
model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
#Flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flatten())

# FC6 Fully Connected Layer
model.add(layers.Dense(84, activation='tanh'))

#Output Layer with softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])

model.fit(x=x_train,y=y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test), verbose=1)

save_dir = syspath + '/data/saved_models/models_mnist'
model_name = 'mnist_lenet_' + str(os.getpid()) + str(time.time())[:10] + ".h5"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model.save(os.path.join(save_dir, model_name))