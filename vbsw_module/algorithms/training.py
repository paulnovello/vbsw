import numpy as np
import pickle as pkl
import sys
import tensorflow as tf
import time

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

syspath = os.path.dirname(os.path.realpath(__file__)) + '/../..'
sys.path.insert(0, syspath)
from vbsw_module.models.fcnn import FCNN
from vbsw_module.io.saving_reading import save_exp




def training(training_set, test_set, N_layers, N_units, activation_hidden, activation_output, N_seeds,
             batch_size, epochs, optimizer, learning_rate, loss_function, test_losses,
             keep_best=True, criterion_for_best=None, saving_period=1, verbose=1, save_file=None):

    try:
        test_losses[0] = test_losses[0]
    except TypeError:
        test_losses = [test_losses]
    verbose_period = saving_period

    x_train, y_train = training_set
    x_test, y_test = test_set

    layers = [N_units for i in range(N_layers)]
    activations = [activation_hidden for i in range(N_layers)]
    activations.append(activation_output)

    best = None
    best_model = None
    for i in range(N_seeds):
        tf.keras.backend.clear_session()
        NN = FCNN(x_train.shape[1], y_test.shape[1], layers, activations)
        if verbose:
            NN.model.summary()
        NN.train(train_set=training_set,
                 batch_size=batch_size,
                 epochs=epochs,
                 optimizer=optimizer,
                 learning_rate=learning_rate,
                 loss_function=loss_function,
                 test_losses=test_losses,
                 saving_period=saving_period,
                 test_set=test_set,
                 verbose=verbose,
                 verbose_period=verbose_period,
                 keep_best=keep_best,
                 criterion_for_best=criterion_for_best)

        if save_file is not None:
            save_exp(NN.results, save_file)

        if keep_best:
            if criterion_for_best is None:
                criterion_for_best = test_losses[0]

            if criterion_for_best == "mse":
                comp = [-a for a in NN.results["data"][criterion_for_best + "_test"]]
            else:
                comp = NN.results["data"][criterion_for_best + "_test"]
            best_i = np.max(comp)
            if (best == None) or (best_i > best):
                best = best_i
                best_model = tf.keras.models.clone_model(NN.model)
                best_model.set_weights(NN.model.get_weights())

    if keep_best:
        return best_model







