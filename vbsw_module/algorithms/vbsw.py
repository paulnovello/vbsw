import numpy as np
import sys
import tensorflow as tf
import time
import os
import pickle as pkl

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

syspath = os.path.dirname(os.path.realpath(__file__)) + '/../..'
sys.path.insert(0, syspath)
from vbsw_module.functions.df import dataset_weighting
from vbsw_module.algorithms.training import training


def vbsw(training_set, test_set, ratio, N_stat, N_layers, N_units, activation_hidden,
         activation_output, N_seeds, batch_size, epochs, optimizer, learning_rate, loss_function, test_losses,
         keep_best=True, criterion_for_best=None, saving_period=1, verbose=1, case_name=None, dataset=None):
    x_train, y_train = training_set
    training_set = dataset_weighting(x_train, y_train, ratio, N_stat, dataset)

    if case_name is not None:
        save_dir = syspath + "/results/" + case_name
        res_name = "res_" + str(os.getpid()) + str(time.time())[:10]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, res_name)
    else:
        save_file = None

    model = training(training_set=training_set,
                     test_set=test_set,
                     N_layers=N_layers,
                     N_units=N_units,
                     activation_hidden=activation_hidden,
                     activation_output=activation_output,
                     N_seeds=N_seeds,
                     batch_size=batch_size,
                     epochs=epochs,
                     optimizer=optimizer,
                     learning_rate=learning_rate,
                     loss_function=loss_function + "_w",
                     test_losses=test_losses,
                     keep_best=keep_best,
                     criterion_for_best=criterion_for_best,
                     saving_period=saving_period,
                     verbose=verbose,
                     save_file=save_file)

    if case_name is not None:
        with open(save_file, "rb") as f:
            results = pkl.load(f)

        results["hyperparams"]["ratio"] = ratio
        results["hyperparams"]["N_stat"] = N_stat
        results["hyperparams"]["vbsw"] = 1

        with open(save_file, "wb") as f:
            pkl.dump(results, f)

    if keep_best:
        return model


def vbsw_for_dl(model_init, training_set, test_set, ratio, N_stat, N_seeds,
                activation_output, batch_size, epochs, optimizer, learning_rate, loss_function, test_losses,
                keep_best=True, criterion_for_best=None, saving_period=1, verbose=1, case_name=None, dataset=None):
    x_train_init, y_train = training_set
    x_test_init, y_test = test_set
    loss_init = model_init.evaluate(x_test_init, y_test)[1]

    inputs = model_init.input
    output_latent = model_init.layers[-2].output
    model_trunc = tf.keras.Model(inputs, output_latent, name="model_trunc")


    x_train = model_trunc.predict(x_train_init)
    x_test = model_trunc.predict(x_test_init)

    training_set = dataset_weighting(x_train, y_train, ratio, N_stat, dataset)
    test_set = (x_test, y_test)

    if case_name is not None:
        save_dir = syspath + "/results/" + case_name
        res_name = "res_" + str(os.getpid()) + str(time.time())[:10]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, res_name)
    else:
        save_file = None

    model = training(training_set=training_set,
                     test_set=test_set,
                     N_layers=0,
                     N_units=0,
                     activation_hidden="",
                     activation_output=activation_output,
                     N_seeds=N_seeds,
                     batch_size=batch_size,
                     epochs=epochs,
                     optimizer=optimizer,
                     learning_rate=learning_rate,
                     loss_function=loss_function + "_w",
                     test_losses=test_losses,
                     keep_best=keep_best,
                     criterion_for_best=criterion_for_best,
                     saving_period=saving_period,
                     verbose=verbose,
                     save_file=save_file)

    if case_name is not None:
        with open(save_file, "rb") as f:
            results = pkl.load(f)

        results["misc"] = {}
        results["misc"]["loss_init"] = loss_init
        results["hyperparams"]["ratio"] = ratio
        results["hyperparams"]["N_stat"] = N_stat
        results["hyperparams"]["vbsw"] = 1

        with open(save_file, "wb") as f:
            pkl.dump(results, f)

    if keep_best:
        input = model_trunc.input
        x = model_trunc(input)
        output = model(x)
        final_model = tf.keras.models.Model(input, output)
        return final_model
