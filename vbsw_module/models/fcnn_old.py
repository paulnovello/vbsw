import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../../')
import tensorflow as tf
import autograd.numpy as np
import pickle as pkl
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Model
from vbsw_module.functions.optimizers_list import opt_list
from vbsw_module.functions.loss_functions import loss_list


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, n_units, activation, dropout_rate, batch_norm, weights_reg, bias_reg, **kwargs):

        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.batch_norm = batch_norm
        self.dense = Dense(n_units, activation=activation,
                           kernel_regularizer=l1_l2(weights_reg[0], weights_reg[1]),
                           bias_regularizer=l1_l2(bias_reg[0], bias_reg[1]))
        self.dropout = Dropout(self.dropout_rate)
        self.batch_norm_layer = BatchNormalization()
        super(DenseBlock, self).__init__(**kwargs)

    def call(self, inputs):
        x = self.dense(inputs)
        if self.dropout_rate != 0:
            x = self.dropout(x)
        if self.batch_norm != 0:
            x = self.batch_norm_layer(x)
        return x

    def get_config(self):
        #base_config = super(DenseBlock, self).get_config()
        base_config = {}
        base_config['n_units'] = self.n_units
        base_config['activation'] = self.activation
        base_config['dropout_rate'] = self.dropout_rate
        base_config['batch_norm'] = self.batch_norm
        return base_config
#def DenseBlock(n_units, activation, dropout_rate, batch_norm)


class FCNN:

    def __init__(self, input_dim, output_dim, n_units,
                 activations, dropout=None, batch_norm=None,
                 weights_reg=None, bias_reg=None):

        if dropout is None:
            dropout = [0 for i in range(len(n_units))]
        if batch_norm is None:
            batch_norm = [0 for i in range(len(n_units))]
        if weights_reg is None:
            weights_reg = [[0, 0] for i in range(len(n_units))]
        if bias_reg is None:
            bias_reg = [[0, 0] for i in range(len(n_units))]

        input_layer = Input((1, input_dim))

        for i in range(len(n_units)):
            if i == 0:
                x = DenseBlock(n_units[0], activations[0],
                               dropout[0], batch_norm[0],
                               weights_reg[0], bias_reg[0])(input_layer)
            else:
                x = DenseBlock(n_units[i], activations[i],
                           dropout[i], batch_norm[i],
                           weights_reg[i], bias_reg[i])(x)
        if len(n_units) == 0:
            output_layer = Dense(output_dim, activation=activations[-1])(input_layer)
        else:
            output_layer = Dense(output_dim, activation=activations[-1])(x)

        self.model = Model(input_layer, output_layer)

        self.n_units = n_units
        self.activations = activations
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights_reg = weights_reg
        self.bias_reg = bias_reg
        self.results = {}

        self.block_dict = {}

        hyperparams_dict = {}
        hyperparams_dict['n_units'] = n_units
        hyperparams_dict['activations'] = activations
        hyperparams_dict['dropout'] = dropout
        hyperparams_dict['batch_norm'] = batch_norm
        hyperparams_dict['weights_reg'] = weights_reg
        hyperparams_dict['bias_reg'] = bias_reg
        self.results["hyperparams"] = hyperparams_dict
        self.results["data"] = {}
        self.results["training"] = {}
        self.results["boundaries"] = {}
        self.results['params'] = {}

        #for i in range(len(self.n_units)):
        #    self.block_dict[i] = DenseBlock(self.n_units[i], self.activations[i],
        #                                    self.dropout[i], self.batch_norm[i])
        #self.output_layer = Dense(self.output_dim, activation=activations[-1])

    #def call(self, x):

    #    for i in range(len(self.n_units)):
    #        x = self.block_dict[i](x)
    #    return self.output_layer(x)

    @tf.function
    def train_step(self, x, y, loss_function, optimizer):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = loss_function(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train(self, train_set, batch_size, epochs,
              optimizer, loss_function, test_losses, saving_period,
              test_set=None, validation_set=None, learning_rate=None,
              verbose=0, verbose_period=1,
              plot=0, plot_period=1, training_plot=None,
              plot_params=None, sampler=None):
        #
        training_dict = {}
        training_dict['n'] = train_set[0].shape[0]
        training_dict['n_test'] = None if test_set is None else test_set[0].shape[0]
        training_dict['n_val'] = None if validation_set is None else validation_set[0].shape[0]
        training_dict['batch_size'] = batch_size
        training_dict['loss_function'] = loss_function
        training_dict['optimizer'] = optimizer
        training_dict['learning_rate'] = learning_rate
        training_dict['epochs'] = epochs
        training_dict['test_losses'] = test_losses
        training_dict['sampler'] = sampler
        self.results["training"] = training_dict

        optimizer = opt_list(optimizer, learning_rate)
        # optimizer = tf.keras.optimizers.Adam()

        if 'train_loss (' + loss_function + ')' not in self.results["data"].keys():
            self.results["data"]['train_loss (' + loss_function + ')'] = []
        #for test_loss in test_losses:
        #    if test_loss + '_train' not in self.results["data"].keys():
        #        self.results["data"][test_loss + '_train'] = []

        x_train = np.array(train_set[0], dtype='float32')
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        y_train = np.array(train_set[1], dtype='float32')
        y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
        train_ds = tf.data.Dataset.from_tensor_slices((x_train,
                                                       y_train)).batch(batch_size)

        if test_set != None:
            x_test = np.array(test_set[0], dtype='float32')
            x_test = np.reshape(x_test, (x_test.shape[0], 1, self.input_dim))
            y_test = np.array(test_set[1], dtype='float32')
            y_test = np.reshape(y_test, (y_test.shape[0], 1, self.output_dim))
            for test_loss in test_losses:
                if test_loss + '_test' not in self.results["data"].keys():
                    self.results["data"][test_loss + '_test'] = []

        if validation_set != None:
            x_validation = np.array(validation_set[0], dtype='float32')
            x_validation = np.reshape(x_validation, (x_validation.shape[0], 1, self.input_dim))
            y_validation = np.array(validation_set[1], dtype='float32')
            y_validation = np.reshape(y_validation, (y_validation.shape[0], 1, self.output_dim))
            for test_loss in test_losses:
                if test_loss + '_validation' not in self.results["data"].keys():
                    self.results["data"][test_loss + '_validation'] = []

        for epoch in range(epochs):
            for x, y in train_ds:
                self.train_step(x, y, loss_list(loss_function), optimizer)


            if epoch % saving_period == 0:
                training_error = np.mean(loss_list(loss_function)(y_train, self.model(x_train)))
                self.results["data"]['train_loss (' +
                             loss_function + ')'
                             ].append(float(training_error))
                for test_loss in test_losses:
                    #metric = np.mean(loss_list(test_loss)(y_train, self.model(x_train)))
                    #self.results["data"][test_loss + '_train'
                    #             ].append(float(metric))

                    if test_set is not None:
                        metric = np.mean(loss_list(test_loss)(y_test, self.model(x_test)))
                        self.results["data"][test_loss + '_test'
                                     ].append(float(metric))

                    if validation_set is not None:
                        metric = np.mean(loss_list(test_loss)(y_validation, self.model(x_validation)))
                        self.results["data"][test_loss + '_validation'
                                     ].append(float(metric))

            if verbose & (epoch % verbose_period == 0):
                to_print = str(epoch)
                for k in self.results["data"].keys():
                    to_print += k + ': ' + str(self.results["data"][k][-1]) + ', '
                print(to_print)

            if plot & (epoch % plot_period == 0):
                pass#plot = training_plot(self, plot_params)

    def save_results(self, path_result, problem):

        problem_id = problem[0]
        problem_boundaries_id = problem[1]
        problem_params_id = problem[2]

        path = os.path.dirname(os.path.realpath(__file__)) +\
                        "/../../params/params_" +  problem_id + "/table_" + \
                        problem_id + "_boundaries.txt"
        params_df = pd.read_csv(path, sep="\t", index_col=0)
        params_df = params_df.astype("str")
        boundaries_dict = params_df.loc[problem_boundaries_id].to_dict()

        path = os.path.dirname(os.path.realpath(__file__)) +\
                        "/../../params/params_" + problem_id + "/table_" + \
                        problem_id + "_params.txt"
        params_df = pd.read_csv(path, sep="\t", index_col=0)
        params_df = params_df.astype("str")
        params_dict = params_df.loc[problem_params_id].to_dict()



        try:
            with open(path_result, "rb") as f:
                results = pkl.load(f)

            results["data"].append(self.results["data"])
            results["hyperparams"] = self.results["hyperparams"]
            results["training"] = self.results["training"]
            results["boundaries"] = boundaries_dict
            results["params"] = params_dict

            with open(path_result, "wb") as f:
                pkl.dump(results, f)

        except FileNotFoundError:
            results = {}
            results["data"] = [self.results["data"]]
            results["hyperparams"] = self.results["hyperparams"]
            results["training"] = self.results["training"]
            results["boundaries"] = boundaries_dict
            results["params"] = params_dict

            with open(path_result, "wb") as f:
                pkl.dump(results, f)

