import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import time


syspath = os.path.dirname(os.path.realpath(__file__)) + '/..'
sys.path.insert(0, syspath)

from vbsw_module.functions.basic_functions import fun_list
from vbsw_module.data_generation.samplers import tbs
from vbsw_module.models.fcnn_old import FCNN

from vbsw_module.data_generation.data_generator import DataGenerator
from vbsw_module.data_generation.samplers import grid_sampler
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



model_id = 'fcnn'
problem_id = sys.argv[1]
sampler = sys.argv[2]
problem_boundaries_id = 1
problem_params_id = 1
if len(sys.argv) == 4:
    N_seeds = int(sys.argv[3])
else:
    N_seeds = 50
saving_period = 100
test_losses = ['mse', 'sup']




n = 16
n_units = [8]
activations = ['relu', 'linear']
dropout = [0]
batch_norm = [0]
weights_reg = [[0, 0]]
bias_reg = [[0, 0]]

n_test = 1000
n_val = 0


loss_function = "mse"
optimizer = "adam"
epochs = 50000
learning_rate = None

hyperparams_dict = {}
hyperparams_dict['n_units'] = n_units
hyperparams_dict['activations'] = activations
hyperparams_dict['dropout'] = dropout
hyperparams_dict['batch_norm'] = batch_norm
hyperparams_dict['weights_reg'] = weights_reg
hyperparams_dict['bias_reg'] = bias_reg



training_dict = {}
training_dict['n'] = n
training_dict['n_test'] = n_test
training_dict['n_val'] = n_val
training_dict['batch_size'] = n
training_dict['loss_function'] = loss_function
training_dict['optimizer'] = optimizer
training_dict['learning_rate'] = learning_rate
training_dict['epochs'] = epochs
training_dict['sampler'] = sampler
if training_dict['sampler'] == "None":
    training_dict['sampler'] = None
training_dict['test_losses'] = test_losses

training_dict['saving_period'] = saving_period
training_dict['verbose'] = 1
training_dict['verbose_period'] = 100
training_dict['plot'] = 0
training_dict['plot_period'] = 1
training_dict['training_plot'] = None
training_dict['plot_params'] = None

#### CREATE SAVING ENV
dir_list = os.listdir(syspath + "/results/")
dirname = "tbs" + str(problem_id) + str(problem_boundaries_id )+ \
          "-" + str(problem_params_id)

if dirname not in dir_list:
    os.system("mkdir " + syspath + "/results/" + dirname)

path_result = syspath + "/results/" + dirname + "/results_" + str(problem_id) + \
              str(problem_boundaries_id) + "-" + str(problem_params_id) + \
              "_" + str(model_id) + str(os.getpid()) + str(time.time())[:10]


for i in range(N_seeds):
    #### ADAPTIVE SAMPLING
    gen = DataGenerator(problem_id, problem_params_id, problem_boundaries_id,
                        grid_sampler, fun_list(problem_id))

    X_train, Y_train = gen.generate(int(training_dict['n']) // 2)
    boundaries = pd.read_csv(syspath + "/params/params_" + str(problem_id) + "/table_" + \
                             str(problem_id) + "_boundaries.txt",
                             sep="\t", index_col=0)
    boundaries = np.array([pd.eval(boundaries.loc[problem_boundaries_id, "min"]),
                           pd.eval(boundaries.loc[problem_boundaries_id, "max"])])

    tf.keras.backend.clear_session()
    fcnn = [hyperparams_dict, training_dict]
    problem = [problem_id, problem_boundaries_id, problem_params_id]



    if training_dict['sampler'] == "tbs":
        X_ada = tbs(boundaries, X_train, [fun_list(problem_id)], int(training_dict['n']) // 2, 5e-4, 100, 3)
        Y_ada = fun_list(problem_id)(X_ada)
        X_train = np.r_[X_train, X_ada]
        Y_train = np.r_[Y_train, Y_ada]
    else:
        X_train, Y_train = gen.generate(int(training_dict['n']))




    train_set = [X_train, Y_train]

    X_test, Y_test = gen.generate(int(training_dict['n_test']))
    test_set = [X_test, Y_test]


    #### TRAINING
    NN = FCNN(input_dim=int(X_train.shape[1]),
              output_dim=int(Y_train.shape[1]),
              n_units=hyperparams_dict['n_units'],
              activations=hyperparams_dict['activations'],
              dropout=pd.eval(str(hyperparams_dict['dropout'])),
              batch_norm=pd.eval(str(hyperparams_dict['batch_norm'])),
              weights_reg=pd.eval(str(hyperparams_dict['weights_reg'])),
              bias_reg=pd.eval(str(hyperparams_dict['bias_reg'])))

    print(epochs)
    NN.train(train_set=train_set,
             batch_size=int(training_dict['batch_size']),
             epochs=int(training_dict['epochs']),
             optimizer=training_dict['optimizer'],
             loss_function=training_dict['loss_function'],
             test_losses=training_dict['test_losses'],
             saving_period=int(training_dict['saving_period']),
             verbose=bool(training_dict['verbose']),
             verbose_period=int(training_dict['verbose_period']),
             test_set=test_set,
             learning_rate=training_dict['learning_rate'],
             plot=bool(training_dict['plot']),
             plot_period=int(training_dict['plot_period']),
             training_plot=training_dict['training_plot'],
             plot_params=training_dict['plot_params'],
             sampler=training_dict['sampler'])

    problem = [problem_id, problem_boundaries_id, problem_params_id]
    NN.save_results(path_result, problem)
