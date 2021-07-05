# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../../')
path = os.path.dirname(os.path.realpath(__file__))
from models_training.params.querry import add_params


########### Ueta ############

problem_id = "Ueta"

boundaries_dict = {}
boundaries_dict['min'] = [0, 0, 0]
boundaries_dict['max'] = [4, 2, 10]

params_dict = {}
params_dict['sig_a'] = -0.45
params_dict['sig_r'] = -0.45
params_dict['v'] = 1
# number of T per (U_0,q_0,U_T)
params_dict['T_div'] = 10

add_params(path, problem_id, boundaries_dict, params_dict)

########### tau ############

problem_id = "tau"

boundaries_dict = {}
boundaries_dict['min'] = [0, 0, 0, 0]
boundaries_dict['max'] = [4, 2, 10, 1]

params_dict = {}
params_dict['sig_a'] = -0.45
params_dict['sig_r'] = -0.45
params_dict['sig_s'] = 1.45
params_dict['v'] = 1
# number of T per (U_0,q_0,U_T)
params_dict['tau_max'] = 5
params_dict['T_div'] = 10

add_params(path, problem_id, boundaries_dict, params_dict)

########### Bateman ############

problem_id = "Bateman"

boundaries_dict = {}
boundaries_dict['min'] = [0 for i in range(13)]
boundaries_dict['max'] = [5] + [1 for i in range(12)]


params_dict = {}
params_dict['sig_r_0'] = 1
params_dict['sig_r_1'] = 5
params_dict['sig_r_2'] = 3
params_dict['sig_r_3'] = 0.1
params_dict['v'] = 1
# number of T per (U_0,q_0,U_T)
params_dict['T_div'] = 10

add_params(path, problem_id, boundaries_dict, params_dict)

########### runge ############

problem_id = "runge"

boundaries_dict = {}
boundaries_dict['min'] = [0]
boundaries_dict['max'] = [1]

params_dict = {}
params_dict['shift'] = 0.5
params_dict['coeff'] = 25

add_params(path, problem_id, boundaries_dict, params_dict)

########### tanh ############

problem_id = "tanh"

boundaries_dict = {}
boundaries_dict['min'] = [0]
boundaries_dict['max'] = [1]

params_dict = {}
params_dict['shift'] = 0.5
params_dict['coeff'] = 10

add_params(path, problem_id, boundaries_dict, params_dict)

########### Michalewitz ############

problem_id = "mich"

boundaries_dict = {}
boundaries_dict['min'] = [0, 0]
boundaries_dict['max'] = [3, 3]

params_dict = {}
params_dict['m'] = 5

add_params(path, problem_id, boundaries_dict, params_dict)


