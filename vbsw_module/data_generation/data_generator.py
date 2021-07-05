import os
import sys
syspath = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.insert(0, syspath)

import numpy as np
import pickle as pkl
import pandas as pd
from importlib import import_module



class DataGenerator:

    def __init__(self, problem_id, problem_params_id, problem_boundaries_id,
                 sampler, generator):
        self.problem_id = problem_id
        self.problem_params_id = problem_params_id
        self.problem_boundaries_id = problem_boundaries_id
        boundaries = pd.read_csv(syspath + "params/params_" + str(self.problem_id) + "/table_" + \
                                 str(problem_id) + "_boundaries.txt",
                                 sep = "\t", index_col = 0)
        self.boundaries = np.array([pd.eval(boundaries.loc[problem_boundaries_id, "min"]),
                                    pd.eval(boundaries.loc[problem_boundaries_id, "max"])])
        params = pd.read_csv(syspath +"params/params_" + str(self.problem_id) + "/table_" + \
                                 str(problem_id) + "_params.txt",
                                 sep = "\t", index_col = 0)
        self.params = np.array(params.loc[problem_params_id])
        self.sampler = sampler
        self.generator = generator
        self.inputs = None
        self.outputs = None

    def generate(self, n):
        if n == 0:
            return (None, None)
        else:
            self.inputs = self.sampler(self.boundaries, n)
            self.outputs = self.generator(self.inputs, self.params)
            return self.inputs, self.outputs

    def save(self):
        path = os.path.dirname(os.path.realpath(__file__)) +\
            "/../datasets/data_" + str(self.problem_id) +\
            str(self.problem_boundaries_id) + "-" + str(self.problem_params_id) +\
            str(self.sampler.__name__)[:4]
        try:
            with open(path, 'rb') as f:
                data = pkl.load(f)
            X = data[0]
            Y = data[1]
            X = np.append(X, self.inputs, axis = 0)
            Y = np.append(Y, self.outputs, axis = 0)
            data = [X, Y]
            with open(path, 'wb') as f:
                pkl.dump(data, f)
        except:
            data = [self.inputs, self.outputs]
            with open(path, 'wb') as f:
                pkl.dump(data, f)



