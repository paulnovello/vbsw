import numpy as np
import pandas as pd
import os

def get_add_row(params_dict, path):
    for k in params_dict.keys():
        params_dict[k] = str(params_dict[k])

    try:
        params_df = pd.read_csv(path, sep="\t", index_col=0)
        params_df = params_df.astype("str")

        if params_dict not in params_df.to_dict('index').values():
            ind = np.max(params_df.index) + 1
            params = pd.DataFrame(params_dict, columns=params_dict.keys(), index=[ind])
            params_df = params_df.append(params)
            params_df.to_csv(path, sep="\t")
            return ind
        else:
            ind = np.where(np.array(list(params_df.to_dict('index').values()))== params_dict)[0][0] + 1
            return ind
    except pd.errors.EmptyDataError:
        ind = 1
        params = pd.DataFrame(params_dict, columns=params_dict.keys(), index=[ind])
        params.to_csv(path, sep="\t")

def add_params(path, problem_id, boundaries_dict, params_dict):

    if "params_" + str(problem_id) not in os.listdir(path):
        os.system("mkdir params_" + str(problem_id))

    path_boundaries = path + "/params_" + str(problem_id) + "/table_" + str(problem_id) + "_boundaries.txt"
    path_params = path + "/params_" + str(problem_id) + "/table_" + str(problem_id) + "_params.txt"

    try:
        get_add_row(boundaries_dict, path_boundaries)
        get_add_row(params_dict, path_params)
    except FileNotFoundError:
        os.system("touch " + path_boundaries)
        os.system("touch " + path_params)
        get_add_row(boundaries_dict, path_boundaries)
        get_add_row(params_dict, path_params)