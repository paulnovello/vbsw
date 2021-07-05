import os
import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

syspath = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.insert(0, syspath)

problem_id = sys.argv[1]
res_path = syspath + "/results/tbs" + str(problem_id) + "1-1/"

res_list = os.listdir(res_path)

bins = 30
MSE_uni = []
MSE_tbs = []
MSE_lvbs = []
SUP_uni = []
SUP_tbs = []
SUP_lvbs = []


for res in res_list:
    try:
        with open(res_path + res, "rb") as f:
            data = pkl.load(f)


    except (EOFError, pkl.UnpicklingError, IsADirectoryError):
        continue

    if data["training"]["sampler"] is None:
        MSE_uni += [np.min(data["data"][i]["mse_test"]) for i in range(len(data["data"]))]
    elif data["training"]["sampler"] == "tbs":
        MSE_tbs += [np.min(data["data"][i]["mse_test"]) for i in range(len(data["data"]))]
        
    if data["training"]["sampler"] is None:
        SUP_uni += [np.min(data["data"][i]["sup_test"]) for i in range(len(data["data"]))]
    elif data["training"]["sampler"] == "tbs":
        SUP_tbs += [np.min(data["data"][i]["sup_test"]) for i in range(len(data["data"]))]


def CI(var):
    return np.sqrt(var/50)


print(MSE_uni)
print("BS - mean MSE: ", np.mean(MSE_uni))
print("BS - MSE standard error: ", CI(np.var(MSE_uni)))
print("BS - mean L_inf: ", np.mean(SUP_uni))
print("BS - L_inf standard error: ", CI(np.var(SUP_uni)))

print("TBS - mean MSE: ", np.mean(MSE_tbs))
print("TBS - MSE standard error: ", CI(np.var(MSE_tbs)))
print("TBS - mean L_inf: ", np.mean(SUP_tbs))
print("TBS - L_inf standard error: ", CI(np.var(SUP_tbs)))





