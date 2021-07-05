import os
import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

syspath = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.insert(0, syspath)


res_path = syspath + "/results/double_moon/"

res_list = os.listdir(res_path)


for res in res_list:
    try:
        with open(res_path + res, "rb") as f:
            data = pkl.load(f)
    except (EOFError, pkl.UnpicklingError, IsADirectoryError):
        continue

    if res[8] == "u":
        MSE_uni = [np.max(data["data"][i]["mse_test"]) for i in range(len(data["data"]))]
    else:
        MSE_lvbs = [np.max(data["data"][i]["mse_test"]) for i in range(len(data["data"]))]

    try:
        if res[8] == "u":
            ACC_uni = [np.max(data["data"][i]["binary_accuracy_test"]) for i in range(len(data["data"]))]
        else:
            ACC_lvbs = [np.max(data["data"][i]["binary_accuracy_test"]) for i in range(len(data["data"]))]
    except KeyError:
        pass

def CI(var):
    return np.sqrt(var/50)

print("baseline - mean accuracy: ", np.mean(ACC_uni))
print("baseline - standard error: ", CI(np.var(ACC_uni)))
print("baseline - max accuracy: ", np.max(ACC_uni))
print("VBSW - mean accuracy: ", np.mean(ACC_lvbs))
print("VBSW - standard error: ", CI(np.var(ACC_lvbs)))
print("VBSW - max accuracy", np.max(ACC_lvbs))


