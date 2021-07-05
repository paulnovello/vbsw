import pickle as pkl
import numpy as np
import os
import sys
syspath = os.path.dirname(os.path.realpath(__file__)) + '/../..'
sys.path.insert(0, syspath)

def save_exp(results_training, path):
    try:
        with open(path, "rb") as f:
            results = pkl.load(f)

        results["data"].append(results_training["data"])

        with open(path, "wb") as f:
            pkl.dump(results, f)

    except FileNotFoundError:
        results = {}
        results["data"] = [results_training["data"]]
        results["hyperparams"] = results_training["hyperparams"]
        results["training"] = results_training["training"]

        with open(path, "wb") as f:
            pkl.dump(results, f)

def read(results_dir):
    res_list = os.listdir(syspath + "/results/" + results_dir)
    results_dict = {}
    for res in res_list:
        if res == "old":
            continue
        with open(syspath + "/results/" + results_dir + "/" + res, "rb") as f:
            results_dict[res] = pkl.load(f)
    return results_dict


def read(dataset, testf, key):
    res = os.listdir(dataset + "/")
    vbsw = []
    diff = []
    uni = []

    for r in res:
        if r == "old":
            continue
        with open(dataset + "/" + r, "rb") as f:
            data = pkl.load(f)

        if key in list(data["hyperparams"].keys()):
            vbsw.append([np.max(a[testf + "_test"]) for a in data["data"]])
            if "misc" in list(data.keys()):
                diff.append(data["misc"]["loss_init"])
        else:
            uni.append([np.max(a[testf + "_test"]) for a in data["data"]])

    return np.array(uni), np.array(vbsw), np.array(diff)


def stats(dataset, testf):
    uni, vbsw, diff = read(dataset, testf, "vbsw")
    N = uni.shape[1]
    mean_v = np.mean(vbsw[0])
    mean_u = np.mean(uni[0])
    max_v = np.max(vbsw[0])
    max_u = np.max(uni[0])
    ci_v = np.var(vbsw[0]) ** 0.5 / N ** 0.5
    ci_u = np.var(uni[0]) ** 0.5 / N ** 0.5
    print("for " + dataset + ": \n"
                             "-------- metric: " + testf + " -------- \n"
          "baseline: " + str(max_u) + " (" + str(mean_u) + " pm " + str(ci_u) + ") \n"
          "vbsw: " + str(max_v) + " (" + str(mean_v) + " pm " + str(ci_v) + ") \n")

def stats_dl(dataset, testf):
    uni, vbsw, diff = read(dataset, testf, "vbsw")
    vbsw = np.max(vbsw, axis=1)
    N = vbsw.shape[0]
    mean_v = np.mean(vbsw)
    mean_u = np.mean(diff)
    max_v = np.max(vbsw)
    max_u = np.max(diff)
    ci_v = np.var(vbsw) ** 0.5 / N ** 0.5
    ci_u = np.var(diff) ** 0.5 / N ** 0.5
    print("for " + dataset + ": \n"
                             "-------- metric: " + testf + " -------- \n"
          "baseline: " + str(max_u) + " (" + str(mean_u) + " pm " + str(ci_u) + ") \n"
          "vbsw: " + str(max_v) + " (" + str(mean_v) + " pm " + str(ci_v) + ") \n")