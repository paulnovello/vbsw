import argparse
import os
import sys

syspath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, syspath )
print(syspath)
from vbsw_module.io.saving_reading import stats, stats_dl

parser = argparse.ArgumentParser()
parser.add_argument("task", help="chose between launching an experiment, reading results or displaying plots. Possible values: "
                                 "launch, read, plot")
parser.add_argument("case_study", help="the case study to launch or to read. Possible_values: "
                                       "cifar10, mnist, double_moon, boston, rte, stsb, mrpc, runge, tanh")

parser.add_argument("-i", "--init", help="if task=launch: initial training of DNN on mnist or cifar10. "
                                         "if task=plot: initialize new plots for tbs and double_moon", action="store_true")
parser.add_argument("-n", "--n_seed", help="number of random seeds to use for a \"launch\" task or for double moon plots")
args = parser.parse_args()

if args.task == "launch":
    if args.n_seed:
        if args.init:
            for i in range(int(args.n_seed)):
                exec(open(syspath + "/init_scripts/" + args.case_study + "_init.py").read())
        elif args.case_study in ["runge", "tanh"]:
            case_name = "tbs" + args.case_study + "1-1"
            if os.path.isdir(syspath + "/results/" + case_name):
                if len(os.listdir(syspath + "/results/" + case_name)) > 1:
                    if not os.path.isdir(syspath + "/results/" + case_name + "/old"):
                        os.makedirs(syspath + "/results/" + case_name + "/old")
                    os.system(
                        "mv " + syspath + "/results/" + case_name + "/* " + syspath + "/results/" + case_name + "/old")
            os.system("python paper_experiments/tbs.py " + args.case_study + " tbs " + args.n_seed)
            os.system("python paper_experiments/tbs.py " + args.case_study + " None " + args.n_seed)
        elif args.case_study in ["cifar10", "mnist", "double_moon", "boston"]:
            os.system("python paper_experiments/" + args.case_study + ".py " + args.n_seed)
        elif args.case_study in ["stsb", "rte", "mrpc"]:
            os.system("python paper_experiments/bert_glue.py " + args.case_study + " " + args.n_seed)
        else:
            print("Invalid case_study argument: " + args.case_study)
    else:
        print("No specified values for n_seed. Choosing default values (see README).")
        if args.init:
            n_seed = "1"
            for i in range(int(n_seed)):
                exec(open(syspath + "/init_scripts/" + args.case_study + "_init.py").read())
        elif args.case_study in ["runge", "tanh"]:
            n_seed = "50"
            os.system("python paper_experiments/tbs.py " + args.case_study + " tbs " + n_seed)
            os.system("python paper_experiments/tbs.py " + args.case_study + " None " + n_seed)
        elif args.case_study in ["cifar10", "mnist", "double_moon", "boston"]:
            n_seed = "10"
            os.system("python paper_experiments/" + args.case_study + ".py " + n_seed)
        elif args.case_study in ["stsb", "rte", "mrpc"]:
            n_seed = "50"
            os.system("python paper_experiments/bert_glue.py " + args.case_study + " " + n_seed)
        else:
            print("Invalid case_study argument: " + args.case_study)

if args.task == "plot":
    if args.n_seed is None:
        print("No specified values for n_seed. Choosing default values (see README).")
    if args.case_study == "double_moon":
        list_fig = os.listdir(syspath + "/figures/double_moon/decision_boundary")
        n_seed = args.n_seed if args.n_seed else "1"
        if (len(list_fig)) < 3 or (args.init):
            os.system("python misc/plots_double_moon.py " + n_seed)
        os.system("display " + syspath + "/figures/double_moon/data.png")
        os.system("display " + syspath + "/figures/double_moon/weighted_data.png")
        pic_list = os.listdir(syspath + "/figures/double_moon/decision_boundary")
        n = len(pic_list)
        print(str(n // 2) + " images found for baseline and VBSW, plotting the last ones")
        os.system("display " + syspath + "/figures/double_moon/decision_boundary/unif" + str(n // 2 - 1) + ".png")
        os.system("display " + syspath + "/figures/double_moon/decision_boundary/vbsw" + str(n // 2 - 1) + ".png")
    elif args.case_study in ["runge", "tanh"]:
        list_fig = os.listdir(syspath + "/figures/tbs")
        if (len(list_fig) < 3) or (args.init):
            os.system("python misc/plots_tbs.py")
        os.system("display " + syspath + "/figures/tbs/" + args.case_study + "-sample.png")
    else:
        print("No plots for " + args.case_study + " case study")

if args.task == "read":
    if args.case_study in ["runge", "tanh"]:
        print("Results for " + args.case_study +": \n")
        os.system("python " + syspath + "/paper_experiments/extraction_tbs.py " + args.case_study)
    elif args.case_study == "double_moon":
        print("Results for " + args.case_study +": \n")
        os.system("python " + syspath + "/paper_experiments/extraction_dm.py" )
    elif args.case_study in ["cifar10", "mnist"]:
        print("Results for " + args.case_study + ": \n")
        stats_dl(syspath + "/results/" + args.case_study + "_paper_results", "categorical_accuracy")
    elif args.case_study == "rte":
        print("Results for " + args.case_study + ": \n")
        stats(syspath + "/results/bert_paper_results_" + args.case_study, "binary_accuracy")
    elif args.case_study == "mrpc":
        print("Results for " + args.case_study + ": \n")
        stats(syspath + "/results/bert_paper_results_" + args.case_study, "binary_accuracy")
        stats(syspath + "/results/bert_paper_results_" + args.case_study, "F1")
    elif args.case_study == "stsb":
        print("Results for " + args.case_study + ": \n")
        stats(syspath + "/results/bert_paper_results_" + args.case_study, "spearman_correlation")
        stats(syspath + "/results/bert_paper_results_" + args.case_study, "pearson_correlation")
    elif args.case_study == "boston":
        print("Results for " + args.case_study + ": \n")
        stats(syspath + "/results/" + args.case_study + "_paper_results", "mse")
    else:
        print("Invalid case_study argument: " + args.case_study)


