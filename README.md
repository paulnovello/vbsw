# Implementation of Variance Based Samples Weighting for Supervised Deep Learning

This code is associated to the submission of the paper "Variance Based Samples Weighting for Supervised Deep Learning". It has two purposes. First, you can reproduce the results of the paper using the script `experiments.py.` Second, you can use Variance Based Samples Weighting (VBSW) for your own test cases. 

## Requirements

The code has been tested on ubuntu 18, on a computer with a  Nvidia GTX1060 GPU. To run the code, you have to use anaconda to build an environment satisfying all the dependencies. In the root of the source_code folder, use

~~~shell
conda env create -f vbsw.yml
~~~

This environment includes tensorflow 2.1, which has its own requirements. These are:

* Cuda > 10.1
* Nvidia GPU driver > 418.x

Before executing any command, activate the environment using

~~~shell
conda activate vbsw
~~~

## Paper experiments

To reproduce most of the paper experiments results, you have to run the command:

~~~shell
python experiments.py [-h] [-i] [-n N_SEED] task case_study
~~~

Use 

~~~shell
python experiments.py -h
~~~

To see a quick indication of the options to specify and their effect.

### General usage

For most of the experiments, the following commands are sufficient. In the case of Cifar10 and MNIST, please read the dedicated parts before using it.

To launch the computations for a case study `CASE_STUDY` and for `N_SEED_VBSW` different random seeds:

~~~
python experiments.py launch CASE_STUDY -n N_SEED_VBSW
~~~

Then, you can read the results using:

~~~
python experiments.py read CASE_STUDY
~~~

`CASE_STUDY` and their default `N_SEED_VBSW` values are gathered in  the following table.

| `CASE_STUDY`  | double_moon | boston | mnist | cifar10 | stsb | mrpc | rte  | runge | tanh |
| ------------- | ----------- | ------ | ----- | ------- | ---- | ---- | ---- | ----- | ---- |
| `N_SEED_VBSW` | 10          | 10     | 10    | 10      | 50   | 50   | 50   | 50    | 50   |
| `N_SEED_INIT` | -           | -      | 1     | 1       | -    | -    | -    | -     | -    |

 Runge and tanh are experiments of the appendix.

### Cifar10 and MNIST

For these experiments, you need to first train a network before being able to apply VBSW. A ResNet20 and a LeNet 5 are implemented for Cifar10 and MNIST respectively, like in the paper. To train `N_SEED_INIT` initial networks on `CASE_STUDY`, use:

~~~
python experiments.py launch CASE_STUDY -i -n N_SEED_INIT 
~~~

Then you can use the General Usage commands. Note that unlike other experiments, VBSW will then be applied to `N_SEED_INIT` networks for `N_SEED_VBSW` different random seeds. The complete experiments for Cifar10 and MNIST is way more computationally costly than the other.

### Plots

You have the possibility to generate and display most of the plots that can be found in the paper and the appendix. The plots are those of double moon and runge/tanh experiments. 

#### double moon

You first have to initialize computations required to produce the plots using

~~~
python experiments.py plot double_moon -i -n N_SEED_DM
~~~

Since every training is different and Figures 1b and 1d depends on one training, you can specify `N_SEED_DM` to produce `N_SEED_DM` different plots. You can display all the obtained plots using

~~~
python experiments.py plot double_moon 
~~~

Among the `N_SEED_DM` plots produced, only the last will be displayed. If you want to see all the plots produced, you can find it in `./figures/double_moon/decision_boundar`. You can use the first command to create new plots. Be careful that the plots will be stacked, which could lead to memory problems. You can remove the plots using 

~~~
rm ./figures/double_moon/decision_boundary/*
~~~

This will remove all the plots and you will have to use the first command again to produce new ones.

#### TBS: runge and tanh

In this case, you can use the commands

~~~
python experiments.py plot CASE_STUDY -i 
~~~

and

~~~
python experiments.py plot CASE_STUDY
~~~

safely, because no training is involved. At each execution using `-i`, the plots will be overwritten.

## Using VBSW for other experiments

You can use VBSW by yourself, for any case study. Implementation example scripts can be found in the root folder: `example_vbsw.py` and `example_vbsw_dl.py`. The only requirement is that the content of `vbsw_module` folder needs to be in `PYTHON_PATH`.  Note that in `example_vbsw.py` and `example_vbsw_dl.py`, the number of epochs has been set very low for the user to quickly test the implementation.

