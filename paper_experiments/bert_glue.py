import bert
import numpy as np
import sys
import os
import time

import tensorflow as tf
import tensorflow_datasets as tfds

syspath = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.insert(0, syspath)

from vbsw_module.algorithms.vbsw import vbsw
from vbsw_module.algorithms.training import training

dataset = sys.argv[-2]
case_name = "bert_paper_results_" + dataset
import datetime
if os.path.isdir(syspath + "/results/" + case_name):
    listdir = os.listdir(syspath + "/results/" + case_name)
    if len(listdir) > 0:
        if not os.path.isdir(syspath + "/results/" + case_name + "/old"):
            os.makedirs(syspath + "/results/" + case_name + "/old")
        today = str(datetime.datetime.today()).replace(" ", "")
        os.makedirs(syspath + "/results/" + case_name + "/old/" + today)
        for file in listdir:
            if file != "old":
                os.system("mv ~/scratch/results/" + case_name + "/" + file + " ~/scratch/results/" + case_name + "/old/" + today)


### Load the data
model_dir = syspath + "/data/saved_models/bert_model"
model_ckpt = os.path.join(model_dir, "bert_model.ckpt")

bert_params = bert.params_from_pretrained_ckpt(syspath + "/data/saved_models/bert_model")
max_seq_len = 512

l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

l_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')
l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')

# using the default token_type/segment id 0
output = l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
model = tf.keras.Model(inputs=l_input_ids, outputs=output)
model.build(input_shape=(None, max_seq_len))

bert.load_bert_weights(l_bert, model_ckpt)

data = tfds.load("glue/" + dataset)
train_data, test_data, val_data = data['train'], data['test'], data["validation"]

do_lower_case = True
bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
vocab_file = os.path.join(model_dir, "vocab.txt")
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


def read_glue(input_data, dataset):
    y = []

    input_ids_batch = []
    token_type_ids_batch = []
    for data in tfds.as_numpy(input_data):
        if dataset == "cola":
            input_str_1 = data["sentence"]
        else:
            input_str_1 = data["sentence1"]
            input_str_2 = data["sentence2"]
        y.append(data["label"])

        input_tokens_1 = tokenizer.tokenize(input_str_1)
        if dataset in ["cola"]:
            input_tokens = ["[CLS]"] + input_tokens_1 + ["[SEP]"]
        else:
            input_tokens_2 = tokenizer.tokenize(input_str_2)
            input_tokens = ["[CLS]"] + input_tokens_1 + ["[SEP]"] + input_tokens_2 + ["[SEP]"]

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids = input_ids + [0] * (max_seq_len - len(input_tokens))
        if dataset in ["cola"]:
            token_type_ids = [0] * len(input_tokens_1) + [0] + [1] * (max_seq_len - 1 - len(input_tokens_1))
        else:
            token_type_ids = [0] * len(input_tokens_1) + [0] * (max_seq_len - len(input_tokens_1))

        token_type_ids = token_type_ids[:max_seq_len]
        input_ids = input_ids[:max_seq_len]

        input_ids_batch.append(input_ids)
        token_type_ids_batch.append(token_type_ids)

    input_ids = np.array(input_ids_batch, dtype=np.int32)
    token_type_ids = np.array(token_type_ids_batch, dtype=np.int32)
    y = np.array(y)

    return input_ids, y, token_type_ids


x_train_init, y_train, ids_train = read_glue(train_data, dataset)
x_test_init, y_test, ids_test = read_glue(val_data, dataset)

"""

trunc = 1000
div_train = x_train_init.shape[0] // trunc
div_test = x_test_init.shape[0] // trunc
x_train = model.predict(x_train_init[:trunc])[:, 0, :]
x_test = model.predict(x_test_init[:trunc])[:, 0, :]

for i in range(1, div_train):
    x_train = np.concatenate([x_train, model.predict(x_train_init[trunc * i:trunc * (i + 1)])[:, 0, :]], axis=0)

for i in range(1, div_test):
    x_test = np.concatenate([x_test, model.predict(x_test_init[trunc * i:trunc * (i + 1)])[:, 0, :]], axis=0)
"""

x_train = model.predict(x_train_init)[:,0,:]
x_test = model.predict(x_test_init)[:,0,:]
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

dataset_list = ["mrpc", "stsb", "rte"]

ratio_dict = {"mrpc": 75,
              "stsb": 30,
              "rte": 20}

N_stat_dict = {"mrpc": 25,
               "stsb": 30,
               "rte": 10}

blr_dict = {"mrpc": [3e-4, 8],
            "stsb": [3e-4, 8],
            "rte": [3e-4, 16]}



if dataset == "rte":
    loss_function = "binary_crossentropy"
    test_losses = ["binary_accuracy"]
elif dataset == "mrpc":
    loss_function = "binary_crossentropy"
    test_losses = ["binary_accuracy", "F1"]
else:
    loss_function = "mse"
    test_losses = ["spearman_correlation", "pearson_correlation"]



### Train a linear model with VBSW
model_vbsw = vbsw(training_set=(x_train, y_train),
                  test_set=(x_test, y_test),
                  ratio=ratio_dict[dataset],
                  N_stat=N_stat_dict[dataset],
                  N_seeds=int(sys.argv[-1]),
                  N_layers=0,
                  N_units=0,
                  activation_hidden="",
                  activation_output="linear",
                  batch_size=blr_dict[dataset][1],
                  epochs=100,
                  optimizer="adam",
                  learning_rate=blr_dict[dataset][0],
                  loss_function=loss_function,
                  test_losses=test_losses,
                  keep_best=True,
                  saving_period=100,
                  case_name=case_name,
                  dataset=dataset)

### Train a linear model without VBSW
save_dir = syspath + "/results/" + case_name
res_name = "res_" + str(os.getpid()) + str(time.time())[:10]
save_file = os.path.join(save_dir, res_name)

model = training(training_set=(x_train, y_train),
                 test_set=(x_test, y_test),
                 N_seeds=int(sys.argv[-1]),
                 N_layers=0,
                 N_units=0,
                 activation_hidden="",
                 activation_output="linear",
                 batch_size=blr_dict[dataset][1],
                 epochs=100,
                 optimizer="adam",
                 learning_rate=blr_dict[dataset][0],
                 loss_function=loss_function,
                 test_losses=test_losses,
                 keep_best=True,
                 saving_period=100,
                 save_file=save_file)
