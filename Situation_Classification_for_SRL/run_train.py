# -*- coding:utf-8 -*-
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from transformers import BertTokenizer, RobertaTokenizer
from tensorflow.compat.v1 import reset_default_graph
from tf_bert_base_uncase import *
from __init__ import BertBaseUnCaseV1, score_direct_for_triple, TrainModelConfig, PATH_TRANS_INPUT

import os
import re


def train_process(fold_count):
    test_count = 't' + str(fold_count)
    if model_type in TrainModelConfig.BERT_LIST:
        train, _ = encode_examples_bert(train_data.loc[train_index], tokenizer, model_type)
        val, _ = encode_examples_bert(train_data.loc[val_index], tokenizer, model_type)
    elif (model_type == "roberta-base") or (model_type == "roberta-large"):
        train, _ = encode_examples_roberta(train_data.loc[train_index], tokenizer, model_type)
        val, _ = encode_examples_roberta(train_data.loc[val_index], tokenizer, model_type)

    ds_train_encoded = train.batch(BertBaseUnCaseV1.BATCH_SIZE)
    ds_val_encoded = val.batch(BertBaseUnCaseV1.BATCH_SIZE)

    print('training {}/{} ...'.format(fold_count, BertBaseUnCaseV1.FOLD_NUM))

    model_ver = BuildModels(BertBaseUnCaseV1.m_ver, model_type)
    model = model_ver.build_model_1(verbose=BertBaseUnCaseV1.SHOW_MODE)
    # training the model
    train_model(model, ds_train_encoded, ds_val_encoded, test_count, model_type, checkpoint_path)
    # loss_temp, acc_temp = model.evaluate(ds_test_encoded)
    # print("In training, temp_loss: {}; temp_acc: {}".format(loss_temp, acc_temp))

    K.clear_session()
    del(model, model_ver)
    gc.collect()


def test_process(i):
    global test_result
    test_count = 't' + str(i)
    print('evaluating {}/{} ...'.format(i, BertBaseUnCaseV1.FOLD_NUM))
    model_ver = BuildModels(BertBaseUnCaseV1.m_ver, model_type)
    model = model_ver.build_model_1(verbose=BertBaseUnCaseV1.SHOW_MODE)
    model.load_weights('{}{}{}.h5'.format(checkpoint_path, BertBaseUnCaseV1.m_ver, test_count))
    model.evaluate(ds_test_encoded)
    test_result += model.predict(ds_test_encoded)

    K.clear_session()
    del model
    gc.collect()


if __name__ == '__main__':
    # load the dataset
    print("loading the dataest ...")

    # gpu_config()
    for is_train in TrainModelConfig.TRAIN_OR_TEST:  # train all the models and then evaluate them.
        for model_type in TrainModelConfig.TRANS_MODEL_LIST:
            if BertBaseUnCaseV1.VER == "v5":
                df_train_load = triple_label_v3(pd.read_csv(BertBaseUnCaseV1.PATH), model_type)
            else:
                df_train_load = triple_label_v2(pd.read_csv(BertBaseUnCaseV1.PATH), model_type)
            # else:
            #     df_train_load = triple_label_v1(pd.read_csv(BertBaseUnCaseV1.PATH))

            # values_count = df_train_load['tri_label'].value_counts()
            # print("Literal, rq, sarcasm count:", values_count)

            bert_init = BertBaseUnCaseV1(model_type, BertBaseUnCaseV1.VER)
            model_name, trans_path, version, checkpoint_path = bert_init.model_init()
            train_data, test_data = split_dataset(df_train_load)

            # get tokenizer
            if model_type in TrainModelConfig.BERT_LIST:
                tokenizer = BertTokenizer.from_pretrained("{}{}".format(PATH_TRANS_INPUT, model_type),
                                                          do_lower_case=True)
                test, test_label = encode_examples_bert(test_data, tokenizer, model_type)
            elif model_type in TrainModelConfig.ROBERTA_LIST:
                tokenizer = RobertaTokenizer.from_pretrained("{}{}".format(PATH_TRANS_INPUT, model_type),
                                                             lowercase=True, add_prefix_space=True)
                test, test_label = encode_examples_roberta(test_data, tokenizer, model_type)

            ds_test_encoded = test.batch(BertBaseUnCaseV1.BATCH_SIZE)

            print("Starting training ... ")

            for each in TrainModelConfig.MODEL_LIST:
                test_result = np.zeros((len(test_label), BertBaseUnCaseV1.N_CLASS))
                BertBaseUnCaseV1.m_ver = each
                print(model_type, BertBaseUnCaseV1.m_ver)
                # processs = []
                if is_train == 0:
                    fold_count = 0
                    # use k-fold to split train set.
                    kf = KFold(n_splits=BertBaseUnCaseV1.FOLD_NUM)
                    for train_index, val_index in kf.split(train_data):
                        fold_count += 1
                        train_process(fold_count)
                else:
                    for i in range(1, BertBaseUnCaseV1.FOLD_NUM + 1):
                        # evaluate the model and predict the test dataset
                        test_process(i)

                    nd_pre_y = test_result.argmax(axis=1)
                    score_direct_for_triple(nd_pre_y, test_label, model_type)
