# -*- coding:utf-8 -*-

import torch
from os import makedirs, getcwd
from os.path import exists
from __init__ import PATH_TRANS_INPUT, TrainModelConfig, PATH_DATASET, CONTEXT_LENGTH, PSD_VERSION, IS_BALANCE
from process_sarc_data import load_rq_unbalance_dataset, load_rq_balance_dataset_response, load_rq_unbalance_dataset_v8
from convert_torch_to_tf import Convert_Model


def dl_models():
    """download transformers pre-training models"""
    print("a")
    for each in TrainModelConfig.BERT_LIST:
        if not exists(PATH_TRANS_INPUT + each):
            makedirs(PATH_TRANS_INPUT + each)
        if not exists(PATH_TRANS_INPUT + each + "/config.json"):
            Convert_Model.dl_bert(each, PATH_TRANS_INPUT)
        else:
            print("Download failed, because {} exists in {}".format(each, PATH_TRANS_INPUT))
    for each in TrainModelConfig.ROBERTA_LIST:
        if not exists(PATH_TRANS_INPUT + each):
            makedirs(PATH_TRANS_INPUT + each)
        if not exists(PATH_TRANS_INPUT + each + "/config.json"):
            Convert_Model.dl_roberta(each, PATH_TRANS_INPUT)
        else:
            print("Download failed, because {} exists in {}".format(each, PATH_TRANS_INPUT))


if __name__ == '__main__':
    """download transformers pre-training models"""
    dl_models()

    """pre-processing the raw dataset"""
    # load_data_merge_triple(PATH_DATASET, CONTEXT_LENGTH, PSD_VERSION, IS_BALANCE)
    # load_rq_unbalance_dataset(PATH_DATASET, PSD_VERSION, IS_BALANCE)
    # load_rq_balance_dataset_response(PATH_DATASET, PSD_VERSION, IS_BALANCE)
    # load_rq_unbalance_dataset_v8(PATH_DATASET, PSD_VERSION, 0)