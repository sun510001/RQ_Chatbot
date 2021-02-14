# -*- coding:utf-8 -*-

from os import getcwd


PATH_ROOT = getcwd()
PATH_RQ = PATH_ROOT + "/RQ_generator/"
PATH_LG = PATH_ROOT + "/Literal_genrator/"
PATH_SG = PATH_ROOT + "/SarcasmGeneration-ACL2020/"
PATH_SRL = PATH_ROOT + "Situation_Classification_for_SRL"

PATH_PYTHON = '/home/aquamarine/sunqifan/anaconda3/envs/r_cla/bin/python3.6'

"""----------------------"""
"""Literal generation config"""
class LoadDialogueDataset:
    PATH_ROOT = getcwd()
    DIA_PATH = PATH_ROOT + "/Literal_generator/ijcnlp_dailydialog/dialogues_text.txt"
    DIA_JSON_PATH = PATH_ROOT + "/Literal_generator/ijcnlp_dailydialog/dialogues_text.json"
    SARC_DATASET_PATH = PATH_ROOT + "/Situation_Classification_for_SRL/data/sarcasm_merge_triple_v8.csv"


class SentimentClassifier2:
    PATH_ROOT = getcwd()
    DATA_PATH = PATH_ROOT + "/RQ_generator/data/token_v4.pickle"
    MODEL_PATH = PATH_ROOT + "/RQ_generator/data/dummy_model_v4_embed_128_batch_size_32_final.hdf5"
    embed_dim = 128
    max_features = 20000
    maxlen = 60
    model_version = "v4"  # v2: maxlen=30 v3: maxlen=60 v4: neg pos
    batch_size = 32
    epochs = 10
    token_file_path = "token_" + model_version
    model_name = "dummy_model_" + model_version + "_embed_" + str(embed_dim) + "_batch_size_" + str(batch_size)
    SAVE_FEATURES_PREFIX = model_name + "_features_"
    feature_dir = "./features"

