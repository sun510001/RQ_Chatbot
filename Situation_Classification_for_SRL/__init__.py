# -*- coding:utf-8 -*-

from os import makedirs, getcwd
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, \
    confusion_matrix


"""-------------------------"""
"""Situation classification for srl config"""
"""preprocessing configures"""
PATH_ROOT = getcwd()
PATH_DATASET = PATH_ROOT + "/data/"
PATH_PIC = PATH_ROOT + "/pictures/"
PATH_TRANS_INPUT = PATH_DATASET + "/input/"
if not exists(PATH_DATASET):
    makedirs(PATH_DATASET)
if not exists(PATH_TRANS_INPUT):
    makedirs(PATH_TRANS_INPUT)


class TrainModelConfig:
    """
    Transformers models(TRANS_MODEL_LIST):
    ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]

    fine-tuning model(MODEL_LIST): ['v1', 'v2']
    v1: normal
    v2: lstm
    v3: dense(hidden_size)
    v4: bilstm x2 + dense

    TRAIN_OR_TEST:
    0: train the models;
    1: test the models
    """
    TRANS_MODEL_LIST = ["roberta-base"]
    ROBERTA_LIST = ["roberta-base", "roberta-large"]
    BERT_LIST = ["bert-base-uncased", "bert-large-uncased"]
    MODEL_LIST = ['v4']
    TRAIN_OR_TEST = [0, 1]


"""
preprocessing sarcasm dataset from Twitter and Reddit
PSD_VERSION:
v1: context_length=2, join two context together;
v2: context_length=3, join three context together;
v3: unbalance dataset, context_length=2
v4: merge train and test dataset
v5: add rq annotations which are made by mturk workers; unbalanced dataset
v6: balance dataset
v7: v6 add response
v8: v6 split response, last context, last2 context to each columns
"""
CONTEXT_LENGTH = 2
PSD_VERSION = "v8"
IS_BALANCE = 1


class PredictContext:
    """
    predict result init
    version same as BertBaseUncaseV1
    """
    PSD_VERSION = "v8"
    VER = 'v7'
    M_VER = 'v4'
    FOLD_NUM = 5
    MODEL_COUNT = 't1'
    MODEL_NAME = 'roberta-base'
    TRANS_PATH = PATH_TRANS_INPUT + MODEL_NAME
    VERSION = MODEL_NAME + '-' + PSD_VERSION + VER
    checkpoint_path = PATH_DATASET + MODEL_NAME + '-model/' + VERSION


class BertBaseUnCaseV1:
    """
    training models
    v1v1: bert-base-uncase simple (TFBertModel;Flatten;Dropout0.2;Dense),
    v1v2: bert-base-uncase simple (TFBertModel;LSTM(100, dropout=0.2, recurrent_dropout=0.2);Dense),
    [x]v1v3: bert-base-uncase simple (TFBertModel;LSTM(100, dropout=0.2, recurrent_dropout=0.2)x2;Dense),
    v1v4: model same to v1v1, fix the bug of mistaking for attention mask and token type id.
    v1v5: model same to v1v2, fix the bug of mistaking for attention mask and token type id.
    v3v1-2: max_len=200, test for training response only
    v4v1-2: some test for new model; lr = 5e-5; N_CLASS = 3;
    v7: new lstm
    """

    VER = 'v7'
    m_ver = ''
    FOLD_NUM = 5
    PATH = PATH_DATASET + 'sarcasm_merge_triple_{}.csv'.format(PSD_VERSION)

    N_CLASS = 3
    lr = 5e-5
    SEED = 1830126
    BATCH_SIZE = 32
    MAXLEN = 200
    EPOCHS = 12
    hidden_size = 768

    def __init__(self, model_type, ver):
        self.model_type = model_type
        self.ver = ver

    def model_init(self):
        model_name = self.model_type
        trans_path = PATH_TRANS_INPUT + model_name
        version = model_name + '-' + PSD_VERSION + self.ver

        if not exists(PATH_DATASET + model_name + '-model/'):
            makedirs(PATH_DATASET + model_name + '-model/')
        checkpoint_path = PATH_DATASET + model_name + '-model/' + version
        return model_name, trans_path, version, checkpoint_path

    # Set SHOW_MODE True if you want to show model summary
    SHOW_MODE = True
    if SHOW_MODE:
        SHOW_MODE_NUM = 1
    else:
        SHOW_MODE_NUM = 0
    # epsilon = 1e-08


def plot_confusion_matrix(cm, class_names, model_type, *args):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    plt.clf()
    plt.close('all')
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    labels = cm

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if args:
        figure.savefig('{}{}confusion_matrix_{}{}{}{}.png'.format(PATH_PIC, model_type, PSD_VERSION, BertBaseUnCaseV1.VER,
                                                                BertBaseUnCaseV1.m_ver, args))
    else:
        figure.savefig('{}{}confusion_matrix_{}{}{}.png'.format(PATH_PIC, model_type, PSD_VERSION, BertBaseUnCaseV1.VER,
                                                              BertBaseUnCaseV1.m_ver))
    # return figure


# def plot_heatmap(cm, class_names, *args):
#     if args:
#         sns.heatmap(cm)


def model_predict_scores(model, nd_test_x, nd_true_y):
    result = model.predict(nd_test_x, verbose=1)

    nd_pre_y = result.argmax(axis=1)
    # # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(nd_true_y, nd_pre_y)
    # # precision tp / (tp + fp)
    # precision = precision_score(nd_true_y, nd_pre_y, pos_label=1)
    # # recall: tp / (tp + fn)
    # recall = recall_score(nd_true_y, nd_pre_y, pos_label=1)
    # # f1: 2 tp / (2 tp + fp + fn)
    # f1 = f1_score(nd_true_y, nd_pre_y)

    # print('accuracy: {0:.2f}  Precision: {1:.2f}  Recall: {2:.2f}  F1: {3:.2f}'.format(accuracy, precision, recall, f1))
    print(classification_report(nd_true_y, nd_pre_y))
    return 0


def model_predict_scores_for_triple(model, nd_test_x, nd_true_y, test_count):
    result = model.predict(nd_test_x, verbose=1)
    nd_pre_y = result.argmax(axis=1)
    print(classification_report(nd_true_y, nd_pre_y))
    confusion = confusion_matrix(nd_true_y, nd_pre_y)
    plot_confusion_matrix(confusion, ['Literal', 'RQ', 'Sarcasm'], test_count)


def score_direct_for_triple(nd_test_y, nd_true_y, model_type):
    print(classification_report(nd_true_y, nd_test_y))
    confusion = confusion_matrix(nd_true_y, nd_test_y)
    plot_confusion_matrix(confusion, ['Literal', 'RQ', 'Sarcasm'], model_type)
