# -*- coding:utf-8 -*-
import re
from os import makedirs, getcwd
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix


"""------------------------"""
"""RQ generation config"""
"""for rq detection"""
PATH_ROOT = getcwd()
PATH_DATASET_RQ = PATH_ROOT + "/data/"
PATH_PIC_RQ = PATH_ROOT + "/pictures/"
PATH_TRANS_INPUT = PATH_DATASET_RQ + "../../Situation_Classification_for_SRL/data/input/"

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


class TrainModelConfigV2:
    """
    Transformers models(TRANS_MODEL_LIST):
    ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]

    fine-tuning model(MODEL_LIST): ['v1', 'v2']
    v1: normal
    v2: lstm

    TRAIN_OR_TEST:
    0: train the models;
    1: test the models
    """
    TRANS_MODEL_LIST = ["bert-base-uncased"]
    ROBERTA_LIST = ["roberta-base", "roberta-large"]
    BERT_LIST = ["bert-base-uncased", "bert-large-uncased"]
    MODEL_LIST = ['v1']
    TRAIN_OR_TEST = [1]


class PredictContextV2(object):
    """
    predict result init
    version same as BertBaseUncaseV2
    """
    PSD_VERSION = 'v8'
    VER = 'v6'
    M_VER = 'v1'
    FOLD_NUM = 5
    MODEL_COUNT = 't1'
    MODEL_NAME = 'bert-base-uncased'
    TRANS_PATH = PATH_TRANS_INPUT + MODEL_NAME
    VERSION = MODEL_NAME + '-' + PSD_VERSION + VER
    checkpoint_path = PATH_DATASET_RQ + MODEL_NAME + '-model/' + VERSION


class BertBaseUnCaseV2:
    """
    training models
    v5: only for rq detection; N_CLASS = 2; lr = 5e-5
    """
    VER = 'v6'
    m_ver = ''
    FOLD_NUM = 5
    PATH = PATH_DATASET_RQ + 'sarcasm_merge_triple_{}.csv'.format(PSD_VERSION)

    N_CLASS = 2
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

        if not exists(PATH_DATASET_RQ + model_name + '-model/'):
            makedirs(PATH_DATASET_RQ + model_name + '-model/')
        checkpoint_path = PATH_DATASET_RQ + model_name + '-model/' + version
        return model_name, trans_path, version, checkpoint_path

    # Set SHOW_MODE True if you want to show model summary
    SHOW_MODE = True
    if SHOW_MODE:
        SHOW_MODE_NUM = 1
    else:
        SHOW_MODE_NUM = 0


class SentimentClassifier:
    PATH_ROOT = getcwd()
    DATA_PATH = PATH_ROOT + "/data/"
    TRAIN_DATA_PATH = DATA_PATH + "sentiment_data.csv"
    embed_dim = 128
    max_features = 20000
    maxlen = 200
    model_version = "v4"  # v2: maxlen=30 v3: maxlen=60 v4: neg pos
    batch_size = 32
    epochs = 10
    token_file_path = "token_" + model_version
    model_name = "dummy_model_" + model_version + "_embed_" + str(embed_dim) + "_batch_size_" + str(batch_size)
    SAVE_FEATURES_PREFIX = model_name + "_features_"
    feature_dir = "../features/"
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                  "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                  "these",
                  "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                  "do",
                  "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                  "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
                  "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                  "again",
                  "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "both",
                  "each",
                  "few", "more", "most", "other", "some", "such", "own", "same", "so",
                  "than",
                  "too", "very", "s", "t", "can", "will", "just", "should", "now", "only"]
    denial_words = ["any", "no", "nor", "not"]


class LoadCommonSense:
    PATH_ROOT = getcwd()
    DIC_TXT_PATH = PATH_ROOT + "/data/train600k.txt"
    DIC_DF_PATH = PATH_ROOT + "/data/commonsense_dataset.csv"
    DIC_DF_PATH_2 = PATH_ROOT + "/data/commonsense_dataset_merged.csv"
    DIC_PATH = PATH_ROOT + "/data/ckbc-demo/Bilinear_cetrainSize300frac1.0dSize200relSize150acti0.001.1e-05.800.RAND.tanh.txt19.pickle"
    STF_PATH = PATH_ROOT + "/data/stanford-parser-full-2020-11-17/"

    CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    left_to_right = {"ReceivesAction": "is", "AtLocation": "in", "HasA": "has a", "IsA": "is a",
                     "NotCapableOf": "can not", "Causes": "makes people", "CausesDesire": "make people want to",
                     "HasProperty": "is", "Desires": "want to", "InheritsFrom": "is inherited from",
                     "CapableOf": "can", "NotIsA": "is not a", "NotHasProperty": "is not",
                     "NotDesires": "do not want to", "DesireOf": "have a desire of", "LocationOfAction": "in",
                     "MotivatedByGoal": "because", "PartOf": "is a part of", "MadeOf": "is made of",
                     "RelatedTo": "is related to", "DefinedAs": "is", "NotHasA": "does not have",
                     "SymbolOf": "is a symbol of", "CreatedBy": "is created by", "UsedFor": "is used of"}
    right_to_left = {"HasSubevent": "to", "HasPrerequisite": "will", "HasLastSubevent": "after",
                     "HasFirstSubevent": "before", "InstanceOf": "such as"}
    not_use = {"LocatedNear", "HasPainCharacter", "HasPainIntensity", "NotMadeOf"}


class GrammaticalErrorCorrection:
    PATH_ROOT = getcwd()
    CODE_PATH = PATH_ROOT + "/bert-gec/scripts/"
    OUTPUT_PATH = PATH_ROOT + "/bert-gec/bert-fuse/output/"


class NltkNer:
    ner_dic = {"Who": ["PERSON"], "Where": ["NORP", "FAC", "ORG", "GPE", "LOC"],
               "What": ["PRODUCT", "WORK_OF_ART", "LAW"], "What event": ["EVENT"], "What language": ["LANGUAGE"],
               "When": ["DATE", "TIME"]}


def model_predict_scores(model, nd_test_x, nd_true_y):
    result = model.predict(nd_test_x, verbose=1)
    nd_pre_y = result.argmax(axis=1)
    print(classification_report(nd_true_y, nd_pre_y))
    return 0


def model_predict_scores_for_triple(model, nd_test_x, nd_true_y, test_count):
    result = model.predict(nd_test_x, verbose=1)
    nd_pre_y = result.argmax(axis=1)
    print(classification_report(nd_true_y, nd_pre_y))
    confusion = confusion_matrix(nd_true_y, nd_pre_y)
    plot_confusion_matrix_for_rq(confusion, ['Literal', 'RQ', 'Sarcasm'], test_count)


def score_direct_for_triple(nd_test_y, nd_true_y, model_type):
    print(classification_report(nd_true_y, nd_test_y))
    confusion = confusion_matrix(nd_true_y, nd_test_y)
    plot_confusion_matrix_for_rq(confusion, ['Literal', 'RQ', 'Sarcasm'], model_type)


def plot_confusion_matrix_for_rq(cm, class_names, model_type, *args):
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
        figure.savefig('{}{}confusion_matrix_{}{}{}{}.png'.format(PATH_PIC_RQ, model_type, PSD_VERSION, BertBaseUnCaseV2.VER,
                                                                BertBaseUnCaseV2.m_ver, args))
    else:
        figure.savefig('{}{}confusion_matrix_{}{}{}.png'.format(PATH_PIC_RQ, model_type, PSD_VERSION, BertBaseUnCaseV2.VER,
                                                              BertBaseUnCaseV2.m_ver))


def expand_contractions(sentence, contraction_mapping):
    """convert contraction phrases to normal phrases"""
    sentence = re.sub(r' n\'t', 'n\'t', sentence)
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


def clean_text(sentence, move_stopwords=True):
    utt = re.sub(r"[^\w ']", "", sentence).lower()
    expanded_corpus = expand_contractions(utt, LoadCommonSense.CONTRACTION_MAP)

    utt_list = word_tokenize(expanded_corpus)
    if move_stopwords:
        out_put = []
        for word in utt_list:
            if word in SentimentClassifier.stop_words:
                out_put.append("<del>")
            elif word in SentimentClassifier.denial_words:
                out_put.append("<den>")
            else:
                out_put.append(word)

        return out_put
    else:
        return utt_list


def analyzing_sentence_structure(sentence):
    scp = StanfordParser(path_to_jar=LoadCommonSense.STF_PATH + "stanford-parser.jar",
                         path_to_models_jar=LoadCommonSense.STF_PATH + "stanford-parser-4.2.0-models.jar")
    sentence = " ".join(clean_text(sentence, False))
    try:
        result = list(scp.raw_parse(sentence))
    except BaseException:
        return {"subject": "", "verb": "", "object": ""}, sentence, 1
    tree_result = result[0].subtrees()
    sentence_json = {"subject": "", "verb": "", "object": ""}
    temp_object = []
    for each in tree_result:
        flag = 0
        tree_label = each.label()

        find_verb = re.findall(r"VB.*", tree_label)
        if tree_label == "NP" and flag != 1:
            flag += 1
            sentence_json["subject"] = each.leaves()
        elif find_verb:
            sentence_json["verb"] = each.leaves()
            break
        elif tree_label == 'ROOT':
            temp_object = each.leaves()

    if not sentence_json["subject"]:
        sentence_json["subject"] = ["i"]
        temp_object = sentence_json["subject"] + temp_object
        sentence_new = "i " + sentence
    else:
        sentence_new = sentence

    error = 0
    try:
        sentence_json["object"] = [word for word in temp_object if
                                   word not in (sentence_json["subject"] + sentence_json["verb"])]
    except BaseException:
        error = 1

    return sentence_json, sentence_new, error


def wash_sentence(utterances):
    """
    wash the sentences
    :param utterances:
    :return: normal_list, cleaned_list, tree_list
    """
    utt_list = clean_text(utterances)
    analyze_result, sentence_new, error = analyzing_sentence_structure(utterances)

    print("wash sentence:", sentence_new, utt_list, analyze_result)
    return sentence_new, utt_list, analyze_result, error

