# -*- coding:utf-8 -*-
import logging
import os
import pickle

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import warnings

warnings.filterwarnings('ignore')

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras_self_attention import SeqSelfAttention
from __init__ import SentimentClassifier2


def find_sentiment(input_list):
    complete_token_path = SentimentClassifier2.DATA_PATH
    with open(complete_token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    maxlen = SentimentClassifier2.maxlen
    X = tokenizer.texts_to_sequences([input_list])
    X = pad_sequences(X, maxlen=maxlen)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    model = load_model(SentimentClassifier2.MODEL_PATH, custom_objects={'SeqSelfAttention': SeqSelfAttention})
    result = model.predict(X)  # neg: 1; pos: 0

    return result