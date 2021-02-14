# -*- coding:utf-8 -*-
import os
import pickle

import re
import nltk
import numpy as np
import yaml
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras_self_attention import SeqSelfAttention
from __init__ import SentimentClassifier, analyzing_sentence_structure


# nltk.download('vader_lexicon')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def loadConfigForROV():
    with open("./data/config.yaml") as f:
        docs = yaml.load_all(f, Loader=yaml.FullLoader)
        for doc in docs:
            for k, v in doc.items():
                if k == "exception_vadarneg_words":
                    exception_vadarneg_words = v
                elif k == "missing_vadarneg_words":
                    missing_vadarneg_words = v
                elif k == "connectives_addition_of_ideas":
                    connectives_addition_of_ideas = v
    return exception_vadarneg_words, missing_vadarneg_words, connectives_addition_of_ideas


def getWordNetAntonyms():
    m = {}
    for line in open('./data/antonyms.txt'):
        m[line.strip().split()[0]] = line.strip().split()[1]
    return m


def findIfnegationPresent(utterance):
    words = utterance.split()
    for w in words:
        if w == 'not' or w == 'never' or w == 'Not' or w == 'Never':
            return w, True
    return '', False


def findIfendingwithnt(utterance):
    d = {"didn't": "did", "don't": "do", "doesn't": "does", "can't": "can",
         "cannot": "can", "wouldn't": "would", "shouldn't": "should"}
    words = utterance.split()
    for w in words:
        if w in d:
            return w, d[w], True
        if w.lower() in d:
            return w, d[w.lower()], True
    return '', '', False


def get_antonym(word):
    antonyms = getWordNetAntonyms()
    if word.lower() not in antonyms:
        synonymsset = []
        antonymsset = []
        for syn in wn.synsets(word.lower()):
            for l in syn.lemmas():
                synonymsset.append(l.name())
                if l.antonyms():
                    antonymsset.append(l.antonyms()[0].name())
        if len(antonymsset) == 0:
            for w in synonymsset:
                if w in antonyms:
                    return antonyms[w.lower()]
            return "not " + word
        else:
            return antonymsset[0]
    else:
        return antonyms[word.lower()]


def bi_neg_involve(utterance):
    exception_vadarneg_words, missing_vadarneg_words, connectives_addition_of_ideas = loadConfigForROV()
    utterance = utterance.replace(',', '')
    sid = SentimentIntensityAnalyzer()
    arr = []
    sent = word_tokenize(utterance)
    for i in range(len(sent)):
        w = sent[i]
        if w == 'no':
            continue
        ss = sid.polarity_scores(w)
        # TODO cannot reverse neutral words when connective addition of ideas are exist
        if (ss['neg'] == 1.0 or w in missing_vadarneg_words) and (w not in exception_vadarneg_words):
            arr.append((w, i, abs(ss['compound'])))
    # check the words which between two negative words are connectives or not

    if len(arr) == 2:
        check_words_list = sent[arr[0][1] + 1:arr[1][1]]
        check = any(ele in check_words_list for ele in connectives_addition_of_ideas)
        if check:
            return [arr[0][0], arr[1][0]], True
        else:
            return [arr[0][0]], True
    else:
        return [], False


def bi_pos_involve(utterance):
    exception_vadarneg_words, missing_vadarneg_words, connectives_addition_of_ideas = loadConfigForROV()
    utterance = utterance.replace(',', '')
    sid = SentimentIntensityAnalyzer()
    arr = []
    sent = word_tokenize(utterance)
    for i in range(len(sent)):
        w = sent[i]
        ss = sid.polarity_scores(w)
        if ss['pos'] == 1.0:
            arr.append((w, i, abs(ss['compound'])))
    # check the words which between two positive words are connectives or not
    if len(arr) == 2:
        check_words_list = sent[arr[0][1] + 1:arr[1][1]]
        check = any(ele in check_words_list for ele in connectives_addition_of_ideas)
        if check:
            return [arr[0][0], arr[1][0]], True
        elif arr[0][2] >= arr[1][2]:
            return [arr[0][0]], True
        else:
            return [arr[1][0]], True
        # else:
        #     return [arr[0][0]], True
    else:
        return [], False


def unique_neg_involve(utterance):
    exception_vadarneg_words, missing_vadarneg_words, _ = loadConfigForROV()
    sid = SentimentIntensityAnalyzer()
    count = 0
    word = ''
    arr = []
    for w in word_tokenize(utterance):
        if w == 'no':
            continue
        ss = sid.polarity_scores(w)
        if ss['neg'] == 1.0 and w not in exception_vadarneg_words:
            count = count + 1
            if count <= 1:
                word = w
            arr.append(word)
        elif w in missing_vadarneg_words and count == 0:
            count = count + 1
            if count <= 1:
                word = w
    if count == 1:
        return word, True
    return 'cant_change', False


def unique_positive_involve(utterance):
    sid = SentimentIntensityAnalyzer()
    count = 0
    word = ''
    arr = []
    for w in word_tokenize(utterance):
        ss = sid.polarity_scores(w)
        if ss['pos'] == 1.0 and w:
            count = count + 1
            if count <= 1:
                word = w
            arr.append(word)
    if count == 1:
        return word, True
    return 'cant_change', False


# Current style/sentiment transfer techniques are still at low accuracy
def reverse_valence(utterance, common_flag):
    """reverse the utterance that is only verb can be reversed, if the object or subject can be reversed,
     keep the original version"""
    # directly handle these without going for complicated logic
    utterance = utterance.lower()
    utterance = utterance.replace(' i ', ' I ')
    backup_utt = utterance

    if 'should be' in utterance or 'would be' in utterance:
        utterance = utterance.replace(' be ', ' not be ')
    if ' need to ' in utterance:
        utterance = utterance.replace(' need to ', ' need not ')
    # if 'hate' in utterance:
    #     return utterance.replace('hate', 'love')
    if 'least' in utterance:
        utterance = utterance.replace('least', 'most')
    if utterance.endswith('lies.'):
        utterance = utterance.replace('lies', 'truth')

    tree_json, _, _ = analyzing_sentence_structure(utterance)

    # check if negation present , in terms of single or double words or not/n't words
    word, verdict = findIfnegationPresent(utterance)
    negword, replneg, verdict1 = findIfendingwithnt(utterance)
    words, verdict3 = bi_neg_involve(utterance)
    negative, verdict2 = unique_neg_involve(utterance)
    words_2, verdict5 = bi_pos_involve(utterance)
    positive, verdict4 = unique_positive_involve(utterance)

    # handle case by case , give priority to remove not first
    # print("here1",utterance)
    if verdict:
        utterance = utterance.replace(word + ' ', '')
    elif verdict1 and not verdict2:
        utterance = utterance.replace(negword, replneg)
    elif verdict5:
        for w in words_2:
            if get_antonym(w).startswith('not'):
                continue
            utterance = utterance.replace(w, get_antonym(w))
        # return utterance
    elif verdict4:
        utterance = utterance.replace(positive, get_antonym(positive))
        # return utterance
    elif verdict3:
        for w in words:
            if get_antonym(w).startswith('not'):
                continue
            utterance = utterance.replace(w, get_antonym(w))

        # return utterance
    else:
        prev_utterance = utterance
        utterance = utterance.replace(negative, get_antonym(negative))
        # incase algorithm could not handle still try to negate
        # cases replace present tense verbs by appending a don't
        # cases replace unique words prefixing with un
        if utterance == prev_utterance:
            text = word_tokenize(utterance)
            pos_text = pos_tag(text)
            for a, b in pos_text:
                if b == 'VBP' and a != 'am':
                    utterance = utterance.replace(a, "don't " + a)
                    break
                elif a == 'am':
                    utterance = utterance.replace(a, "won't")
                    break
                if a.startswith('un'):
                    utterance = utterance.replace(a, a[2:])
                    break


    tree_json_2, _, _ = analyzing_sentence_structure(utterance)
    if tree_json['verb'] != tree_json_2['verb']:
        change_tag = 'v'
    elif tree_json['object'] != tree_json_2['object']:
        change_tag = 'o'
    elif tree_json['subject'] != tree_json_2['subject']:
        change_tag = 's'
    else:
        change_tag = 'e'
    error = 0
    if backup_utt == utterance:
        error = 1
    if change_tag in ['o', 's'] and not common_flag:
        utterance = backup_utt
    return [utterance, change_tag, error]