# -*- coding:utf-8 -*-

import importlib
import spacy
from nltk import pos_tag
from nltk import word_tokenize
import re
nlp = spacy.load("en_core_web_sm")

from __init__ import NltkNer, GrammaticalErrorCorrection


def yes_no_question(utterance):
    utterance = re.sub(r' n\'t', 'n\'t', utterance)
    word_token = word_tokenize(utterance)
    pt = pos_tag(word_token)
    if pt[0][1] == "TO":
        del word_token[0]
        list_out = ["should we"] + word_token
        return " ".join(list_out) + " ?"

    for i in range(len(pt)-1):
        if pt[i][1] == "MD":
            md_word = pt[i][0]
            del word_token[i]
            list_out = [md_word] + word_token
            return " ".join(list_out) + " ?"
        elif pt[i][1] == "VBD":
            list_out = ["did"] + word_token
            return " ".join(list_out) + " ?"
        elif pt[i][1] in ["VBN", "VBZ"]:

            list_out = ["do"] + word_token
            return " ".join(list_out) + " ?"
        elif pt[i][1] == "VBP":
            vbp_word = pt[i][0]
            del word_token[i]
            list_out = [vbp_word] + word_token
            return " ".join(list_out) + " ?"
        # else:
    list_out = ["is it that"] + word_token
    return " ".join(list_out) + " ?"


def wh_question(utterance):
    utterance = re.sub(r"nothing|something", "anything", utterance.lower())
    utterance = re.sub(r"no one|noone|nobody", "any one", utterance.lower())
    word_token = word_tokenize(utterance)
    pt = pos_tag(word_token)
    ner = nlp(utterance)
    utt_ner = [(X.text, X.label_) for X in ner.ents]

    for i in range(len(pt)-1):
        if utt_ner:
            for index, value in NltkNer.ner_dic.items():
                if utt_ner[0][1] in value:
                    word_token = word_token[i+1:]
                    list_out = [index] + word_token
                    return " ".join(list_out) + " ?"

        if pt[i][1] in ["NNP", "NNPS", "NN", "NNS"]:
            if pt[i][0] in ['i', 'I']:
                word_token = word_token[i + 1:]
                list_out = ["Who"] + word_token
                return " ".join(list_out) + " ?"
            find_body = re.findall(r".*body|everyone|any one", pt[i][0].lower())
            find_one = re.findall(r"everything|nothing|none|anything", pt[i][0].lower())
            if find_body:
                word_token = word_token[i+1:]
                list_out = ["Who"] + word_token
                return " ".join(list_out) + " ?"
            elif find_one:
                word_token = word_token[i+1:]
                list_out = ["What"] + word_token
                return " ".join(list_out) + " ?"
            else:
                word_token = word_token[i+1:]
                list_out = ["what"] + word_token
                return " ".join(list_out) + " ?"
        elif pt[i][1] in ["PRP"]:
            word_token = word_token[i+1:]
            list_out = ["Who"] + word_token
            return " ".join(list_out) + " ?"
        elif pt[i][1] in ["PRP$", "POS"]:
            word_token = word_token[i+1:]
            list_out = ["Whose"] + word_token
            return " ".join(list_out) + " ?"
    return ""


def convert_question(utterance, input, rov):
    if rov == 'v':
        q = wh_question(input)
    else:
        q = yes_no_question(utterance)
    return q
