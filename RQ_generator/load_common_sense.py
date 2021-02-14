# -*- coding:utf-8 -*-

import re

import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from scipy import spatial
import numpy as np
import pickle
import math
from sentence_transformers import SentenceTransformer

from __init__ import LoadCommonSense, clean_text
from reverse_utterance import reverse_valence


def getVec(We, words, t):
    t = t.strip()
    array = t.split('_')
    if array[0] in words:
        vec = We[words[array[0]], :]
    else:
        vec = We[words['UUUNKKK'], :]
        print('can not find corresponding vector:', array[0].lower())
    for i in range(len(array) - 1):
        if array[i + 1] in words:
            vec = vec + We[words[array[i + 1]], :]
        else:
            print('can not find corresponding vector:', array[i + 1].lower())
            vec = vec + We[words['UUUNKKK'], :]
    vec = vec / len(array)
    return vec


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(term1, term2, words, We, rel, Rel, Weight, Offset, evaType):
    v1 = getVec(We, words, term1)
    v2 = getVec(We, words, term2)
    result = {}

    del_rels = ['HasPainIntensity', 'HasPainCharacter', 'LocationOfAction', 'LocatedNear',
                'DesireOf', 'NotMadeOf', 'InheritsFrom', 'InstanceOf', 'RelatedTo', 'NotDesires',
                'NotHasA', 'NotIsA', 'NotHasProperty', 'NotCapableOf']

    for del_rel in del_rels:
        del rel[del_rel.lower()]

    for k, v in rel.items():
        v_r = Rel[rel[k], :]
        gv1 = np.tanh(np.dot(v1, Weight) + Offset)
        gv2 = np.tanh(np.dot(v2, Weight) + Offset)

        temp1 = np.dot(gv1, v_r)
        score = np.inner(temp1, gv2)
        result[k] = (sigmoid(score))

    if (evaType.lower() == 'max'):
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        for k, v in result[:1]:
            print(k, 'score:', v)
        return result[:1]
    if (evaType.lower() == 'topfive'):
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        for k, v in result[:5]:
            print(k, 'score:', v)
        return result[:5]
    if (evaType.lower() == 'sum'):
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        total = 0
        for i in result:
            total = total + i[1]
        print('total score is:', total)
        return total
    if (evaType.lower() == 'all'):
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        for k, v in result[:]:
            print(k, 'score:', v)
        return result
    else:
        tar_rel = evaType.lower()
        if result.get(tar_rel) == None:
            print('illegal relation, please re-enter a valid relation')
            return 'None'
        else:
            print(tar_rel, 'relation score:', result.get(tar_rel))
            return result.get(tar_rel)


def load_commonsense(word_1, word_2):
    """code from https://ttic.uchicago.edu/~kgimpel/commonsense.html ckbc-demo"""
    model_path = LoadCommonSense.DIC_PATH
    model = pickle.load(open(model_path, "rb"), encoding="latin1")

    Rel = model['rel']
    We = model['embeddings']
    Weight = model['weight']
    Offset = model['bias']
    words = model['words_name']
    rel = model['rel_name']

    result = score(word_1, word_2, words, We, rel, Rel, Weight, Offset, "max")
    return result[0][1]


def process_common_sense():
    """
    laod train600.txt commonsense knowledge representation and save it to csv file
    load it as Dataframe object
    :return: df
    """
    # step 1 ---------
    with open(LoadCommonSense.DIC_TXT_PATH, "r") as f:
        contexts = [s.strip() for s in f.readlines()]
    all_data = []
    for each in contexts:
        split_list = re.split(r'\t', each)
        all_data.append(split_list)
    df = pd.DataFrame(all_data, columns=['relation', 'left_term', 'right_term', 'score'])
    df.to_csv(LoadCommonSense.DIC_DF_PATH, index=False)

    # step 2 ---------
    df = pd.read_csv(LoadCommonSense.DIC_DF_PATH)
    b = df['relation'].value_counts(ascending=True)

    """merge left term and right term"""
    def convert_commonsense_vector(row):
        if row['relation'] in LoadCommonSense.left_to_right:
            text = "{} {} {}".format(row['left_term'], LoadCommonSense.left_to_right[row['relation']] ,row['right_term'])
            row['useful'] = 1
        elif row['relation'] in LoadCommonSense.right_to_left:
            text = "{} {} {}".format(row['right_term'], LoadCommonSense.right_to_left[row['relation']] ,row['left_term'])
            row['useful'] = 1
        else:
            text = row['left_term'] + " " + row['right_term']
        text = re.sub(r' a a ', ' a ', text)
        row['merge'] = text
        return row

    df['useful'] = 0
    df = df.dropna()
    df = df.apply(convert_commonsense_vector, axis=1)
    df = df.loc[df['useful'] == 1]
    df = df.drop(columns=['useful'])
    df.to_csv(LoadCommonSense.DIC_DF_PATH_2, index=False)

    # step 3 ------
    df = pd.read_csv(LoadCommonSense.DIC_DF_PATH_2)
    return df


def cosine(u, v):
    """calculate cosine"""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def find_similarity(df, with_out_sub, cleaned_list, tree_json, tag):
    """
    compare the targe sentence in commonsense dataset by using sent2vec
    score the result
    :param df:
    :param targe_sentence:
    :return: return top three score result list type
    """
    if tag == 'o':
        search_list = []
        verb = pos_tag(tree_json['verb'])
        for each in verb:
            verb_type = each[1]
            verb_temp = re.findall(r'VB.*', verb_type)
            if verb_temp:
                search_list.append(each[0])
        object = pos_tag(tree_json['object'])
        for each in object:
            if each[1].startswith("NN"):
                search_list.append(each[0])
                break
        word_for_search = [word for word in search_list if word in cleaned_list]
    elif tag == 'v':
        word_for_search = []
    else:
        word_for_search = []

    merge_list = df['merge'].str.lower().values.tolist()
    match_rows = [with_out_sub]
    for i in range(len(merge_list)):
        if len(word_for_search) == 1:
            pattern = "{}[ ].*".format(word_for_search[0])
        elif len(word_for_search) == 0:
            return []
        else:
            pattern = "[ ].*".join(word_for_search)
        result = re.findall(pattern, merge_list[i])
        if result:
            match_rows.append(merge_list[i])

    if len(match_rows) > 1:
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        sentence_embeddings = sbert_model.encode([match_rows[0]])[0]

        scores = []
        match_rows_o = match_rows[1:]
        for sent in match_rows_o:
            sim = cosine(sentence_embeddings, sbert_model.encode([sent])[0])
            scores.append(sim)

        index_ss = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:10]
        best_result = [match_rows_o[i] for i in index_ss]
    else:
        best_result = []

    return best_result


if __name__ == '__main__':
    load_commonsense("hate", "late")