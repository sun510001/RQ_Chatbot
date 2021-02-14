# -*- coding:utf-8 -*-

import pandas as pd
import json
from __init__ import LoadDialogueDataset
from Literal_generator import literal_generation_for_evaluation
import pandas as pd
from sklearn.model_selection import train_test_split
from sentiment_check import find_sentiment
import re


def load_dialogue_dataset():
    """not use dialogue dataset"""
    # with open(LoadDialogueDataset.DIA_PATH, 'r') as f:
    #     all = f.read().splitlines()
    #
    # dias = []
    # for i in range(len(all)):
    #     dia = all[i].split("__eou__")
    #     dias.append(dia[:-1])
    #     # print(dia)
    #
    # with open(LoadDialogueDataset.DIA_JSON_PATH, 'w') as w:
    #     json.dump(dias, w)

    # with open(LoadDialogueDataset.DIA_JSON_PATH, 'r') as json_file:
    #     data = json.load(json_file)
    #     # literal_generation_for_evaluation(data)
    #     temp_list = []
    #     length_d = len(data)
    #     print("all data:", length_d)
    #     # data = data[7:10]
    #
    #     for j in range(len(data)):
    #         flag = 0
    #         print("data", j)
    #         if len(data) == 2:
    #             flag = -1
    #         elif len(data) < 2:
    #             flag = -1
    #         else:
    #             list_len = len(data[j])
    #             for i in reversed(range(list_len)):
    #                 if i != 0 and i != list_len-1:
    #                     sentence_list = data[j][i].split(' ')
    #                     find_question = re.findall(r'[/?]', data[j][i])
    #                     if len(find_question) == 0 and len(sentence_list) >= 10:
    #                         flag = i + 1
    #                         break
    #             # if flag == 0:
    #
    #         print('flag', flag)
    #         for each in data[j]:
    #             print(each)
    #
    #         temp_list.append([data[j], flag])
    #         print()
    #
    # df = pd.DataFrame(data=temp_list, columns=["data", "flag"])
    # df.to_json("dialogue_for_train_daily_question.json", orient="values")

    # df = pd.read_json("dialogue_for_train_daily_question.json", orient="values")
    # def test(row):
    #     # for each in row[0]:
    #     flag = row[1]
    #     sentence_list = row[0]
    #     literal_generation_for_evaluation(sentence_list[:flag+1])
    #     print()
    #
    # df = df.loc[df[1] != 0]
    # df.apply(test, axis=1)
    # df.to_csv("dialogue_for_train_daily_robot_literal.csv")
    # print()


    """use sarcasm detection shared task conversation"""
    df = pd.read_csv(LoadDialogueDataset.SARC_DATASET_PATH)

    def merge_label(input):
        out = 0  # literal
        if input['rq'] == 1 or (input['sarcasm'] == 1 and input['question'] == 1):
            out = 1  # rq
        elif input['sarcasm'] == 1:
            out = 2  # sarcasm
        return out

    df['tri_label'] = df[['sarcasm', 'question', 'rq']].apply(merge_label, axis=1)
    train_set, test_set = train_test_split(df,
                                           stratify=df['tri_label'],
                                           test_size=0.2,
                                           random_state=43)
    print(test_set.shape[0])
    test_set.reset_index(drop=True)

    data_list = test_set[["sarcasm", "question", "rq", "context_2", "context_1", "response"]].values.tolist()
    result_all = []
    count = 0
    for one_dialogue in data_list:
        count += 1
        print("No.{} | sarcasm {} | question {} | rq {}".format(count, one_dialogue[0], one_dialogue[1], one_dialogue[2]))
        result = literal_generation_for_evaluation(one_dialogue)

        senti = find_sentiment(result[-1])
        # make sure that dot is not in the middle of the sentences
        check_dot = re.findall(r'\. *[a-zA-Z0-9]', result[-1])
        # plan to delete literal question
        check_question = re.findall(r'\?', result[-1])
        # check legth of literal generation
        length_check = re.sub(r"[^\w]+", " ", result[-1])
        length_check = re.sub(r" *$", "", length_check)
        length = len(length_check.split(" "))
        if senti[0][0] > 0.8 and senti[0][1] <= 0.8:
            result.append("pos")
        elif senti[0][1] > 0.8:
            result.append("neg")
        else:
            result.append("neu")
        result.append(length)
        if check_dot:
            result.append(1)
        else:
            result.append(0)
        if check_question:
            result.append(1)
        else:
            result.append(0)
        result_all.append(result)

    df_out = pd.DataFrame(result_all, columns=["sarcasm", "question", "rq", "context_2", "context_1", "response",
                                               "robot_res", "sentiment", "len", "dot", "is_question"])
    df_out.to_csv("dialogue_for_train_2.csv", index=False)


    df = pd.read_csv("dialogue_for_train.csv")
    df_proc = df.loc[(df['len'] >= 8) & (df['dot'] == 0) & (df['question'] == 0)]

    def delete_question(input):
        check_question = re.findall(r'\?', input['robot_res'])
    df_proc = df_proc.apply(delete_question, axis=1)
    print()


if __name__ == '__main__':
    """laod conversation dataset for evaluation and commonsense dictionary"""
    load_dialogue_dataset()

