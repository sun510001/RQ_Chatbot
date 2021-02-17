# -*- coding:utf-8 -*-
import numpy
import pandas as pd
import os, sys
import subprocess
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from load_common_sense import process_common_sense, find_similarity, load_commonsense
# from reverse_module import train_classifier, find_sentiment, reverse_valence
from reverse_utterance import reverse_valence
from __init__ import GrammaticalErrorCorrection, wash_sentence, PredictContextV2, analyzing_sentence_structure,\
    expand_contractions, LoadCommonSense
from convert_to_question import convert_question
from tf_bert_base_uncase import merge_response_context
from run_predict import predict_result
import re
from emoji import demojize


def grammatical_error_correction(utterance):
    """
    Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction
    From paper: "Can Encoder-Decoder Models Benefit from Pre-trained Language Representation in Grammatical Error
     Correction?" (In ACL 2020).
    """
    path_input = "{}test.txt".format(GrammaticalErrorCorrection.CODE_PATH)
    path_output = "{}test.best.tok".format(GrammaticalErrorCorrection.OUTPUT_PATH)
    with open(path_input, 'w') as f:
        for i in range(len(utterance)):
            if i == len(utterance) - 1:
                f.write(utterance[i])
            else:
                f.write(utterance[i] + "\n")
    # command = "./generate.sh test.txt"
    p = subprocess.Popen(["sh", "generate.sh", "test.txt"], cwd=GrammaticalErrorCorrection.CODE_PATH, shell=False,
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.wait()
    with open(path_output, 'r') as f:
        lines = [s.strip() for s in f.readlines()]
    return lines


def rq_generator(input):
    common_flag = False
    rov = reverse_valence(input, common_flag)
    print("reverse_valence:", rov)
    common_flag = True

    if rov[1] in ['o', 's']:
        df = pd.read_csv(LoadCommonSense.DIC_DF_PATH_2)
        # normal_list, cleaned_list, tree_list, error = wash_sentence(reverse_valence_output_list[i][0])
        normal_list, cleaned_list, tree_list, error = wash_sentence(rov[0])
        if tree_list["verb"] == "" or tree_list["object"] == "":
            return [], rov[1], 0
        elif tree_list["subject"] == "":
            with_out_sub = normal_list
        else:
            subject = ' '.join(tree_list["subject"])
            with_out_sub = re.sub(subject, '', normal_list, count=1)
            with_out_sub = re.sub(r'^ +', '', with_out_sub)
        scores_best_list = find_similarity(df, with_out_sub, cleaned_list, tree_list, 'o')
        print("process_common_sense (df): ", scores_best_list)
        print()

        # TODO convert utterance to question
        rq_list = []
        gramma_checked_list = grammatical_error_correction(scores_best_list)
        for each in gramma_checked_list:
            try:
                each = re.sub(r' n\'t', 'n\'t', each)
                rev = reverse_valence(each, common_flag)[0]
                rq = convert_question(rev, input, rov[1])
                print("reverse_valence:", rq)
            except BaseException as e:
                print(e)
                rq = ""

            if rq != "":
                rq_list.append(rq)
        out = grammatical_error_correction(rq_list)
        if len(out) == 0:
            return [], rov[1], 0
        return out, rov[1], -1

    if rov[2] == 0:
        if rov[1] == 'v':
            _, cleaned_list_ori, _, _ = wash_sentence(input)
            search_list = []
            search_list_2_1 = []
            search_list_2_2 = []
            tree_json, _, _ = analyzing_sentence_structure(rov[0])
            tree_json_ori, _, _ = analyzing_sentence_structure(input)

            if tree_json_ori['verb']:
                verb = pos_tag(tree_json_ori['verb'])
                for each in verb:
                    search_list.append(each[0])

            if tree_json_ori['object']:
                object = pos_tag(tree_json_ori['object'])
                count = 0
                for each in object:
                    object_type = each[1]
                    if len(search_list_2_1) == 0:
                        object_temp = re.findall(r'NN.*', object_type)
                        if object_temp:
                            search_list_2_1.append(each[0])
                    if len(search_list_2_2) == 0:
                        adj_temp = re.findall(r'JJ.*', object_type)
                        if adj_temp:
                            #  Adjective comparative and superlative normalization
                            wnl = WordNetLemmatizer()
                            normalize = wnl.lemmatize(each[0], 'a')
                            search_list_2_2.append(normalize)

            word_for_cs_score_1 = [word for word in search_list if word in cleaned_list_ori]
            word_for_cs_score_2 = [word for word in search_list_2_1 if word in cleaned_list_ori]
            word_for_cs_score_3 = [word for word in search_list_2_2 if word in cleaned_list_ori]

            if len(word_for_cs_score_1) >= 1 and (len(word_for_cs_score_2) >= 1 or len(word_for_cs_score_3) >= 1):
                word_1 = "_".join(word_for_cs_score_1)
                word_2_1 = "_".join(word_for_cs_score_2)
                word_2_2 = "_".join(word_for_cs_score_3)
                score = load_commonsense(word_1, word_2_1)
                score_2 = load_commonsense(word_1, word_2_2)
                max_score = max([score, score_2])
                rq = convert_question([], rov[0], rov[1])
                out = grammatical_error_correction([rq])
                return out, rov[1], max_score
            else:
                return [], rov[1], 0
        else:
            return [], rov[1], 0
    else:
        return [], rov[1], 0


if __name__ == '__main__':
    """the evaluation of RQ generation"""
    """
    create the commonsense knowledge sentences
    if there is not commonsense_dataset_merged.csv in data/ uncomment the code below
    """
    try:
        df = pd.read_csv(LoadCommonSense.DIC_DF_PATH_2)
    except BaseException:
        process_common_sense()
        pass

    """you can run a part of data if your GPU memory is not enough"""
    df = pd.read_csv('../dialogue_for_train_after_sg.csv')
    # df = df[:100]  # run for a part of data
    df = df.dropna(subset=["robot_res"])
    df = df.loc[(df['len'] >= 6) & (df['len'] <= 20)]
    model_type = PredictContextV2.MODEL_NAME
    result = []
    df['rq_res'] = ""
    df['rov'] = ""
    df['score'] = 0
    def generate(row):
        print("+++++===>", row.name)
        robot_res = row['robot_res']
        context_2 = row['context_2']
        context_1 = row['context_1']
        robot_res = expand_contractions(robot_res, LoadCommonSense.CONTRACTION_MAP)
        word_token = word_tokenize(robot_res)
        if word_token[-1] == '.':
            del word_token[-1]
        if '.' in word_token[:-4]:
            dot_index = word_token.index('.')
            robot_res = " ".join(word_token[dot_index+1:])
        else:
            robot_res = " ".join(word_token)

        output_list, rov, score = rq_generator(robot_res)
        if len(output_list) != 0:
            row['score'] = score
            print("score:", score)
            df_list = []
            if rov in ['o', 's']:
                for each in output_list:
                    merge_list = [context_2,context_1,each]
                    merge_list_2 = merge_response_context(merge_list, model_type)
                    df_list.append(merge_list_2)
                df_rank_merge = pd.DataFrame({"context_merge": df_list})
                index = predict_result(df_rank_merge, model_type)
                rq = output_list[index]
                robot_res = re.sub(r'([\w]) *$', r'\g<1>.', robot_res)  # add a period after the robot response.
                print("---> [o/s]", robot_res, rq)
                row['rq_res'] = '{} {}'.format(robot_res, rq)
            elif rov == 'v':
                for each in output_list:
                    out_sent = " ".join(output_list)
                    print("---> [v]", out_sent)
                row['rq_res'] = out_sent
            else:
                print("---> [none]")
        row['rov'] = rov
        return row

    df = df.apply(generate, axis=1)
    # df.to_csv('../dialogue_for_train_after_rq_1.csv', index=False)  # run for a part of data
    # sys.exit()
    df.to_csv('../dialogue_for_train_after_rq.csv', index=False)


    """if you ran for parts of data, merge the csv files by the code below"""
    # df_all = pd.DataFrame()
    # for i in range(1, 6):
    #     path = '../dialogue_for_train_after_rq_{}.csv'.format(i)
    #     df = pd.read_csv(path)
    #     df = df.loc[(df['len'] >= 6) & (df['len'] <= 11)]
    #     if i == 1:
    #         df_all = df
    #     else:
    #         df_all = df_all.append(df, ignore_index=True)
    # df_all.to_csv('../dialogue_for_train_after_rq.csv', index=False)


    """get evaluation data"""
    path = '../dialogue_for_train_after_rq.csv'
    df = pd.read_csv(path)
    df = df.dropna(subset=["robot_res",'sg_res','rq_res']).reset_index(drop=True)
    df.to_csv('../dialogue_for_train_after_rq.csv')

    path = '../dialogue_for_train_after_rq.csv'
    df = pd.read_csv(path)
    def del_emoji(text):
        for i in range(len(text)):
            # out = re.sub(r'<URL>', ':URL:', text[i])
            out = re.sub(r'@USER *', '', text[i])
            out = demojize(out)
            text[i] = out
        return text

    # df = df[['context_2','context_1','response','robot_res','sg_res','rq_res']].apply(del_emoji, axis=1)
    df2 = df[['context_2', 'context_1', 'response', 'robot_res', 'sg_res', 'rq_res']].apply(del_emoji, axis=1)
    df3 = df[['sarcasm', 'question', 'rq', 'sentiment', 'len', 'dot', 'is_question', 'triple_result', 'rov', 'score']]
    df4 = pd.concat([df3, df2], axis=1)

    # path = '../dialogue_for_train_after_rq_proc.csv'
    # df = pd.read_csv(path)
    df2.to_csv('../dialogue_for_train_after_rq_proc_sec.csv', index=False)
    df4.to_csv('../dialogue_for_train_after_rq_proc_sec_complete.csv', index=False)

    """for chatbot"""
    input = list(sys.argv)
    data = input[1:]
    print("------<>",data)
    df = pd.DataFrame(data=[data], columns=['context_2', 'context_1', 'robot_res'])
    length = len(data[-1])
    if length < 6 and length > 20:
        out_rq = ''
    model_type = PredictContextV2.MODEL_NAME
    result = []
    df['rq_res'] = ""
    df['rov'] = ""
    df['score'] = 0
    first_flag = []  # do apply on the first row at once
    def generate(row):
        if len(first_flag) == 0:
            print("+++++===>", row.name)
            robot_res = row['robot_res']
            context_2 = row['context_2']
            context_1 = row['context_1']
            robot_res = expand_contractions(robot_res, LoadCommonSense.CONTRACTION_MAP)
            word_token = word_tokenize(robot_res)
            if word_token[-1] == '.':
                del word_token[-1]
            if '.' in word_token[:-4]:
                dot_index = word_token.index('.')
                robot_res = " ".join(word_token[dot_index + 1:])
            else:
                robot_res = " ".join(word_token)

            output_list, rov, score = rq_generator(robot_res)
            if len(output_list) != 0:
                row['score'] = score
                print("score:", score)
                df_list = []
                if rov in ['o', 's']:
                    for each in output_list:
                        merge_list = [context_2, context_1, each]
                        merge_list_2 = merge_response_context(merge_list, model_type)
                        df_list.append(merge_list_2)
                    df_rank_merge = pd.DataFrame({"context_merge": df_list})
                    index = predict_result(df_rank_merge, model_type)
                    rq = output_list[index]
                    robot_res = re.sub(r'([\w]) *$', r'\g<1>.', robot_res)  # add a period after the robot response.
                    print("---> [o/s]", robot_res, rq)
                    row['rq_res'] = '{} {}'.format(robot_res, rq)
                elif rov == 'v':
                    for each in output_list:
                        out_sent = " ".join(output_list)
                        print("---> [v]", out_sent)
                    row['rq_res'] = out_sent
                else:
                    print("---> [none]")
            row['rov'] = rov
            row['rq_res'] = re.sub(r' \. \?', ' ?', row['rq_res'])
            first_flag.append(row['rq_res'])
        return row

    df = df.apply(generate, axis=1)
    out_rq = first_flag[0]
    print(out_rq)
    with open("chatbot_out_rq.txt", 'w') as f:
        f.write(out_rq)