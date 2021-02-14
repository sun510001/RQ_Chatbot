# -*- coding:utf-8 -*-

import ast
import re
from os import getcwd
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_context_to_list(input):
    if input[0] == "SARCASM":
        input['sarcasm'] = 1
    else:
        input['sarcasm'] = 0
    input['context_count'] = len(input[2])
    # clean the text
    clean_context_list = []
    # count the words in the longest sentence.
    max_word_sentence = 0
    sum_context_merge = 0
    # count = 0
    for i in reversed(range(len(input[2]))):
        out = re.sub(r'@USER|<URL>', '', input[2][i])
        out = re.sub(r'^ +', '', out)

        len_string = len(input[2][i].split())
        # global count
        # global statistic_words
        # if len_string > max_word_sentence:
        #     max_word_sentence = len_string
        # statistic_words.append(len_string)
        # list_of_context = clean_context_list
        # input['context_1'] = (list_of_context[-2])
        # input['context_2'] = (list_of_context[-1])
        # if count < context_len:
        # count += 1
        sum_context_merge += len_string
        clean_context_list.append(out)

    input['context_merge'] = ' '.join(clean_context_list)
    input['max_word_sentence'] = max_word_sentence
    input['sum_context_merge'] = sum_context_merge
    # check response is question or not
    resp = input['response']
    if re.findall(r'\?', resp):
        input['question'] = 1
    else:
        input['question'] = 0
    return input


def load_each_data(data_source, path, context_len):
    path_sarc = path + 'sarcasm_detection_shared_task_' + data_source
    path_sarc_train = path_sarc + '_training.jsonl'
    path_sarc_test = path_sarc + '_testing.jsonl'
    df_train = pd.read_json(path_sarc_train, orient='records', lines=True)
    df_test = pd.read_json(path_sarc_test, orient='records', lines=True)
    df_sarc = (pd.concat([df_train, df_test])).reset_index(drop=True)

    # df_proceed = pd.DataFrame(columns=['label_num', 'context_count', 'context_1', 'context_2'])
    # add_col = ['label_num', 'context_1', 'context_2']
    df_sarc['sarcasm'], df_sarc['context_count'], df_sarc['sum_context_merge'], df_sarc['context_merge'], \
    df_sarc['question'] = [np.nan, np.nan, np.nan, np.nan, np.nan]

    df_sarc_twi = df_sarc.apply(process_context_to_list, axis=1)
    # df_sarc_twi = df_sarc.apply(process_context_to_list, args=(context_len,), axis=1)

    # df_rq = df_sarc_twi.loc[(df_sarc_twi['rq'] == 1) & (df_sarc_twi['label'] == 'SARCASM')]
    df_save = df_sarc_twi[['sarcasm', 'question', 'context_count', 'sum_context_merge', 'response',
                           'context', 'context_merge']]

    '''draw the image for counting the words'''
    plt.hist(df_save['sum_context_merge'].values)
    plt.savefig('pictures/sum_context_merge_len_' + str(context_len) + '_' + data_source + '.png')
    plt.close('all')
    plt.clf()

    '''draw the image for contexts length'''
    plt.hist(df_save['context_count'].values)
    plt.savefig('pictures/context_count_len_' + str(context_len) + '_' + data_source + '.png')
    plt.clf()

    return df_save


# def split_df_sarc(df):
#     df_sarc_train, df_sarc_test = train_test_split(df, test_size=0.2)
#     return df_sarc_train, df_sarc_test
#
#
# def split_df_for_srq(df):
#     df_srq = df.loc[(df['question'] == 1) & (df['label_num'] == 1)]
#     df_non_srq = df[(df['question'] == 0) & (df['label_num'] == 0)]
#     # make df_non_srq the same length with df_srq
#     df_non_srq = df_non_srq.sample(n=(df_srq.shape[0]))
#
#     df_srq_train, df_srq_test = train_test_split(df_srq, test_size=0.2)
#     df_non_srq_train, df_non_srq_test = train_test_split(df_non_srq, test_size=0.2)
#
#     df_train = pd.concat([df_srq_train, df_non_srq_train], axis=0)
#     df_test = pd.concat([df_srq_test, df_non_srq_test], axis=0)
#
#     return df_train, df_test
#
#
# def split_df_for_q(df):
#     """only deal with questions which have no sarcastic meanings"""
#     df_literal = df.loc[(df['question'] == 0) & (df['label_num'] == 0)]
#     df_rq = df[(df['question'] == 1) & (df['label_num'] == 0)]
#     # make df_non_srq the same length with df_srq
#     df_literal = df_literal.sample(n=(df_rq.shape[0]))
#
#     df_literal_train, df_literal_test = train_test_split(df_literal, test_size=0.2)
#     df_rq_train, df_rq_test = train_test_split(df_rq, test_size=0.2)
#
#     df_train = pd.concat([df_literal_train, df_rq_train], axis=0)
#     df_test = pd.concat([df_literal_test, df_rq_test], axis=0)
#
#     return df_train, df_test
#
#
# def split_df_for_rq(df):
#     """
#     only deal with rhetorical questions which have no sarcastic meanings
#     process questions, only keep rhetorical questions
#     """
#     df_literal = df.loc[(df['question'] == 0) & (df['label_num'] == 0)]
#     # df_rq = df[(df['question'] == 1) & (df['label_num'] == 0)]
#     path_save = '../data_cla/sarcasm_merge_rq_v7.csv'
#     df_rq = pd.read_csv(path_save)
#     # df_rq.to_csv('../data_cla/sarcasm_merge_rq_v5.csv', index=False)
#
#     # make df_non_srq the same length with df_srq
#     df_literal = df_literal.sample(n=(df_rq.shape[0]))
#
#     df_literal_train, df_literal_test = train_test_split(df_literal, test_size=0.2)
#     df_rq_train, df_rq_test = train_test_split(df_rq, test_size=0.2)
#
#     df_train = pd.concat([df_literal_train, df_rq_train], axis=0)
#     df_test = pd.concat([df_literal_test, df_rq_test], axis=0)
#     df_train = df_train.fillna(0)
#     df_test = df_test.fillna(0)
#     return df_train, df_test
#
#
# def split_df_for_sarc_literal(df):
#     """
#     deal with rhetorical questions which have no sarcastic meanings and sarcastic response
#     process questions, only keep rhetorical questions
#     """
#     df_literal = df.loc[df['label_num'] == 0]
#     df_srq = df[(df['question'] == 1) & (df['label_num'] == 1)]
#
#     # make df_non_srq the same length with df_srq
#     df_literal = df_literal.sample(n=(df_srq.shape[0]))
#
#     df_literal_train, df_literal_test = train_test_split(df_literal, test_size=0.2)
#     df_rq_train, df_rq_test = train_test_split(df_srq, test_size=0.2)
#
#     df_train = pd.concat([df_literal_train, df_rq_train], axis=0)
#     df_test = pd.concat([df_literal_test, df_rq_test], axis=0)
#
#     return df_train, df_test
#
#
# def split_df_for_triple(df):
#     """
#     deal with rhetorical questions which have no sarcastic meanings and sarcastic response
#     process questions, only keep rhetorical questions
#     """
#     df_literal = df.loc[(df['question'] == 0) & (df['label_num'] == 0)]
#     df_sarc = df[(df['question'] == 0) & (df['label_num'] == 1)]
#     df_srq = df[(df['question'] == 1) & (df['label_num'] == 1)]
#     # path_load = '../data_cla/sarcasm_merge_rq_v7.csv'
#     # df_rq = pd.read_csv(path_load)
#     # df_rq.to_csv('../data_cla/sarcasm_merge_rq_v5.csv', index=False)
#
#     # make df_non_srq the same length with df_srq
#     df_literal = df_literal.sample(n=(df_srq.shape[0]))
#     df_sarc = df_sarc.sample(n=(df_srq.shape[0]))
#
#     # df_literal_train, df_literal_test = train_test_split(df_literal, test_size=0.2)
#     # df_rq_train, df_rq_test = train_test_split(df_srq, test_size=0.2)
#     # df_sarc_train, df_sarc_test = train_test_split(df_sarc, test_size=0.2)
#
#     df_merge = pd.concat([df_literal, df_sarc, df_srq], axis=0)
#     # df_test = pd.concat([df_literal_test, df_rq_test, df_sarc_test], axis=0)
#
#     return df_merge
#
#
# def split_df_v12(df):
#     df_literal = df.loc[(df['question'] == 0) & (df['label_num'] == 0)]
#     df_sarc = df[(df['question'] == 0) & (df['label_num'] == 1)]
#     df_srq = df[(df['question'] == 1) & (df['label_num'] == 1)]
#     path_load = '../data_cla/sarcasm_merge_rq_v7.csv'
#     df_rq = pd.read_csv(path_load)
#     # df_rq.to_csv('../data_cla/sarcasm_merge_rq_v5.csv', index=False)
#
#     # make df_non_srq the same length with df_srq
#     df_rq_merge = pd.concat([df_rq, df_srq], axis=0)
#     df_literal = df_literal.sample(n=(df_rq_merge.shape[0]))
#     df_sarc = df_sarc.sample(n=(df_rq_merge.shape[0]))
#
#     # df_literal_train, df_literal_test = train_test_split(df_literal, test_size=0.2)
#     # df_rq_train, df_rq_test = train_test_split(df_srq, test_size=0.2)
#     # df_sarc_train, df_sarc_test = train_test_split(df_sarc, test_size=0.2)
#
#     df_merge = pd.concat([df_literal, df_sarc, df_rq_merge], axis=0)
#     # df_test = pd.concat([df_literal_test, df_rq_test, df_sarc_test], axis=0)
#     df_merge['rq'] = df_merge['rq'].fillna(0)
#     return df_merge
#
#
# def split_df_v13(df):
#     df_literal = df.loc[(df['question'] == 0) & (df['label_num'] == 0)]
#     df_sarc = df[(df['question'] == 0) & (df['label_num'] == 1)]
#     df_q = df[(df['question'] == 1)]
#     # path_load = '../data_cla/sarcasm_merge_rq_v7.csv'
#     # df_rq = pd.read_csv(path_load)
#     # df_rq.to_csv('../data_cla/sarcasm_merge_rq_v5.csv', index=False)
#
#     # make df_non_srq the same length with df_srq
#     # df_rq_merge = pd.concat([df_rq, df_srq], axis=0)
#     df_literal = df_literal.sample(n=(df_q.shape[0]))
#     df_sarc = df_sarc.sample(n=(df_q.shape[0]))
#
#     # df_literal_train, df_literal_test = train_test_split(df_literal, test_size=0.2)
#     # df_rq_train, df_rq_test = train_test_split(df_srq, test_size=0.2)
#     # df_sarc_train, df_sarc_test = train_test_split(df_sarc, test_size=0.2)
#
#     df_merge = pd.concat([df_literal, df_sarc, df_q], axis=0)
#     # df_test = pd.concat([df_literal_test, df_rq_test, df_sarc_test], axis=0)
#     # df_merge['rq'] = df_merge['rq'].fillna(0)
#     return df_merge


def split_df_v14(df, is_balance):
    """add rq labels"""
    df['rq'] = 0
    df2 = df.loc[(df['question'] == 1) & (df['sarcasm'] == 0)]
    path_load = 'data/sarcasm_merge_rq_v7.csv'
    df_rq = pd.read_csv(path_load)

    list_rq_response = df_rq['response'].values.tolist()

    def match_rq(input):
        if input['response'] in list_rq_response:
            input['rq'] = 1
        else:
            input['rq'] = 0
        return input

    df2 = df2.apply(match_rq, axis=1)
    df_rq = df2.loc[df2['rq'] == 1]
    df_non_rq = df2.loc[df2['rq'] == 0]

    df_literal = df.loc[(df['question'] == 0) & (df['sarcasm'] == 0)]
    df_sarc = df[(df['question'] == 0) & (df['sarcasm'] == 1)]
    df_srq = df[(df['question'] == 1) & (df['sarcasm'] == 1)]

    df_literal = pd.concat([df_literal, df_non_rq], axis=0)
    df_rq_merge = pd.concat([df_rq, df_srq], axis=0)

    if is_balance == 1:
        # make df_non_srq the same length with df_srq
        df_literal = df_literal.sample(n=(df_rq_merge.shape[0]))
        df_sarc = df_sarc.sample(n=(df_rq_merge.shape[0]))

    df_merge = pd.concat([df_literal, df_sarc, df_rq_merge], axis=0)

    return df_merge[['sarcasm', 'question', 'rq', 'context_count', 'max_word_sentence', 'sum_context_merge',
                     'context_merge', 'response', 'context']]


def split_df_v15(df, is_balance):
    """add rq labels"""
    df_literal_rq = df.loc[df['rq'] == 1]
    df_literal_non_rq = df.loc[(df['rq'] == 0) & (df['sarcasm'] == 0)]
    df_sarc = df[(df['question'] == 0) & (df['sarcasm'] == 1)]
    df_srq = df[(df['question'] == 1) & (df['sarcasm'] == 1)]

    df_literal = df_literal_non_rq
    df_rq_merge = pd.concat([df_literal_rq, df_srq], axis=0)

    if is_balance == 1:
        # make df_non_srq the same length with df_srq
        df_literal = df_literal.sample(n=(df_rq_merge.shape[0]))
        df_sarc = df_sarc.sample(n=(df_rq_merge.shape[0]))

    df_merge = pd.concat([df_literal, df_sarc, df_rq_merge], axis=0)

    return df_merge[['sarcasm', 'question', 'rq', 'context_count', 'sum_context_merge',
                     'context_merge', 'response', 'context']]


def split_df_v16(df, is_balance):
    """add rq labels; add response to merge context"""
    df_literal_rq = df.loc[df['rq'] == 1]
    df_literal_non_rq = df.loc[(df['rq'] == 0) & (df['sarcasm'] == 0)]
    df_sarc = df[(df['question'] == 0) & (df['sarcasm'] == 1)]
    df_srq = df[(df['question'] == 1) & (df['sarcasm'] == 1)]

    df_literal = df_literal_non_rq
    df_rq_merge = pd.concat([df_literal_rq, df_srq], axis=0)

    if is_balance == 1:
        # make df_non_srq the same length with df_srq
        df_literal = df_literal.sample(n=(df_rq_merge.shape[0]))
        df_sarc = df_sarc.sample(n=(df_rq_merge.shape[0]))

    df_merge = pd.concat([df_literal, df_sarc, df_rq_merge], axis=0)

    def x(input):
        return "{} {}".format(input[1], input[0])

    df_merge['context_merge'] = df_merge[['context_merge', 'response']].apply(x, axis=1)

    return df_merge[['sarcasm', 'question', 'rq', 'context_count', 'sum_context_merge',
                     'context_merge', 'response', 'context']]


def split_df_v17(df, is_balance):
    """add rq labels; add response; context(-1,-2) to each columns"""
    # df_literal_rq = df.loc[df['rq'] == 1]
    # df_literal_non_rq = df.loc[(df['rq'] == 0) & (df['sarcasm'] == 0)]
    # df_sarc = df[(df['question'] == 0) & (df['sarcasm'] == 1)]
    # df_srq = df[(df['question'] == 1) & (df['sarcasm'] == 1)]

    # df_literal = df_literal_non_rq
    # df_rq_merge = pd.concat([df_literal_rq, df_srq], axis=0)

    # if is_balance == 1:
    #     # make df_non_srq the same length with df_srq
    #     df_literal = df_literal.sample(n=(df_rq_merge.shape[0]))
    #     df_sarc = df_sarc.sample(n=(df_rq_merge.shape[0]))

    # df_merge = pd.concat([df_literal, df_sarc, df_rq_merge], axis=0)

    df['context_1'] = ''
    df['context_2'] = ''

    def x(input):
        context_list = ast.literal_eval(input[0])
        input[1] = context_list[-1]  # context_1 is the last context
        input[2] = context_list[-2]  # context_2 is the second last context
        return input
        # return "{} {}".format(input[1], input[0])

    df[['context', 'context_1', 'context_2']] = df[['context', 'context_1', 'context_2']].apply(x, axis=1)

    return df[['sarcasm', 'question', 'rq', 'context_count', 'context_2', 'context_1', 'response']]


# def other(df):
#     df_literal = df.loc[(df['question'] == 1) & (df['label_num'] == 0)]
#     print()
#
#
# def load_data_merge():
#     df_t = load_each_data('twitter')
#     df_r = load_each_data('reddit')
#     """
#     v1: sarcasm vs literal; v2: srq vs other; v3: srq vs normal sarcasm; v4: question vs literal (non-sarcasm)
#     v5: rq (processed) vs literal (non-sarcasm); v7: rq (use sarcasm_merge_rq_v7) vs literal (non-sarcasm);
#     v8: rq (use sarcasm_merge_rq_v7) vs sarcasm; v9: rq vs sarcasm vs literal; v10: rq(sarcasm) vs literal
#
#     """
#     version = 'v10'
#     df_save = pd.concat([df_t, df_r], axis=0)
#     df_train, df_test = split_df_for_srq(df_save)
#
#     '''draw the image for counting the words'''
#     # plt.hist(statistic_words)
#     # plt.savefig('statistic_for_words_merge.png')
#     # plt.show()
#
#     df_train.to_csv('../data_cla/sarcasm_merge_training_train_' + version + '.csv', index=False)
#     df_test.to_csv('../data_cla/sarcasm_merge_training_test_' + version + '.csv', index=False)
#
#     # print()
#     # with open('../data_cla/sarcasm_detection_shared_task_twitter_training.jsonl') as f:
#     #     data = json.load(f)
#     #     for each in data['labe']:


def load_rq_unbalance_dataset_v8(path, context_len, version, is_balance):
    if exists(path + 'sarcasm_merge_triple_' + version + '.csv'):
        df_t = load_each_data('twitter', path, context_len)
        df_r = load_each_data('reddit', path, context_len)

        df_save = (pd.concat([df_t, df_r])).reset_index(drop=True)
        df_save.to_json(path + 'sarcasm_merge_triple_' + version + '.json', orient='records', lines=True)

        # df_save = pd.read_csv(path + 'sarcasm_merge_triple_' + version + '.csv')
        df_out = split_df_v14(df_save, is_balance)

        '''draw the image for counting the words'''
        # plt.hist(statistic_words)
        # plt.savefig('statistic_for_words_merge.png')
        # plt.show()

        df_out.to_csv(path + 'sarcasm_merge_triple_' + version + '.csv', index=False)
        print('Create sarcasm_merge_triple_{}.csv successful.')
    else:
        print('Create sarcasm_merge_triple_{}.csv failed, because it exists in {}/{}'.format(version, getcwd(), path))


def load_rq_unbalance_dataset(path, version, is_balance):
    """process for sarcasm_merge_triple_v5.csv"""
    df = pd.read_csv(path + 'sarcasm_merge_triple_v5.csv')
    df_out = split_df_v15(df, is_balance)
    df_out.to_csv(path + 'sarcasm_merge_triple_' + version + '.csv', index=False)
    print('Create sarcasm_merge_triple_' + version + '.csv successful.')


def load_rq_balance_dataset_response(path, version, is_balance):
    """process for sarcasm_merge_triple_v6.csv"""
    df = pd.read_csv(path + 'sarcasm_merge_triple_v6.csv')
    df_out = split_df_v16(df, is_balance)
    df_out.to_csv(path + 'sarcasm_merge_triple_' + version + '.csv', index=False)
    print('Create sarcasm_merge_triple_' + version + '.csv successful.')


def load_rq_unbalance_dataset_v8(path, version, is_balance):
    """process for sarcasm_merge_triple_v6.csv"""
    df = pd.read_csv(path + 'sarcasm_merge_triple_v6.csv')
    df_out = split_df_v17(df, is_balance)
    df_out.to_csv(path + 'sarcasm_merge_triple_' + version + '.csv', index=False)
    print('Create sarcasm_merge_triple_' + version + '.csv successful.')

# if __name__ == '__main__':
#     '''
#     For instance, for the following training example :
#     "label": "SARCASM",
#     "response": "Did Kelly just call someone else messy? Baaaahaaahahahaha",
#     "context": ["X is looking a First Lady should . #classact",
#     "didn't think it was tailored enough it looked messy"]
#
#     The response tweet, "Did Kelly..." is a reply to its immediate context
#     "didn't think it was tailored..." which is a reply to "X is looking...".
#     Your goal is to predict the label of the "response" while also using the context
#     (i.e, the immediate or the full context).
#     '''
#     # count = 0
#     # statistic_words = []
#
#     # load_data_merge()
#     # load_data_merge_triple()
#     load_each_data()
