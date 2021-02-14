# -*- coding:utf-8 -*-

import re
from emoji import demojize
import pandas as pd
import numpy as np

from __init__ import PSD_VERSION, PATH_DATASET

count = [1]


def detect_rq(row, max):
    print("No. {}/{}: ".format(count[0], max))
    count[0] += 1
    for i, value in enumerate(row['context']):
        print('Context {} -> {}'.format(i, value))
    print('Response  => {}'.format(row['response']))

    try:
        result = int(input("<1 if it is rq>: "))
    except BaseException as e:
        result = 0
    print("result: {}\n".format(result))
    if result == 1:
        row['rq'] = 1
    else:
        row['rq'] = 0
    # row['rq'] = 1
    return row


def del_emoji(text):
    text = re.sub(r'<URL>', ':URL:', text)
    return demojize(text)
    # return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)


def proc_rq(row):
    temp_str = ""
    row['index'] = row.name
    for i, value in enumerate(row['context']):
        temp_str += "Context {} -> {} <br>".format(i, value)
    temp_str += "Response  => {}".format(row['response'])
    row['mer_context'] = del_emoji(temp_str)
    return row


def creat_csv(path, path_df1, path_df2):
    """
    create csv for amazon mturk
    :param path:
    """
    df_1 = pd.read_json(path, orient='records', lines=True)
    df_1['rq'] = 0
    df_2 = df_1.loc[(df_1['sarcasm'] == 0) & (df_1['question'] == 1)]
    # df_2 = df_2.sample(n=10)
    max_count = df_2.shape[0]

    # create v5 csv
    df_2 = df_2.apply(proc_rq, axis=1)
    # df_2[['mer_context']].to_csv(path_df1, index=False)
    df_2[['mer_context']].to_csv(path_df2)

    # df_2 = df_2.apply(detect_rq, args=(max_count,), axis=1)
    # df_3 = pd.concat([df_1.loc[:, df_1.columns != 'rq'], df_2[['rq']]], axis=1)
    # df_3['rq'] = df_3['rq'].fillna(0).astype('int64')
    # df_4 = df_3.loc[df_3['rq'] == 1]
    # return df_3


def load_proc_csv(path):
    df_1 = pd.read_json(path, orient='records', lines=True)
    df_2 = df_1.loc[df_1['rq'] == 1]
    print(df_2)


def process_mturk_result(path_in, path_out, path_df2, path):
    df = pd.read_csv(path)

    df_simple = df[['HITId', 'Input.mer_context', 'Answer.rq.label']]
    def process_result(input):
        """
        find the most frequent value in the grouped answer.
        :param input:
        :return:
        """
        most_fre_val = input['Answer.rq.label'].value_counts().idxmax()
        output = [input['Input.mer_context'].values[0], most_fre_val]
        return output

    df_data = df_simple.groupby('HITId').apply(process_result).to_list()
    df_result = pd.DataFrame(df_data, columns=['mer_context', 'rq'])

    # load rq index dataset
    df_index_context = pd.read_csv(path_df2)
    mergedStuff = pd.merge(df_result, df_index_context, on=['mer_context'], how='right')
    mergedStuff = mergedStuff.set_index(mergedStuff.columns[2])

    # load orignal data add rq label by index
    df_1 = pd.read_json(path_in, orient='records', lines=True)
    df_rq_yes_index = mergedStuff.loc[mergedStuff['rq'] == 'Yes']
    df_1['rq'] = 0
    # def label_rq():

    df_1['rq'].iloc[df_rq_yes_index.index] = 1
    # df_1.to_json(path_out, orient='records', lines=True)
    df_1.to_csv(path_out, index=False)

    # a_test = list(df_1.loc[df_1['rq'] == 1].index.tolist())
    # b_test = list(df_rq_yes_index.index.tolist())
    # b_test.sort()
    # # c_test = sorted(b_test)
    # d = []
    # for each in b_test:
    #     if each in a_test:
    #         pass
    #     else:
    #         d.append(each)


    return df_result


if __name__ == '__main__':
    """
    v1: annotated by myself
    v7: annotated by amazon mturk workers with index
    v6: v7 without index
    """
    path_in = PATH_DATASET + 'sarcasm_merge_triple_' + PSD_VERSION + '.json'
    path_out = PATH_DATASET + 'sarcasm_merge_triple_v5.csv'
    path_df1 = PATH_DATASET + 'sarcasm_merge_rq_v6.csv'
    path_df2 = PATH_DATASET + 'sarcasm_merge_rq_v7.csv'
    path_mturk_result = PATH_DATASET + 'Batch_4266013_batch_results.csv'

    # df1 = creat_csv(path_in, path_df1, path_df2)
    process_mturk_result(path_in, path_out, path_df2, path_mturk_result)

