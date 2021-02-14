# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, \
    confusion_matrix


def plot_confusion_matrix(cm, class_names, name):
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
    plt.ylabel('Human label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    figure.savefig('confusion_matrix_for_{}.png'.format(name))


def score_direct_for_triple(nd_true_y, nd_test_y, name):
    print(classification_report(nd_true_y, nd_test_y))
    confusion = confusion_matrix(nd_true_y, nd_test_y)
    plot_confusion_matrix(confusion, ['Literal', 'RQ', 'Sarcasm'], name)


def evaluate_chat_result(df, df_com):
    def select(row):
        flag = 0
        if (row['rq'] == 1) or (row['sarcasm'] == 1 and row['question'] == 1):
            flag = 1
        elif row['sarcasm'] == 1:
            flag = 2
        return flag

    df_com['real'] = df_com[['sarcasm', 'question', 'rq']].apply(select, axis=1)
    df_group = df.groupby("HITId", sort=False).mean().reset_index(drop=True)

    """export the evalutation data for the paper"""
    df_1 = df_group[
        ["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a", "Answer.hl_b", "Answer.hl_c",
         "Answer.hl_d", "Answer.is_rq.a", "Answer.is_rq.b", "Answer.is_rq.c", "Answer.is_rq.d", "Answer.is_s.a",
         "Answer.is_s.b", "Answer.is_s.c",
         "Answer.is_s.d"]]
    df_2 = df_com[["context_2", "context_1", "response", "robot_res", "sg_res", "rq_res", "triple_result"]]
    df_3 = pd.merge(df_1, df_2, left_index=True, right_index=True)

    head_list = ["Context 2", "Context 1", "Human", "Literal", "Sarcasm", "RQ", "Classification result",
                 "Appropriateness.human", "Appropriateness.literal", "Appropriateness.sarcasm", "Appropriateness.RQ",
                 "Human-likeness.human", "Human-likeness.literal",
                 "Human-likeness.sarcasm", "Human-likeness.RQ", "Is_RQ.human", "Is_RQ.literal", "Is_RQ.sarcasm",
                 "Is_RQ.RQ",
                 "Is_sarcastic.human", "Is_sarcastic.literal", "Is_sarcastic.sarcasm", "Is_sarcastic.RQ"]

    # df_1 = df_1.reset_index(drop=True, inplace=True)
    # df_2 = df_2.reset_index(drop=True, inplace=True)

    def a(input):
        print(input.name)
        for i in range(len(input)):
            if i < 7:
                print('{} : {}'.format(head_list[i], input[i]))
            else:
                print('{} : {:.2f}'.format(head_list[i], input[i]))
        print("\sun/")

    # df_2.apply(a, axis=1)
    # df_2.to_csv("eva_chat_latex_merge_c.csv")
    # df_1.to_csv("eva_chat_latex_merge_d.csv", float_format='%.2f')

    merge_df = df_3[["context_2", "context_1", "response", "robot_res", "sg_res", "rq_res", "triple_result",
                         "Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a", "Answer.hl_b",
                         "Answer.hl_c", "Answer.hl_d", "Answer.is_rq.a", "Answer.is_rq.b", "Answer.is_rq.c",
                         "Answer.is_rq.d", "Answer.is_s.a", "Answer.is_s.b", "Answer.is_s.c", "Answer.is_s.d"]]
    merge_df.apply(a, axis=1)

    """for the slide in presentation, data from df_app and df_hl"""
    # plt.figure(figsize=(12, 4))
    # columns = ["Literal", "Sarcasm", "RQ"]
    data_app = [3.39, 3.40, 3.47]
    data_hl = [3.56, 3.43, 3.39]

    # df_s_app = pd.DataFrame(data=[data_app], columns=columns)
    # df_s_hl = pd.DataFrame(data=[data_hl], columns=columns)

    # app_c = df_s_app['flag_app'].value_counts()
    # hl_c = df_s_hl['flag_hl'].value_counts()

    index = ['appropriateness', 'human-likeness']
    # app = app_c.sort_index().tolist()
    # hl = hl_c.sort_index().tolist()
    # plt.figure(figsize=(12, 3))
    labels = ["Literal", "Sarcasm", "RQ"]
    df_pic = pd.DataFrame([data_app, data_hl], columns=labels, index=index)
    df_pic.plot(kind='bar', rot=0, figsize=(12, 6), ylim=[3.3, 3.6])  #
    plt.legend()
    plt.savefig('result_pic/eva_slice_present.png')
    plt.clf()


    """get precision recall and f1"""
    df_mean_rows = df_group[["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a",
                             "Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].mean()
    df_is_rows = df_group[["Answer.is_rq.a", "Answer.is_rq.b", "Answer.is_rq.c", "Answer.is_rq.d", "Answer.is_s.a",
                           "Answer.is_s.b", "Answer.is_s.c", "Answer.is_s.d"]].round(0).reset_index(drop=True)
    df_is_rows = df_is_rows.sum(axis=0)

    df_app = df_group[["Answer.app_b", "Answer.app_c", "Answer.app_d"]].idxmax(axis=1).reset_index(drop=True)
    df_hl = df_group[["Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].idxmax(axis=1).reset_index(drop=True)

    def clean_col(row):
        """from the label of situation classifier: literal=0; rq=1; sarcasm=2"""
        if row == "Answer.app_b":
            return 0
        elif row == "Answer.app_c":
            return 2
        else:
            return 1

    ser_app = df_app.map(clean_col)
    ser_app.name = "anno_app"

    def clean_col_2(row):
        if row == "Answer.hl_b":
            return 0
        elif row == "Answer.hl_c":
            return 2
        else:
            return 1

    ser_hl = df_hl.map(clean_col_2)
    ser_hl.name = "anno_hl"
    df_list = [df_com['real'], df_com['triple_result'], ser_app, ser_hl]
    df_4 = pd.concat(df_list, axis=1)

    app_score = score_direct_for_triple(df_4['anno_app'], df_4['triple_result'], "app")
    hl_score = score_direct_for_triple(df_4['anno_hl'], df_4['triple_result'], "hl")
    true_score = score_direct_for_triple(df_4['real'], df_4['triple_result'], "real")

    """for report, when literal get four app and hl results of human, literal, rq, sarcasm; when rq, sarcasm"""
    df_group_merge = pd.concat([df_group[["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a",
                                          "Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]], df_com['triple_result']],
                               axis=1)

    d_literal = df_group_merge[["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a",
                                "Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].loc[df_com['triple_result'] == 0].mean()

    d_rq = df_group_merge[["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a",
                           "Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].loc[
        df_com['triple_result'] == 1].mean()

    d_sarc = df_group_merge[["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a",
                             "Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].loc[
        df_com['triple_result'] == 2].mean()

    """get the rows when situaiton classifier result equals human annotation"""
    df_4['flag_app'] = -1
    df_4['flag_hl'] = -1
    df_4['mean_app'] = 0
    df_4['mean_hl'] = 0

    def find_same(row):
        if row['triple_result'] == row['anno_app']:
            row['flag_app'] = row['anno_app']
        if row['triple_result'] == row['anno_hl']:
            row['flag_hl'] = row['anno_hl']

        return row

    df4 = df_4.apply(find_same, axis=1)
    df4_r_app = df4.loc[df4['flag_app'] != -1]
    df4_r_app_index = df4_r_app.index.values.tolist()
    df4_r_hl = df4.loc[df4['flag_hl'] != -1]
    df4_r_hl_index = df4_r_hl.index.values.tolist()

    df_g_app = df_group[["Answer.app_b", "Answer.app_c", "Answer.app_d"]].iloc[df4_r_app_index, :]
    df_g_hl = df_group[["Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].iloc[df4_r_hl_index, :]
    df_g_app = pd.concat([df_g_app, df4_r_app['flag_app']], axis=1)
    df_g_hl = pd.concat([df_g_hl, df4_r_hl['flag_hl']], axis=1)

    # df_g_app['mean'] = 0
    # df_g_hl['mean'] = 0
    #
    # def cal_mean(row):
    #     if row[3] == 1:  # rq
    #         row[4] = row[2]
    #     elif row[3] == 2:  # sarcasm
    #         row[4] = row[1]
    #     else:  # literal
    #         row[4] = row[0]
    #     return row
    #
    # df_g_app = df_g_app.apply(cal_mean, axis=1)
    # df_g_hl = df_g_hl.apply(cal_mean, axis=1)
    df_g_app['mean'] = df_g_app[["Answer.app_b", "Answer.app_c", "Answer.app_d"]].max(axis=1)
    df_g_hl['mean'] = df_g_hl[["Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].max(axis=1)

    df_g_app_mean = df_g_app.groupby('flag_app', as_index=False)['mean'].mean()
    df_g_hl_mean = df_g_hl.groupby('flag_hl', as_index=False)['mean'].mean()

    app_c = df4_r_app['flag_app'].value_counts()
    hl_c = df4_r_hl['flag_hl'].value_counts()

    df_group.to_csv("dialogue_for_train_after_rq_proc_sec_mean.csv")

    # is_list = df_is_rows.values.tolist()
    # slice_list = df_mean_rows.values.tolist()
    # r = df_mean_rows.index + w * (i - len(col_list) / 2)

    # """scale 1-5 for appropriateness and human-likeness"""
    # plt.figure(figsize=(12, 4))
    # data = is_list[:4]
    # labels = ["human response", "literal response", "sarcasm", "rq"]
    # plt.bar(range(len(data)), data, tick_label=labels)
    # plt.legend()
    # plt.savefig('eva_is_rq_for_pre.png')
    # plt.clf()
    #
    # data = is_list[4:]
    # labels = ["human response", "literal response", "sarcasm", "rq"]
    # plt.bar(range(len(data)), data, tick_label=labels)
    # plt.legend()
    # plt.savefig('eva_is_sarc_for_pre.png')
    # plt.clf()

    # data = slice_list[:4]
    # labels = ["human response", "literal response", "sarcasm", "rq"]
    # plt.bar(range(len(data)), data, tick_label=labels)
    # plt.ylim((3, 4))
    # plt.legend()
    # plt.savefig('eva_slice_app.png')
    # plt.clf()
    #
    # data = slice_list[4:]
    # labels = ["human response", "literal response", "sarcasm", "rq"]
    # plt.bar(range(len(data)), data, tick_label=labels)
    # plt.ylim((3, 4))
    # plt.legend()
    # plt.savefig('eva_slice_hl.png')
    # plt.clf()

    # """=========get count of current output========="""
    # index = ['appropriateness', 'human-likeness']
    # app = app_c.sort_index().tolist()
    # hl = hl_c.sort_index().tolist()
    # # plt.figure(figsize=(12, 3))
    # labels = ["Literal", "RQ", "Sarcasm"]
    # df_pic = pd.DataFrame([app, hl], columns=labels, index=index)
    # df_pic.plot(kind='bar', rot=0, figsize=(12, 6))  # , ylim=[0, 13]
    # plt.legend()
    # plt.savefig('result_pic/eva_slice_pred.png')
    # plt.clf()

    """=========get means of current output========="""
    # index = ['appropriateness', 'human-likeness']
    # app = df_g_app_mean['mean'].tolist()
    # hl = df_g_hl_mean['mean'].tolist()
    # # plt.figure(figsize=(12, 3))
    # labels = ["Literal", "RQ", "Sarcasm"]
    # df_pic = pd.DataFrame([app, hl], columns=labels, index=index)
    # df_pic.plot(kind='bar', ylim=[3, 5], rot=0, figsize=(12, 6))  # , ylim=[0, 13]
    # plt.legend()
    # plt.savefig('result_pic/eva_slice_mean_pred.png')
    # plt.clf()
    #
    # print()


if __name__ == '__main__':
    """analyze the evaluation result of generated data after all generation and human evaluation"""
    df_complete = pd.read_csv('dia_for_mturk.csv')
    df = pd.read_csv("Batch_from_mturk.csv")

    df = df.loc[~df[["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d"]].eq(
        df[["Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d"]].iloc[:, 0], axis=0).all(1)]
    df = df.loc[~df[["Answer.hl_a", "Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].eq(
        df[["Answer.hl_a", "Answer.hl_b", "Answer.hl_c", "Answer.hl_d"]].iloc[:, 0], axis=0).all(1)]

    df = df[["HITId", "Answer.app_a", "Answer.app_b", "Answer.app_c", "Answer.app_d", "Answer.hl_a", "Answer.hl_b",
             "Answer.hl_c", "Answer.hl_d", "Answer.is_rq.a", "Answer.is_rq.b", "Answer.is_rq.c", "Answer.is_rq.d",
             "Answer.is_s.a", "Answer.is_s.b", "Answer.is_s.c", "Answer.is_s.d"]]

    evaluate_chat_result(df, df_complete)
