# -*- coding:utf-8 -*-

from __init__ import PredictContext, BertBaseUnCaseV1, score_direct_for_triple, TrainModelConfig
import torch
from tf_bert_base_uncase import BuildModels, encode_examples_bert, encode_examples_roberta,\
    triple_label_v1, split_dataset, merge_response_context
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np
import gc, sys


def predict_result(input, model_type):
    if model_type in TrainModelConfig.BERT_LIST:
        tokenizer = BertTokenizer.from_pretrained(PredictContext.TRANS_PATH, do_lower_case=True)
        x = encode_examples_bert(input, tokenizer, model_type, train=False).batch(1)
    elif model_type in TrainModelConfig.ROBERTA_LIST:
        tokenizer = RobertaTokenizer.from_pretrained(PredictContext.TRANS_PATH, do_lower_case=True)
        x = encode_examples_roberta(input, tokenizer, model_type, train=False).batch(1)

    # evaluate the model and predict the test dataset
    test_count = 't1'
    # print('evaluating {}/{} ...'.format(i, PredictContext.FOLD_NUM))
    model_ver = BuildModels(PredictContext.M_VER, model_type)
    model2 = model_ver.build_model_1()
    model2.load_weights('{}{}{}.h5'.format(PredictContext.checkpoint_path, PredictContext.M_VER, test_count))
    test_result = model2.predict(x, verbose=BertBaseUnCaseV1.SHOW_MODE_NUM)

    nd_pre_y = test_result.argmax(axis=1)
    return nd_pre_y


def predict_result_fold(input, model_type):
    if model_type in TrainModelConfig.BERT_LIST:
        tokenizer = BertTokenizer.from_pretrained(PredictContext.TRANS_PATH, do_lower_case=True)
        x = encode_examples_bert(input, tokenizer, model_type, train=False).batch(1)
    elif model_type in TrainModelConfig.ROBERTA_LIST:
        tokenizer = RobertaTokenizer.from_pretrained(PredictContext.TRANS_PATH, do_lower_case=True)
        x = encode_examples_roberta(input, tokenizer, model_type, train=False).batch(1)
    test_result = np.zeros((input.shape[0], 3))

    for i in range(1, PredictContext.FOLD_NUM + 1):
        # evaluate the model and predict the test dataset
        test_count = 't' + str(i)
        print('evaluating {}/{} ...'.format(i, PredictContext.FOLD_NUM))
        model_ver = BuildModels(PredictContext.M_VER, model_type)
        model2 = model_ver.build_model_1()
        model2.load_weights('{}{}{}.h5'.format(PredictContext.checkpoint_path, PredictContext.M_VER, test_count))
        test_result += model2.predict(x, verbose=BertBaseUnCaseV1.SHOW_MODE_NUM)
        del model2
        gc.collect()

    nd_pre_y = test_result.argmax(axis=1)
    return nd_pre_y


def cla_predict(input, model_type):
    if model_type in TrainModelConfig.BERT_LIST:
        tokenizer = BertTokenizer.from_pretrained(PredictContext.TRANS_PATH, do_lower_case=True)
        x = encode_examples_bert(input, tokenizer, model_type, train=False).batch(1)
    elif model_type in TrainModelConfig.ROBERTA_LIST:
        tokenizer = RobertaTokenizer.from_pretrained(PredictContext.TRANS_PATH, do_lower_case=True)
        x = encode_examples_roberta(input, tokenizer, model_type, train=False).batch(1)
    test_result = np.zeros((input.shape[0], 3))

    for i in range(1, PredictContext.FOLD_NUM + 1):
        # evaluate the model and predict the test dataset
        test_count = 't' + str(i)
        print('evaluating {}/{} ...'.format(i, PredictContext.FOLD_NUM))
        model_ver = BuildModels(PredictContext.M_VER, model_type)
        model2 = model_ver.build_model_1()
        model2.load_weights('{}{}{}.h5'.format(PredictContext.checkpoint_path, PredictContext.M_VER, test_count))
        test_result += model2.predict(x, verbose=BertBaseUnCaseV1.SHOW_MODE_NUM)
        del model2
        gc.collect()

    nd_pre_y = test_result.argmax(axis=1)
    return nd_pre_y


if __name__ == '__main__':
    # """predict test data"""
    # # load the dataset
    # df_train_load = triple_label_v1(pd.read_csv(BertBaseUnCaseV1.PATH))
    # values_count = df_train_load['tri_label'].value_counts()
    # print("Literal, rq, sarcasm count:", values_count)
    #
    # # get bert tokenizer
    # tokenizer = BertTokenizer.from_pretrained(BertBaseUnCaseV1.TRANS_PATH, do_lower_case=True)
    # train_data, test_data = split_dataset(df_train_load)
    # model_type = PredictContext.MODEL_NAME
    # # test dataset
    # if (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):
    #     test, test_label = encode_examples_bert(test_data, tokenizer, model_type)
    # elif (model_type == "roberta-base") or (model_type == "roberta-large"):
    #     test, test_label = encode_examples_roberta(test_data, tokenizer, model_type)
    # ds_test_encoded = test.batch(BertBaseUnCaseV1.BATCH_SIZE)
    #
    # output = predict_result(ds_test_encoded, model_type)
    # print(output)

    """predict evaluation"""
    # df = pd.read_csv('../dialogue_for_train.csv')
    # model_type = PredictContext.MODEL_NAME
    # df['context_merge'] = df[['context_2', 'context_1']].apply(merge_response_context, args=(model_type,), axis=1)
    # result = []
    # # for each in predict_list:
    # output = predict_result_fold(df, model_type)
    # print(output)
    # df['triple_result'] = output
    # print()
    # df.to_csv('../dialogue_for_train_after_cla.csv', index=False)

    """for chatbot"""
    # test_list = ['Not being a douchenozzle', 'do you like apple', "I like to eat apples but I'm a ginger so it's not for me."]
    input = list(sys.argv)
    print(input)
    col = ["context_2", "context_1"]
    df = pd.DataFrame(data=[input[1:]], columns=col)
    model_type = PredictContext.MODEL_NAME
    df['context_merge'] = df[['context_2', 'context_1']].apply(merge_response_context, args=(model_type,), axis=1)
    y = cla_predict(df, model_type)
    print(y[0])
    with open("chatbot_out.txt", 'w') as f:
        f.write(str(y[0]))