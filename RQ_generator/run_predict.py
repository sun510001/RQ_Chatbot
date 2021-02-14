# -*- coding:utf-8 -*-

from __init__ import PredictContextV2, BertBaseUnCaseV2, TrainModelConfigV2
import torch
from tf_bert_base_uncase import BuildModels, encode_examples_bert, encode_examples_roberta, triple_label_v2, split_dataset
import pandas as pd
from transformers import BertTokenizer, logging
import numpy as np
import gc

logging.set_verbosity_warning()
logging.set_verbosity_error()
logging.set_verbosity_error()


def predict_result(input, model_type):
    tokenizer = BertTokenizer.from_pretrained(PredictContextV2.TRANS_PATH, do_lower_case=True)
    if model_type in TrainModelConfigV2.BERT_LIST:
        x = encode_examples_bert(input, tokenizer, model_type, train=False).batch(1)
    elif model_type in TrainModelConfigV2.ROBERTA_LIST:
        x = encode_examples_roberta(input, tokenizer, model_type, train=False).batch(1)

    model_ver = BuildModels(PredictContextV2.M_VER, model_type)
    model2 = model_ver.build_model_1()
    model2.load_weights('{}{}{}.h5'.format(PredictContextV2.checkpoint_path, PredictContextV2.M_VER, PredictContextV2.MODEL_COUNT))
    test_result = model2.predict(x, verbose=BertBaseUnCaseV2.SHOW_MODE_NUM)
    test_index = test_result[:, 1].argmax()
    return test_index


# def predict_rq(input):
#     """predict single context"""
#     model_type = PredictContextV2.MODEL_NAME
#     output = predict_result(context, model_type)
#     # print(output)
#     return output

if __name__ == '__main__':
    context = "It's literally in the US Constitution. Trump said he didn't agree with that in the primary debates, which is scary as fuck. Okay, even ignoring the racist and/or xenophobic bullshit, anyone born in the US is a US citizen, and you can't deport US citizens."
    # print(predict_rq(context))

    # df = pd.read_csv('../dialogue_for_train_2.csv')
    # model_type = PredictContextV2.MODEL_NAME
    # df['context_merge'] = df[['context_2', 'context_1', 'robot_res']].apply(merge_response_context, args=(model_type,),
    #                                                                         axis=1)
    # result = []
    # output = predict_result(df, model_type)
    # print(output)
    # # result.append(output)
    # df['rq_result'] = output
