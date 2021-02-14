# -*- coding:utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from transformers import TFBertModel, BertConfig, RobertaConfig, TFRobertaModel
from __init__ import BertBaseUnCaseV1, PATH_PIC, PSD_VERSION, TrainModelConfig, PATH_TRANS_INPUT


if not BertBaseUnCaseV1.SHOW_MODE:
    from transformers import logging

    logging.set_verbosity_warning()
    logging.set_verbosity_error()
    logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, verbose=1):
        self.model = model
        self.path = path
        self.best_loss = np.inf
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            if self.verbose == 1:
                print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss


def merge_label(input):
    out = 0  # literal
    if input['rq'] == 1 or (input['sarcasm'] == 1 and input['question'] == 1):
        out = 1  # rq
    elif input['sarcasm'] == 1:
        out = 2  # sarcasm
    return out


def merge_response_context(input, model_type):
    """
    just add sep between two sentences, [CLS] and the last [SEP] will be added by encode_plus add_special_tokens,
    only for bert.
    roberta special token is </s>
    """
    if model_type in TrainModelConfig.BERT_LIST:
        output = "[SEP]".join(input)
    elif model_type in TrainModelConfig.ROBERTA_LIST:
        output = "</s></s>".join(input)
    return output


def triple_label_v1(df):
    """convert the contexts"""
    df['tri_label'] = df[['sarcasm', 'question', 'rq']].apply(merge_label, axis=1)
    # df['context_merge'] = df['response']

    return df[['context_merge', 'tri_label']]


def triple_label_v2(df, model_type):
    """
    use the splited response and contexts
    :param df:
    :return:
    """
    df['tri_label'] = df[['sarcasm', 'question', 'rq']].apply(merge_label, axis=1)
    df_temp = df.loc[df['tri_label'] == 1]
    df['context_merge'] = df[['context_2', 'context_1']].apply(merge_response_context, args=(model_type,),
                                                                           axis=1)
    return df[['context_merge', 'tri_label']]


def split_dataset(df):
    train_set, test_set = train_test_split(df,
                                           stratify=df['tri_label'],
                                           test_size=0.2,
                                           random_state=43)

    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)


def bert_map_example_to_dict(input_ids, attention_masks, token_type_ids, *args):
    if args:
        return {
                   "input_ids": input_ids,
                   "attention_mask": attention_masks,
                   "token_type_ids": token_type_ids,
               }, args
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids,
        }


def roberta_map_example_to_dict(input_ids, attention_masks, *args):
    if args:
        return {
                   "input_ids": input_ids,
                   "attention_mask": attention_masks,
               }, args
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }


def convert_example_to_feature(review, tokenizer, model_type):
    # combine step for tokenization, WordPiece vector mapping,
    # adding special tokens as well as truncating reviews longer than the max length
    if model_type in TrainModelConfig.BERT_LIST:
        return tokenizer.encode_plus(review,
                                     add_special_tokens=True,  # add [CLS], [SEP]
                                     max_length=BertBaseUnCaseV1.MAXLEN,  # max length of the text that can go to BERT
                                     # pad_to_max_length=True,  # add [PAD] tokens
                                     padding='max_length',
                                     return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                     truncation=True,
                                     )

    elif model_type in TrainModelConfig.ROBERTA_LIST:
        return tokenizer.encode_plus(review,
                                     add_special_tokens=True,
                                     max_length=BertBaseUnCaseV1.MAXLEN,
                                     padding='max_length',
                                     return_attention_mask=True,
                                     truncation=True,
                                     )


def encode_examples_bert(ds, tokenizer, model_type, train=True):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    if train:
        label_list = []

        # if (limit > 0):
        #     ds = ds.take(limit)

        for index, row in ds.iterrows():
            review = row["context_merge"]
            label = row["tri_label"]
            bert_input = convert_example_to_feature(review, tokenizer, model_type)

            input_ids_list.append(bert_input['input_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            label_list.append([label])

        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(bert_map_example_to_dict), \
               label_list
    else:
        for index, row in ds.iterrows():
            review = row["context_merge"]
            bert_input = convert_example_to_feature(review, tokenizer, model_type)

            input_ids_list.append(bert_input['input_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            token_type_ids_list.append(bert_input['token_type_ids'])

        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list)).map(bert_map_example_to_dict)


def encode_examples_roberta(ds, tokenizer, model_type, train=True):
    input_ids_list = []
    # token_type_ids_list = np.zeros((ds.shape[0], BertBaseUnCaseV1.MAXLEN), dtype='int32').tolist()
    attention_mask_list = []

    if train:
        label_list = []

        for index, row in ds.iterrows():
            review = row["context_merge"]
            label = row["tri_label"]
            bert_input = convert_example_to_feature(review, tokenizer, model_type)

            input_ids_list.append(bert_input['input_ids'])
            attention_mask_list.append(bert_input['attention_mask'])

            label_list.append([label])

        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, label_list)).map(roberta_map_example_to_dict), label_list
    else:
        for index, row in ds.iterrows():
            review = row["context_merge"]
            bert_input = convert_example_to_feature(review, tokenizer, model_type)

            input_ids_list.append(bert_input['input_ids'])
            attention_mask_list.append(bert_input['attention_mask'])

        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list)).map(roberta_map_example_to_dict)


class BuildModels:
    def __init__(self, version, model_type):
        """different version use different model"""
        self.version = version
        self.model_type = model_type

    def build_model_1(self, verbose=False):
        """initialization the model"""
        if self.model_type in TrainModelConfig.BERT_LIST:
            config = BertConfig.from_pretrained("{}{}/config.json".format(PATH_TRANS_INPUT, self.model_type),
                                                num_labels=BertBaseUnCaseV1.N_CLASS)
            bert_model = TFBertModel.from_pretrained("{}{}/tf_model.h5".format(PATH_TRANS_INPUT, self.model_type),
                                                     config=config)
            bert_model.trainable = False
            input_ids_layer = Input(shape=(BertBaseUnCaseV1.MAXLEN,), dtype=np.int32, name='input_ids')
            input_mask_layer = Input(shape=(BertBaseUnCaseV1.MAXLEN,), dtype=np.int32, name='attention_mask')
            input_token_type_layer = Input(shape=(BertBaseUnCaseV1.MAXLEN,), dtype=np.int32, name='token_type_ids')
            input_layer_list = [input_ids_layer, input_mask_layer, input_token_type_layer]
            bert_layer = bert_model(input_layer_list)[0]

        elif self.model_type in TrainModelConfig.ROBERTA_LIST:
            config = RobertaConfig.from_pretrained("{}{}/config.json".format(PATH_TRANS_INPUT, self.model_type),
                                                   num_labels=BertBaseUnCaseV1.N_CLASS)
            bert_model = TFRobertaModel.from_pretrained("{}{}/tf_model.h5".format(PATH_TRANS_INPUT, self.model_type),
                                                        config=config)
            bert_model.trainable = False
            input_ids_layer = Input(shape=(BertBaseUnCaseV1.MAXLEN,), dtype=np.int32, name='input_ids')
            input_mask_layer = Input(shape=(BertBaseUnCaseV1.MAXLEN,), dtype=np.int32, name='attention_mask')

            input_layer_list = [input_ids_layer, input_mask_layer]
            bert_layer = bert_model(input_layer_list)[0]

            # input_layer_list = input_ids_layer
            # bert_layer = bert_model(input_layer_list)[0]

        if self.version == "v1":
            flat_layer = Flatten()(bert_layer)
            out = Dropout(0.2)(flat_layer)
            # out = Dense(256, activation='relu')(dropout_layer)
        elif self.version == "v2":
            out = LSTM(BertBaseUnCaseV1.hidden_size, dropout=0.2)(bert_layer)
        elif self.version == "v3":
            flat_layer = Flatten()(bert_layer)
            dense_layer = Dense(BertBaseUnCaseV1.hidden_size, activation='relu')(flat_layer)
            out = Dropout(0.2)(dense_layer)
        elif self.version == "v4":
            bi_layer = Bidirectional(LSTM(BertBaseUnCaseV1.hidden_size, dropout=0.2, return_sequences=True))(bert_layer)
            bi_layer = Bidirectional(LSTM(BertBaseUnCaseV1.hidden_size))(bi_layer)
            dropout_layer = Dropout(0.2)(bi_layer)
            out = Dense(256, activation='relu')(dropout_layer)

        if BertBaseUnCaseV1.VER == 'v5':
            dense_output = Dense(BertBaseUnCaseV1.N_CLASS, activation='sigmoid')(out)
        else:
            dense_output = Dense(BertBaseUnCaseV1.N_CLASS, activation='softmax')(out)
        model = Model(inputs=input_layer_list, outputs=dense_output)

        # compile and fit
        if BertBaseUnCaseV1.VER == 'v5':
            optimizer = optimizers.Adam(learning_rate=BertBaseUnCaseV1.lr)
            loss = losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = metrics.SparseCategoricalAccuracy('accuracy')
        else:
            optimizer = optimizers.Adam(learning_rate=BertBaseUnCaseV1.lr)
            loss = losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        if verbose:
            model.summary()
            dot_img_file = 'model_{}_{}{}.png'.format(self.model_type, BertBaseUnCaseV1.VER, self.version)
            plot_model(model, to_file=dot_img_file, show_shapes=False, show_layer_names=False)
            print("output the picture of the model.")
        return model


def train_model(model, train, val, test_count, model_type, checkpoint_path):
    """training model"""
    cp_path = '{}{}{}.h5'.format(checkpoint_path, BertBaseUnCaseV1.m_ver, test_count)
    ckpt = CustomModelCheckpoint(model, cp_path, verbose=BertBaseUnCaseV1.SHOW_MODE_NUM)

    es = EarlyStopping(monitor='val_loss', patience=2, verbose=BertBaseUnCaseV1.SHOW_MODE_NUM)
    bert_his = model.fit(train, epochs=BertBaseUnCaseV1.EPOCHS, validation_data=val, callbacks=[ckpt, es],
                         verbose=BertBaseUnCaseV1.SHOW_MODE_NUM)


    history_dict = bert_his.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}{}-{}{}{}{}.png'.format(PATH_PIC, model_type, PSD_VERSION, BertBaseUnCaseV1.VER,
                                           BertBaseUnCaseV1.m_ver, test_count))
    plt.clf()
    plt.close('ALL')
